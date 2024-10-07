# Typical
import cv2
import math
import numpy as np
import time as Time
import threading
import sys
import os

# FaceMesh
from FaceMeshModule import FaceMeshDetector as facelms

# QtWindow UI
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QGridLayout, QLabel, QSlider, QVBoxLayout, QFileDialog, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import json
import threading

# OF
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyqtgraph as pg
from collections import deque
from threading import Thread
from sklearn.mixture import GaussianMixture


# MSRCR
# from retinex import retinex_MSRCR


# =====================================================
# ===================== SETUP =========================
# =====================================================

# Initialize background subtractor
print("BG Subtractor")
_bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=20, detectShadows=True)

# Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization)
print("CLAHE")
clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(2, 2))

# HSV ranges for color isolation of skin-tone colors (above/below the reference color)
lower_bound = np.array([0, 100, 100])
upper_bound = np.array([10, 255, 255])

# Fields
_fps = 30

# Default skin tone reference color
def get_bgr_from_reference_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.mean(image_rgb, axis=(0, 1))

# Get the average BGR color from the reference image
bgr_color_ref = get_bgr_from_reference_image("reference.png")
# Convert the average BGR color to HSV
average_color_bgr = np.uint8([[[bgr_color_ref[0], bgr_color_ref[1], bgr_color_ref[2]]]])
HsvRef = cv2.cvtColor(average_color_bgr, cv2.COLOR_BGR2HSV)[0][0]

CombinedGridFrame = None
TOTAL_FRAMES = 5
COLUMNS_FRAMES = 3


# =====================================================
# ================== CGF WINDOW =======================
# =====================================================

# Declare global variable
config_window = None
config_app = None
# Global configuration dictionary
config_lock = threading.Lock()
current_config = {
    'hue_range': 20,
    'hue_offset': 180,
    'sat_low': 25,
    'sat_high': 255,
    'val_low': 85,
    'val_high': 255,
    'of_smoothing': 50,
    'eye_area_smoothing': 50,
    'eye_bounds_expand': 10,
    'of_process_stage': 1,
    # 'area_process_stage': 3,
    'median_blur_knl': 3,
    'gauss_blur_knl_pre': 3,
    'dilation_knl_mask': 3,
    'gauss_blur_knl_dil': 3,
}

def get_config_vals():
    global config_window
    if config_window is not None:
        return config_window.get_window_config()
    print("Config window is not initialized. Returning None!")
    return None
            

class ConfigWindow(QMainWindow):
    def __init__(self):
        global config_window
        
        # 1. Initialize the main window
        super().__init__()
        config_window = self # SINGLETON

        self.setWindowTitle("FT/OF Preproc Testing")
        self.setGeometry(100, 100, 800, 600)

        scroll = QScrollArea()
        self.setCentralWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)

        layout = QVBoxLayout(content)

        slider_widget = QWidget()
        slider_layout = QGridLayout(slider_widget)
        layout.addWidget(slider_widget)

        # 2. Define sliders with (Name, Minimum Value, Maximum Value)
        sliders = [
            # HSV color isolation
            ('Hue Range', 0, 90),
            ('Hue Offset', 0, 360),
            # HSV thresholds
            ('Saturation Low', 0, 255),
            ('Saturation High', 0, 255),
            ('Value Low', 0, 255),
            ('Value High', 0, 255),
            # OF
            ('OF Smoothing Factor', 10, 100),       # divided by 10
            ('Eye-Area Smoothing Factor', 10, 100), # divided by 10
            ('Eye Bounds Expand', 10, 50),
            # ('Area Process Stage', 1, 4),
            ('OF Process Stage', 1, 4),
            # Kernels
            ('Median Blur K (HSV mask)', 0, 25),
            ('Gauss Blur K (OG image)', 0, 25),
            ('Dilation K (HSV-Masked)', 0, 25),
            ('Gauss Blur K (dilated eyes)', 0, 25),
            #('CLAHE Clip Limit', 20, 100),
            #('CLAHE Grid Size', 8, 16),
            #('BG Sub Learning Rate', 5, 100),
            #('BG Sub History', 500, 1000),
            #('BG Sub Var Threshold', 16, 100),
        ]

        self.sliders = {}       # Dictionary to store sliders
        self.value_labels = {}  # Dictionary to store value labels

        # 3. Add sliders to the layout
        for i, (name, default, maximum) in enumerate(sliders):
            # Create slider
            label = QLabel(name)
            slider = QSlider(Qt.Horizontal)
            
            # Set slider range
            slider.setRange(0, maximum)

            # Get global default value
            config_key = self.slider_name_to_config_key(name)
            value = current_config.get(config_key, default)
            print(f"The default value for {config_key} is {value}")

            # Set slider value and store it
            if value is not None:
                # Set global def value
                slider.setValue(int(value))
                
                # Store slider 
                self.sliders[name] = slider
                
                # Create value label
                value_label = QLabel(str(value))
                self.value_labels[name] = value_label
                
                # Callback handlers
                slider.valueChanged.connect(lambda value, name=name: self.update_value_label(name, value))
                slider.valueChanged.connect(self.update_config)

            # Layout
            row = i % 8
            col = i // 8 * 3
            slider_layout.addWidget(label, row, col)
            slider_layout.addWidget(slider, row, col + 1)
            slider_layout.addWidget(value_label, row, col + 2)

        # 5. Add Save and Load buttons
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        button_layout = QVBoxLayout()
        save_button = QPushButton("Save Configuration")
        load_button = QPushButton("Load Configuration")
        save_button.clicked.connect(self.save_config)
        load_button.clicked.connect(self.load_config)
        button_layout.addWidget(save_button)
        button_layout.addWidget(load_button)
        layout.addLayout(button_layout)

    # === CLASS FUNCTIONS ===
    def update_value_label(self, name, value):
        if name in ['OF Smoothing Factor', 'Eye-Area Smoothing Factor']:
            display_value = value / 10.0
            self.value_labels[name].setText(f"{display_value:.1f}")
        else:
            self.value_labels[name].setText(str(value))
    
    def update_config(self):
        global current_config
        with config_lock:
            current_config = self.get_window_config()

    @staticmethod
    def ensure_odd(value):
        if value == 0:
            return 0  # Return 0 if the input is 0
        return value if value % 2 == 1 else value + 1
    
    def get_window_config(self):
        """Retrieve current values from all sliders."""
        config = {
            # HSV color isolation
            'hue_range': self.sliders['Hue Range'].value(),
            'hue_offset': self.sliders['Hue Offset'].value(),
            # HSV thresholds
            'sat_low': self.sliders['Saturation Low'].value(),
            'sat_high': self.sliders['Saturation High'].value(),
            'val_low': self.sliders['Value Low'].value(),
            'val_high': self.sliders['Value High'].value(),
            # OF
            'of_smoothing': max(10, self.sliders['OF Smoothing Factor'].value()),
            'eye_area_smoothing': max(10, self.sliders['Eye-Area Smoothing Factor'].value()),
            'eye_bounds_expand': self.sliders['Eye Bounds Expand'].value(),
            'of_process_stage': self.sliders['OF Process Stage'].value(),
            # 'area_process_stage': self.sliders['Area Process Stage'].value(),
            # Kernels
            'median_blur_knl': self.ensure_odd(self.sliders['Median Blur K (HSV mask)'].value()),
            'gauss_blur_knl_pre': self.ensure_odd(self.sliders['Gauss Blur K (OG image)'].value()),
            'dilation_knl_mask': max(1, self.sliders['Dilation K (HSV-Masked)'].value()),
            'gauss_blur_knl_dil': self.ensure_odd(self.sliders['Gauss Blur K (dilated eyes)'].value()),
            #'clahe_clip_limit': self.sliders['CLAHE Clip Limit'].value() / 10.0,
            #'clahe_grid_size': self.sliders['CLAHE Grid Size'].value(),
            #'bg_sub_learning_rate': self.sliders['BG Sub Learning Rate'].value() / 1000.0,
            #'bg_sub_history': self.sliders['BG Sub History'].value(),
            #'bg_sub_var_threshold': self.sliders['BG Sub Var Threshold'].value(),
        }
        return config
    
    def slider_name_to_config_key(self, slider_name):
        # Map slider names to configuration keys
        name_map = {
            'Hue Range': 'hue_range',
            'Hue Offset': 'hue_offset',
            'Saturation Low': 'sat_low',
            'Saturation High': 'sat_high',
            'Value Low': 'val_low',
            'Value High': 'val_high',
            'OF Smoothing Factor': 'of_smoothing',
            'Eye-Area Smoothing Factor': 'eye_area_smoothing',
            'Eye Bounds Expand': 'eye_bounds_expand',
            'OF Process Stage': 'of_process_stage',
            # 'Area Process Stage': 'area_process_stage',
            'Median Blur K (HSV mask)': 'median_blur_knl',
            'Gauss Blur K (OG image)': 'gauss_blur_knl_pre',
            'Dilation K (HSV-Masked)': 'dilation_knl_mask',
            'Gauss Blur K (dilated eyes)': 'gauss_blur_knl_dil',
        }
        return name_map.get(slider_name, slider_name)

    # Save/Load 
    def save_config(self):
        config = self.get_window_config()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'w') as f:
                json.dump(config, f)

    def load_config(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'r') as f:
                config = json.load(f)
            self.apply_config(config)
    
    def apply_config(self, config):
        for key, value in config.items():
            #use the map to get the slider name
            slider_name = self.slider_name_to_config_key(key)
            if slider_name and slider_name in self.sliders:
                self.sliders[slider_name].setValue(value)
                self.update_value_label(slider_name, value)
                

# =====================================================
# ==================== RENDER =========================
# =====================================================

WindowIndex = 0

def DISPLAY_FRAME(window_name, img_frame, combined_frame, num_images):
    """
    Displays all frames in a single window, filling left to right, then creating new rows as needed.

    Parameters:
    - window_name (str): The name of the window.
    - img_frame (numpy.ndarray): The image/frame to display.
    - combined_frame (numpy.ndarray, optional): The combined frame containing all images.
    - num_images (int): Total number of images being processed.
    - num_columns (int): Number of columns in the grid.

    Returns:
    - combined_frame (numpy.ndarray): The updated combined frame with the new image added.
    """
    global WindowIndex

    # Get image dimensions
    img_height, img_width = img_frame.shape[:2]

    # Initialize combined_frame if it's None
    if combined_frame is None:
        # Calculate number of rows needed based on total images
        num_rows = math.ceil(num_images / COLUMNS_FRAMES)
        # Calculate combined frame size
        combined_height = num_rows * img_height
        combined_width = COLUMNS_FRAMES * img_width
        # Create a black canvas
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Determine row and column for the current image
    row = WindowIndex // COLUMNS_FRAMES
    col = WindowIndex % COLUMNS_FRAMES

    # Calculate starting x and y positions
    start_x = col * img_width
    start_y = row * img_height

    # Check if the image fits within the combined_frame
    if start_x + img_width > combined_frame.shape[1] or start_y + img_height > combined_frame.shape[0]:
        print(f"Image {WindowIndex + 1} does not fit in the combined frame... skipping!")
        return combined_frame

    # Convert grayscale images to BGR
    if len(img_frame.shape) == 2:
        img_frame = cv2.cvtColor(img_frame, cv2.COLOR_GRAY2BGR)

    # Optionally, add the window name as a label on the image
    cv2.putText(img_frame, window_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Insert the image into the combined frame
    combined_frame[start_y:start_y + img_height, start_x:start_x + img_width] = img_frame

    # Increment the window index for the next image
    WindowIndex += 1

    return combined_frame


# =====================================================
# ====================== WINDOW =======================
# =====================================================

# Queue for Plotting Data
plot_queue = queue.Queue()

class PlotWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eyelid Motion and Area")
        self.setGeometry(1200, 100, 600, 800)  # Increased height to accommodate two plots

        # Create a layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Initialize pyqtgraph PlotWidgets
        self.motion_plot = pg.PlotWidget()
        self.area_plot = pg.PlotWidget()
        layout.addWidget(self.motion_plot)
        layout.addWidget(self.area_plot)

        # Configure the motion plot
        self.motion_plot.setTitle("Eyelid Motion Magnitude Over Time")
        self.motion_plot.setLabel('left', 'Motion Magnitude')
        self.motion_plot.setLabel('bottom', 'Frame')
        self.motion_plot.addLegend()

        # Configure the area plot
        self.area_plot.setTitle("Eye Area Over Time")
        self.area_plot.setLabel('left', 'Area')
        self.area_plot.setLabel('bottom', 'Frame')
        self.area_plot.addLegend()

        # Initialize data lines for left and right eyes
        self.left_curve = self.motion_plot.plot(pen=pg.mkPen(color='b', width=2), name='Left Eye')
        self.right_curve = self.motion_plot.plot(pen=pg.mkPen(color='r', width=2), name='Right Eye')
        self.area_curve = self.area_plot.plot(pen=pg.mkPen(color='g', width=2), name='Eye Area')

        # Data buffers
        self.left_data = deque(maxlen=100)
        self.right_data = deque(maxlen=100)
        self.area_data = deque(maxlen=100)

        # Timer to update the plot
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)  # Update every 100 ms

    def update_plot(self):
        """Fetch data from the queue and update the plot."""
        while not plot_queue.empty():
            left_motion, right_motion, area = plot_queue.get()
            self.left_data.append(left_motion)
            self.right_data.append(right_motion)
            self.area_data.append(area)

        # Update the plot data
        self.left_curve.setData(list(self.left_data))
        self.right_curve.setData(list(self.right_data))
        self.area_curve.setData(list(self.area_data))


# =====================================================
# ============ LIGHTING NORMALIZATION =================
# =====================================================


def apply_gmm_skin_modeling(hsv_samples):
    # Fit a Gaussian Mixture Model to the HSV samples to model skin tone distribution
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(hsv_samples)
    means = gmm.means_
    covariances = gmm.covariances_

    # Set HSV bounds based on GMM means and variances
    lower_bound = means[0] - 2 * np.sqrt(np.diag(covariances[0]))
    upper_bound = means[0] + 2 * np.sqrt(np.diag(covariances[0]))

    lower_bound = np.clip(lower_bound, 0, 255).astype(np.uint8)
    upper_bound = np.clip(upper_bound, 0, 255).astype(np.uint8)

    # Visualization: Create GMM visualization texture
    gmm_visualization = np.zeros((256, 256, 3), dtype=np.uint8)
    for i, mean in enumerate(means):
        cv2.circle(gmm_visualization, (int(mean[0]), int(mean[1])), 10, (0, 255, 0) if i == 0 else (255, 0, 0), -1)

    # Create a before-after delta image
    delta_image = np.zeros_like(gmm_visualization)
    delta_image[:, :, 1] = cv2.absdiff(gmm_visualization[:, :, 1], lower_bound[1])

    return lower_bound, upper_bound, gmm_visualization, delta_image


def sample_skin_tone(frame, face_landmarks):
    '''
    Samples skin tone from specific face landmark sets, dynamically determining the skin tone range.
    Optionally saves a debug image of the determined skin color.
    '''
    """
    Sets:
        Forehead: 108, 151, 337
        Nose: 5, 51, 281
        Left cheek: 50
        Right cheek: 280
    """
    
    left_landmark_sets = [
        [108, 151, 337],  # Forehead (left side)
        [50],  # Left cheek
    ]
    right_landmark_sets = [
        [108, 151, 337],  # Forehead (right side)
        [280],  # Right cheek
    ]
    
    samples = { 'left': [], 'right': [] }

    # Convert frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Sample left side
    for landmark_set in left_landmark_sets:
        for idx in landmark_set:
            if idx < len(face_landmarks):
                x = int(face_landmarks[idx][0] * frame.shape[1])
                y = int(face_landmarks[idx][1] * frame.shape[0])
                x = min(max(x, 0), frame.shape[1] - 1)
                y = min(max(y, 0), frame.shape[0] - 1)

                # Sample a small circular region around the landmark
                for img_i in range(-2, 3):
                    for j in range(-2, 3):
                        if np.sqrt(img_i**2 + j**2) <= 2:
                            sample_x = min(max(x + img_i, 0), frame.shape[1] - 1)
                            sample_y = min(max(y + j, 0), frame.shape[0] - 1)
                            samples['left'].append(hsv_frame[sample_y, sample_x])

    # Sample right side
    for landmark_set in right_landmark_sets:
        for idx in landmark_set:
            if idx < len(face_landmarks):
                x = int(face_landmarks[idx][0] * frame.shape[1])
                y = int(face_landmarks[idx][1] * frame.shape[0])
                x = min(max(x, 0), frame.shape[1] - 1)
                y = min(max(y, 0), frame.shape[0] - 1)

                # Sample a small circular region around the landmark
                for img_i in range(-2, 3):
                    for j in range(-2, 3):
                        if np.sqrt(img_i**2 + j**2) <= 2:
                            sample_x = min(max(x + img_i, 0), frame.shape[1] - 1)
                            sample_y = min(max(y + j, 0), frame.shape[0] - 1)
                            samples['right'].append(hsv_frame[sample_y, sample_x])

    # Calculate average color for left and right side
    avg_colors = {}
    for side in ['left', 'right']:
        side_samples = np.array(samples[side])
        if len(side_samples) == 0:
            avg_colors[side] = None
            continue

        # Compute mean and standard deviation for each channel
        mean = np.mean(side_samples, axis=0)
        std = np.std(side_samples, axis=0)

        # Define the range as mean ± 2 standard deviations for each channel
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std

        # Filter samples (get rid of outliers)
        mask = np.all((side_samples >= lower_bound) & (side_samples <= upper_bound), axis=1)
        filtered_samples = side_samples[mask]

        if len(filtered_samples) == 0:
            avg_colors[side] = None
        else:
            # Compute the final average color
            avg_colors[side] = np.mean(filtered_samples, axis=0)

    return avg_colors['left'], avg_colors['right']


def save_col_to_img(image_name, final_color_bgr, overwrite=False):
    img_i = 1
    while os.path.exists(f"{image_name}_{img_i}.png"): img_i += 1
    file_name = f"{image_name}_{img_i}.png"
    debug_image = np.full((128, 128, 3), final_color_bgr, dtype=np.uint8)
    cv2.imwrite(file_name, debug_image)


def apply_clahe_region(frame, region):
    try:
        # Extract bounding box from landmarks (get bounding box of the eye region)
        x, y, w, h = cv2.boundingRect(np.array(region, dtype=np.float32))
        # Crop the eye region
        eye_region = frame[y:y+h, x:x+w]
        # Apply CLAHE
        eye_region_clahe = clahe.apply(eye_region)
        # Replace the original eye region in the frame
        frame[y:y+h, x:x+w] = eye_region_clahe
    except Exception as e:
        print("Error applying CLAHE to eye region:", e)
    return frame

def match_histogram(source, reference):
    # Match the histogram of the source image to that of the reference
    matched = cv2.calcHist([source], [0], None, [256], [0, 256])
    reference_hist = cv2.calcHist([reference], [0], None, [256], [0, 256])
    matched = cv2.normalize(matched, reference_hist)
    return matched

def apply_retinex(frame):
    return frame
    retinex_params = {
        'sigma_list': [15, 80, 250],  # Parameters for MSRCR
        'gain': 2.0,
        'offset': 0.0,
        'restoration_factor': 125.0,
        'dynamic_range': 5.0,
    }
    # return retinex_MSRCR(frame, **retinex_params)


# =====================================================
# ================= OPTICAL FLOW ======================
# =====================================================

# EMA
def EMA(prev_val, curr_val, smoothing_factor):
    if prev_val is None:
        return curr_val
    return (smoothing_factor * curr_val) + ((1 - smoothing_factor) * prev_val)

def OF_SMOOTHING_FACTOR():
    config = get_config_vals()
    return 1 - (config['of_smoothing'] / 100.0)

def EYE_AREA_SMOOTHING_FACTOR():
    config = get_config_vals()
    return 1 - (config['eye_area_smoothing'] / 100.0)

# Assume you want to start from the 3rd level and go up to the 5th
start_level = 2  # 0-based index, so 2 means 3rd level
end_level = 5

# Optical Flow Variables
prev_grey = None
prev_left_centroid = None
prev_right_centroid = None
motion_history_left = deque(maxlen=100)  # Stores last 100 motion magnitudes for left eye
motion_history_right = deque(maxlen=100)  # Stores last 100 motion magnitudes for right eye

# Smoothed motion values
smoothed_left_motion = None
smoothed_right_motion = None

# https://chatgpt.com/c/66fc7632-ba54-8000-bcd4-e553a4f7a6dc?model=o1-preview
# OPTICAL FLOW COMPUTE: Compute and plot eyelid motion
def of_motion_compute(left_hull, right_hull, frame_grey, secondary_plot_value):
    """
    Computes the average vertical motion of the eyelids within the eye regions.
    
    Parameters:
    - left_hull (numpy.ndarray): Convex hull of the left eye bounds.
    - right_hull (numpy.ndarray): Convex hull of the right eye bounds.
    - frame_gray (numpy.ndarray): Grayscale current frame.
    
    Returns:
    - smoothed_left_motion (float): Smoothed vertical motion for the left eye.
    - smoothed_right_motion (float): Smoothed vertical motion for the right eye.
    """
    global prev_grey, smoothed_left_motion, smoothed_right_motion, plot_queue, CombinedGridFrame
    if len(frame_grey.shape) == 3:
        print("Image must be grayscale.")
        return
    
    # Downsampling the images via cv2.resize(downsampled_frame, (320, 240))
    # frame_grey = downscale_image(frame_grey, 2)
    CombinedGridFrame = DISPLAY_FRAME("OF INPUT", frame_grey, CombinedGridFrame, TOTAL_FRAMES)
    
    if prev_grey is None:
        # Initialize previous frame
        prev_grey = frame_grey.copy()
        return 0.0, 0.0
    
    try:
        # Create masks for the left and right eye regions
        left_mask = np.zeros_like(frame_grey)
        right_mask = np.zeros_like(frame_grey)
        
        cv2.drawContours(left_mask, [left_hull], -1, 255, -1)
        cv2.drawContours(right_mask, [right_hull], -1, 255, -1)
        
        # Apply masks to the frames
        prev_left = cv2.bitwise_and(prev_grey, prev_grey, mask=left_mask)
        prev_right = cv2.bitwise_and(prev_grey, prev_grey, mask=right_mask)
        
        curr_left = cv2.bitwise_and(frame_grey, frame_grey, mask=left_mask)
        curr_right = cv2.bitwise_and(frame_grey, frame_grey, mask=right_mask)
    except Exception as e:
        print("Error:", e)
        return

    # Compute optical flow for left eye region
    flow_left = cv2.calcOpticalFlowFarneback(
        prev_left, curr_left, 
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Compute optical flow for right eye region
    flow_right = cv2.calcOpticalFlowFarneback(
        prev_right, curr_right, 
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Extract vertical flow components (dy) within the eye regions
    left_dy = flow_left[..., 1][left_mask == 255]
    right_dy = flow_right[..., 1][right_mask == 255]
    
    # Compute average vertical motion
    left_motion = np.mean(left_dy) if left_dy.size > 0 else 0.0
    right_motion = np.mean(right_dy) if right_dy.size > 0 else 0.0

    # Consider this for small motions:
    # if abs(left_motion) < threshold:
    #     left_motion = 0.0
    
    # Apply Exponential Moving Average (EMA) for smoothing
    smoothing = OF_SMOOTHING_FACTOR()
    smoothed_left_motion = EMA(smoothed_left_motion, left_motion, smoothing)
    smoothed_right_motion = EMA(smoothed_right_motion, right_motion, smoothing)

    # Put the smoothed motion magnitudes into the plot queue
    plot_queue.put((smoothed_left_motion, smoothed_right_motion, secondary_plot_value)) # * for unpacking the tuple

    # Update previous frame
    prev_grey = frame_grey.copy()
    
    # Unused but just in case we want it
    return smoothed_left_motion, smoothed_right_motion


def eye_area_compute(frame):
    global prev_area
    area = cv2.countNonZero(frame) # single-channel
    area = EMA(prev_area, area, EYE_AREA_SMOOTHING_FACTOR())
    prev_area = area


# =====================================================
# ===================== IMAGE  ========================
# =====================================================

HsvRef = None

def HUE_OFFSET():
    config = get_config_vals()
    return config['hue_offset'] - 180


def hsv_range(config, hsv_range_override):
    # HSV color gates (skin tone range)
    hsv_range = HsvRef if hsv_range_override is None else hsv_range_override

    lower_color = np.array([
        hsv_range[0] - config['hue_range'] + HUE_OFFSET(), 
        config['sat_low'], 
        config['val_low']], dtype=np.uint8)
    upper_color = np.array([
        hsv_range[0] + config['hue_range'] + HUE_OFFSET(), 
        config['sat_high'], 
        config['val_high']], dtype=np.uint8)
    
    return lower_color, upper_color


def eye_bounds_mask(frame, hull):
    '''Creates a mask from the shape hulls and keeps the eye areas.'''
    # Create a blank mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Draw the eye hulls on the mask
    cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)
    
    # The frame, but pixels outside the eye areas are blacked out
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Return the masked frame
    return masked_frame


def DOWNSCALE(image, levels):
    for _ in range(levels):
        image = cv2.pyrDown(image)
    return image


def CROP(frame, hull):
    '''Crops the eye areas from the frame using convex hulls.'''
    have_hulls = hull is not None and hull is not None
    if have_hulls:
        masked_frame = eye_bounds_mask(frame, hull)
    else:
        print("Eye hulls not found. Proceeding with the original frame.")
        masked_frame = frame
    # Done!
    return masked_frame


def process_face_side(bgr_frame, eye_hull, hsv_ref, config):
    # Preprocessing
    bgr_frame = apply_retinex(bgr_frame)
    gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = clahe.apply(gray_frame)
    
    # Apply CLAHE to the eye region
    eye_lms = get_eye_lms_data(bgr_frame)
    if eye_lms:
        gray_frame = apply_clahe_region(gray_frame, eye_lms)

    # Gaussian blur
    if config['gauss_blur_knl_pre'] > 0:
        bgr_frame = cv2.GaussianBlur(bgr_frame, (config['gauss_blur_knl_pre'], config['gauss_blur_knl_pre']), 0)

    # Convert to HSV
    hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

    # Create mask
    lo_hsv_gate, hi_hsv_gate = hsv_range(config, hsv_range_override=hsv_ref)
    hsv_mask = cv2.inRange(hsv_frame, lo_hsv_gate, hi_hsv_gate)

    # Median blur
    if config['median_blur_knl'] > 0:
        hsv_mask = cv2.medianBlur(hsv_mask, config['median_blur_knl'])

    # Dilate
    if config['dilation_knl_mask'] > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config['dilation_knl_mask'], config['dilation_knl_mask']))
        hsv_mask = cv2.dilate(hsv_mask, kernel)

    # Final mask
    masked_out = cv2.bitwise_and(bgr_frame, bgr_frame, mask=hsv_mask)
    if config['gauss_blur_knl_dil'] > 0:
        masked_out = cv2.GaussianBlur(masked_out, (config['gauss_blur_knl_dil'], config['gauss_blur_knl_dil']), 0)

    # Crop to eye region
    cropped_masked = CROP(masked_out, eye_hull)

    return cropped_masked, hsv_frame, hsv_mask, masked_out


prev_area = 0
def process(bgr_frame):
    global WindowIndex, CombinedGridFrame
    WindowIndex = 0
    CombinedGridFrame = None
    config = get_config_vals()

    # Get face landmarks and eye bounds
    faces_lms = get_faces_lms(bgr_frame)
    if faces_lms and len(faces_lms) > 0:
        left_hull, right_hull = get_eye_bounds(bgr_frame, config, faces_lms=faces_lms[0])

    if not left_hull or not right_hull or not faces_lms:
        print("No eye bounds found. Skipping processing.")
        return bgr_frame

    # Downscale
    bgr_frame = DOWNSCALE(bgr_frame, 1)
    CombinedGridFrame = DISPLAY_FRAME("Original", bgr_frame, CombinedGridFrame, TOTAL_FRAMES)

    # Get HSV references for each side
    hsv_ref_left, hsv_ref_right = sample_skin_tone(bgr_frame, faces_lms[0])

    # Process each side
    left_result, left_hsv, left_mask, left_masked = process_face_side(bgr_frame.copy(), left_hull, hsv_ref_left, config)
    right_result, right_hsv, right_mask, right_masked = process_face_side(bgr_frame, right_hull, hsv_ref_right, config)

    # Combine results
    combined_hsv = np.hstack((left_hsv, right_hsv))
    combined_mask = np.hstack((left_mask, right_mask))
    combined_masked = np.hstack((left_masked, right_masked))
    
    # Display steps
    CombinedGridFrame = DISPLAY_FRAME("HSV", combined_hsv, CombinedGridFrame, TOTAL_FRAMES)
    CombinedGridFrame = DISPLAY_FRAME("HSV Mask", combined_mask, CombinedGridFrame, TOTAL_FRAMES)
    CombinedGridFrame = DISPLAY_FRAME("HSV-Masked Result", combined_masked, CombinedGridFrame, TOTAL_FRAMES)

    # Combine cropped results
    combined_result = np.hstack((left_result, right_result))

    # Motion calculation
    of_frame = cv2.cvtColor(combined_result, cv2.COLOR_BGR2GRAY)
    eye_area_compute(of_frame)
    of_motion_compute(left_hull=left_hull, right_hull=right_hull, frame_grey=of_frame, secondary_plot_value=0)

    # Display
    cv2.namedWindow("Processing Steps", cv2.WINDOW_NORMAL)
    cv2.imshow("Processing Steps", CombinedGridFrame)

    return combined_result



def update(in_frame):
    global WindowIndex

    # Isolate the color
    process(in_frame)
    
    # Reset the indx for the row/column display mapping
    WindowIndex = 0


# =====================================================
# =================== MediaPipe =======================
# =====================================================

# Landmark indexes from MP Detection Manager:
"""
Landmark indices for eye tracking:

Left Eye:
- Upper lid: 159
- Lower lid: 145
- Upper eye socket: [56, 247]
- Lower eye socket: [112, 110]
- Pupil: 468
- Inner reference: 173
- Outer reference: 33
- Left jaw: 172
- Right jaw: 397

Right Eye:
- Upper lid: 386
- Lower lid: 374
- Upper eye socket: [286, 467]
- Lower eye socket: [341, 339]
- Pupil: 473
- Inner reference: 398
- Outer reference: 263
- Left jaw: 172
- Right jaw: 397

Note: Both eyes use the same landmarks for left jaw (172) and right jaw (397).
"""

# Orig indices (good enough):
left_eye_indices = [56, 247, 112, 110, 173, 33]
right_eye_indices = [286, 467, 341, 339, 398, 263]

# Initialize the FaceMeshDetector
print("Creating FaceMeshDetector...")
Detector = facelms()
print("FaceMeshDetector is created!")


def get_faces_lms(frame):
    '''Gets all the landmarks from the frame.'''
    img, faces, lms = Detector.FindFaceMesh(img=frame, draw=False)
    return lms


def get_eye_lms_data(frame, faces_lms = None):
    '''Gets the landmarks that surround the eyes, and returns tuple (left_eye, right_eye).'''
    if faces_lms is None or len(faces_lms) == 0: 
        faces_lms = get_faces_lms(frame)

    # Get the first (and usually only) face's landmark data
    face_lms = faces_lms[0]

    # Get the left eye landmarks
    left_eye = []
    for i in left_eye_indices:
        if i < len(face_lms):
            left_eye.append(face_lms[i])
        else:
            left_eye.append(None)  # or some default value

    # Get the right eye landmarks
    right_eye = []
    for i in right_eye_indices:
        if i < len(face_lms):
            right_eye.append(face_lms[i])
        else:
            right_eye.append(None)  # or some default value

    # Check if we have all landmarks for both eyes
    if None in left_eye or None in right_eye:
        print("Some eye landmarks are missing.")
        return None, None

    return left_eye, right_eye


def expand_eye_bound_lms(eye_points, factor):
    # Convert to numpy array if not already
    eye_points = np.array(eye_points)

    # Calculate the center and expand all points from it
    eye_center = np.mean(eye_points, axis=0)
    expanded_points = eye_center + factor * (eye_points - eye_center)

    # Return as list of lists for consistency with original format
    return expanded_points.tolist()


def get_eye_bounds(frame, config, faces_lms = None):
    '''Main function to process the frame and extract eye landmark data.'''
    # Get the eye landmarks
    if faces_lms is not None and len(faces_lms) > 0:
        left_eye, right_eye = get_eye_lms_data(frame, faces_lms)
    else:
        left_eye, right_eye = get_eye_lms_data(frame)
    
    # If that didn't work, get outta here...
    if left_eye is None or right_eye is None:
        return None, None
    
    # Enlarge the eye boundary landmarks
    left_eye = expand_eye_bound_lms(left_eye, config['eye_bounds_expand'] / 10)        # The slider is from 10 to 20
    right_eye = expand_eye_bound_lms(right_eye, config['eye_bounds_expand'] / 10)      # The slider is from 10 to 20

    # Draw the landmarks and eye borders
    # draw_lms_cv(frame, left_eye)
    # draw_lms_cv(frame, right_eye)
    left_shape_hull = get_eye_border(frame, left_eye)
    right_shape_hull = get_eye_border(frame, right_eye)

    # Done!
    return left_shape_hull, right_shape_hull


# - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Drawing
# - - - - - - - - - - - - - - - - - - - - - - - - - - -

def draw_lms_cv(frame, lms):    
    # Draw the landmarks
    for lm in lms:
        x, y = int(lm[0] * frame.shape[1]), int(lm[1] * frame.shape[0])
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    return frame


def get_eye_border(frame, landmarks, color=(0, 255, 0), thickness=2, draw=False):
    """
    Draw lines around the border of eye landmarks.
    :param frame: The image frame to draw on
    :param landmarks: List of (x, y) coordinates of eye landmarks
    :return: Convex hull of the eye border
    """
    if len(landmarks) < 3:
        print("Not enough landmarks to draw a border")
        return None

    # Scale landmarks to frame dimensions
    scaled_landmarks = []
    for lm in landmarks:
        x, y = int(lm[0] * frame.shape[1]), int(lm[1] * frame.shape[0])
        scaled_landmarks.append([x, y])

    # Convert scaled landmarks to numpy array
    points = np.array(scaled_landmarks, dtype=np.int32)

    # Compute the convex hull of the points
    hull = cv2.convexHull(points)

    # Draw the convex hull
    if draw:
        cv2.polylines(frame, [hull], True, color, thickness)

    return hull


# =====================================================
# ===================== MAIN LOOP =======================
# =====================================================

def video_capture_loop():
    global _fps, WindowIndex, config_app

    # Open the default webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    t = Time.time()  # time in seconds

    print("Webcam is open:", cap.isOpened())
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    # Capture frame-by-frame
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Process the frame
        update(frame)
        _fps = 1 / (Time.time() - t)
        t = Time.time()

        # Wait for 'q' key to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting application...")
            # config_app.quit()
            break

    # When everything is done, release the capture
    config_app.quit()
    cap.release()
    cv2.destroyAllWindows()


def main():
    global _fps, WindowIndex, config_window, config_app, plot_window

    # Initialize QApplication and ConfigWindow
    config_app = QApplication(sys.argv)
    config_window = ConfigWindow()
    config_window.show()

    # Set config defaults
    config_def = get_config_vals()
    config_window.apply_config(config_def)
    
    # Initialize and show the PlotWindow
    plot_window = PlotWindow()
    plot_window.show()

    # Start video processing in a separate daemon thread
    video_thread = threading.Thread(target=video_capture_loop, args=(), daemon=True)
    video_thread.start()

    # Execute the PyQt5 application in the main thread
    sys.exit(config_app.exec_())


if __name__ == "__main__":
    main()
