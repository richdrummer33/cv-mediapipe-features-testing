# Typical
import cv2
import math
import numpy as np
import time as Time
import threading
import sys
import warnings

# FaceMesh
from FaceMeshModule import FaceMeshDetector as facelms

# QtWindow UI
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QGridLayout, QLabel, QSlider, QVBoxLayout, QFileDialog, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import json

# OF
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from threading import Thread
import queue
import pyqtgraph as pg # NU


# =====================================================
# ===================== SETUP =========================
# =====================================================

# Suppress specific SymbolDatabase deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

# Initialize background subtractor
print("BG Subtractor")
_bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=20, detectShadows=True)
_fps = 30

# Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization)
print("CLAHE")
clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(2, 2))

# HSV ranges for color isolation of skin-tone colors (above/below the reference color)
lower_bound = np.array([0, 100, 100])
upper_bound = np.array([10, 255, 255])

# Default skin tone reference color
def get_bgr_from_reference_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.mean(image_rgb, axis=(0, 1))
# Get the average BGR color from the reference image
bgr_color_ref = get_bgr_from_reference_image("reference.png")
# Convert the average BGR color to HSV
average_color_bgr = np.uint8([[[bgr_color_ref[0], bgr_color_ref[1], bgr_color_ref[2]]]])
_average_color_hsv = cv2.cvtColor(average_color_bgr, cv2.COLOR_BGR2HSV)[0][0]
_REF_skin_hsv = _average_color_hsv
print("Average HSV color:", _average_color_hsv)

_CombinedImages = None
TOTAL_FRAMES = 11
COLUMNS_FRAMES = 6


# =====================================================
# ================== CGF WINDOW =======================
# =====================================================

# Declare global variable
config_window = None
config_app = None

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
            ('Hue Offset', -180, 180),
            # HSV thresholds
            ('Saturation Low', 0, 255),
            ('Saturation High', 0, 255),
            ('Value Low', 0, 255),
            ('Value High', 0, 255),
            # OF
            ('OF Smoothing Factor', 10, 100),       # divided by 10
            ('Eye-Area Smoothing Factor', 10, 100), # divided by 10
            ('Eye Bounds Expand', 10, 50),
            ('OF Process Stage', 1, 4),
            # Kernels
            ('Median Blur K (HSV mask)', 1, 25),
            ('Gauss Blur K (OG image)', 1, 25),
            ('Dilation K (HSV Mask)', 1, 20),
            ('Gauss Blur K (dilated eyes)', 1, 25),
            # NEW
            ('Hist Hue Min', 0, 180),
            ('Hist Hue Max', 1, 180),
            ('Clip Threshold', 0, 100),
            ('Retina Threshold', 0, 100),
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
            label = QLabel(name)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, maximum)
            slider.setValue(default)
            self.sliders[name] = slider  # Store slider in dictionary

            value_label = QLabel(str(default))
            self.value_labels[name] = value_label  # Store value label in dictionary

            slider.valueChanged.connect(lambda value, name=name: self.update_value_label(name, value))

            row = i % 8
            col = i // 8 * 3
            slider_layout.addWidget(label, row, col)
            slider_layout.addWidget(slider, row, col + 1)
            slider_layout.addWidget(value_label, row, col + 2)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        # 4. Set default values
        self.apply_defaults()

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
    @staticmethod
    def ensure_odd(value):
        return value if value % 2 == 1 else value + 1
 
    def apply_defaults(self):
        # find the current directory wth "defaut_" prefix, and load the json file
        file_path = [f for f in os.listdir() if f.startswith("default_") and f.endswith(".json")]
        # load the json
        if len(file_path) > 0:
            with open(file_path[0], 'r') as f:
                config = json.load(f)
            self.apply_config(config)
            print("Config window Default values set from file!")
        else:
            default_config = get_config_vals()
            for name, slider in self.sliders.items():
                if name in default_config:
                    slider.setValue(default_config[name])
            print("Config window Default values set hardcoded.")
        
    
    def update_value_label(self, name, value):
        if name in ['OF Smoothing Factor', 'Eye-Area Smoothing Factor']:
            display_value = value / 10.0
            self.value_labels[name].setText(f"{display_value:.1f}")
        else:
            self.value_labels[name].setText(str(value))


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
            # Kernels
            'median_blur_knl': self.ensure_odd(self.sliders['Median Blur K (HSV mask)'].value()),
            'gauss_blur_knl_pre': self.ensure_odd(self.sliders['Gauss Blur K (OG image)'].value()),
            'dilation_knl_mask': max(1, self.sliders['Dilation K (HSV Mask)'].value()),
            'gauss_blur_knl_dil': self.ensure_odd(self.sliders['Gauss Blur K (dilated eyes)'].value()),
            # NEW
            'hist_hue_min': self.sliders['Hist Hue Min'].value(),
            'hist_hue_max': self.sliders['Hist Hue Max'].value(),
            'clip_threshold': self.sliders['Clip Threshold'].value(),
            'retina_threshold': self.sliders['Retina Threshold'].value(),
            #'clahe_clip_limit': self.sliders['CLAHE Clip Limit'].value() / 10.0,
            #'clahe_grid_size': self.sliders['CLAHE Grid Size'].value(),
            #'bg_sub_learning_rate': self.sliders['BG Sub Learning Rate'].value() / 1000.0,
            #'bg_sub_history': self.sliders['BG Sub History'].value(),
            #'bg_sub_var_threshold': self.sliders['BG Sub Var Threshold'].value(),
        }
        return config
    

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

    def get_config():
        global config_window
        if config_window is not None:
            return config_window.get_window_config()
        else:
            # Return default config or handle the error as needed
            return {
                # HSV color isolation
                'hue_range': 20,
                'hue_offset': 180,
                # HSV thresholds
                'sat_low': 25,
                'sat_high': 254,
                'val_low': 85,
                'val_high': 254,
                # OF
                'of_smoothing': 50,
                'eye_area_smoothing': 50,
                'eye_bounds_expand': 10,
                'of_process_stage': 1,
                # Kernels
                'median_blur_knl': 3,
                'gauss_blur_knl_pre': 3,
                'dilation_knl_mask': 3,
                'gauss_blur_knl_dil': 3,
                # NEW
                'hist_hue_min': 0,
                'hist_hue_max': 180,
                'clip_threshold': 50,
                'retina_threshold': 50,
            }
    
    def apply_config(self, config):
        for key, value in config.items():
            # Map config keys back to slider names
            slider_map = {
                'hue_range': 'Hue Range',
                'hue_offset': 'Hue Offset',
                'sat_low': 'Saturation Low',
                'sat_high': 'Saturation High',
                'val_low': 'Value Low',
                'val_high': 'Value High',
                'of_smoothing': 'OF Smoothing Factor',
                'eye_area_smoothing': 'Eye-Area Smoothing Factor',
                'eye_bounds_expand': 'Eye Bounds Expand',
                'of_process_stage': 'OF Process Stage',
                'median_blur_knl': 'Median Blur K (HSV mask)',
                'gauss_blur_knl_pre': 'Gauss Blur K (OG image)',
                'dilation_knl_mask': 'Dilation K (HSV Mask)',
                'gauss_blur_knl_dil': 'Gauss Blur K (dilated eyes)',
                'hist_hue_min': 'Hist Hue Min',
                'hist_hue_max': 'Hist Hue Max',
                'clip_threshold': 'Clip Threshold',
                'retina_threshold': 'Retina Threshold',
            }
            slider_name = slider_map.get(key)
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

    # Mask old label
    cv2.rectangle(img_frame, (0, 0), (img_width, 30), (0, 0, 0), -1)
    cv2.putText(img_frame, window_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 165, 255), 2, cv2.LINE_AA)

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

def plotter():
    """Initializes and displays the PlotWindow."""
    plot_window = PlotWindow()
    plot_window.show()


class PlotWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eyelid Motion Magnitude")
        self.setGeometry(1200, 100, 600, 400)  # Position it separately from ConfigWindow

        # Create a layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Initialize pyqtgraph PlotWidget
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Configure the plot
        self.plot_widget.setTitle("Eyelid Motion Magnitude Over Time")
        self.plot_widget.setLabel('left', 'Motion Magnitude')
        self.plot_widget.setLabel('bottom', 'Frame')
        self.plot_widget.addLegend()

        # Initialize data lines for left and right eyes
        self.left_curve = self.plot_widget.plot(pen=pg.mkPen(color='b', width=2), name='Left Eye')
        self.right_curve = self.plot_widget.plot(pen=pg.mkPen(color='r', width=2), name='Right Eye')
        self.area_curve = self.plot_widget.plot(pen=pg.mkPen(color='g', width=2), name='Eye Area')

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


def downscale_image(image, levels):
    for _ in range(levels):
        image = cv2.pyrDown(image)
    return image


# =====================================================
# ============== REF COLOR SAMPLING ===================
# Sample colors in the image to prepare detection
# =====================================================

from scipy import stats
import os

def sample_pixels_from_lms(frame, face_landmarks, landmark_sets=None):
    '''
    Samples skin tone from specific face landmark sets, dynamically determining the skin tone range.
    Optionally saves a debug image of the determined skin color.
    '''
    """
    Sets:
        Forehead: 108, 151, 337
        Nose: 5, 51, 281
        Left cheek: 280
        Right cheek: 
    """

    if landmark_sets is None:
        landmark_sets = [
            #[337, 108, 151],  # Forehead
            #[5, 51, 281],  # Nose
            [280],  # Left cheek
            [50],  # Right cheek
        ]

    # Convert frame to HSV
    samples = []
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for landmark_set in landmark_sets:
        for idx in landmark_set:
            # Ensure the index is within the landmarks list
            if idx < len(face_landmarks):
                x = int(face_landmarks[idx][0] * frame.shape[1])
                y = int(face_landmarks[idx][1] * frame.shape[0])

                # Ensure x and y are within frame boundaries
                x = min(max(x, 0), frame.shape[1] - 1)
                y = min(max(y, 0), frame.shape[0] - 1)

                # Sample a small circular region around the landmark
                for img_i in range(-2, 3):
                    for j in range(-2, 3):
                        if np.sqrt(img_i**2 + j**2) <= 2:  # Small circle radius
                            sample_x = min(max(x + img_i, 0), frame.shape[1] - 1) 
                            sample_y = min(max(y + j, 0), frame.shape[0] - 1)
                            samples.append(hsv_frame[sample_y, sample_x])

    if not samples:
        return None  # Return None if no samples found
    
    samples = np.array(samples)
    
    # Compute mean and standard deviation for each channel
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)

    # Define the range as mean ± 2 standard deviations for each channel
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std

    # Filter samples (get rid of outliers)
    mask = np.all((samples >= lower_bound) & (samples <= upper_bound), axis=1)
    filtered_samples = samples[mask]

    if len(filtered_samples) == 0:
        return None  # Return None if all samples were filtered out

    # Compute the final average color
    average_color_hsv = np.mean(filtered_samples, axis=0)

    # Convert HSV to BGR correctly for debug image
    # average_color_hsv_uint8 = np.uint8([[average_color_hsv]])
    # bgr_col = cv2.cvtColor(average_color_hsv_uint8, cv2.COLOR_HSV2BGR)[0][0]
    # save_col_to_img("skin_tone_sample", bgr_col)

    return average_color_hsv


def save_col_to_img(image_name, final_color_bgr, overwrite=False):
    img_i = 1
    while os.path.exists(f"{image_name}_{img_i}.png"): img_i += 1
    file_name = f"{image_name}_{img_i}.png"
    debug_image = np.full((128, 128, 3), final_color_bgr, dtype=np.uint8)
    cv2.imwrite(file_name, debug_image)

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
def of_motion_compute(left_hull, right_hull, frame_grey, smoothing_factor, secondary_plot_value):
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
    global prev_grey, smoothed_left_motion, smoothed_right_motion, plot_queue, _CombinedImages
    if len(frame_grey.shape) == 3:
        print("Image must be grayscale.")
        return
    
    # Downsampling the images via cv2.resize(downsampled_frame, (320, 240))
    # frame_grey = downscale_image(frame_grey, 2)
    _CombinedImages = DISPLAY_FRAME("OF INPUT", frame_grey, _CombinedImages, TOTAL_FRAMES)
    
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
    smoothed_left_motion = EMA(smoothed_left_motion, left_motion, smoothing_factor)
    smoothed_right_motion = EMA(smoothed_right_motion, right_motion, smoothing_factor)

    # Put the smoothed motion magnitudes into the plot queue
    plot_queue.put((smoothed_left_motion, smoothed_right_motion, secondary_plot_value)) # * for unpacking the tuple

    # Update previous frame
    prev_grey = frame_grey.copy()
    
    # Unused but just in case we want it
    return smoothed_left_motion, smoothed_right_motion


# =====================================================
# ===================== IMAGE  ========================
# =====================================================

def get_hsv_col_ref(hsv_frame, bgr_frame):

    # Get and check lm results
    faces_lms = get_faces_lms(bgr_frame) # One face only
    if not faces_lms or len(faces_lms) == 0:
        return _average_color_hsv
        
    try:
        return sample_pixels_from_lms(hsv_frame, faces_lms[0])
    except:
        print("Error sampling skin tone (no landmarks... yet)")
    
    return _average_color_hsv


def HUE_OFFSET():
    cfg = get_config_vals()
    return cfg['hue_offset'] - 180


def hsv_range(config, hsv_color_ref):
    # HSV color gates (skin tone range)
    h_lower = max(0, hsv_color_ref[0] - config['hue_range'] + HUE_OFFSET())
    h_upper = min(180, hsv_color_ref[0] + config['hue_range'] + HUE_OFFSET())
    lower_color = np.array([
        h_lower,
        config['sat_low'], 
        config['val_low']], dtype=np.uint8)
    upper_color = np.array([
        h_upper,
        config['sat_high'], 
        config['val_high']], dtype=np.uint8)
    
    return lower_color, upper_color


def eye_bounds_mask(frame, left_hull, right_hull):
    '''Creates a mask from the shape hulls and keeps the eye areas.'''
    # Create a blank mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Draw the eye hulls on the mask
    cv2.drawContours(mask, [left_hull], 0, (255, 255, 255), -1)
    cv2.drawContours(mask, [right_hull], 0, (255, 255, 255), -1)
    
    # The frame, but pixels outside the eye areas are blacked out
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Return the masked frame
    return masked_frame


def CROP(frame, left_hull, right_hull):
    '''Crops the eye areas from the frame using convex hulls.'''
    have_hulls = left_hull is not None and right_hull is not None
    if have_hulls:
        masked_frame = eye_bounds_mask(frame, left_hull, right_hull)
    else:
        print("Eye hulls not found. Proceeding with the original frame.")
        masked_frame = frame
    # Done!
    return masked_frame


def hist_back_projection(hsv_frame, hsv_low = 0, hsv_high = 180):
    
    # Extract the hue channel from HSV frame and apply histogram back-projection
    hue = hsv_frame[:, :, 0]
    hist_size = max(25, 2)  # TODO: Use CONFIG 
    hue_range = hsv_low, hsv_high
    #print("Hue range:", hue_range)

    # Calculate the histogram and normalize it
    hist = cv2.calcHist([hue], [0], None, [hist_size], hue_range)
    hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    # Calculate back-projection
    backproj = cv2.calcBackProject([hue], [0], hist, hue_range, 1)

    return backproj


def of_apply_histogram(config, hsv_result, invert = False):
    # Apply the eye mask to the final hsv-masked frame, and use it for Optical Flow
    inverted_frame = cv2.bitwise_not(hsv_result)    # invert color hsv_masked_out
    of_frame = hist_back_projection(inverted_frame, config['hist_hue_min'], config['hist_hue_max'])
    return of_frame


# ===========================================================
# ===================== EYE PROCESSING ======================
# ===========================================================

# ~~~~~~~~~~ Retina and White Part Color Values ~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
Right Eye: 
  - Pupil: 478
  - Outer Retina: 471
  - White: 469
Left Eye:
 - Pupil: 473
 - Outer Retina: 476
 - White: 474
 """


# def get_eye_color_values(hsv_frame, eye_hull):
#     """
#     Find the retina and the white part of the eye and store the HSV values.
#     
#     :param hsv_frame: HSV image of the face
#     :param eye_hull: Convex hull of the eye region
#     :return: HSV values for retina and white part of the eye
#     """
#     # If no eye hull is provided, return the last value, or skin if no last value
#     global REF_retina_hsv, REF_white_hsv
#     if eye_hull is None:
#         if REF_retina_hsv is not None and REF_white_hsv is not None:
#             return REF_retina_hsv, REF_white_hsv
#         else:
#             return REF_skin_hsv
#     
#     # We have the eye hull, so let's proceed
#     mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
#     cv2.drawContours(mask, [eye_hull], 0, 255, -1)
#     
#     # Extract the eye region
#     eye_region = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)
#     
#     # Separate the eye region into its HSV components
#     h, s, v = cv2.split(eye_region)
#     
#     # Find the darkest area (retina) and the brightest area (white part)
#     retina_mask = (v == np.min(v[np.nonzero(mask)]))
#     white_mask = (v == np.max(v[np.nonzero(mask)]))
#     
#     # Get average HSV values for retina and white part
#     REF_retina_hsv = np.mean(eye_region[retina_mask], axis=0)
#     REF_white_hsv = np.mean(eye_region[white_mask], axis=0)
#     
#     return REF_retina_hsv, REF_white_hsv


_REF_retina_hsv = None
_REF_white_hsv = None
_last_left_hull = None
_last_right_hull = None
_last_mask = None

# NOTE: https://claude.ai/chat/e3322b61-2533-4d26-895c-e2aae3f5a8b3
def process_retina_mask(config, hsv_frame, bgr_frame, left_hull, right_hull):
    """
    Create a mask that isolates the eyeballs by finding pixels most different from the skin tone.
    
    :param config: Configuration dictionary
    :param hsv_frame: HSV image of the face
    :param bgr_frame: BGR image of the face
    :param left_hull: Convex hull of the left eye region
    :param right_hull: Convex hull of the right eye region
    :return: Binary mask of the eyeballs, HSV-masked frame
    """
    global _REF_skin_hsv, _REF_retina_hsv, _REF_white_hsv, _CombinedImages, _last_left_hull, _last_right_hull, _last_mask

    # If the hsv_frame is all black, we can't do anything
    if np.all(hsv_frame == 0):
        print("HSV Mask is all black. Skipping...")
        return np.zeros_like(hsv_frame[:,:,0]), hsv_frame
    
    # if left or right hull are zero or none, set them to the last known values
    if left_hull is None:
        left_hull = _last_left_hull
    if right_hull is None:
        right_hull = _last_right_hull
    _last_right_hull, _last_left_hull = right_hull, left_hull

    # STEP 1: Crop the frame to the eye regions
    cropped_frame = CROP(hsv_frame, left_hull, right_hull)
    DISPLAY_FRAME("Cropped Retina", cropped_frame, _CombinedImages, TOTAL_FRAMES)

    # STEP 2: Find the pixel most different from the skin tone
    h_diff = np.minimum(np.abs(cropped_frame[:,:,0] - _REF_skin_hsv[0]), 
                        180 - np.abs(cropped_frame[:,:,0] - _REF_skin_hsv[0])) / 90.0
    s_diff = np.abs(cropped_frame[:,:,1] - _REF_skin_hsv[1]) / 255.0
    v_diff = np.abs(cropped_frame[:,:,2] - _REF_skin_hsv[2]) / 255.0
    
    diff_map = (h_diff + s_diff + v_diff) / 3.0
    max_diff_coords = np.unravel_index(np.argmax(diff_map), diff_map.shape) # The pixel with the maximum difference

    # STEP 3: Find adjacent pixels that are also significantly different from the skin tone
    # 3.1: Find the maximum difference in the difference map
    max_diff = np.max(diff_map)
    avg_diff = np.mean(diff_map)
    threshold = min(avg_diff, max_diff * config['retina_threshold'] / 100.0) 
    print(f"Max diff: {max_diff}, Avg diff: {avg_diff}, Threshold: {threshold}")

    seed_point = max_diff_coords[::-1]  # This is the coord for cv2.floodFill to start from (roughly the most different pixel)
    mask = np.zeros((cropped_frame.shape[0]+2, cropped_frame.shape[1]+2), np.uint8) # ⭐ UPDATED

    # 3.2: Convert threshold back to 0-255 range for floodFill
    fill_threshold = (int(threshold * 255),) * 3  # (⭐ UPDATED) NOTE: Adjust multiplier as needed.
    # Fills the area around the seed point that is significantly different from the skin tone    
    cv2.floodFill(cropped_frame, mask, seedPoint=seed_point, newVal=(255, 255, 255),    # ⭐ UPDATED
                  loDiff=fill_threshold, upDiff=fill_threshold,                         # threshold should range from 0 to 255
                  flags=8 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)                       # If the threshold value is higher, sensitivity is lower
    # Invert the mask so that the inside is white and the outside is black
    mask = cv2.bitwise_not(mask) # ⭐ I ADDED
    # Remove the added border from the mask
    mask = mask[1:-1, 1:-1] # ⭐ ADDED

    # STEP 4: If the mask is still empty, fall back to a simple threshold
    if np.sum(mask) == 0:
        print("Flood fill failed. Using simple threshold...")
        if _last_mask is not None:
            mask = _last_mask
        else:
            print("No mask found. Skipping...")
            return np.zeros_like(hsv_frame[:,:,0]), hsv_frame
            
    _last_mask = mask
    DISPLAY_FRAME("Final Mask", mask, _CombinedImages, TOTAL_FRAMES)

    # Step 4: Create a mask based on the average value of these pixels
    # eye_pixels = cropped_frame[mask == 255]
    # if len(eye_pixels) > 0:
    #     avg_eye_color = np.mean(eye_pixels, axis=0)
    #     lower_bound = np.array([max(0, avg_eye_color[i] - 20) for i in range(3)], dtype=np.uint8)
    #     upper_bound = np.array([min(255, avg_eye_color[i] + 20) for i in range(3)], dtype=np.uint8)
    #     eye_mask = cv2.inRange(cropped_frame, lower_bound, upper_bound)
    # else:
    #     print("No eye pixels found. Using original mask.")
    #     eye_mask = mask

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) # (config['dilation_knl_mask'], config['dilation_knl_mask']))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Step 5: Apply the mask to the original image
    out_hsv_masked = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask) # ⭐ CHANGED from cropped_frame to hsv_frame
    DISPLAY_FRAME("Retina Final Result", out_hsv_masked, _CombinedImages, TOTAL_FRAMES)
    
    return mask, out_hsv_masked


def process_face_mask(config, bgr_frame, hsv_frame, left_hull, right_hull):
    global _CombinedImages, _REF_skin_hsv

    # == Step 2 == 
    # Convert the image to HSV color space
    _REF_skin_hsv = get_hsv_col_ref(hsv_frame, bgr_frame)
    _CombinedImages = DISPLAY_FRAME("HSV", hsv_frame, _CombinedImages, TOTAL_FRAMES)

    # == Step 3 == 
    # Get HSV range
    lo_hsv_gate, hi_hsv_gate = hsv_range(config, _REF_skin_hsv)
    try:  mask = cv2.inRange(hsv_frame, lo_hsv_gate, hi_hsv_gate) # this outputs a binary mask (colors are either 0 or 255)
    except:
        print("Error creating HSV mask (no landmarks... yet)")
        return bgr_frame
    
    # Create a mask based on the HSV range, and median blur it to remove noise
    mask_blurred = cv2.medianBlur(mask, config['median_blur_knl']) # Median blur differs from gaussian blur in that it takes the median of all the pixels under the kernel area and the central element is replaced with this median value
    _CombinedImages = DISPLAY_FRAME("HSV Mask", mask_blurred, _CombinedImages, TOTAL_FRAMES)

    # == Step 4 == 
    # Dilate the mask to make objects more connected (thick)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config['dilation_knl_mask'], config['dilation_knl_mask']))
    out_mask = cv2.dilate(mask_blurred, kernel)
    _CombinedImages = DISPLAY_FRAME("HSV Mask (dilated)", out_mask, _CombinedImages, TOTAL_FRAMES)

    # == Step 5 == 
    # Bitwise AND the original frame with the mask to get the final output
    out_hsv_masked = cv2.bitwise_and(bgr_frame, bgr_frame, mask=out_mask)
    out_hsv_masked = cv2.GaussianBlur(out_hsv_masked, (config['gauss_blur_knl_dil'], config['gauss_blur_knl_dil']), 0)

    # Done!
    return out_mask, out_hsv_masked
    

def process(bgr_frame):
    '''Converts to HSV to isolate a color in the frame, and removes non-colors'''
    global WindowIndex, _CombinedImages, prev_area
    WindowIndex = 0  # Reset window index

    # Setup
    _CombinedImages = None   # grid display (stacks images each time DISPLAY_FRAME is called)
    config = get_config_vals()
    bgr_frame = downscale_image(bgr_frame, 1)
    of_frame = bgr_frame
    _CombinedImages = DISPLAY_FRAME("Original", bgr_frame, _CombinedImages, TOTAL_FRAMES) 
    
    # Soften and HSV-conversion
    mask_blurred = cv2.GaussianBlur(bgr_frame, (config['gauss_blur_knl_pre'], config['gauss_blur_knl_pre']), 0)
    hsv_frame = cv2.cvtColor(mask_blurred, cv2.COLOR_BGR2HSV)

    # Eye bounds (convex hulls)
    left_hull, right_hull = get_eye_bounds(bgr_frame, config)
    if config['of_process_stage'] == 1:
        print("Applying eye mask to original frame.")
        of_frame = CROP(bgr_frame, left_hull, right_hull)
    
    # PROCESSING #1: HSV Masking 
    mask_face, hsv_face_masked_result = process_face_mask(config, bgr_frame, hsv_frame, left_hull, right_hull)
    _CombinedImages = DISPLAY_FRAME("Face Masked Result", hsv_face_masked_result, _CombinedImages, TOTAL_FRAMES)

    if config['of_process_stage'] == 2:
        # Apply the eye mask to the dilated HSV mask, and use it for Optical Flow
        of_frame = CROP(mask_face, left_hull, right_hull)
    
    # PROCESSING #2: Retina-mask | face-mask (combined masks)
    mask_retina, hsv_retina_masked = process_retina_mask(config, hsv_frame, bgr_frame, left_hull, right_hull)

    if config['of_process_stage'] == 3:
        of_frame = of_apply_histogram(config, hsv_retina_masked, invert=True)
        of_frame = CROP(mask_retina, left_hull, right_hull)
    
    # COMBINE eyeball and skin masks
    mask_combined = cv2.bitwise_or(mask_face, mask_retina)
    _CombinedImages = DISPLAY_FRAME("Combined Mask", mask_combined, _CombinedImages, TOTAL_FRAMES)

    # APPLY the combined mask to with bitwise_and to the original frame
    bgr_combined_masked = cv2.bitwise_and(bgr_frame, bgr_frame, mask=mask_combined)

    if config['of_process_stage'] == 4:
        of_frame = of_apply_histogram(config, bgr_combined_masked, invert=True)
        of_frame = CROP(mask_combined, left_hull, right_hull)

    _CombinedImages = DISPLAY_FRAME("Masked Final Result", bgr_combined_masked, _CombinedImages, TOTAL_FRAMES)
    
    # MOTION calculation
    try:
        calculate_motion(config, of_frame, left_hull, right_hull)
    except Exception as e:
        print("Error calculating motion:", e)

    # DISPLAY images
    cv2.namedWindow("Processing Steps", cv2.WINDOW_NORMAL)
    cv2.imshow("Processing Steps", _CombinedImages)


prev_area = 0
def calculate_motion(config, of_frame, left_hull, right_hull):
    global prev_area
    # Grey version of the frame for Optical Flow
    of_frame = cv2.cvtColor(of_frame, cv2.COLOR_BGR2GRAY) if len(of_frame.shape) == 3 else of_frame # grey

    # 6A: Eye-area (dilated result)
    eye_area_frame = of_frame
    if len(of_frame.shape) == 3: eye_area_frame = cv2.cvtColor(of_frame, cv2.COLOR_BGR2GRAY)
    if config['clip_threshold'] < 1:
        _, eye_area_frame = cv2.threshold(eye_area_frame, config['clip_threshold']/100, 255, cv2.THRESH_BINARY)
        area_sum = np.sum(eye_area_frame)
    else:
        print("Clip threshold is 1. Using 'count non zero' hard-pixel-count.")
        area_sum = cv2.countNonZero(eye_area_frame)
    area_sum = EMA(prev_area, area_sum, EYE_AREA_SMOOTHING_FACTOR())
    prev_area = area_sum
    # 6B: Compute the optical flow 
    of_motion_compute(left_hull, right_hull, of_frame, OF_SMOOTHING_FACTOR(), area_sum)


def update(in_frame):
    global WindowIndex
    process(in_frame)
    WindowIndex = 0 # Reset for row/column display map


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
    img_lm, faces_lm, faces_lms = Detector.FindFaceMesh(frame, False)
    return faces_lms


def get_eye_lms_data(frame):
    '''Gets the landmarks that surround the eyes, and returns tuple (left_eye, right_eye).'''
    lms = get_faces_lms(frame)
    
    # Check if any face landmarks were detected
    if not lms or len(lms) == 0:
        return None, None

    # Get the first (and usually only) face's landmark data
    face_lms = lms[0]

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


def get_eye_bounds(frame, config):
    '''Main function to process the frame and extract eye landmark data.'''
    # Get the eye landmarks
    left_eye, right_eye = get_eye_lms_data(frame)
    if left_eye is None or right_eye is None:
        return None, None
    
    # Enlarge the eye boundary landmarks
    left_eye = expand_eye_bound_lms(left_eye, config['eye_bounds_expand'] / 10)        # The slider is from 10 to 20
    right_eye = expand_eye_bound_lms(right_eye, config['eye_bounds_expand'] / 10)      # The slider is from 10 to 20

    # Draw the landmarks and eye borders
    # draw_lms_cv(frame, left_eye)
    # draw_lms_cv(frame, right_eye)
    left_shape_hull = draw_eye_border(frame, left_eye)
    right_shape_hull = draw_eye_border(frame, right_eye)

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


def draw_eye_border(frame, landmarks, color=(0, 255, 0), thickness=2):
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
    # cv2.polylines(frame, [hull], True, color, thickness)

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
    global _fps, WindowIndex, config_window, config_app

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
