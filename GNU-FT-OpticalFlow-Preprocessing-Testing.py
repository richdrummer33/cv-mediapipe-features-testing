'''
This script performs preprocessing steps for the Optical Flow algorithm, including color isolation, 
face mesh detection, and optical flow computation. It also provides a GUI for configuring various 
parameters using PyQt5.

This is for developing an image-processing flow, to be used for open/closed/blink detection in a Unity-engine game.
The image processing flow that results from developments here will be implemented in Unity using OpenCVForUnity.

Classes:
- ConfigWindow: A PyQt5 QMainWindow subclass for configuring various parameters using sliders.
- PlotWindow: A PyQt5 QWidget subclass for displaying real-time plots of eyelid motion magnitude.

Functions:
- get_bgr_from_reference_image(image_path): Reads an image and returns the average BGR color.
- get_config(): Retrieves the current configuration values from the ConfigWindow.
- DISPLAY_FRAME(window_name, img_frame, combined_frame, num_images): Displays multiple frames in a single window.
- plotter(): Initializes and displays the PlotWindow.
- downscale_image(image, levels): Downscales an image by a specified number of levels.
- EMA(prev_val, curr_val, smoothing_factor): Computes the Exponential Moving Average (EMA) for smoothing.
- of_motion_compute(left_hull, right_hull, frame_grey, smoothing_factor, secondary_plot_value): Computes the average vertical motion of the eyelids within the eye regions.
- hsv_range(config): Returns the lower and upper HSV bounds for color isolation based on the configuration.
- eye_bounds_mask(frame, left_hull, right_hull): Creates a mask from the shape hulls and keeps the eye areas.
- CROP(frame, left_hull, right_hull): Crops the eye areas from the frame using convex hulls.
- process(frame): Processes a frame to isolate a color and remove non-colors.
- update(in_frame): Updates the frame by processing it and displaying the results.
- get_all_lms(frame): Gets all the landmarks from the frame using the FaceMeshDetector.
- get_eye_lms_data(frame): Gets the landmarks that surround the eyes and returns them as a tuple (left_eye, right_eye).
- get_eye_lms_dict(left_eye, right_eye): Processes the eye landmarks to extract the eye socket, pupil, etc., as a dictionary.
- expand_eye_bound_lms(eye_points, factor): Expands the eye boundary landmarks by a specified factor.
- get_eye_bounds(frame, config): Main function to process the frame and extract eye landmark data.

Global Variables:
- _bg_subtractor: Background subtractor for foreground extraction.
- clahe: CLAHE object for contrast limited adaptive histogram equalization.
- lower_bound: Lower HSV bound for color isolation.
- upper_bound: Upper HSV bound for color isolation.
- _fps: Frames per second for video processing.
- _bgr_color_ref_path: File path to the reference image for BGR color extraction.
- bgr_color_ref: Average BGR color from the reference image.
- _average_color_hsv: Average HSV color converted from the BGR reference color.
- CombinedGridFrame: Combined frame for displaying multiple images.
- TOTAL_FRAMES: Total number of frames to be displayed.
- COLUMNS_FRAMES: Number of columns in the grid for displaying frames.
- config_window: Global variable for the configuration window.
- config_app: Global variable for the configuration application.
- WindowIndex: Index for tracking the current window in the grid display.
- plot_queue: Queue for storing data to be plotted.
- prev_grey: Previous grayscale frame for optical flow computation.
- prev_left_centroid: Previous centroid of the left eye for optical flow computation.
- prev_right_centroid: Previous centroid of the right eye for optical flow computation.
- motion_history_left: History of motion magnitudes for the left eye.
- motion_history_right: History of motion magnitudes for the right eye.
- smoothed_left_motion: Smoothed vertical motion for the left eye.
- smoothed_right_motion: Smoothed vertical motion for the right eye.
- prev_area: Previous area of the eye region for motion computation.
- Detector: FaceMeshDetector object for detecting face landmarks.
'''

# Typical
import cv2
import math
import numpy as np
import time as Time
import threading
import sys

# FaceMesh
from FaceMeshModule import FaceMeshDetector as facelms

# QtWindow UI
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QGridLayout, QLabel, QSlider, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

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
_bgr_color_ref_path = "reference.png"

def get_bgr_from_reference_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.mean(image_rgb, axis=(0, 1))

# Get the average BGR color from the reference image
bgr_color_ref = get_bgr_from_reference_image(_bgr_color_ref_path)

# Convert the average BGR color to HSV
average_color_bgr = np.uint8([[[bgr_color_ref[0], bgr_color_ref[1], bgr_color_ref[2]]]])
_average_color_hsv = cv2.cvtColor(average_color_bgr, cv2.COLOR_BGR2HSV)[0][0]

# print the r,g,b vals
print(f"Average BGR color: {bgr_color_ref}")
# print the h,s,v vals
print(f"Average HSV color: {_average_color_hsv}")

CombinedGridFrame = None
TOTAL_FRAMES = 4
COLUMNS_FRAMES = 3


# =====================================================
# ================== CGF WINDOW =======================
# =====================================================

# Declare global variable
config_window = None
config_app = None

class ConfigWindow(QMainWindow):
    def __init__(self):
        super().__init__()
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

        # Define sliders with (Name, Default Value, Maximum Value)
        sliders = [
            # HSV color isolation
            ('Hue Range', 20, 90),
            ('Hue Offset', 0, 180),
            # HSV thresholds
            ('Saturation Low', 25, 255),
            ('Saturation High', 255, 255),
            ('Value Low', 85, 255),
            ('Value High', 255, 255),
            # OF
            ('OF Smoothing Factor', 10, 100),
            ('Eye-Area Smoothing Factor', 10, 100),
            ('Eye Bounds Expand', 10, 50),
            ('OF Process Stage', 1, 4),
            # Kernels
            ('Median Blur K (HSV mask)', 3, 25),
            ('Gauss Blur K (OG image)', 3, 25),
            ('Dilation Kernel', 8, 20),
            ('Gauss Blur K (dilated eyes)', 3, 25),
            #('CLAHE Clip Limit', 20, 100),
            #('CLAHE Grid Size', 8, 16),
            #('BG Sub Learning Rate', 5, 100),
            #('BG Sub History', 500, 1000),
            #('BG Sub Var Threshold', 16, 100),
        ]

        self.sliders = {}  # Dictionary to store sliders

        for i, (name, default, maximum) in enumerate(sliders):
            label = QLabel(name)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, maximum)
            slider.setValue(default)
            self.sliders[name] = slider  # Store slider in dictionary

            row = i % 8
            col = i // 8 * 2
            slider_layout.addWidget(label, row, col)
            slider_layout.addWidget(slider, row, col + 1)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

    @staticmethod
    def ensure_odd(value):
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
            # Kernels
            'median_blur_knl': self.ensure_odd(self.sliders['Median Blur K (HSV mask)'].value()),
            'gauss_blur_knl_pre': self.ensure_odd(self.sliders['Gauss Blur K (OG image)'].value()),
            'dilation_kernel': max(1, self.sliders['Dilation Kernel'].value()),
            'gauss_blur_knl_dil': self.ensure_odd(self.sliders['Gauss Blur K (dilated eyes)'].value()),
            #'clahe_clip_limit': self.sliders['CLAHE Clip Limit'].value() / 10.0,
            #'clahe_grid_size': self.sliders['CLAHE Grid Size'].value(),
            #'bg_sub_learning_rate': self.sliders['BG Sub Learning Rate'].value() / 1000.0,
            #'bg_sub_history': self.sliders['BG Sub History'].value(),
            #'bg_sub_var_threshold': self.sliders['BG Sub Var Threshold'].value(),
        }
        return config

def get_config():
    global config_window
    if config_window is not None:
        return config_window.get_window_config()
    else:
        # Return default config or handle the error as needed
        return {
            # HSV color isolation
            'hue_range': 20,
            'hue_offset': 0,
            # HSV thresholds
            'sat_low': 25,
            'sat_high': 255,
            'val_low': 85,
            'val_high': 255,
            # OF
            'of_smoothing': 25,
            'eye_area_smoothing': 25,
            'eye_bounds_expand': 10,
            'of_process_stage': 1,
            # Kernels
            'median_blur_knl': 5,
            'gauss_blur_knl_pre': 5,
            'dilation_kernel': 8,
            'gauss_blur_knl_dil': 5,
            #'clahe_clip_limit': 2.0,
            #'clahe_grid_size': 8,
            #'bg_sub_learning_rate': 0.005,
            #'bg_sub_history': 500,
            #'bg_sub_var_threshold': 16,
        }
    

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
# ================= OPTICAL FLOW ======================
# =====================================================

# EMA
def EMA(prev_val, curr_val, smoothing_factor):
    if prev_val is None:
        return curr_val
    return (smoothing_factor * curr_val) + ((1 - smoothing_factor) * prev_val)

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


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# [gpt-01] https://chatgpt.com/c/66fda7cd-c828-8000-a3ec-cafa4fe69603
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 
 
def sample_skin_tone(frame, lms):
    '''
    Samples skin tone from specific face landmarks. 
    Returns the average HSV color.
    '''
    col_samples = []

    # Define the neighborhood size
    neighborhood_size = 3

    # Sample pixels around the landmarks
    for key in face_sample_spots:
        # Some sample spots have > 1 point
        sample_point_indexes = face_sample_spots[key]

        # Sample pixels around each point
        for pt_index in sample_point_indexes:
            # Get the landmark point location/vect from the pt_index
            pt = lms[pt_index]
            pt_x, pt_y = int(pt[0]), int(pt[1])
            # sum the pixel values in the neighborhood
            col_sum = np.zeros(3)
            for i in range(-neighborhood_size, neighborhood_size + 1):
                for j in range(-neighborhood_size, neighborhood_size + 1):
                    col_sum += frame[pt_y + i, pt_x + j]
            # Average the pixel values
            col_avg = col_sum / ((2 * neighborhood_size + 1) ** 2)
            col_samples.append(col_avg)

    # Return default if no samples
    if not col_samples:
        return _average_color_hsv 
    
    # Compute the mean color and convert to HSV
    mean_color_bgr = np.mean(col_samples, axis=0)
    average_color_bgr = np.uint8([[mean_color_bgr]])
    ave_hsv_col = cv2.cvtColor(average_color_bgr, cv2.COLOR_BGR2HSV)[0][0]
    return ave_hsv_col


def hsv_range(config, ave_hsv_col):
    # (Same as before, but gpt-o1 modified using "max" operator)
    # HSV color gates (skin tone range)
    lower_color = np.array([
        max(0, ave_hsv_col[0] - config['hue_range'] + config['hue_offset']), 
        config['sat_low'], 
        config['val_low']], dtype=np.uint8)
    upper_color = np.array([
        min(180, ave_hsv_col[0] + config['hue_range'] + config['hue_offset']), 
        config['sat_high'], 
        config['val_high']], dtype=np.uint8)
    return lower_color, upper_color


_prev_area = 0

def process(frame):
    # >>>>>>>>>>>>>>>>>>>>>>> [gpt-01] >>>>>>>>>>>>>>>>>>>>>>>>>>
    '''Converts to HSV to isolate a color in the frame, and removes non-colors'''
    global WindowIndex, CombinedGridFrame, _prev_area, _average_color_hsv
    WindowIndex = 0  # Reset window index
    area = 0

    # ~~~ SET THINGS UP ~~~
    CombinedGridFrame = None   # Grid display
    of_frame = None     # The frame for Optical Flow computation
    config = get_config()

    # ~~~ DOWNSCALE (PERFORMANCE) ~~~
    frame = downscale_image(frame, 1)
    of_frame = frame  # Default to the original frame
    CombinedGridFrame = DISPLAY_FRAME("Original", frame, CombinedGridFrame, TOTAL_FRAMES) 

    # ~~~ MEDIAPIPE FACE MESH ~~~
    # Get bounds of the eyes as convex hulls
    left_hull, right_hull = get_eye_bounds(frame, config)

    # ➡️ **New Code: Get all face landmarks and sample skin tone**
    faces_lms = get_all_lms(frame)
    if faces_lms and len(faces_lms) > 0:
        _average_color_hsv = sample_skin_tone(frame, faces_lms[0])  # Assuming single face

    # ➡️ **Modify hsv_range to accept average_color_hsv**
    lo_hsv_gate, hi_hsv_gate = hsv_range(config, _average_color_hsv)

    # ~~~ CONVERT TO HSV ~~~
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ~~~ CREATE MASK BASED ON DYNAMIC HSV RANGE ~~~
    hsv_mask = cv2.inRange(hsv_frame, lo_hsv_gate, hi_hsv_gate)
    hsv_mask_blur = cv2.medianBlur(hsv_mask, config['median_blur_knl'])
    CombinedGridFrame = DISPLAY_FRAME("HSV Mask", hsv_mask_blur, CombinedGridFrame, TOTAL_FRAMES)
    # <<<<<<<<<<<<<<<<<<<<<< [gpt-01] <<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ~~~ MEDIAPIPE FACE MESH ~~~
    # Get bounds of the eyes as convex hulls (borders the eyes)
    left_hull, right_hull = get_eye_bounds(frame, config)

    if config['of_process_stage'] == 1:
        # Apply the eye mask to the original frame, and use it for Optical Flow
        frame = CROP(frame, left_hull, right_hull)
        of_frame = frame
        print("OF Stage 1 : Original Frame")

    # ~~~ RUN PROCESSING STEPS ~~~
    # == Step 1 == Gaussian blur to soften the image
    hsv_mask_blur = cv2.GaussianBlur(frame, (config['gauss_blur_knl_pre'], config['gauss_blur_knl_pre']), 0)

    # == Step 2 == Convert the image to HSV color space
    hsv_frame = cv2.cvtColor(hsv_mask_blur, cv2.COLOR_BGR2HSV)

    if config['of_process_stage'] == 2:
        # Apply the eye mask to the HSV image, and use it for Optical Flow
        hsv_frame = CROP(hsv_frame, left_hull, right_hull)
        of_frame = hsv_frame
        print("OF Stage 2")

    # == Step 3 == Create a mask based on the HSV range, and median blur it to remove noise
    lo_hsv_gate, hi_hsv_gate = hsv_range(config, _average_color_hsv)
    hsv_mask = cv2.inRange(hsv_frame, lo_hsv_gate, hi_hsv_gate)
    hsv_mask_blur = cv2.medianBlur(hsv_mask, config['median_blur_knl']) # Median blur differs from gaussian blur in that it takes the median of all the pixels under the kernel area and the central element is replaced with this median value
    CombinedGridFrame = DISPLAY_FRAME("HSV Mask", hsv_mask_blur, CombinedGridFrame, TOTAL_FRAMES)

    # == Step 4 == Dilate the mask to make objects more connected (thick)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config['dilation_kernel'], config['dilation_kernel']))
    hsv_mask_dil = cv2.dilate(hsv_mask_blur, kernel)
    CombinedGridFrame = DISPLAY_FRAME("HSV Mask (dilated)", hsv_mask_dil, CombinedGridFrame, TOTAL_FRAMES)

    if config['of_process_stage'] == 3:
        # Apply the eye mask to the dilated HSV mask, and use it for Optical Flow
        hsv_mask_dil = CROP(hsv_mask_dil, left_hull, right_hull)
        of_frame = hsv_mask_dil
        print("OF Stage 3")

    # == Step 5 == Bitwise AND the original frame with the mask to get the final output
    hsv_masked_out = cv2.bitwise_and(frame, frame, mask=hsv_mask_dil)
    hsv_masked_out = cv2.GaussianBlur(hsv_masked_out, (config['gauss_blur_knl_dil'], config['gauss_blur_knl_dil']), 0)
    CombinedGridFrame = DISPLAY_FRAME("HSV-Masked Result", hsv_masked_out, CombinedGridFrame, TOTAL_FRAMES)
    
    if config['of_process_stage'] == 4:
        # Apply the eye mask to the final hsv-masked frame, and use it for Optical Flow
        hsv_masked_out = CROP(hsv_masked_out, left_hull, right_hull)
        of_frame = hsv_masked_out
        print("OF Stage 4")
    
    # == Step 6 == Compute Motion!
    of_frame = cv2.cvtColor(of_frame, cv2.COLOR_BGR2GRAY) if len(of_frame.shape) == 3 else of_frame # grey
    # 6A: Eye-area (dilated result)
    hsv_masked_out_gray = cv2.cvtColor(hsv_masked_out, cv2.COLOR_BGR2GRAY)
    area = cv2.countNonZero(hsv_masked_out_gray) # single-channel
    area = EMA(_prev_area, area, config['eye_area_smoothing'] / 10)
    _prev_area = area
    # 6B: Compute the optical flow 
    of_motion_compute(left_hull, right_hull, of_frame, config['of_smoothing'] / 10, area)

    # ... Finally, DISPLAY all the steps combined
    cv2.namedWindow("Processing Steps", cv2.WINDOW_NORMAL)
    cv2.imshow("Processing Steps", CombinedGridFrame)
    return hsv_masked_out

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# https://chatgpt.com/c/66fda7cd-c828-8000-a3ec-cafa4fe69603
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def update(in_frame):
    global WindowIndex

    # Multi-image display
    combined_frame = None

    # Isolate the color
    processed_frame = process(in_frame)
    
    # Reset the indx for the row/column display mapping
    WindowIndex = 0


# =====================================================
# =================== MediaPipe =======================
# =====================================================

# Landmark indexes from MP Detection Manager:
"""
Face Landmark Indices:

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

Skin Tone Sampling Points:
    Forehead: 108, 151, 337
    Nose: 5, 51, 281
    Left cheek: 50
    Right cheek: 280
"""

face_sample_spots = {
    'forehead': [108, 151, 337],
    'nose': [5, 51, 281],
    'left_cheek': [50],
    'right_cheek': [280]
}


# Initialize the FaceMeshDetector
print("Creating FaceMeshDetector...")
Detector = facelms()
print("FaceMeshDetector is created!")

# Indices for encompassing the eyes for convex hulls
left_eye_indices_bounds = [56, 247, 112, 110, 173, 33]
right_eye_indices_bounds = [286, 467, 341, 339, 398, 263]


def get_eye_lms_data(frame):
    '''Gets the landmarks that surround the eyes, and returns tuple (left_eye, right_eye).'''
    lms = get_all_lms(frame)
    
    # Check if any face landmarks were detected
    if not lms or len(lms) == 0:
        return None, None

    # Get the first (and usually only) face's landmark data
    face_lms = lms[0]

    # Get the left eye landmarks
    left_eye = []
    for i in left_eye_indices_bounds:
        if i < len(face_lms):
            left_eye.append(face_lms[i])
        else:
            left_eye.append(None)  # or some default value

    # Get the right eye landmarks
    right_eye = []
    for i in right_eye_indices_bounds:
        if i < len(face_lms):
            right_eye.append(face_lms[i])
        else:
            right_eye.append(None)  # or some default value

    # Check if we have all landmarks for both eyes
    if None in left_eye or None in right_eye:
        print("Some eye landmarks are missing.")
        return None, None

    return left_eye, right_eye


def get_all_lms(frame):
    '''Gets all the landmarks from the frame.'''
    img_lm, face_lm, landmarks = Detector.FindFaceMesh(frame, False)
    return landmarks


def get_eye_lms_dict(left_eye, right_eye):
    '''Process the eye landmarks to extract the eye socket, pupil, etc as a dictionary.'''
    if left_eye is None or right_eye is None:
        return None

    # Extract specific landmarks for each eye
    left_eye_data = {
        "upper_socket": left_eye[0:2],
        "lower_socket": left_eye[2:4],
        "inner_corner": left_eye[4],
        "outer_corner": left_eye[5],
    }

    right_eye_data = {
        "upper_socket": right_eye[0:2],
        "lower_socket": right_eye[2:4],
        "inner_corner": right_eye[4],
        "outer_corner": right_eye[5],
    }

    return left_eye_data, right_eye_data


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
    draw_lms_cv(frame, left_eye)
    draw_lms_cv(frame, right_eye)
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
    global _fps, WindowIndex, config_window, config_app

    # Initialize QApplication and ConfigWindow
    config_app = QApplication(sys.argv)
    config_window = ConfigWindow()
    config_window.show()

    # get confg and enforce config['of_process_stage'] == 5
    config = get_config()
    config['of_process_stage'] = 1
    
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
