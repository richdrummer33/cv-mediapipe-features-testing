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


# =====================================================
# ===================== SETUP =========================
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
            ('Hue Range', 20, 90),
            ('Hue Offset', 0, 180),
            ('Saturation Low', 25, 255),
            ('Saturation High', 255, 255),
            ('Value Low', 85, 255),
            ('Value High', 255, 255),
            ('Gaussian Kernel', 3, 25),
            ('Median Kernel', 5, 25),
            ('Dilation Kernel', 8, 20),
            ('CLAHE Clip Limit', 20, 100),
            ('CLAHE Grid Size', 8, 16),
            ('BG Sub Learning Rate', 5, 100),
            ('BG Sub History', 500, 1000),
            ('BG Sub Var Threshold', 16, 100),
            ('Eye Bounds Expand', 10, 20)
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

    def get_window_config(self):
        """Retrieve current values from all sliders."""
        config = {
            'hue_range': self.sliders['Hue Range'].value(),
            'hue_offset': self.sliders['Hue Offset'].value(),
            'sat_low': self.sliders['Saturation Low'].value(),
            'sat_high': self.sliders['Saturation High'].value(),
            'val_low': self.sliders['Value Low'].value(),
            'val_high': self.sliders['Value High'].value(),
            'gaussian_kernel': self.sliders['Gaussian Kernel'].value(),
            'median_kernel': self.sliders['Median Kernel'].value(),
            'dilation_kernel': self.sliders['Dilation Kernel'].value(),
            'clahe_clip_limit': self.sliders['CLAHE Clip Limit'].value() / 10.0,
            'clahe_grid_size': self.sliders['CLAHE Grid Size'].value(),
            'bg_sub_learning_rate': self.sliders['BG Sub Learning Rate'].value() / 1000.0,
            'bg_sub_history': self.sliders['BG Sub History'].value(),
            'bg_sub_var_threshold': self.sliders['BG Sub Var Threshold'].value(),
            'eye_bounds_expand': self.sliders['Eye Bounds Expand'].value(),
        }
        return config

def get_config():
    global config_window
    if config_window is not None:
        return config_window.get_window_config()
    else:
        # Return default config or handle the error as needed
        return {
            'hue_range': 20,
            'hue_offset': 0,
            'sat_low': 25,
            'sat_high': 255,
            'val_low': 85,
            'val_high': 255,
            'gaussian_kernel': 3,
            'median_kernel': 5,
            'dilation_kernel': 8,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': 8,
            'bg_sub_learning_rate': 0.005,
            'bg_sub_history': 500,
            'bg_sub_var_threshold': 16,
            'eye_bounds_expand': 10,
        }


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


# =====================================================
# ===================== HELPERS =======================
# =====================================================

def color_bgr_to_hsv(bgr_color):
    # Convert the BGR color to HSV
    hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
    return hsv_color


def frame_bgr_to_hsv(frame):
    # Convert the frame from BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv_frame

_window_index = 0

def display_step(window_name, img_frame, combined_frame, num_images, window_width=300, window_height=300):
    """
    Displays all frames in a single window, filling left to right, then creating new rows as needed.
    
    Parameters:
    - window_name (str): The name of the window.
    - img_frame (numpy.ndarray): The image/frame to display.
    - combined_frame (numpy.ndarray, optional): The combined frame containing all images.
    - num_images (int): Total number of images being processed.
    - window_width (int): Width of the display window.
    - window_height (int): Height of the display window.
    
    Returns:
    - combined_frame (numpy.ndarray): The updated combined frame with the new image added.
    """
    global _window_index

    # Get image dimensions
    img_height, img_width = img_frame.shape[:2]

    # Initialize combined_frame if it's None
    if combined_frame is None:
        # Calculate number of columns that can fit in the window
        num_columns = max(1, window_width // img_width)
        # Calculate number of rows needed based on total images
        num_rows = math.ceil(num_images / num_columns)
        # Calculate combined frame size
        combined_height = min(num_rows * img_height, window_height)
        combined_width = min(num_columns * img_width, window_width)
        # Create a black canvas
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Recalculate number of columns based on combined_frame width and image width
    num_columns = max(1, combined_frame.shape[1] // img_width)

    # Determine row and column for the current image
    row = _window_index // num_columns
    col = _window_index % num_columns

    # Calculate starting x and y positions
    start_x = col * img_width
    start_y = row * img_height

    # Check if the image fits within the combined_frame
    if start_x + img_width > combined_frame.shape[1] or start_y + img_height > combined_frame.shape[0]:
        print(f"Image {_window_index + 1} does not fit in the combined frame. Skipping.")
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
    _window_index += 1

    return combined_frame


# =====================================================
# ===================== IMAGE  ========================
# =====================================================

sproc = False

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
    
    
def mask_preprocess(frame):
    '''Converts to HSV to isolate a color in the frame, and removes non-colors'''
    
    # ~~~ SET THINGS UP ~~~
    # ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
    # Window configuration
    global _window_index
    total_frames = 7  # Adjust this number based on how many steps/images you are displaying
    grid_frame = np.zeros((frame.shape[0] * 2, frame.shape[1] * 3, 3), dtype=np.uint8) # 2x3 grid

    # Get the current configuration
    config = get_config()

    # Display original frame
    if not sproc: grid_frame = display_step("Original", frame, grid_frame, total_frames)


    # ~~~ MEDIAPIPE FACE MESH ~~~
    # ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
    # Get bounds of the eyes as convex hulls (borders the eyes)
    left_hull, right_hull = get_eye_bounds(frame, config)

    # if the hulls are not found, return the original frame
    if left_hull is not None and right_hull is not None:
        masked_frame = eye_bounds_mask(frame, left_hull, right_hull)
        grid_frame = display_step("Eye-Bounds-Masked", masked_frame, grid_frame, total_frames)

    # ~~~ RUN PROCESSING STEPS ~~~
    # ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
    # Step 1: Gaussian blur to soften the image
    blur = cv2.GaussianBlur(frame, (config['gaussian_kernel'], config['gaussian_kernel']), 0)
    if not sproc: grid_frame = display_step("Blurred", blur, grid_frame, total_frames)  # Display the blurred image

    # Step 2: Convert the image to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    if not sproc: grid_frame = display_step("HSV", hsv, grid_frame, total_frames)  # Display the HSV image

    # HSV color gates (skin tone range)
    lower_color = np.array([_average_color_hsv[0] - config['hue_range'] + config['hue_offset'], config['sat_low'], config['val_low']], dtype=np.uint8)
    upper_color = np.array([_average_color_hsv[0] + config['hue_range'] + config['hue_offset'], config['sat_high'], config['val_high']], dtype=np.uint8)

    # Step 3: Create a mask based on the HSV range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    if not sproc: grid_frame = display_step("Mask", mask, grid_frame, total_frames)  # Display the mask (binary image)

    # Step 4: Median blur the mask to remove noise
    blur = cv2.medianBlur(mask, config['median_kernel'])
    if not sproc: grid_frame = display_step("Median Blur on Mask", blur, grid_frame, total_frames)  # Display the median blurred mask

    # Step 5: Dilate the mask to make objects more connected (thick)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config['dilation_kernel'], config['dilation_kernel']))
    hsv_d = cv2.dilate(blur, kernel)
    if not sproc: grid_frame = display_step("Dilated Mask", hsv_d, grid_frame, total_frames) # Display the dilated mask

    # Step 6: Bitwise AND the original frame with the mask to get the final output
    masked = cv2.bitwise_and(frame, frame, mask=hsv_d)
    if not sproc: grid_frame = display_step("Masked Final Frame", masked, grid_frame, total_frames)  # Display the final masked frame

    # finally, display all the steps combined
    if not sproc:
        cv2.namedWindow("Processing Steps", cv2.WINDOW_NORMAL)
        cv2.imshow("Processing Steps", grid_frame)

    return masked


def update(in_frame):
    global _window_index

    # Multi-image display
    combined_frame = None
    total_frames = 5  # Adjust this number based on how many steps/images you are displaying

    # Display original frame
    if sproc: combined_frame = display_step("Original", in_frame, combined_frame, total_frames)

    # Isolate the color
    masked_frame = mask_preprocess(in_frame)
    if sproc: combined_frame = display_step("Masked", masked_frame, combined_frame, total_frames)
    
    # Convert to grayscale
    gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to the grayscale image
    in_frame = clahe.apply(gray_frame)
    if sproc: combined_frame = display_step("CLAHE/grey", in_frame, combined_frame, total_frames)

    # Apply background subtraction
    foreground_mask = _bg_subtractor.apply(in_frame)
    if sproc: combined_frame = display_step("Foreground Mask", foreground_mask, combined_frame, total_frames)

    # Bitwise AND between grayscale and foreground mask
    result = cv2.bitwise_and(gray_frame, foreground_mask)
    result = cv2.equalizeHist(result)
    if sproc: combined_frame = display_step("Equalized", result, combined_frame, total_frames)

    # Named-resizeable window, and display it
    if sproc:
        cv2.namedWindow("Combined Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Combined Frame", combined_frame)

    _window_index = 0
    return result


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


def get_all_lms(frame):
    '''Gets all the landmarks from the frame.'''
    img_lm, face_lm, landmarks = Detector.FindFaceMesh(frame, False)
    return landmarks


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
    global _fps, _window_index, config_app

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
        processed_frame = update(frame)

        # Calculate FPS
        _fps = 1 / (Time.time() - t)
        t = Time.time()

        # Display the processed frame if needed
        # cv2.imshow('Processed', processed_frame)

        # Wait for 'q' key to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting application...")
            # Safely quit the PyQt application from the video thread
            config_app.quit()
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    global _fps, _window_index, config_window, config_app

    # Initialize QApplication and ConfigWindow
    config_app = QApplication(sys.argv)
    config_window = ConfigWindow()
    config_window.show()

    # Start video processing in a separate daemon thread
    video_thread = threading.Thread(target=video_capture_loop, args=(), daemon=True)
    video_thread.start()

    # Execute the PyQt5 application in the main thread
    sys.exit(config_app.exec_())


if __name__ == "__main__":
    main()
