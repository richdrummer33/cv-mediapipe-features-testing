import cv2
import numpy as np
import time as Time

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
# ===================== FUNCTIONS =====================
# =====================================================

def color_bgr_to_hsv(bgr_color):
    # Convert the BGR color to HSV
    hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]

    return hsv_color


def frame_bgr_to_hsv(frame):
    # Convert the frame from BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    return hsv_frame

# GOOD VALUES (Logi fancy cam): 
#   offset=0, range = 20 
#   low=[.. 25, 85], up=[.. 255, 255]
def frame_mask_proproc(frame):
    '''Converts to HSV to isolate a color in the frame, and removes non-colors'''
    # https://stackoverflow.com/questions/8753833/exact-skin-color-hsv-range
    offset = 0
    range = 20

    blur = cv2.GaussianBlur(frame, (3,3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    lower_color = np.array([_average_color_hsv[0] - range + offset, 25, 85], dtype=np.uint8) # np.array([108, 23, 82])
    upper_color = np.array([_average_color_hsv[0] + range + offset, 255, 255], dtype=np.uint8) # np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    blur = cv2.medianBlur(mask, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    hsv_d = cv2.dilate(blur, kernel)

    masked = cv2.bitwise_and(frame, frame, mask=hsv_d)

    return masked


def preprocess(clahe_frame):

    # Isolate the color
    masked_frame = frame_mask_proproc(clahe_frame)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to the grayscale image
    clahe_frame = clahe.apply(gray_frame)

    # Apply background subtraction
    foreground_mask = _bg_subtractor.apply(clahe_frame)

    # Apply Gaussian blur
    # blurred_frame = cv2.GaussianBlur(foreground_mask, (7, 7), 0)

    # Apply dilation
    # dilated_frame = cv2.dilate(blurred_frame, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # Bitwise AND between grayscale and foreground mask
    result = cv2.bitwise_and(gray_frame, foreground_mask) # dilated_frame

    # Apply histogram equalization
    result = cv2.equalizeHist(result)

    # Print fps
    try: cv2.putText(result, f"FPS: {_fps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    except: pass

    return result



def main():
    global _fps

    # Open the default webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    t = Time.time() # time in seconds
    
    print("Webcam is open:", cap.isOpened())
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")
    

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Process the frame
        processed_frame = preprocess(frame)

        # Display the original and processed frames
        cv2.imshow('Original', frame)
        cv2.imshow('Processed', processed_frame)

        # Calculate FPS
        _fps = 1 / (Time.time() - t)
        t = Time.time()

        # Wait for 'q' key to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
