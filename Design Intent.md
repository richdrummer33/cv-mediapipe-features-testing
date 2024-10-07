
*INTENT*
To improve upon and develop an image-processing flow to be used for open/closed/blink detection in a Unity-engine game.

The image processing flow that results from developments here will be implemented in Unity using OpenCVForUnity, 
and passed to a new CV-based detection manager for improving webcam-based detection of eyelid states and movements, which currently relies solely on landmark data from the Homuler Mediapipe plugin.

The classic CV approach in development here is to be designed for the purpose of improving detection by providing 
a source of raw data from classic CV algorithms to enhance the limitations of landmark tracking, which can struggle 
in edge cases (glasses with reflections, glasses rims blocking eyes, low light, etc.).

Modules:
- cv2: OpenCV library for image processing.
- math: Mathematical functions.
- numpy: Numerical operations on arrays.
- time: Time-related functions.
- threading: Thread-based parallelism.
- sys: System-specific parameters and functions.
- FaceMeshModule: Custom module for face mesh detection.
- PyQt5: GUI library for creating the configuration window.
- matplotlib: Plotting library for visualizing data.
- queue: Queue data structure for thread-safe data exchange.
- pyqtgraph: Plotting library for real-time data visualization.