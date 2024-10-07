

*CURRENT APPROACH / DESIGN*
The `process` function is responsible for processing a given frame to isolate a specific color and remove non-colors. 
It performs several preprocessing steps, including downscaling, Gaussian blurring, HSV conversion, median blurring, 
dilation, and bitwise operations to achieve the desired output. Additionally, it integrates with the Mediapipe Face 
Mesh to obtain eye bounds and applies Optical Flow (OF) computation at different stages of the processing pipeline.
- frame (numpy.ndarray): The input frame to be processed.
- None
Steps:
1. Downscale the frame for performance optimization.
2. Obtain eye bounds using the Mediapipe Face Mesh.
3. Apply Gaussian blur to the frame.
4. Convert the frame to HSV color space.
5. Create a mask based on the HSV range and apply median blur to remove noise.
6. Dilate the mask to make objects more connected.
7. Bitwise AND the original frame with the mask to get the final output.
8. Compute Optical Flow (OF) at different stages of the processing pipeline.
Limitations:
- The current approach relies heavily on color isolation, which may not be robust in varying lighting conditions or 
  with different skin tones.
- The HSV range for color isolation is fixed and may need manual adjustment for different environments.
- The Optical Flow computation is sensitive to noise and may produce inaccurate results if the input frames are not 
  preprocessed adequately.
- The performance of the pipeline may be affected by the resolution of the input frames and the complexity of the 
  processing steps.
- The Mediapipe Face Mesh may struggle with detecting eye bounds accurately in certain scenarios, such as when the 
  subject is wearing glasses or in low-light conditions.
Future Improvements:
- Implement adaptive HSV range adjustment based on the environment and subject's skin tone.
- Enhance the robustness of the Optical Flow computation by incorporating additional preprocessing steps or using 
  more advanced algorithms.
- Optimize the performance of the pipeline by parallelizing certain steps or using hardware acceleration.
- Improve the accuracy of the Mediapipe Face Mesh detection by fine-tuning the model or using additional landmarks.