# Design Considerations*

## Lighting Normalization Techniques
- Adaptive Illumination Compensation: Implement adaptive histogram equalization (CLAHE) on the entire face or just around the eyes to minimize the impact of uneven lighting.
- Independent HSV Sampling: 
  - Instead of a single sampled HSV reference for both sides of the face, dynamically adapt the HSV reference for each eye individually. You could update the reference color separately for left and right sides based on the local region.
  - Use smaller ROIs (e.g., cheeks or forehead) on the left and right sides for separate HSV reference calculation to better adapt to uneven lighting across the face.


## Stability
- Landmark Tracking for Stability: Leverage facial landmarks (like forehead or nose bridge) to determine head movement and subtract that motion to keep eyelid velocity detection stable. Kalman filters can also be used to smooth the landmark movement to reduce noise.
- Implement an optimization algorithm, like Particle Swarm Optimization (PSO) or grid search, that runs periodically in the background to tune the HSV and other detection thresholds based on input quality
