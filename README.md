# ObjectTracking
Implementation of object tracking using Python.

# About
The purpose of this repository is to make a simple implementation of an object tracking algorithm. For this purpose,
- A pretrained MobileSSD network (for object detection)
- Euclidean distance (to track objects)
is used.

# Dependencies
- Python
- Numpy
- OpenCV

# Structure
- main.py: Main code to capture image from video and process it.
- /model: Folder containing model files for object detection and face detection (to test the tracking algorithm on faces as well)
- /utils: Folder containing classes and functions for tracking.
  - tracker.py: Contains Tracker class which calculates Euclidean distance between newly detected object and previous objects. Selects the closest one and do the labeling.
  - functions.py: Contains function for calculating the centroid of a given box.
- /cfg: Folder containing the configuration file and writer of configuration file.
- /test: Contains test videos and output of the algorithm.

# Results
![Alt Text](https://github.com/tameraktekin/ObjectTracking/blob/main/test/test_output.gif)
# Drawbacks
Using Euclidean distance to track objects has to major drawbacks.
  - First, object detector must be run for each frame which is time consuming.
  - Second, overlapping objects can change ids resulting for tracker to track wrong objects.
  
# To do
- Implement an object detection model and train it.
