# Stereo Calibration

Repository to implement stereo camera calibration, including histogram equalization, undistortion, and rectification.


## Process
1. Load camera parameters from `config.py`
2. Load stereo images from `original_images/` folder
3. Equalize histogram of stereo images
4. Detect and match key points via ORB
5. Undistortion
  - `stereo_image.py`: undistort entire image
  - `stereo_keypoint.py`: undistort only key points
6. Rectification


## Result
### Rectification
You can see the result images in `result_images` folder.


### Time cost
- `stereo_image.py`: 0.03804 sec
- `stereo_keypoint.py`: 0.03253 sec


## To do

- [X] Convert code from calibrating whole image to calibrating only key points
  - [X] Feature matching first, and undistort key points
  - [X] Calculate R, t and rectify(perspective transform) undistorted key points
  - [X] Plot obtained key points on the rectified images
- [ ] Convert code from Python to C++
- [ ] Refactoring code available for ROS2

## Reference

- [Stereo Calibration](https://github.com/wingedrasengan927/Stereo-Geometry/blob/master/Fundamental%20Matrix%20and%20Stereo%20Rectification.ipynb)
