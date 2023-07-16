# Stereo Calibration

Repository to implement stereo camera calibration, including histogram equalization, undistortion, and rectification.

## To do

- [X] Convert code from calibrating whole image to calibrating only key points
  - [X] Feature matching first, and undistort key points
  - [X] Calculate R, t and rectify(perspective transform) undistorted key points
  - [X] Plot obtained key points on the rectified images
- [ ] Convert code from Python to C++
- [ ] Refactoring code available for ROS2

## Reference

- [Stereo Calibration](https://github.com/wingedrasengan927/Stereo-Geometry/blob/master/Fundamental%20Matrix%20and%20Stereo%20Rectification.ipynb)
