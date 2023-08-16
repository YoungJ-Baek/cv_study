# Stereo Calibration

Repository to implement stereo camera calibration, including histogram equalization, undistortion, and rectification.


## Python
1. Load camera parameters from `config.py`
2. Load stereo images from `original_images/` folder
3. Equalize histogram of stereo images
4. Detect and match key points via ORB
5. Undistortion
  - `stereo_image.py`: undistort entire image
  - `stereo_keypoint.py`: undistort only key points
6. Rectification


## C++
1. Use hard-coded camera parameters in each `stereo_image.cpp` and `stereo_keypoint.cpp`
2. Load stereo images from `original_images/` folder
3. Equalize histogram of stereo images (`stereo_calibration` library)
4. Detect and match key points via ORB (`stereo_calibration` library)
5. Undistortion (`stereo_calibration` library)
  - `stereo_image.cpp`: undistort entire image
  - `stereo_keypoint.cpp`: undistort only key points
6. Rectification (`stereo_calibration` library)


## Result
### Rectification
You can see the result images in `result_images` folder.


### Time cost

Environment: M1 Macbook Pro 14'
- `stereo_image.py`: 0.911449 sec
- `stereo_keypoint.py`: 0.09214 sec
- `stereo_keypoint.cpp`: 0.051377 sec
- `stereo_image.cpp`: 0.085161

~~Needs to optimize C++ code according to the result.~~
Needs to experiment in same environment


## To do

- [X] Convert code from calibrating whole image to calibrating only key points
  - [X] Feature matching first, and undistort key points
  - [X] Calculate R, t and rectify(perspective transform) undistorted key points
  - [X] Plot obtained key points on the rectified images
- [X] Convert code from Python to C++
  - [X] Stereo calibration for whole image
  - [X] Stereo calibration for key points
  - [X] Calculate time cost via C++
- [X] Refactoring code available for ROS2
  - Refactor and merge to [HRRG repository](https://github.com/hrrg/slam-tutorial)

## Reference

- [Stereo Calibration](https://github.com/wingedrasengan927/Stereo-Geometry/blob/master/Fundamental%20Matrix%20and%20Stereo%20Rectification.ipynb)
