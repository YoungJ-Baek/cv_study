cmake_minimum_required(VERSION 3.5)
project(stereo-calibration)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR}
)

set( THIRD_PARTY_LIBS
    ${OpenCV_LIBS})

add_executable(stereo_calibration stereo_calib.cpp)
target_link_libraries(stereo_calibration ${THIRD_PARTY_LIBS})