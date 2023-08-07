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

add_executable(stereo_image stereo_image.cpp)
target_link_libraries(stereo_image ${THIRD_PARTY_LIBS})

add_executable(stereo_keypoint stereo_keypoint.cpp)
target_link_libraries(stereo_keypoint ${THIRD_PARTY_LIBS})