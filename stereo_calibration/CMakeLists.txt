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

add_library(stereo_calibration "")

target_sources(stereo_calibration 
PRIVATE ${CMAKE_CURRENT_LIST_DIR}/stereo_calibration.cpp
PUBLIC ${CMAKE_CURRENT_LIST_DIR}/stereo_calibration.h)
target_link_libraries(stereo_calibration  ${THIRD_PARTY_LIBS} )

target_include_directories(stereo_calibration
PUBLIC ${CMAKE_CURRENT_LIST_DIR})

add_executable(stereo_image stereo_image.cpp)
target_link_libraries(stereo_image ${THIRD_PARTY_LIBS} stereo_calibration)

add_executable(stereo_keypoint stereo_keypoint.cpp)
target_link_libraries(stereo_keypoint ${THIRD_PARTY_LIBS} stereo_calibration)

# add_executable(main main.cpp)
# target_link_libraries(main ${THIRD_PARTY_LIBS})