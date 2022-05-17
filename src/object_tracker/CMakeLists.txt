cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(object_tracker)
SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
find_package(OpenCV REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
set(realsense2_DIR "/usr/lib/x86_64-linux-gnu/cmake/realsense2/")
find_package(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
    std_msgs
    message_filters
    timeArr_msgs
    realsense2
)

if (OpenCV_FOUND)
    # If the package has been found, several variables will
    # be set, you can find the full list with descriptions
    # in the OpenCVConfig.cmake file.
    # Print some message showing some of them
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate OpenCV" \n)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_PREFIX_PATH /home/nesl/Downloads/libtorch/share/cmake)
find_package(Torch REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch library found!")
    message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate Torch" \n)
endif()


catkin_package()
add_executable (testYOLO src/run.cpp)

include_directories(
${catkin_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/src
${OpenCV_INCLUDE_DIRS}
${realsense2_INCLUDE_DIRS}
)
#include_directories( ${OpenCV_INCLUDE_DIRS} )

## Declare a C++ library
add_library(JASON SHARED
src/KalmanTracker.cpp
src/Hungarian.cpp
src/data_structure.cpp
src/SVM/VacuumClassifier.h
src/SVM/FanClassifier.h
src/SVM/WashingClassifier.h
src/SVM/HumanClassifier.h
src/VMD/VMD.h
src/VMD/VMD_Utils.cpp
)
target_link_libraries(testYOLO ${catkin_LIBRARIES})
target_link_libraries(JASON Eigen3::Eigen)
target_link_libraries(testYOLO ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${realsense2_LIBRARY} JASON)
target_link_libraries(testYOLO Eigen3::Eigen)
set_property(TARGET testYOLO PROPERTY CXX_STANDARD 14)
