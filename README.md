# Capricorn
Provide brief explanation of Capricorn here

## System Dependency Installation

### Ros Installation:
Install ROS Melodic from the following link: [ROS Install](http://wiki.ros.org/noetic/Installation)

Note that ROS is only compatible with Linux distributions

### Intel Realsense Installation:
Install Intel Realsense from the following link: [Linux Intel Realsense](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)

Follow the instructions to install the pre-built packages

### Libtorch Installation:
Install libtorch, following the instructions from the link: [Libtorch](https://pytorch.org/cppdocs/installing.html)

Note that installation of a GPU based version of libtorch will require some configuration of existing CUDA environments

### OpenCV Installation:
This project leverages the Aruco AprilTag library from OpenCV, specifically the 36h11 tag family. This was introduced in the 3.4.2 version. Older versions of OpenCV will not be compatible with the current code. 

As a precaution, install the latest version of OpenCV from the following link: [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html). 

**Be sure to also include the OpenCV Contrib Libraries**

## Editing the Code to Run on your Machine

