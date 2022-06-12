# Capricorn
Capricorn is a cross-modal sensor fusion system combining RF sensor data with visual pixels to classify both intrinsic and extrinsic states of subject. The extrinsic sensing pipeline provides the object class information and location within a 3D space while the intrinsic sensing pipeline leverages a UWB sensor to infer "hidden" states of the subject. The seamless integration of these two pipelines allows for determination of not only the type and location of a given object, but also of it's current behavior. 

## System Dependency Installation
Capricorn is a cross-device system, requiring at minimum one host computer connected to a lidar camera sensor, and a Raspberry Pi reading data from the UWB sensor, all connected to the same network to enable ROS communication. The following System Dependency Installations apply mainly to the host computer. ROS installation is the **only** one that applies to the Pi as well. 


### Ros Installation (HOST AND PI):
Install ROS Melodic from the following link: [ROS Installation](http://wiki.ros.org/noetic/Installation)

Note that ROS is only compatible with Linux distributions

### Intel Realsense Installation:
Install Intel Realsense from the following link: [Intel Realsense Installation](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)

Follow the instructions to install the pre-built packages

### Libtorch Installation:
Install libtorch, following the instructions from the link: [Libtorch Installation](https://pytorch.org/cppdocs/installing.html)

Note that installation of a GPU based version of libtorch will require some configuration of existing CUDA environments

### OpenCV Installation:
This project leverages the Aruco AprilTag library from OpenCV, specifically the 36h11 tag family. This was introduced in the 3.4.2 version. Older versions of OpenCV will not be compatible with the current code. 

As a precaution, install the latest version of OpenCV from the following link: [OpenCV Installation](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html). 

**Be sure to also include the OpenCV Contrib Libraries**

### Eigen Installation:
Install the Eigen library by following the instruction in this link: [Eigen Installation](https://eigen.tuxfamily.org/dox/GettingStarted.html)

## Modifying Source Code to Run on your Machine

On the **Raspbery Pi**, download and extract the entire **uwb_workspace** to a desired directory

### Host Machine
On the host computer download and extract the entire **src** folder to a directory of your choice.

**Lidar Streamer:**
Within the src folder, locate the **lidar_streamer** folder, and navigate to the **CMakeLists.txt** file. In **line 10**, change the directory of **realsense2_DIR** to the location the realsense2 cmake file is stored in. 

**Object Tracker** 
Navigate to the **object_tracker** folder, and open the **CMakeLists.txt** file. On **line 7**, change the **realsense2_DIR** to the location of the realsense2 cmake file. On **line 30**, change the **CMAKE_PREFIX_PATH** to the location of the libtorch cmake file installed during the libtorch download. 

