# Capricorn
Capricorn is a cross-modal sensor fusion system combining RF sensor data with visual pixels to classify both intrinsic and extrinsic states of a subject. The extrinsic sensing pipeline provides the object class information and location within a 3D space while the intrinsic sensing pipeline leverages a UWB sensor to infer "hidden" states of the subject. The seamless integration of these two pipelines allows for determination of not only the type and location of a given object, but also of it's current behavior. 

## System Dependency Installation
Capricorn is a cross-device system, requiring at minimum one host computer connected to a lidar camera sensor, and a Raspberry Pi reading data from the UWB sensor, all connected to the same network to enable ROS communication. The following System Dependency Installations apply mainly to the host computer. ROS installation is the **only** one that applies to the Pi as well. We have also provided some rosbag files with prerecorded UWB sensor data as well as Lidar + Camera data, which would only require a single host computer to run. If you are simply using the prerecorded files, skip to the section at the end titled "Prerecorded Data Setup".

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

## Preparing Source Code to Run on your Machine
On the host computer and extract the entire src folder to a directory of your choice. On the Raspberry Pi download and unzip the entire uwb_workspace folder.

### Source Code Modification

**Host computer Lidar Streamer:**
Within the src folder, locate the lidar_streamer folder, and navigate to the CMakeLists.txt file. In line 10, change the directory of realsense2_DIR to the location the realsense2 cmake file is stored in. 

**Host Computer Object Tracker:** 
Navigate to the object_tracker folder, and open the CMakeLists.txt file. On line 7, change the realsense2_DIR to the location of the realsense2 cmake file. On line 30, change the CMAKE_PREFIX_PATH to the location of the libtorch cmake file installed during the libtorch download. 

Locate the run.cpp file inside the object_tracker/src directory. On line 1582 modify the absolute file path to point to **Weights/FullSetGPU.torchscript.pt** within object_tracker. On line 1587 modify the absolute file path to point to **Weights/FullSet.torchscript.pt** within object_tracker. On line 1606 modify the absolute file path to point to the **object_tracker/objects.names** file.


### Making the files
on the host computer, run ```catkin_make``` on the parent directory containing the src folder.

On the Raspberry Pi, run ```catkin_make``` on the uwb_workspace directory.

Common build errors involve a cmake file for realsense2 or libtorch not being found. Verify that the paths are actually pointing to those files. 

## Running the System Live
Connect both devices to the same network switch, a higher speed switch will enable faster streaming of the UWB data from the Pi to the host machine

## Host Computer
Open up **two** terminals and run: 

```
export ROS_MASTER_URI=http://XXX.XXX.XXX.XXX:11311 && export ROS_IP=XXX.XXX.XXX.XXX
```
where XXX.XXX.XXX.XXX represents the IP address of the **host computer** on each terminal.

On both terminals, navigate to the directory containing the src folder, and run 

```
source devel/setup.bash
```

On one terminal, then run

```
rosrun lidar_streamer lidar_streamer
```
If this executes correctly, you will see a constant stream of 0.0025 being printed out. If you receive errors, check if the camera is plugged in, and run ```realsense_viewer```, Intel's provided software to toggle on and off the camera in order to verify that they work. 

On the other terminal, run 

```
rosrun object_tracker testYolo
```
If this executes correctly, the model will be loaded, and any AprilTag present within the scene will be read. After a brief delay (few seconds), a window will pop up showing the camera feed with bounding boxes. There will be no vibration information present as the intrinsic pipeline is awaiting the Pi to start sending data. 

### Raspberry Pi
On the Raspberry Pi, open up a terminal and run ```su``` to enter superuser mode. Then, run the following commmand:
```
export ROS_MASTER_URI=http://XXX.XXX.XXX.XXX:11311 && export ROS_IP=YYY.YYY.YYY.YYY
```
where XXX.XXX.XXX.XXX represents the IP address of the **host computer**, and YYY.YYY.YYY.YYY represents the IP address of the **Raspberry Pi**.

Navigate to the uwb_workspace directory, and run 

```
source devel/setup.bash
```

Finally, start the streaming by running:

```
rosrun x4m05 sender
```

Successful execution will result in "Chunk Sent" being printed. 

The system should now be running in full capacity. If there is uncertainty about whether the vibration data is being sent over, open up a terminal on the host machine and run 

```
export ROS_MASTER_URI=http://XXX.XXX.XXX.XXX:11311 && export ROS_IP=XXX.XXX.XXX.XXX
rostopic list
```
Check to see if /uwb_chunk is among the listed topic

Then, run ```rostopic echo /uwb_chunk```. If data is being properly streamed, then this should print out a huge chunk of numbers every second.

To enable multi view for two uwb sensors, or complex event detection, change the #define flags in lines 89 and 90.

# Working with Prerecorded Data
Even if you do not have the available sensors, you can still test out Capricorn with the provided prerecorded data. Download and clone the **src** folder to a desired directory.

Download the rosbag files in this google drive: https://drive.google.com/file/d/1LOMHybkEZxwewf7A5JH-KVUtQ49OO6X8/view?usp=sharing

## Device Setup
Setting up the host machine involves these following installations:

### Ros Installation:
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

## Source Code Modification
In the src folder, delete the lidar_streamer directory. Navigate to the object_tracker folder within src, and open the CMakeLists.txt file. On line 7, change the realsense2_DIR to the location of the realsense2 cmake file.On line 30, change the CMAKE_PREFIX_PATH to the location of the libtorch cmake file installed during the libtorch download. 

Locate the run.cpp file inside the object_tracker/src directory. On line 1582 modify the absolute file path to point to **Weights/FullSetGPU.torchscript.pt** within object_tracker. On line 1587 modify the absolute file path to point to **Weights/FullSet.torchscript.pt** within object_tracker. On line 1606 modify the absolute file path to point to the **object_tracker/objects.names** file.

Run ```catkin_make``` on the parent directory of the src folder. If errors arise, verify that the file paths are correct.

## Running the System

Open another terminal, you should now have two terminal - one new terminal and one in the parent of the src directory.

In the new terminal, navigate to the folder containing the downloaded, extracted google drive files and run:

```
rosbag play desired_file.bag
```

In the src terminal, run:

```
rosrun object_tracker testYolo
```
You should see a new window pop up containing both vibration information displayed next to bounding boxes overlayed on a video.

To enable multi view for two uwb sensors, or complex event detection, change the #define flags in lines 89 and 90.


