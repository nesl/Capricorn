#include "ros/ros.h"
#include "std_msgs/Float32MultiArray.h"
#include <vector>

/**
 * This tutorial demonstrates simple receipt of messages over the ROS system.
 */

int iq_count = 240; // arr_size
int frame_size = 1000; // num_arrs

void uwbCallback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
  //std::cout << "here 2!" << std::endl;
  ROS_INFO("I heard: [%f]", msg->data[0]);
  //std::vector<float> data = msg->data;
  //std::cout << data.size() << std::endl;
  /*
  for (float i: data)
  {
    std::cout << i << " ";
  }
  */
}

int main(int argc, char **argv)
{

  ros::init(argc, argv, "listener");

  ros::NodeHandle n;

  ros::Subscriber sub = n.subscribe("uwb_chunk", 1000, uwbCallback);

  std::cout << "Subscriber Started!" << std::endl;
  
  ros::spin();
  
  std::cout << "Exited ROS spin." << std::endl;
  
  return 0;
}
