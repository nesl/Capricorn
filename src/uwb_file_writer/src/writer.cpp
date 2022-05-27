#include "ros/ros.h"
#include "timeArr_msgs/FloatArray.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <iostream>
#include <fstream>
std::string filename = "/home/ziqi/Desktop/rawUWB.csv";
std::fstream file;
double startTime = 0;
bool start = true;
void uwb_chunk_callback(const boost::shared_ptr<const timeArr_msgs::FloatArray> msg)
{
    if (start) {
        startTime = ros::Time::now().toSec();
        start = false;
    }
    std::vector<float> uwb_raw_data = msg->data;
    std::cout << uwb_raw_data.size() << std::endl;
    if (file.is_open()) {
        for (int i = 0; i < 1024 ; i++) {
            for (int j = 0; j < 240; j++) {
                if (j != 239) {
                    file << uwb_raw_data[i * 240 + j] << ", ";
                }
                else {
                    file << uwb_raw_data[i * 240 + j] << std::endl;    
                }
            }
        }
    }
    if (ros::Time::now().toSec() - startTime > 15) {
        exit(0);
    }
}

int main(int argc, char **argv)
{
    /**
     * The ros::init() function needs to see argc and argv so that it can perform
     * any ROS arguments and name remapping that were provided at the command line.
     * For programmatic remappings you can use a different version of init() which takes
     * remappings directly, but for most command-line programs, passing argc and argv is
     * the easiest way to do it.  The third argument to init() is the name of the node.
     *
     * You must call one of the versions of ros::init() before using any other
     * part of the ROS system.
     */
    ros::init(argc, argv, "uwb_listener");
    ros::NodeHandle nh;
    ros::Subscriber uwb_sub = nh.subscribe("uwb_chunk", 5, uwb_chunk_callback);
    startTime = ros::Time::now().toSec();
    file.open(filename, std::ios::app);
    ros::spin();
    return 0;
}