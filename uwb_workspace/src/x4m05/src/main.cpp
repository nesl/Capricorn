#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <stdio.h>
#include <string.h>
#include <wiringPi.h>
#include "taskRadar.h"
#include "common.h"
#include "ros/ros.h"
#include "std_msgs/Float32MultiArray.h"

#define DEBUG

using namespace std;

int main(int argc, char *argv[])
{
    printf("raspbian_x4driver start to work!\n""""""\n\n");
    ros::init(argc, argv, "sender");

    std::thread taskRadarThread(taskRadar);
    std::thread publisherThread(sendOutputFrame);
    taskRadarThread.join();
    publisherThread.join();

    printf("raspbian_x4driver done.\n");
    return 0;
}
