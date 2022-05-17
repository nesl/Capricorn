#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <omp.h>
ros::Publisher color_publisher;
ros::Publisher depth_publisher;
using namespace rs2;
int main(int argc, char *argv[])
{
    ros::init(argc, argv, "lidar");
    ros::NodeHandle nh;
    rs2::config cfg;
    color_publisher = nh.advertise<sensor_msgs::Image>("/lidar/color/", 2);
    depth_publisher = nh.advertise<sensor_msgs::Image>("/lidar/depth/", 2);
    /*
    cfg.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_RGB8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 1024, 768, RS2_FORMAT_Z16, 30);
    */
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    rs2::pipeline pipe;
    rs2::pipeline_profile profile = pipe.start(cfg);
    rs2::align align_to_color(RS2_STREAM_COLOR);
    auto const color_intrinsics = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    float fx = color_intrinsics.fx;
    float fy = color_intrinsics.fy;
    float ppx = color_intrinsics.ppx;
    float ppy = color_intrinsics.ppy;
    float camParam[3][3] = {{fx, 0, ppx},
                            {0, fy, ppy},
                            {0, 0, 1}};
    std::cout << "Model is: " << color_intrinsics.model << std::endl;
    std::cout << "Coeffs are: " ;
    for (int i = 0; i < 5; i++) {
        std::cout << color_intrinsics.coeffs[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "The intrinsic matrix is: " << std::endl;
    for(int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << camParam[i][j] << " ";
        }
        std::cout << std::endl;
    }
    double runningAvg = 0;
    int count = 0;
    while (ros::ok()) {
        rs2::frameset frameset = pipe.wait_for_frames();
        frameset = align_to_color.process(frameset);
        auto depth = frameset.get_depth_frame();
        auto color = frameset.get_color_frame();
        double startTime = ros::Time::now().toSec();
        int colorHeight = color.get_height();
        int colorWidth = color.get_width();
        sensor_msgs::Image color_img;
        uint8_t *colorArr = (uint8_t *)(color.get_data());

        int depthHeight = depth.get_height();
        int depthWidth = depth.get_width();
        sensor_msgs::Image depth_img;
        uint8_t *depthArr = (uint8_t *)(depth.get_data());

        
        #pragma omp parallel sections 
        {
            #pragma omp section 
            {
                for (int i = 0; i < colorHeight; i++)
                {
                    int i_scaled = i * colorWidth;
                    for (int j = 0; j < colorWidth; j++)
                    {
                        int baseIndex = 3 * (i_scaled + j);
                        color_img.data.push_back(colorArr[baseIndex]);
                        color_img.data.push_back(colorArr[baseIndex + 1]);
                        color_img.data.push_back(colorArr[baseIndex + 2]);
                    }
                }
            }

            #pragma omp section 
            {
                for (int i = 0; i < depthHeight; i++)
                {
                    int i_scaled = i * depthWidth;
                    for (int j = 0; j < depthWidth; j++)
                    {
                        depth_img.data.push_back(depthArr[2 * (i_scaled + j)]);
                        depth_img.data.push_back(depthArr[2 * (i_scaled + j) + 1]);
                    }
                }
            }
        }

        long timeSec = (int) (ros::Time::now().toSec());
        long timeNSec = (ros::Time::now().toSec() - (int) (ros::Time::now().toSec())) * 1e9;
        color_img.header.stamp.sec = timeSec;
        color_img.header.stamp.nsec = timeNSec;
        color_img.header.frame_id = "color_img";
        color_img.width = colorWidth;
        color_img.height = colorHeight;
        color_img.encoding = "rgb8";
        color_img.is_bigendian = true;
        depth_img.header.frame_id = "depth_img";
        depth_img.width = depthWidth;
        depth_img.height = colorHeight;
        depth_img.encoding = "mono16";
        depth_img.is_bigendian = true;
        depth_img.header.stamp.sec = timeSec;
        depth_img.header.stamp.nsec = timeNSec;
        color_publisher.publish(color_img);
        depth_publisher.publish(depth_img);
        runningAvg += ros::Time::now().toSec() - startTime;
        if (count == 300) {
            runningAvg = 0;
            count = 1;
        }
        rs2::device dev = profile.get_device();
        rs2::depth_sensor ds = dev.query_sensors().front().as<depth_sensor>();
        float scale = ds.get_depth_scale();
        uint16_t c = (((uint16_t) depth_img.data[2 * (640 * 240 + 320) + 1]) << 8) | ((uint16_t)(depth_img.data[2 * (640 * 240 + 320)]));
        double output = c * scale;
        std::cout << scale << std::endl;
    }
}
