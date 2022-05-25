//ROS INCLUDES
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

//C++ Standard Includes
#include <algorithm>
#include <iostream>
#include <time.h>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
//#include <io.h> // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <unistd.h>
#include <set>
#include <string>
#include <thread>
#include <sstream>

//OpenCV Includes
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/video/tracking.hpp"
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/calib3d.hpp>

//Tracker + Yolo includes
#include <torch/script.h>
#include <torch/torch.h>
#include "Hungarian.h"
#include "KalmanTracker.h"
#include "SVM/FanClassifier.h"
#include "SVM/VacuumClassifier.h"
#include "SVM/WashingClassifier.h"
#include "SVM/HumanClassifier.h"
#include "SVM/DrillClassifier.h"
#include "data_structure.cpp"
#include "VMD/VMD.h"


//Eigen Includes
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <Eigen/Eigen>
#include <Eigen/Core>

//Realsense + Messages Includes
#include "std_msgs/Float32MultiArray.h"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "timeArr_msgs/FloatArray.h"

using namespace Eigen;
using namespace cv;
using namespace std;
using namespace message_filters;
using namespace rs2;


#define MULTI_VIEW_UWB 1
#define COMPLEX_EVENT 0

/**
 * Struct definitions
 *
 * */
//Used in the tracker, commonly initilized as tb
typedef struct TrackingBox
{
    int frame;
    int id;
    Rect_<float> box;
    string classifier;
    double score;
} TrackingBox;

//Used in distance clustering and calculation from the lidar depth stream
struct Bin {
    int numOccurrence = 0;
    double lowerBound;
    double totalDepth = 0;
};

//Defines and constants
#define b_size 85
const int chunk_size = 1024;
const int chunk_size2 = 1024;
const int max_age = 5;
const int min_hits = 3;
const double iouThreshold = 0.3;
static double b[b_size] = {0.0095,-0.0010,-0.0012,-0.0015,-0.0019,-0.0024,-0.0028,-0.0032,-0.0035,-0.0036,-0.0035,-0.0031,-0.0024,-0.0015,-0.0003,0.0012,0.0028,0.0045,0.0063,0.0080,0.0094,0.0105,0.0111,0.0112,0.0105,0.0091,0.0068,0.0037,-0.0003,   -0.0050,-0.0105,-0.0166,-0.0232,-0.0300,-0.0369,-0.0437,-0.0501,-0.0560,-0.0611,-0.0652,-0.0683,-0.0702,0.9291,-0.0702,-0.0683,-0.0652,-0.0611,-0.0560,-0.0501,-0.0437,-0.0369,-0.0300,-0.0232,-0.0166,-0.0105   -0.0050,   -0.0003,    0.0037,    0.0068,    0.0091,    0.0105,    0.0112,    0.0111,    0.0105,0.0094,0.0080,0.0063,0.0045,0.0028,0.0012,-0.0003,-0.0015,-0.0024,-0.0031,-0.0035,-0.0036,-0.0035,-0.0032,-0.0028,-0.0024,-0.0019,-0.0015,-0.0012,-0.0010,0.0095};
const int imgW = 320;
const int imgH = 320;

//Variables for complex event detection

int interaction_detected = 0;
int washine_machine_state = -1;
int curr_stage = 0;
std::string curr_stage_text;


//Global variables used for tracking
int frame_count = 0;
int database_id;
vector<KalmanTracker> trackers;
vector<Rect_<float>> predictedBoxes;
vector<string> predictedBoxesClass;
vector<vector<double>> iouMatrix;
vector<int> assignment;
set<int> unmatchedDetections;
set<int> unmatchedTrajectories;
set<int> allItems;
set<int> matchedItems;
vector<cv::Point> matchedPairs;
vector<TrackingBox> frameTrackingResult;
unsigned int trkNum = 0;
unsigned int detNum = 0;

//Yolo model global variables
torch::jit::script::Module module;
std::vector<std::string> classnames;

//In memory database
vector<obj> database;

//Complex array used for UWB, prevents allocation of array each time
std::complex<double> complexArr1[120][chunk_size];
std::complex<double> complexArr2[120][chunk_size];

//Mutex Lock
std::mutex dataLock;
const int deletionTicks = 5; //How many frames before deletion

//Camera Parameters
float fx = 603.824;
float ppx = 315.726;
float ppy = 238.551;
float fy = 604.016;
float camParam[3][3] = {{fx, 0, ppx},
                            {0, fy, ppy},
                            {0, 0, 1}};

Eigen::Matrix3d rotationMatrix;
Eigen::Vector3d translationMatrix(0, 0, 0);
Eigen::Vector3d zeroVectorTransformed;
rs2_intrinsics intrin;

//UWB Location in AprilTag coordinates, uwb1Absolute is set automatically
double uwb1Absolute[3] = {0, 0, 0};
double uwb2Absolute[3] = {-0.9632, -0.1268, 1.406};
double bbox_factor = 0;


double computeMean(double* arr) {
    double mean = 0;
    for (int i = 0; i < chunk_size; i++) {
        mean += arr[i];
    }
    return mean / chunk_size;
}

double computeVariance(double* arr) {
    double mean = computeMean(arr);
    double variance = 0;
    for (int i = 0; i < chunk_size; i++) {
        variance += (arr[i] - mean) * (arr[i] - mean);
    }
    return variance / chunk_size;
}

double computeDistance(double* pt1, double* pt2) {
    return sqrt( (pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]) + (pt1[2] - pt2[2]) * (pt1[2] - pt2[2]));
}

#if MULTI_VIEW_UWB
//UWB function takes in two synchronized messages from two UWB sensors, processes the chunk, and appends the correct slice to the in memory database
//depending on whether there is aliasing in the first UWB sensor
void uwbCallback(const boost::shared_ptr<const timeArr_msgs::FloatArray> msg1, const boost::shared_ptr<const timeArr_msgs::FloatArray> msg2)
{
    //Msg has IQ data, we need to create a complexArray that has the elements matched up
	int vectorCount = 0;
	for (int i = 0; i < chunk_size; i++)
	{
		for (int j = 0; j < 120; j++)
		{
            //Fills complex array
			complexArr1[j][i] = std::complex<double>(msg1->data.at(vectorCount), msg1->data.at(vectorCount + 120));
			vectorCount++;
		}
		vectorCount += 120;
	}
    //Note that complexArr has each distance bin as a row, each column is time (1s)

    //Compute phase difference in first distance bin
	double referencePhase1 = 0;
	for (int i = 0; i < chunk_size; i++)
	{
		referencePhase1 += std::arg(complexArr1[0][i]);
	}

    //Compute average
	referencePhase1 /= chunk_size;

    //Repeat for second UWB chunk, we gotta synch the uwbs :(
    vectorCount = 0;
    for (int i = 0; i < chunk_size; i++)
	{
		for (int j = 0; j < 120; j++)
		{
            //Fills complex array
			complexArr2[j][i] = std::complex<double>(msg2->data.at(vectorCount), msg2->data.at(vectorCount + 120));
			vectorCount++;
		}
		vectorCount += 120;
	}
    //Note that complexArr has each distance bin as a row, each column is time (1s)

    //Compute phase difference in first distance bin
	double referencePhase2 = 0;
	for (int i = 0; i < chunk_size; i++)
	{
		referencePhase2 += std::arg(complexArr2[0][i]);
	}

    //Compute average
	referencePhase2 /= chunk_size;

    //Decide whether uwb1 is acceptable to be used for uwb computation
    dataLock.lock();
    int dataSize = database.size();
    double distanceUWB1[dataSize]; //Array of distances from UWB
    for (int i = 0; i < dataSize; i++) {
        double tempPoint[3] = {database.at(i).x, database.at(i).y, database.at(i).z};
        distanceUWB1[i] = computeDistance(tempPoint, uwb1Absolute);
    }
    dataLock.unlock();
    bool uwb1Acceptable = true;
    std::cout << "Distance of UWB 1 is ";
    for (int i = 0; i < dataSize - 1; i++) {
        std::cout << distanceUWB1[i] << " ";
        for (int j = i + 1; j < dataSize; j++) {
            if (abs(distanceUWB1[i] - distanceUWB1[j]) < 0.3) {
                uwb1Acceptable = false;
            }
        }
    }
    std::cout << std::endl;
    if (uwb1Acceptable) {
        std::cout << "UWB 1 Acceptable" << std::endl;
    }
    else {
        std::cout << "UWB2 is being used" << std::endl;
    }


    dataLock.lock();
	//Take slice of data for each entry in the database
	for (int i = 0; i < database.size(); i++) {
        double tempPoint[3] = {database.at(i).x, database.at(i).y, database.at(i).z};
        //Get distance from the UWB
		double objDepth = uwb1Acceptable ? computeDistance(tempPoint, uwb1Absolute) : computeDistance(tempPoint, uwb2Absolute);
        int adj = 0;
        //Converts to a UWB index
		int index = (int)((objDepth - 0.3) / 0.0514) + adj; // was +4 here
        double* slice = new double[chunk_size];
        double varArr[5];
        for (int j = index - 2; j <= index + 2; j++) {
            for (int k = 0; k < chunk_size; k++) {
                if (uwb1Acceptable) {
                    slice[k] = std::abs(complexArr1[j][k]);
                }
                else {
                    slice[k] = std::abs(complexArr2[j][k]);
                }
            }
            varArr[j - index - 2] = computeVariance(slice);
        }
        double maxVal = 0;
        for (int j = 0; j < 5; j++) {
            if (varArr[j] > maxVal) {
                maxVal = varArr[j];
                index = j + index - 2;
            }
        }

        //Compute phase difference in each element
        for (int j = 0; j < chunk_size; j++)
        {
            if (uwb1Acceptable) {
                double phaseDiff = referencePhase1 - std::arg(complexArr1[0][j]);
                std::complex<double> phaseShift(cos(phaseDiff), sin(phaseDiff));
                std::complex<double> shiftedValue(complexArr1[index][j] * phaseShift);
                slice[j] = std::abs(shiftedValue); //Fill slice with the abs value of the shifted complex numbers

            }
            else {
                double phaseDiff = referencePhase2 - std::arg(complexArr2[0][j]);
                std::complex<double> phaseShift(cos(phaseDiff), sin(phaseDiff));
                std::complex<double> shiftedValue(complexArr2[index][j] * phaseShift);
                slice[j] = std::abs(shiftedValue); //Fill slice with the abs value of the shifted complex numbers

            }
        }
        if (database.at(i).vibration.size() >= 30) {
            delete database.at(i).vibration.front();
            database.at(i).vibration.pop_front();
        }
        database.at(i).vibration.emplace_back(slice);

    }
    dataLock.unlock();
}

#else
void uwbCallback(const boost::shared_ptr<const timeArr_msgs::FloatArray> msg)
{

    //Msg has IQ data, we need to create a complexArray that has the elements matched up
	int vectorCount = 0;
	for (int i = 0; i < chunk_size; i++)
	{
		for (int j = 0; j < 120; j++)
		{
            //Fills complex array
			complexArr1[j][i] = std::complex<double>(msg->data.at(vectorCount), msg->data.at(vectorCount + 120));
			vectorCount++;
		}
		vectorCount += 120;
	}
    //Note that complexArr has each distance bin as a row, each column is time (1s)

    //Compute phase difference in first distance bin
	double referencePhase = 0;
	for (int i = 0; i < chunk_size; i++)
	{
		referencePhase += std::arg(complexArr1[0][i]);
	}

    //Compute average
	referencePhase /= chunk_size;
    dataLock.lock();
	//Take slice of data for each entry in the database
	for (int i = 0; i < database.size(); i++) {
        double tempPoint[3] = {database.at(i).x, database.at(i).y, database.at(i).z};
		double objDepth = computeDistance(tempPoint, uwb1Absolute);
        //Get index corresponding to the given distance from lidar
        int adj = 0;
		int index = (int)((objDepth - 0.3) / 0.0514) + adj; // was +4 here
        //This is probably useless, please ignore
        double* slice = new double[chunk_size];
        //Compute phase difference in each element
        double varArr[5];
        for (int j = index - 2; j <= index + 2; j++) {
            for (int k = 0; k < chunk_size; k++) {
                slice[k] = std::abs(complexArr1[j][k]);
            }
            varArr[j - index - 2] = computeVariance(slice);
        }
        double maxVal = 0;
        for (int j = 0; j < 5; j++) {
            if (varArr[j] > maxVal) {
                maxVal = varArr[j];
                index = j + index - 2;
            }
        }
        for (int j = 0; j < chunk_size; j++)
        {
            double phaseDiff = referencePhase - std::arg(complexArr1[0][j]);
            std::complex<double> phaseShift(cos(phaseDiff), sin(phaseDiff));
            std::complex<double> shiftedValue(complexArr1[index][j] * phaseShift);
            slice[j] = std::abs(shiftedValue); //Fill slice with the abs value of the shifted complex numbers

        }
        if (database.at(i).vibration.size() >= 30) { //It was 120
        /* print all the data in the buffer
        if (database.at(i).type == "washing machine" && database.at(i).state == 1)
        {
            for (int m=0; m<20; m++)
            {
                for (int n=0; n<1024; n++)
                {
                    printf("%.16f ,", database.at(i).vibration.at(m)[n]);
                }
            }
        }
        exit(0);
        */
            delete database.at(i).vibration.front();
            database.at(i).vibration.pop_front();
        }
        database.at(i).vibration.emplace_back(slice);
    }
    dataLock.unlock();
}
#endif

double getDepth(const boost::shared_ptr<const sensor_msgs::Image> depthImg, int lowerX, int upperX, int lowerY, int upperY) {
    int middleX = (lowerX + upperX) / 2;
    int middleY = (lowerY + upperY) / 2;
    double origDepth = 0.00025 * ((((uint16_t) depthImg->data[2 * (depthImg->width * middleY + middleX) + 1]) << 8) | ((uint16_t)(depthImg->data[2 * (depthImg->width * middleY + middleX)])));
    /*
    if (origDepth > 2) {
        double totalDepth = 0;
        int count = 0;
        for(int i = middleY - 4; i < middleY + 4; i++) {
            for (int j = middleX - 4; j < middleX + 4; j++) {
                double depthAtPixel = 0.00025 * ((((uint16_t) depthImg->data[2 * (depthImg->width * i + j) + 1]) << 8) | ((uint16_t)(depthImg->data[2 * (depthImg->width * i + j)])));
                if (std::abs(depthAtPixel - origDepth) < 0.03) {
                    totalDepth += depthAtPixel;
                    count++;
                }
            }
        }
        return totalDepth / count;
    }
    else {
        */
        //Create distance bins of 1 cm
        Bin values[500];
        for (int i = 0; i < 500; i++) {
            values[i].lowerBound = 0.01 * i;
            values[i].totalDepth = 0;
            values[i].numOccurrence = 0;
        }
        //Prevents invalid values from being used for depth processing
        if (upperX > 640 || upperY > 480) {
            upperX = 640;
            upperY = 480;
        }
        
        
        //Loop through the area in the boxes and add the depth to the appropriate bin
        for (int i = lowerY; i < upperY; i++) {
            for (int j = lowerX; j < upperX; j+= 3) {
                uint16_t depthAtPixel = (((uint16_t) depthImg->data[2 * (depthImg->width * i + j) + 1]) << 8) | ((uint16_t)(depthImg->data[2 * (depthImg->width * i + j)]));
                double depth = 0.00025 * depthAtPixel;
                if (depth > 4.3 || depth == 0) {
                    continue;
                }
                int binNum = (int) (depth * 100);
                values[binNum].totalDepth += depth;
                values[binNum].numOccurrence++;
            }
        } 

        //Vector holding the significant clusters found
        std::vector<Bin> sigPts;
        bool newDetection = true;
        bool secondChance = true;
        //Loop through the bins 
        for (int i = 2; i < 500; i++) {
            double avgDistance = values[i].totalDepth / values[i].numOccurrence;
            int peakValue = 20 - i / 100; //Number of points in bin to be considered significant, scales based on distance because further objects are "smaller"
            //Found new peak, push onto sigPts
            if (newDetection && values[i].numOccurrence > peakValue && values[i - 1].numOccurrence > peakValue) {
                sigPts.push_back(values[i - 1]);
                sigPts.at(sigPts.size() - 1).totalDepth += values[i].totalDepth;
                sigPts.at(sigPts.size() - 1).numOccurrence += values[i].numOccurrence;
                newDetection = false;
                continue;
            }
            //Peaking continues, update the element
            if (!newDetection && values[i].numOccurrence > peakValue) {
                sigPts.at(sigPts.size() - 1).totalDepth += values[i].totalDepth;
                sigPts.at(sigPts.size() - 1).numOccurrence += values[i].numOccurrence;
            }
            //No peak found, reset for new peak
            else if (!newDetection) {
                newDetection = true;
            }
        }
        //No peaks found, return
        if (sigPts.size() == 0) {
            return 0;
        }
        //Loop through and find the most significant bins (the one with most points)
        Bin maxBin;
        for (int i = 0; i < sigPts.size(); i++) { //Todo change back to 1
            if (sigPts.at(i).numOccurrence > 5) {
                maxBin = sigPts.at(i);
                break;
            }
        }
        
        //Get the average distance of that bin and return
        return maxBin.totalDepth / maxBin.numOccurrence;
   // }
     
}

//Returns the location of the object's centroid (x,y,z) in APRILTAG coordinates in variable arr
double getCentroidTransformed( const boost::shared_ptr<const sensor_msgs::Image> colorImg, const boost::shared_ptr<const sensor_msgs::Image> depthImg, Rect_<float> box, double* arr) {
    
    int middleU = (int) (box.tl().x + box.br().x) / 2.0;
    int middleV = (int) (box.tl().y + box.br().y) / 2.0;
    /*
    double count = 0;
    double xTotal = 0;
    double yTotal = 0;
    double zTotal = 0;
    double origDepth = 0.00025 * ((((uint16_t) depthImg->data[2 * (depthImg->width * middleV + middleU) + 1]) << 8) | ((uint16_t)(depthImg->data[2 * (depthImg->width * middleV + middleU)])));
    //Find an acceptable point to start at

    while ( (origDepth == 0 || origDepth > 4)  && !(middleU < box.tl().x || middleV < box.tl().y|| middleU > box.br().x  || middleV > box.br().y) ) {
        middleU += rand() % 3 - 1;
        middleV += rand() % 3 - 1;
        origDepth = 0.00025 * ((((uint16_t) depthImg->data[2 * (depthImg->width * middleV+ middleU) + 1]) << 8) | ((uint16_t)(depthImg->data[2 * (depthImg->width * middleV + middleU)])));
    }

    if (middleU < box.tl().x || middleV < box.tl().y|| middleU > box.br().x  || middleV > box.br().y) {
        arr[0] = arr[1] = arr[2] = -1;
        std::cout << "Object not found in bbox"  << std::endl;
        return 0;
    }

    float minDepth = origDepth;
    for (int i = middleV - 1; i < middleV + 2; i++) {
        for (int j = middleU - 1; j < middleU + 2; j++) {
            float currDepth = 0.00025 * ((((uint16_t) depthImg->data[2 * (depthImg->width * i + j) + 1]) << 8)
            | ((uint16_t)(depthImg->data[2 * (depthImg->width * i + j)])));
            if (minDepth < currDepth && currDepth > 0.5) {
                minDepth = currDepth;
            }
        }
    }

    //Go around that point and compute the distance
    for (int j = middleV - 2; j < middleV + 2; j++) {
        for (int k = middleU - 2; k < middleU + 2; k++) {
            double depthAtPixel = 0.00025 * ((((uint16_t) depthImg->data[2 * (depthImg->width * j + k) + 1]) << 8) |
                ((uint16_t)(depthImg->data[2 * (depthImg->width * j + k)])));
            if ((std::abs(depthAtPixel - minDepth) < 0.3)) {
                float point[3];
                float pixel[2] = {float(k), float(j)};
                rs2_deproject_pixel_to_point(point, &intrin, pixel, depthAtPixel);
                // std::cout << point[2] << " " << depthAtPixel << std::endl;
                //Transform to AprilTag coords
                Eigen::Vector3d coordVec(point[0] + translationMatrix(0), point[1] + translationMatrix(1), point[2] + translationMatrix(2));
                coordVec = rotationMatrix * coordVec;
                xTotal += coordVec(0);
                yTotal += coordVec(1);
                zTotal += coordVec(2);
                count++;
            }
        }
    }
    //Place result in arr
    arr[0] = xTotal / count;
    arr[1] = yTotal / count;
    arr[2] = zTotal / count;
    return computeDistance(arr, uwb1Absolute);
    */
    double depth = getDepth(depthImg, box.tl().x, box.br().x, box.tl().y, box.br().y);
    float point[3];
    float pixel[2] = {float(middleU), float(middleV)};
    rs2_deproject_pixel_to_point(point, &intrin, pixel, depth);
    // std::cout << point[2] << " " << depthAtPixel << std::endl;
    //Transform to AprilTag coords
    Eigen::Vector3d coordVec(point[0] + translationMatrix(0), point[1] + translationMatrix(1), point[2] + translationMatrix(2));
    coordVec = rotationMatrix * coordVec;
    arr[0] = coordVec(0);
    arr[1] = coordVec(1);
    arr[2] = coordVec(2);
    return 0.0;
}

//TODO depth estimation function



// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}

//Original NMS
std::vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh = 0.2, float iou_thresh = 0.2)
{
    std::vector<torch::Tensor> output;
    for (size_t i = 0; i < preds.sizes()[0]; ++i)
    {
        torch::Tensor pred = preds.select(0, i);
        // Filter by scores
        torch::Tensor scores = pred.select(1, 4) * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));

        pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
        if (pred.sizes()[0] == 0) {
            output.push_back(pred);
            continue;
        }
        // (center_x, center_y, w, h) to (left, top, right, bottom)
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        // Computing scores and classes
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        pred.select(1, 5) = std::get<1>(max_tuple);

        torch::Tensor dets = pred.slice(1, 0, 6);
        torch::Tensor keep = torch::empty({dets.sizes()[0]});
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
        torch::Tensor v = std::get<0>(indexes_tuple);
        torch::Tensor indexes = std::get<1>(indexes_tuple);
        int count = 0;
        while (indexes.sizes()[0] > 0)
        {
            keep[count] = (indexes[0].item().toInt());
            count += 1;

            // Computing overlaps
            torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
            for (size_t i = 0; i < indexes.sizes()[0] - 1; ++i)
            {
                lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
            }
            torch::Tensor overlaps = widths * heights;

            // Filter by IOUs
            torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
            indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
        }
        keep = keep.toType(torch::kInt64);
        output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
    }
    return output;
}

//Soft NMS for better occluded object detection
std::vector<torch::Tensor> soft_non_max_suppression(torch::Tensor preds, float score_thresh = 0.2, float iou_thresh = 0.6)
{
    // pred: [# of images, bounding boxs, [x,y,w,h,obj_conf, class1_conf, class2_conf...]] e.g.[1, 15000, 10]
    std::vector<torch::Tensor> output;
    for (size_t i = 0; i < preds.sizes()[0]; ++i)
    {
        torch::Tensor pred = preds.select(0, i);

        // Filter by scores
        torch::Tensor scores = pred.select(1, 4) * std::get<0>(torch::max(pred.slice(1, 5, pred.sizes()[1]), 1)); //conf = obj_conf * cls_conf
        pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0)); // remove bounding boxes with too-low threashold
        if (pred.sizes()[0] == 0)
            continue;

        // (center_x, center_y, w, h) to (left, top, right, bottom)
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        // Computing scores and classes
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1); // confidence of the most probable class
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple); // conf = obj_conf * cls_conf
        pred.select(1, 5) = std::get<1>(max_tuple); // class index

        torch::Tensor dets = pred.slice(1, 0, 6);

        torch::Tensor keep = torch::empty({dets.sizes()[0]});
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
        torch::Tensor classes = dets.select(1,5);
        std::vector<int> idx_vec;
        for (int j=0; j< dets.sizes()[0]; j++)
        {
            idx_vec.push_back(j);
        }
        auto opts = torch::TensorOptions().dtype(torch::kInt32);
        torch::Tensor indexes = torch::from_blob(idx_vec.data(), dets.sizes()[0], opts).to(torch::kInt64);
        scores = dets.select(1, 4);

        int count = 0;
        while (1)
        {
            int max_score_arg = torch::argmax(scores).item().toInt();
            keep[count] = (indexes[max_score_arg].item().toInt());
            count += 1;

            // Computing overlaps
            torch::Tensor lefts = torch::empty(indexes.sizes()[0]);
            torch::Tensor tops = torch::empty(indexes.sizes()[0]);
            torch::Tensor rights = torch::empty(indexes.sizes()[0]);
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0]);
            torch::Tensor widths = torch::empty(indexes.sizes()[0]);
            torch::Tensor heights = torch::empty(indexes.sizes()[0]);

            for (size_t i = 0; i < indexes.sizes()[0]; ++i)
            {
                lefts[i] = std::max(dets[indexes[max_score_arg]][0].item().toFloat(), dets[indexes[i]][0].item().toFloat());
                tops[i] = std::max(dets[indexes[max_score_arg]][1].item().toFloat(), dets[indexes[i]][1].item().toFloat());
                rights[i] = std::min(dets[indexes[max_score_arg]][2].item().toFloat(), dets[indexes[i]][2].item().toFloat());
                bottoms[i] = std::min(dets[indexes[max_score_arg]][3].item().toFloat(), dets[indexes[i]][3].item().toFloat());
                widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
            }
            torch::Tensor overlaps = widths * heights;
            // FIlter by IOUs
            torch::Tensor ious = overlaps / (areas.select(0, indexes[max_score_arg].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 0, indexes.sizes()[0])) - overlaps);
            torch::Tensor scores_discount = torch::ones(scores.sizes()[0]) - ious;
            scores_discount.index_select(0, torch::nonzero(ious <= iou_thresh).select(1, 0)) = 1; // no discont for boxes with iou<threshold

            torch::Tensor obj_type_diff = torch::index_select(classes, 0, indexes.slice(0, 0, indexes.sizes()[0])) - classes.select(0, indexes[max_score_arg].item().toInt());
            scores_discount.index_select(0, torch::nonzero(obj_type_diff != 0).select(1, 0)) = 1; // no discont for boxes with differnt classes

            scores = scores * scores_discount;
            if(!torch::nonzero(scores > score_thresh).select(1, 0).sizes()[0]) break;
        }
        keep = keep.toType(torch::kInt64);
        output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
    }
    return output;
}



std::string getState(std::string classifier, int state) {
    if (state == -1) {
        return "Estimating...";
    }
    if (classifier == "vacuum") {
        if (state == 0) {
            return "Idle";
        }
        else {
            return "Sweeping";
        }
    }
    else if (classifier == "washing machine") {
        if (state == 0) {
            return "Idle";
        }
        else if (state == 1) {
            return "Washing";
        }
        else {
            return "Drying";
        }
    }
    else if (classifier == "standing fan") {
        if (state == 0) {
            return "Idle";
        }
        else if (state == 1) {
            return "Speed 1";
        }
        else if (state == 2) {
            return "Speed 2";
        }
        else {
            return "Speed 3";
        }
    }
    else if (classifier == "person") {
        if (state == 0) {
            return "bpm estimating";
        }
        else {
            return std::to_string(state) + " bpm";
        }
    }
    else {
        if (state == 0) {
            return "off";
        }
        else {
            return "on";
        }
    }
}

//Gets the transform matrix from LiDAR to apriltag frame, called once at the beginning of program execution
void getTransform(cv::Mat cameraMatrix, cv::Mat distCoeffs, cv::Mat image)
{
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    std::cout << "About to eval transform" << std::endl;
    cv::aruco::detectMarkers(image, dictionary, corners, ids);
    cv::Mat imageCopy;
    rotationMatrix << 1, 0, 0,
        0, 1, 0,
        0, 0, 1;
    std::cout << "Marker number is:" << ids.size() << std::endl;
    if (ids.size() > 0)
    {
        image.copyTo(imageCopy);
        cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(corners, 0.17145, cameraMatrix, distCoeffs, rvecs, tvecs);
        cv::Mat rotationArr;
        cv::Rodrigues(rvecs.at(0), rotationArr);
        Eigen::Matrix4d tempMatrix;
        rotationMatrix << rotationArr.at<double>(0, 0), rotationArr.at<double>(1, 0), rotationArr.at<double>(2, 0),
            rotationArr.at<double>(0, 1), rotationArr.at<double>(1, 1), rotationArr.at<double>(2, 1),
            rotationArr.at<double>(0, 2), rotationArr.at<double>(1, 2), rotationArr.at<double>(2, 2);
        translationMatrix = Eigen::Vector3d(-tvecs.at(0)[0], -tvecs.at(0)[1], -tvecs.at(0)[2]);
        zeroVectorTransformed = rotationMatrix * translationMatrix;
        std::cout << zeroVectorTransformed << std::endl;
        uwb1Absolute[0] = zeroVectorTransformed(0);
        uwb1Absolute[1] = zeroVectorTransformed(1);
        uwb1Absolute[2] = zeroVectorTransformed(2);
    }

}

//Subscribes to lidar stream, feeds into Yolo, then tracks the object
double averageTime = 0;
int timeCount = 0;
torch::Tensor imgTensor;
bool gotTransform = false;
void trackObjects(const boost::shared_ptr<const sensor_msgs::Image> colorImg2, const boost::shared_ptr<const sensor_msgs::Image> depthImg2)
{
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    // double startTime = ros::Time::now().toSec();

    if (!gotTransform) {
        cv::Mat camMatrix(3, 3, CV_32FC1, camParam);
        cv::Mat distortionMatrix(1, 5, CV_32FC1, intrin.coeffs);
        uint8_t colorArr[colorImg2->height * colorImg2->width * 3];
        for (int i = 0; i < colorImg2->height * colorImg2->width * 3; i++) {
            colorArr[i] = colorImg2->data[i];
        }
        getTransform(camMatrix, distortionMatrix, cv::Mat(colorImg2->height, colorImg2->width, CV_8UC3, colorArr));
        gotTransform = true;
        std::cout << "Computed Transform" << std::endl;
        return;
    }

    KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.
    //Convert to opencv
    cv::Mat img(colorImg2->height, colorImg2->width, CV_8UC3, const_cast<uchar *>(&colorImg2->data[0]), colorImg2->step);
    //Modify images to correct format
    //cv::cvtColor(img, img, cv::COLOR_BGR2RGB, 3);
    cv::Mat img_input = img.clone();
    cv::Mat img_input2 = img.clone();
    cv::resize(img, img_input, cv::Size(imgW, imgH));

    //Conduct inference
    imgTensor = torch::from_blob(img_input.data, {img_input.rows, img_input.cols, 3}, torch::kByte);
    imgTensor = imgTensor.permute({2, 0, 1});
    imgTensor = imgTensor.toType(torch::kFloat);
    imgTensor = imgTensor.div(255);
    imgTensor = imgTensor.unsqueeze(0);
    // preds: [batch_size, #_of_bb_boxes, #_of_classes]
    torch::Tensor preds = module.forward({imgTensor.to(device)}).toTuple()->elements()[0].toTensor().to(torch::kCPU);
    std::vector<torch::Tensor> dets = soft_non_max_suppression(preds, 0.2, 0.7);

    // std::cout << "Model Inference Time is: " << ros::Time::now().toSec() - startTime << std::endl;
    //Return if no objects detected
    if (dets.size() == 0 || dets[0].sizes()[0] == 0) {
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB, 3);
        cv::imshow("Output", img);
        cv::waitKey(1);
        return;
    }
    //Increment the frame count
    frame_count++;
    /**
     * TRACKING
     * */
    //First frame from camera, fill tracker
    if (trackers.size() == 0)
    {
        if (dets.size() > 0)
        {
            //Get bounding box, add to tracker
            for (size_t i = 0; i < dets[0].sizes()[0]; ++i)
            {
                float left = dets[0][i][0].item().toFloat() * img.cols / imgW;
                float top = dets[0][i][1].item().toFloat() * img.rows / imgH;
                float right = dets[0][i][2].item().toFloat() * img.cols / imgW;
                float bottom = dets[0][i][3].item().toFloat() * img.rows / imgH;
                left *= (1 + bbox_factor);
                top *= 1 + bbox_factor;
                right *= 1 - bbox_factor;
                bottom *= 1 - bbox_factor;
                float score = dets[0][i][4].item().toFloat();
                int classID = dets[0][i][5].item().toInt();
                TrackingBox tb;
                tb.id = database_id++; //Assign the id based off a simple counter
                tb.frame = frame_count;
                tb.box = Rect_<float>(Point_<float>(left, top), Point_<float>(right, bottom));
                tb.classifier = classnames.at(classID);
                tb.score = score;
                KalmanTracker trk = KalmanTracker(tb.box, tb.classifier, tb.score, tb.id);
                trackers.push_back(trk);
            }
        }
        //Return out of function, no need to track because first frame
        return;
    }
    ///////////////////////////////////////
    // 3.1. get predicted locations from existing trackers.
    predictedBoxes.clear();
    predictedBoxesClass.clear();
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        Rect_<float> pBox = (*it).predict();
        if (pBox.x >= 0 && pBox.y >= 0)
        {
            predictedBoxes.push_back(pBox);
            predictedBoxesClass.push_back((*it).m_classifier);
            it++;
        }
        else
        {
            it = trackers.erase(it);
            //cerr << "Box invalid at frame: " << frame_count << endl;
        }
    }

    ///////////////////////////////////////
    // 3.2. associate detections to tracked object (both represented as bounding boxes)
    // dets : detFrameData[fi]
    trkNum = predictedBoxes.size();
    detNum = dets[0].sizes()[0]; //detFrameData[fi].size();

    iouMatrix.clear();
    iouMatrix.resize(trkNum, vector<double>(detNum, 0));
    for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++)
        {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            //iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[fi][j].box);

            float left = dets[0][j][0].item().toFloat() * img.cols / imgW;
            float top = dets[0][j][1].item().toFloat() * img.rows / imgH;
            float right = dets[0][j][2].item().toFloat() * img.cols / imgW;
            float bottom = dets[0][j][3].item().toFloat() * img.rows / imgH;
            left *= (1 + bbox_factor);
            top *= 1 + bbox_factor;
            right *= 1 - bbox_factor;
            bottom *= 1 - bbox_factor;

            /*
            if ((right - left) * (bottom - top) < 10000) {
                iouMatrix[i][j] = 1;
                continue;
            }
            */

            if (predictedBoxesClass[i] == classnames.at(dets[0][j][5].item().toInt()))
            {
                iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], Rect_<float>(Point_<float>(left, top), Point_<float>(right, bottom)));
            }
            else
            {
                iouMatrix[i][j] = 1;
            }
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    assignment.clear();
    HungAlgo.Solve(iouMatrix, assignment);

    //Clear all vectors to prepare for new information, easier than updating
    unmatchedTrajectories.clear();
    unmatchedDetections.clear();
    allItems.clear();
    matchedItems.clear();

    //New individual enters scene
    if (detNum > trkNum) //	there are unmatched detections
    {

        for (unsigned int n = 0; n < detNum; n++)
            allItems.insert(n);

        for (unsigned int i = 0; i < trkNum; ++i)
            matchedItems.insert(assignment[i]);


        set_difference(allItems.begin(), allItems.end(),
                        matchedItems.begin(), matchedItems.end(),
                        insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    }
    //Ppl be gone
    else if (detNum < trkNum) // there are unmatched trajectory/predictions
    {
        for (unsigned int i = 0; i < trkNum; ++i)
            if (assignment[i] == -1) { // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTrajectories.insert(i);
            }
    }
    else
        ;

    //Find matched pairs, and unmatched bounding boxes based on IOU
    matchedPairs.clear();
    for (unsigned int i = 0; i < trkNum; ++i)
    {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
        {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        }
        else
            matchedPairs.push_back(cv::Point(i, assignment[i]));
    }

    ///////////////////////////////////////
    // 3.3. updating trackers

    //Iterate through the matched trackers and update them with the new information
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++)
    {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        float left = dets[0][detIdx][0].item().toFloat() * img.cols / imgW;
        float top = dets[0][detIdx][1].item().toFloat() * img.rows / imgH;
        float right = dets[0][detIdx][2].item().toFloat() * img.cols / imgW;
        float bottom = dets[0][detIdx][3].item().toFloat() * img.rows / imgH;
        left *= (1 + bbox_factor);
        top *= 1 + bbox_factor;
        right *= 1 - bbox_factor;
        bottom *= 1 - bbox_factor;
        float score = dets[0][detIdx][4].item().toFloat();
        int classID = dets[0][detIdx][5].item().toInt();
        trackers[trkIdx].update(Rect_<float>(Point_<float>(left, top), Point_<float>(right, bottom)),  classnames.at(classID), score);
    }

    // create and initialise new trackers for unmatched detections
    for (auto umd : unmatchedDetections)
    {
        float left = dets[0][umd][0].item().toFloat() * img.cols / imgW;
        float top = dets[0][umd][1].item().toFloat() * img.rows / imgH;
        float right = dets[0][umd][2].item().toFloat() * img.cols / imgW;
        float bottom = dets[0][umd][3].item().toFloat() * img.rows / imgH;
        left *= (1 + bbox_factor);
        top *= 1 + bbox_factor;
        right *= 1 - bbox_factor;
        bottom *= 1 - bbox_factor;
        float score = dets[0][umd][4].item().toFloat();
        int classID = dets[0][umd][5].item().toInt();
        /*
        if ((right - left) * (bottom - top) < 8000) {
            std::cout << "Too small :D" << std::endl;
            continue;
        }
        */
        TrackingBox tb;
        tb.id = database_id++; //classID;
        tb.box = Rect_<float>(Point_<float>(left, top), Point_<float>(right, bottom));
        tb.classifier = classnames.at(classID);
        tb.score = score;
        KalmanTracker tracker = KalmanTracker(tb.box, tb.classifier, tb.score, tb.id);
        trackers.push_back(tracker);
    }

    //Fill frametrackingresult with the tracker array information, removing dead trackers that have not been updated in a while
    frameTrackingResult.clear();
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        //Hit streak must be greater than min_hits
        if (((*it).m_time_since_update < max_age) &&
				((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
        {
            TrackingBox res;
            res.box = (*it).get_state();
            res.id = (*it).m_tracker_id;
            res.frame = frame_count;
            res.classifier = (*it).m_classifier;
            res.score = (*it).m_probability;
            frameTrackingResult.push_back(res);
            it++;
        }
        //Remove dead trackers that haven't been updated in 5 frames
        else if (it != trackers.end() && (*it).m_time_since_update > max_age) {
            it = trackers.erase(it);
        }
        else {
            it++;
        }
    }

    //Increment the inactive ticks, every update resets ticksInactive to zero
    dataLock.lock();
    for (int i = 0; i < database.size(); i++) {
        database.at(i).ticksInactive++;
    }
    dataLock.unlock();
    //Loop through the vector generated by the tracker, and compute depth, adding to databse if significant enough
    for (auto tb : frameTrackingResult) {
        double centroid[3] = {0, 0, 0};
        getCentroidTransformed(colorImg2, depthImg2, tb.box, centroid); //RETURNS CENTROID IN APRILTAG COORDS
        if (centroid[0] == -1 || centroid[1] == -1 || centroid[2] == -1) {
            continue;
        }
        bool objectFound = false;
        //Update elements and delete old ones
        dataLock.lock();
        for (int i = 0; i < database.size(); i++) {
            //If same object, update it
            if (tb.id == database.at(i).ID) {
                database.at(i).timestamp = ros::Time::now().toSec();
                database.at(i).x = centroid[0];
                database.at(i).y = centroid[1];
                database.at(i).z = centroid[2];
                //Add depth potentially
                database.at(i).box = tb.box;
                database.at(i).ticksInactive = 0;
                database.at(i).score = tb.score;
                objectFound = true;
                break;
            }
        }
        if (!objectFound) {
            database.push_back(obj(tb.id, ros::Time::now().toSec(), tb.box, tb.classifier, centroid[0], centroid[1], centroid[2], 0, 0, tb.score));
        }
        dataLock.unlock();
    }

    dataLock.lock();

    //Update depth of objects that are still in database but not identified
    for (int i = 0; i < database.size(); i++) {
        double centroid[3] = {0, 0, 0};
        obj temp = database.at(i);
        if (temp.ticksInactive > 1) {
            getCentroidTransformed(colorImg2, depthImg2, temp.box, centroid);
            if (centroid[0] == -1 || centroid[1] == -1 || centroid[2] == -1) {
                continue;
            }
            database.at(i).x = centroid[0];
            database.at(i).y = centroid[1];
            database.at(i).z = centroid[2];
        }
    }
    dataLock.unlock();

    if (COMPLEX_EVENT) {
        bool washing_machine_detected = 0;
        double washing_machine_coords[3] = {0, 0, 0};
        for (int i = 0; i < database.size(); i++) {
            obj temp = database.at(i);
            if (temp.type == "washing machine") {
                washing_machine_detected = 1;
                washing_machine_coords[0] = temp.x;
                washing_machine_coords[1] = 0;
                washing_machine_coords[2] = temp.z;
            }
            if (washing_machine_detected) break;
        }
        if (!washing_machine_detected)
        {
            interaction_detected = 0;
        }
        else
        {
            for (int i = 0; i < database.size(); i++)
            {
                obj temp = database.at(i); // person?
                double person_coords[3] = {0, 0, 0};
                if (temp.type == "person") {
                    washing_machine_detected = 1;
                    person_coords[0] = temp.x;
                    person_coords[1] = 0;
                    person_coords[2] = temp.z;
                    double distance = computeDistance(person_coords, washing_machine_coords);
                    if (distance <= 0.5) //TODO change constant
                    {
                        interaction_detected = 1;
                    }
                }
            }
        }
        dataLock.unlock();
    }

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB, 3);

    //Iterate through database and extract information
    dataLock.lock();
    for (int i = 0; i < database.size(); i++) {
        obj elem = database.at(i);
        //Do not display objects that have been inactive for a while
        if (elem.ticksInactive > 5) {
            continue;
        }
        Rect_<float> box = elem.box;
        rectangle(img, elem.box, Scalar(0), 5);
        string state = getState(elem.type, elem.state);
        double centroid[3] = {elem.x, elem.y, elem.z};
        double distance = computeDistance(centroid, uwb1Absolute);
        std::stringstream ss;
        ss << ((int)(distance * 1000)) / 1000.0 << "m";
        cv::putText(img, //target image
            ss.str(), //text
            cv::Point(box.x + 5,  box.y + 20), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            0.8,
            CV_RGB(255, 0, 0), //font color
            2);
        cv::putText(img, //target image
            elem.type, //text
            (elem.type.length() > 13) ? cv::Point(box.x - 60,  box.y + 70)
            : cv::Point(box.x + 5, box.y + 70), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            0.8,
            CV_RGB(255, 0, 0), //font color
            2);
        cv::putText(img, //target image
            state, //text
            cv::Point(box.x + 5 ,  box.y + 120), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            0.8,
            CV_RGB(255, 0, 0),
            2);
    }

    dataLock.unlock();

    if (COMPLEX_EVENT) {
        switch(curr_stage) {
            case 0:
                curr_stage_text = "Idle";
                if (washine_machine_state==1) curr_stage += 1;
                interaction_detected = 0;
                break;
            case 1:
                curr_stage_text = "Washing";
                if (washine_machine_state==0) curr_stage += 1;
                interaction_detected = 0;
                break;
            case 2:
                curr_stage_text = "Wash Done (Alarm)";
                if (interaction_detected) curr_stage += 1;
                interaction_detected = 0;
                break;
            case 3:
                curr_stage_text = "Wash Done";
                if (washine_machine_state==2)  curr_stage += 1;
                interaction_detected = 0;
                break;
            case 4:
                curr_stage_text = "Drying";
                if (washine_machine_state==0) curr_stage += 1;
                interaction_detected = 0;
                break;
            case 5:
                curr_stage_text = "Drying Done";
                if (interaction_detected) curr_stage = 0;
                interaction_detected = 0;
                break;
        }
        /*
        cv::putText(img, //target image
            "Current Stage is:" + curr_stage_text, //text
            cv::Point(0,  30), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(255, 0, 0),
            2);
            */
    }
    //Display
    cv::imshow("Output", img);
    cv::waitKey(1);
    if (++timeCount >= 100) {
        timeCount = 1;
        averageTime = 0;
    }
    //averageTime += ros::Time::now().toSec() - startTime;
    //std::cout << "Leaving with elapsed " << averageTime / timeCount << std::endl;

}

//TODO function measuring performance of single SVM computation
void calling_svm(obj& elem)
{

    double slice[3 * chunk_size2 - 2] = {0.0}; //If this isnt zero filled does it cause any issues?
    if (elem.vibration.empty()) {
        return;
    }
    double *tempArr = elem.vibration.back();
    for (int i = 0; i < chunk_size2; i++) {
        slice[i] = tempArr[chunk_size2-1-i];
    }
    for (int i = 0; i < chunk_size2; i++) {
        slice[i+chunk_size2-1] = tempArr[i];
    }
    for (int i = 0; i < chunk_size2; i++) {
        slice[i+chunk_size2-2] = tempArr[chunk_size2-1-i];
    }

    //low pass filtering cut_low=20, cut_high=70
    Eigen::VectorXd zi(b_size);
    double largeZVector[3 * chunk_size2 - 2];
    double newZVector[chunk_size2];
    zi(0) = 0;
    for (int j = 0; j < 3 * chunk_size2 - 2; j++)
    {
        largeZVector[j] = (b[0] * slice[j]) + zi(0);
        for (int k = 1; k < b_size; k++)
        {
            zi[k - 1] = b[k] * slice[j] + zi[k];
        }
    }

    for(int i = 0; i < chunk_size2; i++) {
        newZVector[i] = largeZVector[i + chunk_size2-1];
        /* Debug: print the filtered signal
        if (elem.type == "washing machine") {
            if (i < chunk_size2 - 1) {
                printf("%.16f ,", newZVector[i]);
            }
            else if (i == chunk_size2 - 1){
                 printf("%.16f]\n", newZVector[i]);
            }
        }
        */
    }
    // printf("\n\n\n");

    // Removed: Normalize the filtered signal to [0,1]
    // double min_val = *std::min_element(newZVector, newZVector + chunk_size2);
    // double max_val = *std::max_element(newZVector, newZVector + chunk_size2);
    Eigen::VectorXcd sig_amp(chunk_size2);
    // for (int i=0;i<chunk_size2;i++)
    // {
    //     sig_amp[i] = (newZVector[i]- min_val) / (max_val - min_val + 1e-32);
    // }
    for (int i=0;i<chunk_size2;i++)
    {
        sig_amp[i] = newZVector[i];
    }

    //Perform FFT
    Eigen::FFT<double> fft;
    Eigen::VectorXcd Y(chunk_size2);
    fft.fwd(Y, sig_amp);

    int half_length = chunk_size2 / 2;
    double P1[half_length];
    for(int i=0;i<half_length;i++)
    {
        P1[i] = std::abs(Y[i+1]);
    }

    int feature_size = 32;
    double features[feature_size] = {0.0};
    int intevals = half_length/feature_size;
    for(int i=0; i<32; i++)
    {
        double temp_max = 0.0;
        for(int j=0; j<intevals;j++)
        {
            if (P1[i*intevals+j] > temp_max)
            {
                temp_max = P1[i*intevals+j];
            }
        }
        features[i] += temp_max;
    }
    // Removed: Normalize the extracted frequency features
    double min_val = *std::min_element(features, features+feature_size);
    double max_val = *std::max_element(features, features+feature_size);
    for (int i=0;i<feature_size;i++)
    {
       features[i] = (features[i]- min_val) / (max_val - min_val + 1e-32);
    }

    /* Debug: Printing the extracted features
    if (elem.type == "washing machine") {
        printf("[");
        for(int i = 0; i < 32; i++) {
                if (i < 32-1) {
                    printf("%.16f ,", features[i]);
                }
                else if (i == 32-1){
                        printf("%.16f]\n\n\n", features[i]);
                }
            }
    }
    */

    int state = -1;
    string classifier = elem.type;
    if (classifier == "standing fan") {
        state = FanClassifier::fan_predict(features);
    }
    else if (classifier == "vacuum") {
        state = VacuumClassifier::vacuum_predict(features);
    }
    else if (classifier == "washing machine") {
        state = WashingClassifier::washing_predict(features);
        //cout << state<<endl<<endl<<endl;
    }
    else if (classifier == "drill") {
        state = DrillClassifier::drill_predict(features);
    }
    else {
        std::cout << classifier << std::endl;
        std::cout << "SOMETHING IS TERRIBLY WRONG" << std::endl;
    }

    dataLock.lock();
    // elem.state = state; //if no buffer is used
    // dataLock.unlock();
    if (elem.state_history.size() >= 5) { // 500ms * 30 = 15s
        elem.state_history.pop_front();
    }
    elem.state_history.push_back(state);
    int max_state_defined = 4;
    float temp_state = 0.0;
    int state_cnt[max_state_defined] = {0}; // max # of states of any object
    for (int i=0; i<elem.state_history.size(); i++)
    {
        state_cnt[elem.state_history.at(i)] += 1;
    }
    /* Debug: Buffer health
    for (int i=0; i<max_state_defined; i++)
    {
       cout << state_cnt[i] << " | ";
    }
    cout << endl;
    */
    elem.state = std::distance(state_cnt, std::max_element(state_cnt, state_cnt+max_state_defined));
    if (classifier == "washing machine") {
        washine_machine_state = elem.state;
    }
    dataLock.unlock();

    //std::cout << elem.type << "'s state is " << elem.state << std::endl;
}


//TODO function for single VMD computation
int calling_vmd(obj& elem)
{
    double startTime = ros::Time::now().toSec();
    int Fs = 1000;
    std::vector<double> unfiltered_data = elem.flattenArr();
    if (unfiltered_data.size() <= 20 * chunk_size) {
        return 0;
    }
    int sig_length = unfiltered_data.size();
    const double alpha = 50.0, tau = 0, tol = 1e-7, eps = 2.2204e-16;
	const int K = 4, DC = 0, init = 1;
	MatrixXd u, omega;
	MatrixXcd u_hat;

	runVMD(u, u_hat, omega, unfiltered_data, alpha, tau, K, DC, init, tol, eps);

    Eigen::VectorXd new_sig = u.row(0);
	Eigen::VectorXd new_sig1 = new_sig.array() - new_sig.mean();

	//fft
	Eigen::FFT<double> fft;
    Eigen::VectorXcd Y(sig_length);
    fft.fwd(Y, new_sig1);

    int half_length = sig_length/2;
    int ignored_freq_bins =  int(0.13/1000 * sig_length) + 1;

    double P1[half_length - ignored_freq_bins];
    double max_val = -1;
    int max_idx = -1;
    for(int j=ignored_freq_bins;j<half_length;j++)
    {
        P1[j-ignored_freq_bins] = std::abs(Y[j]);
        if (P1[j-ignored_freq_bins] > max_val)
        {
            max_val = P1[j-ignored_freq_bins];
            max_idx = j;
        }
    }

    float bpm = 60 * float(max_idx) / float(sig_length) * Fs;
    dataLock.lock();
    elem.state = (int) bpm;
    dataLock.unlock();
    std::cout << ros::Time::now().toSec() - startTime << std::endl;
    return 0;
}

int state_classifiers()
{
    std::vector<std::thread> ThreadVector;
    while(1)
    {
        //TODO conduct timing analysis here to measure one pass of SVM on a frame
        dataLock.lock();
        for (int i = 0; i < database.size(); i++) {
            int temp = i;
            obj& elem = database.at(i);
            if (elem.type != "person")
            {
                if (elem.ticksInactive > deletionTicks) {
                    std::cout << "Element is being deleted" << std::endl;
                    while (!database.at(i).vibration.empty()) {
                        delete database.at(i).vibration.front();
                        database.at(i).vibration.pop_front();
                    }
                    database.erase(database.begin() + i);
                    i--;
                }
                else {
                    //std::function<void()> t1 = []() {calling_svm(elem, temp);};
                    //ThreadVector.emplace_back(t1);
                    ThreadVector.emplace_back([&](){calling_svm(elem);});
                }
            }

        }
        dataLock.unlock();

        ThreadVector.emplace_back([&](){usleep(500000);}); //guardian thread, 100ms(10Hz)
        if (ThreadVector.size())
        {
            for(auto& t: ThreadVector)
            {
                t.join();
            }
        }
        ThreadVector.clear();
        // std::cout << std::endl;
        /*
        while (ThreadVector.size())
        {
            std::cout << "Erasing" << std::endl;
            ThreadVector.erase(ThreadVector.begin());
        }
        */
    }
    return 0;
}

int respiration_est()
{
    std::vector<std::thread> ThreadVector;
    while(1)
    {
        //TODO conduct timing analysis here for VMD running on one frame
        dataLock.lock();
        for (int i = 0; i < database.size(); i++) {
            obj& elem = database.at(i);
            if (elem.type == "person")
            {
                if (elem.ticksInactive > deletionTicks) {
                    while (!database.at(i).vibration.empty()) {
                        delete database.at(i).vibration.front();
                        database.at(i).vibration.pop_front();
                    }
                    database.erase(database.begin() + i);
                }
                else {
                    ThreadVector.emplace_back([&](){calling_vmd(elem);});
                }
            }
        }
        dataLock.unlock();
        ThreadVector.emplace_back([&](){usleep(1000000);}); //guardian thread, 1s(1Hz)
        if (ThreadVector.size())
        {
            for(auto& t: ThreadVector)
            {
                t.join();
            }
        }
        while (ThreadVector.size())
        {
            ThreadVector.erase(ThreadVector.begin());
        }

    }
    return 0;
}

int start_ros_spin()
{
    ros::spin();
    return 0;
}



int main(int argc, char* argv[]) {
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
        module = torch::jit::load("/home/ziqi/Desktop/Capricorn_debug_ws/src/object_tracker/Weights/640_Medium.torchscript.pt", device_type);
    } else {
        device_type = torch::kCPU;
        module = torch::jit::load("/home/nesl/Desktop/final_capricorn_ws/src/object_tracker/Weights/FullSet.torchscript.pt", device_type);
    }
    ros::init(argc, argv, "object_tracker");
    ros::NodeHandle nh;
    message_filters::Subscriber<sensor_msgs::Image> color_sub(nh, "/lidar/color", 5);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/lidar/depth", 5);
    typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicyLidar;
    Synchronizer<MySyncPolicyLidar> syncLidar(MySyncPolicyLidar(10), color_sub, depth_sub);
    syncLidar.registerCallback(boost::bind(&trackObjects, _1, _2));
    #if MULTI_VIEW_UWB
        message_filters::Subscriber<timeArr_msgs::FloatArray> uwb1_sub(nh, "/uwb_chunk", 5);
        typedef sync_policies::ApproximateTime<timeArr_msgs::FloatArray, timeArr_msgs::FloatArray> MySyncPolicyUWB;
        message_filters::Subscriber<timeArr_msgs::FloatArray> uwb2_sub(nh, "/uwb_chunk2", 5);
        Synchronizer<MySyncPolicyUWB> syncUWB(MySyncPolicyUWB(10), uwb1_sub, uwb2_sub);
        syncUWB.registerCallback(boost::bind(&uwbCallback, _1, _2));
    #else
        ros::Subscriber uwb_sub = nh.subscribe("uwb_chunk", 5, uwbCallback);
    #endif

    std::ifstream f("/home/nesl/Desktop/final_capricorn_ws/src/object_tracker/objects.names");
    std::string name = "";
    while (std::getline(f, name))
    {
        classnames.push_back(name);
    }
    intrin.width = 640;
    intrin.height = 480;
    intrin.fx = fx;
    intrin.fy = fy;
    intrin.ppx = ppx;
    intrin.ppy = ppy;
    intrin.model = rs2_distortion::RS2_DISTORTION_BROWN_CONRADY;
    intrin.coeffs[0] = 0.169433;
    intrin.coeffs[1] = -0.511254;
    intrin.coeffs[2] = 0.000683141;
    intrin.coeffs[3] = 8.51166e-05;
    intrin.coeffs[4] = 0.462624;
    std::thread sensor_fusion(start_ros_spin);
    std::thread state_classifications(state_classifiers);
    std::thread state_regressions(respiration_est);
    state_classifications.detach();
    //state_regressions.detach();
    sensor_fusion.join();
}
