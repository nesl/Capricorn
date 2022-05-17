#include<iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
//#include <Eigen/Dense>
#include <complex>
//#include <unsupported/Eigen/FFT>
#include <array>
//using Eigen::MatrixXd;
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
using namespace std;
using namespace cv;
#define VIBRATION 1024

class obj{
	
	public:
	int ID;
	double timestamp;
	std::string type;
	double x;
	double y;
	double z;
	float score;

	//double vibration[VIBRATION] ={0};
	//std::array<double, VIBRATION> vibration ={0};
	std::deque<double*> vibration;
	int state;
	std::deque<int> state_history;
	int ticksInactive;
	Rect_<float> box;
	
	obj(int id, double now, Rect_<float> box, string type="", double x=0, double y = 0, double z = 0, int state=0, int ticksInactive=0, float score = 0){
	
		this->ID = id;
		this->timestamp = now;
		this->type = type;
		this->x = x;
		this->y = y;
		this->z = z;
		this->ticksInactive=ticksInactive;
		this->state = state;
		this->box = box;
		this->score = score;
	}

	//Should work note that I changed it to a deque to have access to the individual elements, if there are compilation issues please check that
	std::vector<double> flattenArr() {
		std::vector<double> flat;
		if (vibration.empty()) {
			return flat;
		}
		for (int i = 0; i < vibration.size(); i++) {
			for (int j = 0; j < VIBRATION; j++) {
				flat.push_back(vibration[i][j]);
			}
		}
		return flat;
	}
	/*
	//set functions
	void setType(int type){
		this->type = type;
	}
	void setTime(std::time_t now){
	this->timestamp = now;
	}
	void setDistance(double distance){
	this->distance = distance;
	}
	
	void setVibration(std::array<double, VIBRATION> b){
	
	this->vibration = b;
	
	}
	
	void setState(int state){
	
	this->state = state;
	}
	
	void setStatus(bool status){
	
	this->state = status;
	}
	
	
	//get functions
	int getType(int type){
		return this->type;
	}
	std::time_t getTime(std::time_t now){
	return this->timestamp;
	}
	double getDistance(double distance){
	return this->distance;
	}
	
	std::array<double, VIBRATION> getVibration(std::array<double, VIBRATION> b){
	
	return this->vibration;
	
	}
	
	int getState(int state){
	
	return this->state;
	}
	
	bool getStatus(bool status){
	
	return this->state;
	}

	Rect<float> getBox() {
		return this->box;
	}
	*/
};

/*
int main(){

	printf("Hello World\n");
	printf("ABC\n");
	
	auto end = std::chrono::system_clock::now();
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	
	//vector of objects
	vector<obj> listofObjs;
	
	obj obj1(1,end_time);
	obj obj2(2,end_time);
	
	listofObjs.push_back(obj1);
	listofObjs.push_back(obj2);
	
	//insert
	auto iter = listofObjs.begin();
	int pos = 1;
	listofObjs.insert(listofObjs.begin()+pos,obj1);
	
	//remove 
	
	listofObjs.erase(std::next(listofObjs.begin(), 2));
	
	
	//iterator to sequentially access items from the list
	vector<obj>::iterator it;
    	for (it = listofObjs.begin(); it != listofObjs.end(); ++it) {
        	cout << it->ID << "\n";
        	
    	}
    	
    	for(obj it: listofObjs){//another iterator
    		cout << it.ID << "\n";
    	
    	}
	
	//possible to use reverse iterator as well
	
	
	
	
	return 0;

}
*/