#pragma once
#ifndef DEEPSORT
#define DEEPSORT


#include "yolo_manager.h"
#include <time.h>
#include "sort_manager.h"
#include "tracker.h"

#define NOMINMAX


extern std::mutex img_lock;
extern std::queue<ImgStruct> img_list;

class Deepsort
{

public:

	std::vector<std::string> obj_names;
	std::string  names_file = "../../others/coco.names";
public:
	Deepsort(){};
	bool init(int argc, char** argv);
	bool process();
	bool process2();
	bool videoprocess();
	int maxBudget;
	float maxCosineDist;
	vector<RESULT_DATA> result;
	tracker* objTracker;


private:
	SampleOnnxYolo*  yoloservice;
	SampleOnnxSort*  sortservice;
	const std::string gSampleName = "TensorRT.sample_onnx_yolov7_deepsort";
	sample::Logger::TestAtom* sampleTest;

	bool extractInput(ImgStruct img_data, Box box, float* modelinput);
	bool extractInput2(ImgStruct img_data, cv::Rect box, unsigned char* modelinput);
	
};


#endif