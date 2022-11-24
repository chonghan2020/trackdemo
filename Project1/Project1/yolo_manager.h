#pragma once

#ifndef MODEL_MANAGER
#define MODEL_MANAGER
#include <opencv2/opencv.hpp>

#include "tensorrt_struct.h"

#include "NvOnnxParser.h"

#include "argsParser.h"
#include "buffers.h"
#include "trt_common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include "util.h"
#include "aippcommon.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <algorithm> 
#include <vector>




using namespace nvonnxparser;
using namespace nvinfer1;
using namespace std;

using samplesCommon::SampleUniquePtr;

extern std::mutex img_lock;
extern std::queue<ImgStruct> img_list;

static float sigmod(float);
static bool cmpScore(Box, Box);
static float get_iou_value(Box, Box);
void nms(std::vector<Box>&, float);


class SampleOnnxYolo
{
public:
	const std::string gSampleName = "TensorRT.sample_onnx_yolov5";

	SampleOnnxYolo(const commonparams::OnnxParams& params)
		: mParams(params)
		, mEngine(nullptr)
	{
	}

	//!
	//! \brief Function builds the network engine
	//!
	bool build();

	//!
	//! \brief Runs the TensorRT inference engine for this sample
	//!
	bool infer(ImgStruct,vector<Box>&);
	bool infer(unsigned char* img_, vector<Box>& results, cv::Rect Box);
	std::vector<int> needKind = {0,2,3,4,5,6,7};

	std::vector<std::string> className = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
								"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
								"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
								"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
								"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
								"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
								"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
								"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
								"hair drier", "toothbrush" };

private:
	commonparams::OnnxParams mParams; //!< The parameters for the sample.
	aipp::processparam params;
	std::vector< nvinfer1::Dims > input_dims; // we expect only one input
	std::vector< nvinfer1::Dims > output_dims;
	nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
	nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
	int mNumber{ 0 };             //!< The number to classify

	std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
	std::shared_ptr<nvinfer1::IExecutionContext> mcontext;
	//!
	//! \brief Parses an ONNX model for MNIST and creates a TensorRT network
	//!
	bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
		SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
		SampleUniquePtr<nvonnxparser::IParser>& parser);

	//!
	//! \brief Reads the input  and stores the result in a managed buffer
	//!
	bool processinput(float* gpu_input, const nvinfer1::Dims& dims, unsigned char* img_data, cv::Rect box);
	bool processinput(const samplesCommon::BufferManager& buffers,ImgStruct img, aipp::processparam& params);
	bool processinput(const samplesCommon::BufferManager& buffers, unsigned char* img_data, cv::Rect box);
	bool processinput(float* gpu_input, const nvinfer1::Dims& dims, unsigned char* img_data);
	//!
	//! \brief Classifies digits and verify result
	//!
	bool verifyOutput(const samplesCommon::BufferManager& buffers, vector<Box>&, aipp::processparam param);
	bool verifyOutput(std::vector< float* > buffers, vector<Box>& results, aipp::processparam param);
	void converttoOrgimg(std::vector<Box>&, aipp::processparam);
};

#endif

