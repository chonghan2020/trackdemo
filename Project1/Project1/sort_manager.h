#pragma once
#ifndef SORTMANAGER
#define SORTMANAGER

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
using namespace cv;
using namespace std;

using samplesCommon::SampleUniquePtr;

extern std::mutex img_lock;
extern std::queue<ImgStruct> img_list;


class SampleOnnxSort
{
public:
	const std::string gSampleName = "TensorRT.sample_onnx_deepsort";

	SampleOnnxSort(const commonparams::OnnxParams& params)
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
	bool infer(float* sortinput, float* sortoutput, int nBatch);


private:
	commonparams::OnnxParams mParams; //!< The parameters for the sample.

	bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
		SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
		SampleUniquePtr<nvonnxparser::IParser>& parser);

	nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
	nvinfer1::Dims mOutputDims;

	std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
	SampleUniquePtr<nvinfer1::IExecutionContext> mContext;
	template <typename T>
	SampleUniquePtr<T> makeUnique(T* t)
	{
		return SampleUniquePtr<T>{t};
	}
};




#endif
