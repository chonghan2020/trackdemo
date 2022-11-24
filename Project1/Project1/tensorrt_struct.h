#pragma once

#include <string>
#include <vector>
#include <iostream>
#include "./windows/getopt.h"

namespace commonparams
{

	struct Args
	{
		bool runInInt8{ false };
		bool runInFp16{ false };
		bool help{ false };
		int32_t useDLACore{ -1 };
		int32_t batch{ 1 };
		std::vector<std::string> imgPath;
		std::string saveEngine;
		std::string loadEngine;
		bool useILoop{ false };
	};

	struct Params
	{
		int32_t batchSize{ 1 };              //!< Number of inputs in a batch
		int32_t dlaCore{ -1 };               //!< Specify the DLA core to run network on.
		bool int8{ false };                  //!< Allow runnning the network in Int8 mode.
		bool fp16{ false };                  //!< Allow running the network in FP16 mode.
		std::vector<std::string> imgPath;  //!< Directory paths where sample data files are stored
		std::vector<std::string> inputTensorNames;
		std::vector<std::string> outputTensorNames;
	};

	struct OnnxParams : public Params
	{
		std::string onnxFileName; //!< Filename of ONNX file of a network
	};

	inline bool parseArgs(Args& args)
	{

		args.runInInt8 = true;
		args.runInFp16 = true;
		args.useDLACore = 0;
		return true;
	}
}