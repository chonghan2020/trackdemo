#include "sort_manager.h"

bool SampleOnnxSort::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
	SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
	SampleUniquePtr<nvonnxparser::IParser>& parser)
{
	auto parsed = parser->parseFromFile(mParams.onnxFileName.c_str(),
		static_cast<int>(sample::gLogger.getReportableSeverity()));
	if (!parsed)
	{
		return false;
	}

	if (mParams.fp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}
	if (mParams.int8)
	{
		config->setFlag(BuilderFlag::kINT8);
		samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
	}

	samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

	return true;
}

// 获取模型输入输出 设置runtime

bool SampleOnnxSort::build()
{
	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
	if (!builder)
	{
		return false;
	}

	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
	if (!network)
	{
		return false;
	}

	auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		return false;
	}

	IOptimizationProfile* profile = builder->createOptimizationProfile();
	// 这里有个OptProfileSelector，这个用来设置优化的参数,比如（Tensor的形状或者动态尺寸），

	profile->setDimensions("input", OptProfileSelector::kMIN, Dims4(1, 3, 128, 64));
	profile->setDimensions("input", OptProfileSelector::kOPT, Dims4(5, 3, 128, 64));
	profile->setDimensions("input", OptProfileSelector::kMAX, Dims4(10, 3, 128, 64));

	config->addOptimizationProfile(profile);

	builder->setMaxBatchSize(SORTBATCH);
	auto parser
		= SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
	if (!parser)
	{
		return false;
	}

	auto constructed = constructNetwork(builder, network, config, parser);
	if (!constructed)
	{
		return false;
	}

	// CUDA stream used for profiling by the builder.
	auto profileStream = samplesCommon::makeCudaStream();
	if (!profileStream)
	{
		return false;
	}
	config->setProfileStream(*profileStream);

	SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
	if (!plan)
	{
		return false;
	}

	SampleUniquePtr<IRuntime> runtime{ createInferRuntime(sample::gLogger.getTRTLogger()) };
	if (!runtime)
	{
		return false;
	}
	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));

	if (!mEngine)
	{
		return false;
	}


	ASSERT(network->getNbInputs() == 1);
	mInputDims = network->getInput(0)->getDimensions();
	ASSERT(mInputDims.nbDims == 4);

	ASSERT(network->getNbOutputs() == 1);
	mOutputDims = network->getOutput(0)->getDimensions();
	ASSERT(mOutputDims.nbDims == 2);

	return true;
}

// 模型推理

bool SampleOnnxSort::infer(float* sortinputs, float* sortoutputs, int nBatch)
{

	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if (!context)
	{
		return false;
	}
	// Read the input data into the managed buffers
	std::vector<void*> vecBuffers;
	vecBuffers.resize(2);

	
	// 在cuda上创建内存空间
	(cudaMalloc(&vecBuffers[0], SORT_INPUTSIZE * nBatch * sizeof(float)));
	(cudaMalloc(&vecBuffers[1], nBatch * SORT_SHAPE * sizeof(float)));

	cudaMemcpy((float *)vecBuffers[0], sortinputs, SORT_INPUTSIZE * nBatch * sizeof(float), cudaMemcpyHostToDevice);
	context->setBindingDimensions(0, Dims4(nBatch, 3, 128, 64));
	if (!context->allInputDimensionsSpecified())
	{
		return false;
	}
	bool status = context->executeV2(vecBuffers.data());
	if (!status)
	{
		return false;
	}
	(cudaMemcpy(sortoutputs, vecBuffers[1], nBatch * SORT_SHAPE * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(vecBuffers[0]);
	cudaFree(vecBuffers[1]);
	return true;

}
