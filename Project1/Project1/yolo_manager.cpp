#include "yolo_manager.h"

//onnx 建立模型 指定模型精度 dla设置

bool SampleOnnxYolo::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
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

bool SampleOnnxYolo::build()
{

	std::ifstream engineFile("./yolov7_640.trt", std::ios::binary);
	if (!engineFile)
	{
		return false;
	}

	engineFile.seekg(0, engineFile.end);
	long int fsize = engineFile.tellg();
	engineFile.seekg(0, engineFile.beg);

	std::vector<char> engineData(fsize);
	engineFile.read(engineData.data(), fsize);

	SampleUniquePtr<IRuntime> runtime{ createInferRuntime(sample::gLogger.getTRTLogger()) };

	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr));

	if (!mEngine)
	{
		return false;
	}

	return true;
}

// 模型推理


size_t getSizeByDim(const nvinfer1::Dims& dims)
{
	size_t size = 1;
	for (size_t i = 0; i < dims.nbDims; ++i)
	{
		size *= dims.d[i];
	}
	return size;
}

bool SampleOnnxYolo::infer(ImgStruct img_, vector<Box>& results)
{

	mcontext = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

	if (!mcontext)
	{
		return false;
	}

	std::vector< nvinfer1::Dims > input_dims; // we expect only one input
	std::vector< nvinfer1::Dims > output_dims; // and one output
	std::vector<int> output_size;
	std::vector< void* > buffers(mEngine->getNbBindings()); // buffers for input and output data
	for (size_t i = 0; i < mEngine->getNbBindings(); ++i)
	{
		auto binding_size = getSizeByDim(mEngine->getBindingDimensions(i)) * 1 * sizeof(float);
		cudaMalloc(&buffers[i], binding_size);
		if (mEngine->bindingIsInput(i))
		{
			input_dims.emplace_back(mEngine->getBindingDimensions(i));
		}
		else
		{
			output_dims.emplace_back(mEngine->getBindingDimensions(i));
			output_size.push_back(binding_size);
		}
	}
	if (input_dims.empty() || output_dims.empty())
	{
		std::cerr << "Expect at least one input and one output for networkn";
		return -1;
	}

	float* inputdata_deal = new float[getSizeByDim(mEngine->getBindingDimensions(0))];
	if (!processinput(inputdata_deal, input_dims[0], img_.data))
	{
		return false;
	}
	//cv::Mat m(1280, 1280, CV_8UC3);
	//int gird_len = 1280 * 1280;
	//for (int i = 0; i < 1280; i++)
	//{
	//	for (int j = 0; j < 1280; j++)
	//	{
	//		m.at<cv::Vec3b>(i, j)[2] = int(inputdata_deal[i * 1280 + j + 2 * gird_len]*255);
	//		m.at<cv::Vec3b>(i, j)[1] = int(inputdata_deal[i * 1280 + j + gird_len] * 255);
	//		m.at<cv::Vec3b>(i, j)[0] = int(inputdata_deal[i * 1280 + j] * 255);
	//	}
	//}
	//cv::imshow("hc",m);
	//cv::waitKey(0);

	CHECK(cudaMemcpy(buffers[0], (unsigned char *)inputdata_deal, getSizeByDim(input_dims[0]) * sizeof(float), cudaMemcpyHostToDevice));

	int batch_size = 1;

	bool status = mcontext->enqueueV2(buffers.data(), 0, nullptr);
	if (!status)
	{
		return false;
	}
	vector<float*> outputs_cpu;
	float* out1 = new float[getSizeByDim(output_dims[0])];
	float* out2 = new float[getSizeByDim(output_dims[1])];
	float* out3 = new float[getSizeByDim(output_dims[2])];

	cudaMemcpy(out1, buffers[1], getSizeByDim(output_dims[0]) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(out2, buffers[2], getSizeByDim(output_dims[1]) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(out3, buffers[3], getSizeByDim(output_dims[2]) * sizeof(float), cudaMemcpyDeviceToHost);
	outputs_cpu.push_back(out1);
	outputs_cpu.push_back(out2);
	outputs_cpu.push_back(out3);



	if (!verifyOutput(outputs_cpu, results, params))
	{
		return false;
	}

	cudaStreamDestroy(0);
	cudaFree(buffers[0]);
	cudaFree(buffers[1]);
	cudaFree(buffers[2]);
	cudaFree(buffers[3]);

	delete[] out1;
	delete[] out2;
	delete[] out3;
	delete[] inputdata_deal;
	return true;


}


bool SampleOnnxYolo::infer(unsigned char* img_, vector<Box>& results, cv::Rect Box)
{

	mcontext = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

	if (!mcontext)
	{
		return false;
	}

	std::vector< nvinfer1::Dims > input_dims; // we expect only one input
	std::vector< nvinfer1::Dims > output_dims; // and one output
	std::vector<int> output_size;
	std::vector< void* > buffers(mEngine->getNbBindings()); // buffers for input and output data
	for (size_t i = 0; i < mEngine->getNbBindings(); ++i)
	{
		auto binding_size = getSizeByDim(mEngine->getBindingDimensions(i)) * 1 * sizeof(float);
		cudaMalloc(&buffers[i], binding_size);
		if (mEngine->bindingIsInput(i))
		{
			input_dims.emplace_back(mEngine->getBindingDimensions(i));
		}
		else
		{
			output_dims.emplace_back(mEngine->getBindingDimensions(i));
			output_size.push_back(binding_size);
		}
	}
	if (input_dims.empty() || output_dims.empty())
	{
		std::cerr << "Expect at least one input and one output for networkn";
		return -1;
	}

	float* inputdata_deal = new float[getSizeByDim(mEngine->getBindingDimensions(0))];
	if (!processinput(inputdata_deal, input_dims[0], img_, Box))
	{
		return false;
	}

	CHECK(cudaMemcpy(buffers[0], (unsigned char *)inputdata_deal, getSizeByDim(input_dims[0]) * sizeof(float), cudaMemcpyHostToDevice));

	int batch_size = 1;

	bool status = mcontext->enqueueV2(buffers.data(), 0, nullptr);
	if (!status)
	{
		return false;
	}
	vector<float*> outputs_cpu;
	float* out1 = new float[getSizeByDim(output_dims[0])];
	float* out2= new float[getSizeByDim(output_dims[1])];
	float* out3= new float[getSizeByDim(output_dims[2])];

	cudaMemcpy(out1, buffers[1], getSizeByDim(output_dims[0]) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(out2, buffers[2], getSizeByDim(output_dims[1]) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(out3, buffers[3], getSizeByDim(output_dims[2]) * sizeof(float), cudaMemcpyDeviceToHost);
	outputs_cpu.push_back(out1);
	outputs_cpu.push_back(out2);
	outputs_cpu.push_back(out3);



	if (!verifyOutput(outputs_cpu, results, params))
	{
		return false;
	}

	cudaStreamDestroy(0);
	cudaFree(buffers[0]);
	cudaFree(buffers[1]);
	cudaFree(buffers[2]);
	cudaFree(buffers[3]);

	delete[] out1;
	delete[] out2;
	delete[] out3;
	delete[] inputdata_deal;
	return true;
}


//模型输入处理
bool SampleOnnxYolo::processinput(const samplesCommon::BufferManager& buffers, ImgStruct img_data, aipp::processparam& params)
{
	const int inputH = mInputDims.d[2];
	const int inputW = mInputDims.d[3];
	vector<int> dstsize = { inputW,inputH };
	vector<int> srcsize{ INPUTWIDTH,INPUTHEIGHT};
	params=aipp::getparams(srcsize, dstsize);
	float* modelinput=new float[inputH *inputW *3];
	aipp::imgprocess(img_data.data, modelinput, params);
	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
	memcpy(hostDataBuffer, modelinput, inputH * inputW * 3);


	delete[] modelinput;

	return true;
}


bool SampleOnnxYolo::processinput(float* gpu_input, const nvinfer1::Dims& dims, unsigned char* img_data, cv::Rect box)
{
	const int inputH = dims.d[2];
	const int inputW = dims.d[3];
	vector<int> dstsize = { inputW,inputH };
	vector<int> srcsize{ box.width,box.height };
	params = aipp::getparams(srcsize, dstsize);

	aipp::imgprocess(img_data, gpu_input, params);

	return true;
}

bool SampleOnnxYolo::processinput(float* gpu_input, const nvinfer1::Dims& dims, unsigned char* img_data)
{
	const int inputH = dims.d[2];
	const int inputW = dims.d[3];
	vector<int> dstsize = { inputW,inputH };
	vector<int> srcsize{ INPUTWIDTH,INPUTHEIGHT };
	params = aipp::getparams(srcsize, dstsize);

	aipp::imgprocess(img_data, gpu_input, params);

	return true;
}


bool SampleOnnxYolo::processinput(const samplesCommon::BufferManager& buffers, unsigned char* img_data, cv::Rect box)
{
	const int inputH = mInputDims.d[2];
	const int inputW = mInputDims.d[3];
	vector<int> dstsize = { inputW,inputH };
	vector<int> srcsize{ box.width,box.height };
	params = aipp::getparams(srcsize, dstsize);
	float* modelinput = new float[inputH * inputW * 3];
	aipp::imgprocess(img_data, modelinput, params);

	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
	memcpy(hostDataBuffer, modelinput, inputH * inputW * 3);


	delete[] modelinput;

	return true;
}
//yolov5 后处理
bool process(vector<float*> result, vector<Box>& boxes, vector<vector<int>> featuremap_dm, vector<int> stride, vector<vector<int>> anchor, float threshold = 0.3, float iou = 0.45)
{
	int layernum = result.size();
	//float fan_thre=-log(1/threshold-1);
	for (int i = 0; i < layernum; i++)
	{
		float* result_i = result[i];
		vector<int> map_dm = featuremap_dm[i];
		int stride_i = stride[i];
		vector<int> anchor_i = anchor[i];
		for (int z = 0; z < 3; z++)
		{

			for (int j = 0; j < map_dm[2]; j++)
			{
				for (int k = 0; k < map_dm[3]; k++)
				{
					int grid_len = map_dm[2] * map_dm[3];

					//反向sigmod 
					int offset = z * PRO_BOX_SIZE*grid_len + j * map_dm[3] * PRO_BOX_SIZE + k * PRO_BOX_SIZE;
					float boxconfidence = sigmod(result_i[offset + 4]);
					if (boxconfidence< threshold)
					{
						continue;
					}
					float max_prob = 0;
					int idx = 0;
					for (int t = 5; t < 85; ++t)
					{
						float tp = sigmod(result_i[offset + t]);
						if (tp > max_prob) {
							max_prob = tp;
							idx = t - 5;
						}
					}
					float cof = boxconfidence * max_prob;
					if (cof < threshold)
					{
						continue;

					}
					else
					{
						Box box;
						float* res_off = result_i + offset;
						box.x = (sigmod((*res_off)) * 2 - 0.5 + k)*stride_i;
						box.y = (sigmod((*(res_off + 1))) * 2 - 0.5 + j)*stride_i;
						box.w = pow(sigmod((*(res_off + 2))) * 2, 2)*anchor_i[2 * z];
						box.h = pow(sigmod((*(res_off + 3))) * 2, 2)*anchor_i[2 * z + 1];
						if (box.w > 1000 || box.h > 1000)
						{
							continue;
						}
						box.score = cof;
						box.cls = idx;
						box.trackid = -1;
						boxes.push_back(box);
					}
				}
			}
		}

	}
	nms(boxes, iou);
	return true;
}

//模型输出处理

void SampleOnnxYolo::converttoOrgimg(std::vector<Box>& results, aipp::processparam param)
{
	int pad_l, pad_t;
	float scale_x;
	float scale_y;
	pad_l = param.pad_para.padleft;
	pad_t = param.pad_para.padtop;
	scale_x = param.res_para.scale_x;
	scale_y = param.res_para.scale_y;

	for (int i = 0; i < results.size(); i++)
	{
		//if (find(needKind.begin(), needKind.end(), results[i].cls) == needKind.end())
		//{
		//	results.erase(results.begin() + i);
		//}
		int w = results[i].w / scale_x;
		int h = results[i].h / scale_y;
		int x1 = min(max(0, int((results[i].x - pad_l) / scale_x - w / 2)), INPUTWIDTH);
		int y1 = min(max(0, int((results[i].y - pad_t) / scale_y - h / 2)), INPUTHEIGHT);
		int x2 = min(max(0, int(x1 + w)), INPUTWIDTH);
		int y2 = min(max(0, int(y1 + h)), INPUTHEIGHT);
		//if ((x2 - x1 <= 0) || (y2 - y1 <= 0))
		//{
		//	results.erase(results.begin() + i);
		//}
		//else
		//{
		results[i].x1 = x1;
		results[i].x2 = x2;
		results[i].y1 = y1;
		results[i].y2 = y2;
		//}
	}
}


bool SampleOnnxYolo::verifyOutput(const samplesCommon::BufferManager& buffers, vector<Box>& results, aipp::processparam param)
{
	//vector<vector<int>> featuremap_dm = { {1,3,80,80,85} ,{1,3,40,40,85} ,{1,3,20,20,85} };
	vector<vector<int>> featuremap_dm = { {1,3,160,160,85} ,{1,3,80,80,85} ,{1,3,40,40,85} };
	vector<int> stride = { 8,16,32 };
	vector<float*> result;
	vector<vector<int>> anchor = { {12,16, 19,36, 40,28},{36,75, 76,55, 72,146},{142,110, 192,243, 459,401} };
	//vector<vector<int>> anchor = { {10,13, 16,30, 33,23},{30,61, 62,45, 59,119},{116,90, 156,198, 373,326} };


	float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
	result.push_back(output);
	float* output1 = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));
	result.push_back(output1);
	float* output2 = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[2]));
	result.push_back(output2);

	process(result, results, featuremap_dm, stride, anchor);
	converttoOrgimg(results, param);

	return true;
}

bool SampleOnnxYolo::verifyOutput(std::vector< float* > buffers, vector<Box>& results, aipp::processparam param)
{
	vector<vector<int>> featuremap_dm = { {1,3,80,80,85} ,{1,3,40,40,85} ,{1,3,20,20,85} };
	vector<int> stride = { 8,16,32 };
	vector<float*> result;
	//vector<vector<int>> anchor = { {12,16, 19,36, 40,28},{36,75, 76,55, 72,146},{142,110, 192,243, 459,401} };
	vector<vector<int>> anchor = { {12,16, 19,36, 40,28},{36,75, 76,55, 72,146},{142,110, 192,243, 459,401} };


	process(buffers, results, featuremap_dm, stride, anchor);
	converttoOrgimg(results, param);

	return true;
}

static float sigmod(float x)
{
	return 1.0 / (1 + exp(-x));
}


static bool cmpScore(Box l1, Box l2)
{
	if (l1.score > l2.score)
	{
		return true;
	}
	else
	{
		return false;
	}
}


static float get_iou_value(Box b1, Box b2)
{
	int b1x1 = std::max((int)(b1.x - b1.w / 2), 0);
	int b1x2 = std::min((int)(b1.x + b1.w / 2), IMG_WIDTH);
	int b1y1 = std::max((int)(b1.y - b1.h / 2), 0);
	int b1y2 = std::min((int)(b1.y + b1.h / 2), IMG_HEIGHT);

	int b2x1 = std::max((int)(b2.x - b2.w / 2), 0);
	int b2x2 = std::min((int)(b2.x + b2.w / 2), IMG_WIDTH);
	int b2y1 = std::max((int)(b2.y - b2.h / 2), 0);
	int b2y2 = std::min((int)(b2.y + b2.h / 2), IMG_HEIGHT);

	int xx1 = std::max(b1x1, b2x1);
	int xx2 = std::min(b1x2, b2x2);
	int yy1 = std::max(b1y1, b2y1);
	int yy2 = std::min(b1y2, b2y2);

	int insection_width, insection_height;
	insection_width = std::max(0, xx2 - xx1 + 1);
	insection_height = std::max(0, yy2 - yy1 + 1);

	float insection_area, union_area, iou;
	insection_area = float(insection_width) * insection_height;
	union_area = float(b1.w*b1.h + b2.w*b2.h - insection_area);
	iou = insection_area / union_area;

	return iou;
}


void nms(std::vector<Box>& bboxes, float iou_threshold)
{
	if (bboxes.size() < 2)
	{
		return;
	}
	sort(bboxes.begin(), bboxes.end(), cmpScore);
	int updated_size = bboxes.size();

	for (int i = 0; i < updated_size - 1; i++)
	{
		for (int j = i + 1; j < updated_size; j++)
		{
			if (bboxes[i].cls != bboxes[j].cls)
			{
				continue;
			}
			float iou = get_iou_value(bboxes[i], bboxes[j]);
			if (iou > iou_threshold)
			{
				bboxes.erase(bboxes.begin() + j);
				updated_size = bboxes.size();
				j--;
			}
		}
	}

}