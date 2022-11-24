#include "deepsort.h"


bool getBox = false;
bool areainit = false;
bool drawing_box = false;
cv::Rect box_init;
std::vector<cv::Point> polygon;
std::vector<cv::Point> polygon_init;


commonparams::OnnxParams initializeYoloParams(const commonparams::Args& args)
{
	commonparams::OnnxParams params;
	params.onnxFileName = "./yolov7_1280.onnx";
	params.inputTensorNames.push_back("images");

	params.outputTensorNames.push_back("output");
	params.outputTensorNames.push_back("516");
	params.outputTensorNames.push_back("530");

	return params;
}

commonparams::OnnxParams initializeSortParams(const commonparams::Args& args)
{
	commonparams::OnnxParams params;
	params.onnxFileName = "./deepsort_dynamic.onnx";
	params.inputTensorNames.push_back("input");
	params.outputTensorNames.push_back("output");
	/*params.fp16 = true;*/
	return params;
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
	std::ifstream file(filename);
	std::vector<std::string> file_lines;
	if (!file.is_open()) return file_lines;
	for (std::string line; getline(file, line);) file_lines.push_back(line);
	std::cout << "object names loaded \n";
	return file_lines;
}

bool Deepsort::init(int argc, char** argv)
{
	obj_names = objects_names_from_file(names_file);
	commonparams::Args args;
	bool argsOK = commonparams::parseArgs(args);
	if (!argsOK)
	{
		sample::gLogError << "Invalid arguments" << std::endl;
		printHelpInfo();
		return false;
	}

	if (args.help)
	{
		printHelpInfo();
		return false;
	}

	sampleTest = &(sample::gLogger.defineTest(gSampleName, argc, argv));

	sample::gLogger.reportTestStart(*sampleTest);
	yoloservice = new SampleOnnxYolo(initializeYoloParams(args));
	sample::gLogInfo << "Building and running a GPU inference engine for Onnx Deepsort" << std::endl;

	if (!yoloservice->build())
	{
		return sample::gLogger.reportFail(*sampleTest);
	}

	sortservice = new SampleOnnxSort(initializeSortParams(args));
	if (!sortservice->build())
	{
		return sample::gLogger.reportFail(*sampleTest);
	}

	this->maxBudget = 100;
	this->maxCosineDist = 0.2;

	objTracker = new tracker(maxCosineDist, maxBudget);
	
	return true;
}

bool Deepsort::extractInput(ImgStruct img_data, Box box, float* modelinput)
{
	const int inputH = SORT_HEIGHT;
	const int inputW = SORT_WIDTH;
	vector<int> dstsize = { inputW,inputH };
	vector<int> srcsize{ box.x2 - box.x1,box.y2 - box.y1 };
	aipp::processparam params = aipp::getparams(srcsize, dstsize);

	int x1 = box.x1;
	int y1 = box.y1;
	unsigned char* cropimg = new unsigned char[srcsize[0] * srcsize[1] * 3];
	int dst_gridlen = 3 * srcsize[0];
	int src_gridlen = INPUTWIDTH * 3;
	for (int i = 0; i < srcsize[1]; i++)
	{
		for (int j = 0; j < srcsize[0]; j++)
		{
			cropimg[i*dst_gridlen + j*3] = img_data.data[(y1 + i) * src_gridlen + (x1 + j)*3];
			cropimg[i*dst_gridlen + j*3 + 1] = img_data.data[(y1 + i)* src_gridlen + (x1 + j)*3 + 1];
			cropimg[i*dst_gridlen + j*3 + 2] = img_data.data[(y1 + i)* src_gridlen + (x1 + j)*3 + 2];
		}
	}

	aipp::imgprocess2(cropimg, modelinput, params);

	delete[] cropimg;

	return true;
}


bool Deepsort::extractInput2(ImgStruct img_data, cv::Rect box, unsigned char* modelinput)
{
	const int inputH = SORT_HEIGHT;
	const int inputW = SORT_WIDTH;
	vector<int> dstsize = { box.width,box.height };
	vector<int> srcsize{ INPUTWIDTH,INPUTHEIGHT };
	aipp::processparam params = aipp::getparams(srcsize, dstsize);

	int x1 = box.x;
	int y1 = box.y;
	int dst_gridlen = 3 * dstsize[0];
	int src_gridlen = INPUTWIDTH * 3;
	for (int i = 0; i < box.height; i++)
	{
		for (int j = 0; j < box.width; j++)
		{
			modelinput[i * dst_gridlen + j * 3] = img_data.data[(y1 + i) * src_gridlen + (x1 + j) * 3];
			modelinput[i * dst_gridlen + j * 3 + 1] = img_data.data[(y1 + i) * src_gridlen + (x1 + j) * 3 + 1];
			modelinput[i * dst_gridlen + j * 3 + 2] = img_data.data[(y1 + i) * src_gridlen + (x1 + j) * 3 + 2];
		}
	}



	return true;
}


void mouseHandler(int event, int x, int y, int flags, void *param) {
	float distance=0;
	switch (event) {
	case cv::EVENT_LBUTTONUP:
		if (polygon.size()>=1)
		{
			distance = sqrtf(powf((x - polygon[0].x), 2) + powf((y - polygon[0].y), 2));
			if (polygon.size() >= 3 && distance < 20)
			{
				areainit = true;
				getBox = true;
				break;
			}
		}
		polygon.push_back(cv::Point(x, y));

		break;

	case cv::EVENT_RBUTTONUP:
		if (polygon.size() <= 2)
		{
			break;
		}
		areainit = true;
		getBox = true;

	}
}

void drawLine(cv::Mat& image, std::vector<cv::Point> points, cv::Scalar color, int thick, bool isClose=true) {
	if (points.size() == 0)
	{
		return;
	}
	if (points.size()==1)
	{
		cv::circle(image, points[0], 2, cv::Scalar(0, 0, 255), -1);
		return;
	}
	if (points.size() == 2)
	{
		cv::line(image, points[0], points[1], cv::Scalar(0, 0, 255), 2);
		return;
	}

	cv::polylines(image, points, isClose, cv::Scalar(0, 0, 255),2, 8, 0);
}

void shifttooriimg(vector<Box>& results, cv::Rect box)
{
	for (auto& result : results)
	{
		result.x1 = result.x1 + box.x;
		result.y1 = result.y1 + box.y;
		result.x2 = result.x2 + box.x;
		result.y2 = result.y2 + box.y;
	}
}

bool Deepsort::videoprocess()
{
	cv::VideoWriter output;
	cv::VideoCapture cap;
	int codec = VideoWriter::fourcc('M', 'P', '4', 'V');
	cap.open("night_stitch_airplane_modify.mp4");
	output.open("night_stitch_out2.mp4", codec, 25, cv::Size(INPUTWIDTH, INPUTHEIGHT));

	if (!cap.isOpened()) {   //检查是否能正常打开视频文件
		std::cout << "fail to open video" << std::endl;
	}
	int gird_len = 3 * INPUTWIDTH;
	cv::Mat frame;
	while (cap.read(frame)) {
		ImgStruct img_data;
		img_data.height = frame.rows;
		img_data.width = frame.cols;
		for (int i=0;i<INPUTHEIGHT;i++)
		{
			for (int j = 0; j < INPUTWIDTH; j++)
			{
				img_data.data[i * gird_len + j * 3 + 2] = frame.at<cv::Vec3b>(i, j)[2];
				img_data.data[i * gird_len + j * 3 + 1] = frame.at<cv::Vec3b>(i, j)[1];
				img_data.data[i * gird_len + j * 3] = frame.at<cv::Vec3b>(i, j)[0];
			}
		}

		vector<Box> results;
		yoloservice->infer(img_data, results);
		if (results.size()==0)
		{
			cv::Mat m1;
			cv::resize(frame, m1, cv::Size(1080, 720));
			cv::imshow("ss", m1);
			output.write(frame);
			cv::waitKey(1);
			continue;
		}

		int nBatch = min(SORTBATCH, int(results.size()));
		float* sortinputs = new float[nBatch * SORT_INPUTSIZE];
		float* sortoutputs = new float[nBatch * SORT_SHAPE];
		string label;
		for (int i = 0; i < nBatch; i++)
		{
			float* sortinput = new float[SORT_INPUTSIZE];
			extractInput(img_data, results[i], sortinput);
			label = obj_names[results[i].cls];
			memcpy(sortinputs + i * SORT_INPUTSIZE, sortinput, SORT_INPUTSIZE * sizeof(float));
			delete[] sortinput;
		}


		sortservice->infer(sortinputs, sortoutputs, nBatch);

		DETECTIONS detections;
		for (int i = 0; i < nBatch; i++)
		{
			DETECTBOX box(results[i].x1, results[i].y1, results[i].x2 - results[i].x1, results[i].y2 - results[i].y1);
			TrackBoxConvert d;
			d.confidence = results[i].score;
			d.tlwh = box;
			if (*max_element(sortoutputs + i * SORT_SHAPE, sortoutputs + i * SORT_SHAPE + SORT_SHAPE) == 0 || isnan(*min_element(sortoutputs + i * SORT_SHAPE, sortoutputs + i * SORT_SHAPE + SORT_SHAPE)))
			{
				continue;
			}
			memcpy(d.feature.data(), sortoutputs + i * SORT_SHAPE, SORT_SHAPE * sizeof(float));
			detections.push_back(d);
		}

		objTracker->predict();
		objTracker->update(detections);


		result.clear();
		for (Track& track : objTracker->tracks) {
			if (!track.is_confirmed() || track.time_since_update > 1)
				continue;
			result.push_back(make_pair(track.track_id, track.to_tlwh()));
		}

		for (int i = 0; i < result.size(); i++)
		{

			float x1 = result[i].second(0, 0);
			float y1 = result[i].second(0, 1);
			float w = result[i].second(0, 2);
			float h = result[i].second(0, 3);
			//	cv::rectangle(frame, cv::Rect(x1, y1,w, h), cv::Scalar(0, 255, 0), 4);
			cv::putText(frame, label, cv::Point(x1, y1 + 40), cv::FONT_HERSHEY_PLAIN, 5, cv::Scalar(0, 255, 0), 2);
			cv::rectangle(frame, cv::Rect(x1, y1, w, h), cv::Scalar(0, 255, 0), 4);

		}

		delete[] sortinputs;
		delete[] sortoutputs;


		cv::Mat m1;
		cv::resize(frame, m1, cv::Size(1080, 720));
		cv::imshow("ss", m1);
		output.write(frame);
		cv::waitKey(1);
	}

	output.release();
	cap.release();
	cv::waitKey(0);
	return true;

}

bool Deepsort::process()
{


	
	while (true)
	{

		if (img_list.empty())
		{
			Sleep(0.1);
			continue;
		}

		ImgStruct img_;
		img_lock.lock();
		int s = img_list.size();
		for (int i = 0; i < s - 1; i++)
		{
			img_list.pop();
		}
		img_ = img_list.front();
		img_list.pop();
		img_lock.unlock();


		cv::Mat m(INPUTHEIGHT, INPUTWIDTH, CV_8UC3);
		int gird_len = 3 * INPUTWIDTH;
		for (int i = 0; i < INPUTHEIGHT; i++)
		{
			for (int j = 0; j < INPUTWIDTH; j++)
			{
				m.at<cv::Vec3b>(i, j)[0] = int(img_.data[i * gird_len + j * 3]);
				m.at<cv::Vec3b>(i, j)[1] = int(img_.data[i * gird_len + j * 3 + 1]);
				m.at<cv::Vec3b>(i, j)[2] = int(img_.data[i * gird_len + j * 3 + 2]);
			}
		}

		vector<Box> results;
		time_t start, end;
		start = clock();
		yoloservice->infer(img_, results);
		end = clock();
		cout << end - start << endl;


		std::map<int, int> cls_result;
		int total_num = 0;
		for (int i = 0; i < results.size(); i++)
		{

			float x1 = results[i].x1;
			float y1 = results[i].y1;
			float w = results[i].x2- results[i].x1;
			float h = results[i].y2 - results[i].y1;
			int cls = results[i].cls;
			if (cls==9)
			{
				continue;
			}

			cv::rectangle(m,cv::Rect(x1,y1,w,h),(0,0,255),4);

		}

		cv::imshow("ss",m);
		cv::waitKey(1);



		}
		return true;
}

bool Deepsort::process2()
{
	cv::namedWindow("ATTACK", cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback("ATTACK", mouseHandler, NULL);
	while (true)
	{

		if (img_list.empty())
		{
			Sleep(0.1);
			continue;
		}

		ImgStruct img_;
		img_lock.lock();
		int s = img_list.size();
		for (int i = 0; i < s - 1; i++)
		{
			img_list.pop();
		}
		img_ = img_list.front();
		img_list.pop();
		img_lock.unlock();


		cv::Mat m(INPUTHEIGHT, INPUTWIDTH, CV_8UC3);
		int gird_len = 3 * INPUTWIDTH;
		for (int i = 0; i < INPUTHEIGHT; i++)
		{
			for (int j = 0; j < INPUTWIDTH; j++)
			{
				m.at<cv::Vec3b>(i, j)[0] = int(img_.data[i * gird_len + j * 3]);
				m.at<cv::Vec3b>(i, j)[1] = int(img_.data[i * gird_len + j * 3 + 1]);
				m.at<cv::Vec3b>(i, j)[2] = int(img_.data[i * gird_len + j * 3 + 2]);
			}
		}


		if (!getBox)
		{
			drawLine(m, polygon, (0, 0, 255), 2, false);
			cv::imshow("ATTACK", m);
			cv::waitKey(1);
			cv::setMouseCallback("ATTACK", mouseHandler, NULL);
			continue;
		}



		cv::setMouseCallback("ATTACK", NULL, NULL);
		drawLine(m, polygon, (0, 0, 255), 2, true);

		cv::Rect ori = cv::boundingRect(polygon);

		vector<Box> results;
		unsigned char* yoloinput = new unsigned char[ori.width * ori.height * 3];
		extractInput2(img_, ori, yoloinput);


		yoloservice->infer(yoloinput, results, ori);

		shifttooriimg(results, ori);

		if (results.size() == 0)
		{

			cv::imshow("ATTACK", m);
			if (cv::waitKey(1) == 'r')
			{
				polygon.clear();
				cv::setMouseCallback("ATTACK", mouseHandler, NULL);
				getBox = false;
				areainit = false;
			}
			continue;
		}


		std::map<int, int> cls_result;
		int total_num = 0;
		for (int i = 0; i < results.size(); i++)
		{

			float x1 = results[i].x1;
			float y1 = results[i].y1;
			float w = results[i].x2 - results[i].x1;
			float h = results[i].y2 - results[i].y1;
			int cls = results[i].cls;
			if (cls == 9)
			{
				continue;
			}
			vector<cv::Point2f> rect_point;
			rect_point.push_back(cv::Point2f(x1, y1));
			rect_point.push_back(cv::Point2f(x1, y1 + h));
			rect_point.push_back(cv::Point2f(x1 + w, y1 + h));
			rect_point.push_back(cv::Point2f(x1 + w, y1));
			for (auto point : rect_point)
			{
				int inner = cv::pointPolygonTest(polygon, point, false);
				if (inner >= 0)
				{
					total_num++;
					//string label = "分类:" + to_string(cls);
					cv::rectangle(m, cv::Rect(x1, y1, w, h), cv::Scalar(0, 0, 255), 1);
					string label = obj_names[cls];
					cv::putText(m, label, cv::Point(x1, y1), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2);

					//putTextZH(m, label.c_str(), cv::Point(x1, y1), cv::Scalar(0, 0, 255), 20);
					auto it = cls_result.find(cls);
					if (it == cls_result.end())
					{
						cls_result.insert(pair<int, int>(cls, 1));
					}
					else
					{
						cls_result[cls]++;
					}
					break;

				}

			}

		}


		string label_totalnums = "Total_nums:" + to_string(total_num);
		cv::putText(m, label_totalnums, cv::Point(30, 30), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 2);
		int cr_no = 0;
		for (auto cr : cls_result)
		{
			int y = 30 * (cr_no + 2);
			string label = obj_names[cr.first] + ":" + to_string(cr.second);
			cv::putText(m, label, cv::Point(30, y), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2);
			cr_no++;
		}


		cv::imshow("ATTACK", m);
		if (cv::waitKey(1) == 'r')
		{
			polygon.clear();
			cv::setMouseCallback("ATTACK", mouseHandler, NULL);
			getBox = false;
			areainit = false;
		}

		//保存图片
		time_t t;
		struct tm* tmp;
		char buf[20];
		time(&t);
		tmp = localtime(&t);
		strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", tmp);

		string savetime(buf);
		string savepath = "D:/Program_c/result/" + savetime + ".jpg";
		cv::imwrite(savepath, m);



	}
	return true;
}
