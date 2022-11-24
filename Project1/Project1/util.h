#pragma once
#ifndef RUNWAY_UNIT
#define RUNWAY_UNIT


#include <string>
#include <iostream>
#include <mutex>
#include <queue>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#define INPUTWIDTH 1920
#define INPUTHEIGHT 1080
#define OBJECT_NUM 80
#define PRO_BOX_SIZE 85
#define IMG_WIDTH 640
#define IMG_HEIGHT 640
#define SORT_SHAPE 512
#define SORTBATCH 10
#define SORTTOTAL 50
#define SORT_WIDTH 64
#define SORT_HEIGHT 128
#define SORT_INPUTSIZE (SORT_WIDTH*SORT_HEIGHT*3)
#define T0TAL_STEP 50

void printHelpInfo();

struct HistoryInfo
{
	int cls;
	float w;
	float h;
};



struct ImgStruct
{
	~ImgStruct() {
		delete[] data;
	};

	ImgStruct() {};

	ImgStruct(const ImgStruct& p)
	{
		height = p.height;
		width = p.width;
		memcpy(this->data, p.data, INPUTHEIGHT*INPUTWIDTH * 3);
	}

	ImgStruct& operator=(const ImgStruct& p)
	{
		this->height = p.height;
		this->width = p.width;
		memcpy(this->data, p.data, INPUTHEIGHT*INPUTWIDTH * 3);
		return *this;
	}

	int height;
	int width;
	unsigned char* data=new unsigned char[INPUTHEIGHT*INPUTWIDTH * 3];
};

struct urlParams
{
	std::string url = "output.mp4";
	int img_height;
	int img_width;
};


struct Box
{
	float x;
	float y;
	float w;
	float h;
	int x1;
	int x2;
	int y1;
	int y2;
	float score;
	int cls;
	int trackid;
};


typedef Eigen::Matrix<float, 1, SORT_SHAPE, Eigen::RowMajor> FEATURE;
typedef Eigen::Matrix<float, Eigen::Dynamic, SORT_SHAPE, Eigen::RowMajor> FEATURESS;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;

using RESULT_DATA = std::pair<int, DETECTBOX>;
using MATCH_DATA = std::pair<int, int>;
using TRACKER_DATA = std::pair<int, FEATURESS>;

typedef struct t {
	std::vector<MATCH_DATA> matches;
	std::vector<int> unmatched_tracks;
	std::vector<int> unmatched_detections;
}TRACHER_MATCHD;


struct TrackBox
{
	TrackBox() {};
	TrackBox(Box b) :x(b.x), y(b.y), w(b.w), h(b.h), score(b.score), cls(b.cls) {
	};

	float x;
	float y;
	float w;
	float h;
	float score;
	int cls;
	int trackid;
	float feature[SORT_SHAPE];
};

const float kRatio = 0.5;
enum DETECTBOX_IDX { IDX_X = 0, IDX_Y, IDX_W, IDX_H };

class TrackBoxConvert
{
public:
	DETECTBOX tlwh;
	float confidence;
	FEATURE feature;
	DETECTBOX to_xyah() const {
		//(centerx, centery, ration, h)
		DETECTBOX ret = tlwh;
		ret(0, IDX_X) += (ret(0, IDX_W)*kRatio);
		ret(0, IDX_Y) += (ret(0, IDX_H)*kRatio);
		ret(0, IDX_W) /= ret(0, IDX_H);
		return ret;
	}
	DETECTBOX to_tlbr() const {
		//(x,y,xx,yy)
		DETECTBOX ret = tlwh;
		ret(0, IDX_X) += ret(0, IDX_W);
		ret(0, IDX_Y) += ret(0, IDX_H);
		return ret;
	}
};

typedef std::vector<TrackBoxConvert> DETECTIONS;




#endif

