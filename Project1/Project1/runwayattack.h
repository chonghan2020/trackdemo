#pragma once

#include "VideoProcess.h"
#include "util.h"
#include "deepsort.h"


extern std::queue<ImgStruct> img_list;


class RunwayAttack
{
public:
	RunwayAttack() {};
	bool init(urlParams, int, char**);
	bool process();
	~RunwayAttack()
	{
		delete vpservice;
		delete sortservice;
	}	
private:
	VideoProcess* vpservice;
	Deepsort* sortservice;

};
