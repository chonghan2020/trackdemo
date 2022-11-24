#include "runwayattack.h"



bool RunwayAttack::init(urlParams urlp,int argc,char** argv)
{
	vpservice = new VideoProcess(urlp);
	bool ret = vpservice->init();
	if (!ret)
	{
		printf("init video process failed");
	}
	sortservice = new Deepsort();
	ret = sortservice->init(argc, argv);
	if (!ret)
	{
		printf("init deepsort process failed");
	}

	return true;
}


bool RunwayAttack::process()
{
	std::thread videothread(&VideoProcess::process,vpservice,ref(img_list));
	videothread.detach();
	std::thread modelinferthread(&Deepsort::process2, sortservice);
	modelinferthread.detach();
	while (true);
	return true;
}