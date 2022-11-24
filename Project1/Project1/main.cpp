#include "runwayattack.h"
#include "util.h"


std::mutex img_lock;
std::queue<ImgStruct> img_list;

int main(int argc, char** argv)
{
	urlParams urlp;
	RunwayAttack client;
	client.init(urlp,argc,argv);
	client.process();
}