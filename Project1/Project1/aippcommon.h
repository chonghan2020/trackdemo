#ifndef AIPP
#define AIPP
#include <vector>
#include <iostream>
#include <algorithm>

namespace aipp
{
	struct padparam
	{
		padparam() {};
		padparam(int l, int r, int t, int b) :padleft(l), padright(r), padtop(t), padbottom(b) {};
		int padleft;
		int padright;
		int padtop;
		int padbottom;
	};
	struct resizeparam
	{
		resizeparam() {};
		resizeparam(int w, int h, float x, float y, int sw, int sh) :width(w), height(h), scale_x(x), scale_y(y), src_width(sw), src_height(sh) {};
		int width;
		int height;
		float scale_x;
		float scale_y;
		int src_width;
		int src_height;
	};

	struct modelinputparams
	{
		modelinputparams() {};
		modelinputparams(int w, int h) :width(w), height(h) {};
		int width;
		int height;
	};
	struct processparam
	{
		processparam() {};
		processparam(resizeparam rp, padparam pp, modelinputparams mm) :res_para(rp), pad_para(pp), input_para(mm) {};
		resizeparam res_para;
		padparam pad_para;
		modelinputparams input_para;
	};

	processparam getparams(std::vector<int>, std::vector<int>);
	void imgprocess(unsigned char*, float*, processparam);
	void imgprocess2(unsigned char*, float*, processparam);
}

#endif