#include "aippcommon.h"

aipp::processparam aipp::getparams(std::vector<int> srcSize, std::vector<int> dstSize)
{
	int res_width;
	int res_height;
	int src_width = srcSize[0];
	int src_height = srcSize[1];
	int dst_width = dstSize[0];
	int dst_height = dstSize[1];
	float src_ratio = (float)src_width / src_height;
	float dst_ratio = (float)dst_width / dst_height;

	if (src_ratio > dst_ratio)
	{
		res_width = dst_width;
		res_height = int(res_width / src_ratio);
	}
	else
	{
		res_height = dst_height;
		res_width = int(res_height * src_ratio);
	}
	float scale_x = (float)res_width / src_width;
	float scale_y = (float)res_height / src_height;
	resizeparam res_para(res_width, res_height, scale_x, scale_y, src_width, src_height);
	int pad_l = (dst_width - res_width) / 2;
	int pad_r = std::max(dst_width - res_width - pad_l, 0);
	int pad_t = (dst_height - res_height) / 2;
	int pad_b = std::max(dst_height - res_height - pad_t, 0);
	padparam pad_para(pad_l, pad_r, pad_t, pad_b);
	modelinputparams in_para(dst_width, dst_height);
	processparam imgpara(res_para, pad_para, in_para);

	return imgpara;
}

void aipp::imgprocess(unsigned char* src, float* dst, processparam paras)
{
	float scale_x = paras.res_para.scale_x;
	float scale_y = paras.res_para.scale_y;
	int dst_width = paras.res_para.width;
	int dst_height = paras.res_para.height;
	int pad_l = paras.pad_para.padleft;
	int pad_t = paras.pad_para.padtop;
	int pad_r = paras.input_para.width - paras.pad_para.padright;
	int pad_b = paras.input_para.height - paras.pad_para.padbottom;
	int src_width = paras.res_para.src_width;
	int src_height = paras.res_para.src_height;

	int grid_len_src = src_width * 3;
	int modelin_width = paras.input_para.width;
	int modelin_height = paras.input_para.height;
	int grid_len_dst = modelin_width * modelin_height;

	for (int i = 0; i < modelin_height; i++)
	{
		for (int j = 0; j < modelin_width; j++)
		{
			if (i < pad_t || i >= pad_b || j < pad_l || j >= pad_r)
			{
				dst[i*modelin_width + j] = 0;
				dst[i*modelin_width + j + grid_len_dst] = 0;
				dst[i*modelin_width + j + 2 * grid_len_dst] = 0;
			}
			else
			{
				int src_i = std::max(0, std::min((int)floor((i - pad_t) / scale_x), src_height - 1));
				int src_j = std::max(0, std::min((int)floor((j - pad_l) / scale_y), src_width - 1));
				dst[i* modelin_width + j] = (float)src[src_i * grid_len_src + src_j * 3+2] / 255 ;
				dst[i* modelin_width + j + grid_len_dst] = (float)src[src_i * grid_len_src + src_j * 3 + 1] / 255 ;
				dst[i* modelin_width + j + 2 * grid_len_dst] = (float)src[src_i * grid_len_src + src_j * 3] / 255;
			}

		}
	}
}

void aipp::imgprocess2(unsigned char* src, float* dst, processparam paras)
{
	float scale_x = paras.res_para.scale_x;
	float scale_y = paras.res_para.scale_y;
	int dst_width = paras.res_para.width;
	int dst_height = paras.res_para.height;
	int pad_l = paras.pad_para.padleft;
	int pad_t = paras.pad_para.padtop;
	int pad_r = paras.input_para.width - paras.pad_para.padright;
	int pad_b = paras.input_para.height - paras.pad_para.padbottom;
	int src_width = paras.res_para.src_width;
	int src_height = paras.res_para.src_height;

	int grid_len_src = src_width * 3;
	int modelin_width = paras.input_para.width;
	int modelin_height = paras.input_para.height;
	int grid_len_dst = modelin_width * modelin_height;
	for (int i = 0; i < modelin_height; i++)
	{
		for (int j = 0; j < modelin_width; j++)
		{
			if (i < pad_t || i >= pad_b || j < pad_l || j >= pad_r)
			{
				dst[i*modelin_width + j] = 0;
				dst[i*modelin_width + j + grid_len_dst] = 0;
				dst[i*modelin_width + j + 2 * grid_len_dst] = 0;
			}
			else
			{
				int src_i = std::max(0, std::min((int)floor((i - pad_t) / scale_x), src_height - 1));
				int src_j = std::max(0, std::min((int)floor((j - pad_l) / scale_y), src_width - 1));
				dst[i* modelin_width + j] = ((float)src[src_i * grid_len_src + src_j * 3 + 2] / 255 - 0.485) / 0.229;;
				dst[i* modelin_width + j + grid_len_dst] = ((float)src[src_i * grid_len_src + src_j * 3 + 1] / 255 - 0.456) / 0.224;
				dst[i* modelin_width + j + 2 * grid_len_dst] = ((float)src[src_i * grid_len_src + src_j * 3] / 255 - 0.406) / 0.225;
			}

		}
	}

}