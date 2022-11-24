#pragma once

#include<math.h>
#include<queue>

#define NOMINMAX

#define RECV_DEVICE_PORT 5012
#define RECV_PORT 6013
#define MQTT_PORT 1883
#define KEEPALIVE 60
#define MQTT_IP "192.168.20.217"


#ifndef OPENCV
#define OPENCV
#include<opencv2/opencv.hpp>
#pragma comment(lib, "opencv_world460.lib")//“˝»Î¡¥Ω”ø‚
#endif


#include "WinSock2.h"
#include "util.h"

#include <Windows.h>
#include <process.h>
#include <queue>


extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}

#define av_err2str(errnum) av_make_error_string(av_error, AV_ERROR_MAX_STRING_SIZE, errnum)

extern std::mutex img_lock;


class VideoProcess
{
public:
	VideoProcess(urlParams urlp) :url(urlp.url){};
	bool init();
	bool init_ffmpeg();
	bool process(std::queue<ImgStruct>&);

private:
	//SOCKET sockSend;
	//SOCKET sockSendControl;
	//SOCKADDR_IN addrClient;
	//SOCKADDR_IN addrControl;
	int lenClient;
	int lenControl;
	AVFormatContext *av_format_ctx = NULL;
	AVCodecParameters* av_codec_params = NULL;
	const AVCodec* av_codec = NULL;
	AVCodecContext* av_codec_ctx = NULL;
	AVFrame* av_frame = NULL;
	AVFrame* av_frame_RGB = NULL;
	AVPacket* av_packet = NULL;
	uint8_t *out_buffer;
	int numBytes;
	struct SwsContext *img_convert_ctx;
	char av_error[AV_ERROR_MAX_STRING_SIZE] = { 0 };
	int video_stream_index = -1;
	std::string url;
};