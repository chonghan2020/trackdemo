#include "Videoprocess.h"

bool VideoProcess::init()
{
	return true;
}

bool VideoProcess::init_ffmpeg()
{

	int ret = 0;
	avformat_free_context(av_format_ctx);
	av_format_ctx = NULL;
	avformat_network_init(); //
	av_format_ctx = avformat_alloc_context();

	AVDictionary* avdic = nullptr;
	char option_key[] = "rtsp_transport";
	char option_value[] = "tcp";
	av_dict_set(&avdic, option_key, option_value, 0);

	char option_key2[] = "max_delay";
	char option_value2[] = "1000000";

	av_dict_set(&avdic, option_key2, option_value2, 0);

	av_dict_set(&avdic, "timeout", "10000", 0);

	do
	{
		//char r[100] = "L:\\wxp\\ÏîÄ¿ÊÓÆµ\\1.avi";

		ret = avformat_open_input(&av_format_ctx, url.c_str(), nullptr, nullptr);

		//if (ret < 0)
		//{
		//	std::cout << "can't open the file." << std::endl;
		//	return;
		//}

	} while (ret < 0);

	if (avformat_find_stream_info(av_format_ctx, nullptr) < 0)
	{
		std::cout << "Could't find stream infomation." << std::endl;

		return false;
	}

	video_stream_index = -1;

	video_stream_index = av_find_best_stream(av_format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);//

																						  //
	if (video_stream_index == -1)
	{
		printf("Didn't find a video stream.\n");
		return false;
	}

	//
	av_codec_ctx = avcodec_alloc_context3(nullptr);

	avcodec_parameters_to_context(av_codec_ctx,
		av_format_ctx->streams[video_stream_index]->codecpar
	);//

	av_codec = const_cast<AVCodec*>(avcodec_find_decoder(av_codec_ctx->codec_id));

	if (av_codec == nullptr)
	{
		printf("Codec not found.\n");
		return false;
	}

	av_codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

	//
	if (avcodec_open2(av_codec_ctx, av_codec, nullptr) < 0)
	{
		printf("Could not open codec.\n");
		return false;
	}


	av_frame = av_frame_alloc();

	av_frame_RGB = av_frame_alloc();


	img_convert_ctx =
		sws_getContext(av_codec_ctx->width, av_codec_ctx->height,
			av_codec_ctx->pix_fmt, av_codec_ctx->width, av_codec_ctx->height,
			AV_PIX_FMT_BGR24, SWS_BICUBIC, nullptr, nullptr, nullptr);

	numBytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, av_codec_ctx->width, av_codec_ctx->height, 1);

	out_buffer = (uint8_t*)av_malloc(numBytes * sizeof(uint8_t));

	av_image_fill_arrays(av_frame_RGB->data, av_frame_RGB->linesize, out_buffer,
		AV_PIX_FMT_BGR24, av_codec_ctx->width, av_codec_ctx->height, 1);

	av_packet = (AVPacket*)malloc(sizeof(AVPacket)); //


	av_init_packet(av_packet); //
	return true;
}


bool VideoProcess::process(std::queue<ImgStruct>& img_list)
{

	int ret = 0;

	avformat_network_init(); //
	av_format_ctx = avformat_alloc_context();

	AVDictionary *avdic = nullptr;
	char option_key[] = "rtsp_transport";
	char option_value[] = "tcp";
	av_dict_set(&avdic, option_key, option_value, 0);

	char option_key2[] = "max_delay";
	char option_value2[] = "1000000";

	av_dict_set(&avdic, option_key2, option_value2, 0);

	av_dict_set(&avdic, "timeout", "10000", 0);

	do
	{

		ret = avformat_open_input(&av_format_ctx, url.c_str(), nullptr, &avdic);

	} while (ret < 0);

	if (avformat_find_stream_info(av_format_ctx, nullptr) < 0)
	{
		std::cout << "Could't find stream infomation." << std::endl;

		//return false;
	}

	video_stream_index = -1;

	video_stream_index = av_find_best_stream(av_format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);//

																						  //
	if (video_stream_index == -1)
	{
		printf("Didn't find a video stream.\n");
		//return false;
	}

	//
	av_codec_ctx = avcodec_alloc_context3(nullptr);

	avcodec_parameters_to_context(av_codec_ctx,
		av_format_ctx->streams[video_stream_index]->codecpar
	);//

	av_codec = const_cast<AVCodec*>(avcodec_find_decoder(av_codec_ctx->codec_id));

	if (av_codec == nullptr)
	{
		printf("Codec not found.\n");
		//return false;
	}

	av_codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

	//
	if (avcodec_open2(av_codec_ctx, av_codec, nullptr) < 0)
	{
		printf("Could not open codec.\n");
		//return false;
	}


	av_frame = av_frame_alloc();

	av_frame_RGB = av_frame_alloc();


	img_convert_ctx =
		sws_getContext(av_codec_ctx->width, av_codec_ctx->height,
			av_codec_ctx->pix_fmt, av_codec_ctx->width, av_codec_ctx->height,
			AV_PIX_FMT_BGR24, SWS_BICUBIC, nullptr, nullptr, nullptr);

	numBytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, av_codec_ctx->width, av_codec_ctx->height, 1);

	out_buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));

	av_image_fill_arrays(av_frame_RGB->data, av_frame_RGB->linesize, out_buffer,
		AV_PIX_FMT_BGR24, av_codec_ctx->width, av_codec_ctx->height, 1);

	av_packet = (AVPacket *)malloc(sizeof(AVPacket)); //


	av_init_packet(av_packet); //
	while (true)
	{
		int result = av_read_frame(av_format_ctx, av_packet);
		if (result < 0)
		{

			std::cout <<result<< std::endl;
			bool cont= init_ffmpeg();
			std::cout << cont << std::endl;
			if (cont==false)
			{
				Sleep(1);
			}
			continue; //
		}

		if (av_packet->stream_index == video_stream_index)
		{
			if (avcodec_send_packet(av_codec_ctx, av_packet) != 0)
			{
				std::cout << "input AVPacket to decoder failed!\n";
				continue;
			}

			while (avcodec_receive_frame(av_codec_ctx, av_frame) == 0)
			{

				sws_scale(img_convert_ctx,
					av_frame->data,
					av_frame->linesize, 0, av_codec_ctx->height,
					av_frame_RGB->data,
					av_frame_RGB->linesize);
				
				ImgStruct img_data;
				img_data.height = av_codec_ctx->height;
				img_data.width = av_codec_ctx->width;
				memcpy(img_data.data, out_buffer, numBytes);

				img_lock.lock();
				if (img_list.size() <= 5)
				{
					img_list.push(img_data);
				}
				else
				{
					img_list.pop();
					img_list.push(img_data);
				}
				img_lock.unlock();
				//delete[] data;
				av_packet_unref(av_packet); //
				Sleep(1);
			}
		}
	}
	return true;
}