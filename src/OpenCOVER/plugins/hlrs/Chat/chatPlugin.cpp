/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef WIN32
#include <SDKDDKVer.h>
#include <winsock2.h>
#include <windows.h>
#include <direct.h>
#include <conio.h>
#include <mmsystem.h>
#endif
#include <stdio.h>
#include "cover/OpenCOVER.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/coVRTui.h>
#include <config/CoviseConfig.h>
#include "ChatPlugin.h"
#include <net/udpMessage.h>
#include <net/udp_message_types.h>
extern "C"
{
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
}




#ifndef WIN32
 // for chdir
#include <unistd.h>
#endif

#ifdef _WINDOWS
#include <direct.h>
#include <mmeapi.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

ChatPlugin *ChatPlugin::plugin = NULL;

//------------------------------------------------------------------------------
void ChatPlugin::key(int type, int keySym, int mod)
{
	if (type == osgGA::GUIEventAdapter::KEYDOWN)
	{
		//fprintf(stdout,"--- coVRKey called (KeyPress, keySym=%d, mod=%d)\n",
		//	keySym,mod);
		return;
		//}else{
		//fprintf(stdout,"--- coVRKey called (KeyRelease, keySym=%d)\n",keySym);
	}

	switch (keySym)
	{

	case ('r'): /* r: reset animation */
		break;

	}
}

//-----------------------------------------------------------------------------
ChatPlugin::ChatPlugin()
	: ui::Owner("ChatPlugin", cover->ui)
{
	CHATTab = NULL;

}
ChatPlugin * ChatPlugin::instance()
{
	return plugin;
}

bool ChatPlugin::init()
{
	QAudioFormat format;
	format.setSampleRate(128000);
	format.setChannelCount(1);
	format.setSampleSize(16);
	format.setCodec("audio/pcm");
	format.setByteOrder(QAudioFormat::LittleEndian);
	format.setSampleType(QAudioFormat::UnSignedInt);

	//If format isn't supported find the nearest supported
	QAudioDeviceInfo info(QAudioDeviceInfo::defaultInputDevice());
	if (!info.isFormatSupported(format))
		format = info.nearestFormat(format);
	input = nullptr;
	output = nullptr;
	inStream = nullptr;
	outStream = nullptr;

	if (coVRMSController::instance()->isMaster())
	{
		int CHATPort = coCoviseConfig::getInt("InPort", "COVER.Plugin.CHAT", 0);
		int CHATPortOut = coCoviseConfig::getInt("OutPort", "COVER.Plugin.CHAT", 1);
		input = new QAudioInput(format);
		output = new QAudioOutput(format);
		inStream = input->start();
		outStream = output->start();
	}
	CHATtab_create();
	initialize_encoding_audio();
	return true;
}

bool ChatPlugin::destroy()
{
	return true;
}

//------------------------------------------------------------------------------
ChatPlugin::~ChatPlugin()
{

}

bool ChatPlugin::update()
{
	if(inStream)
	{
		int bufferSize = 64 * 1024;
		QByteArray ba;
		vrb::UdpMessage audioMessage;
		audioMessage.type = vrb::AUDIO_STREAM;
		//while (inStream->bytesAvailable() > bufferSize)
		do{
			ba = inStream->read(bufferSize);
			audioMessage.length = ba.length();
			audioMessage.data = (char*)ba.constData();
			cover->sendVrbUdpMessage(&audioMessage);
		} while (ba.length() > 0);
		audioMessage.data = nullptr;
	}
	return true;
}

//------------------------------------------------------------------------------
void ChatPlugin::preFrame()
{


}
//------------------------------------------------------------------------------
void ChatPlugin::postFrame()
{
	// we do not need to care about animation (auto or step) here,
	// because it's in the main program
}
void ChatPlugin::UDPmessage(int type, int length, const void* data)
{
	if (type == vrb::AUDIO_STREAM)
	{
		if(outStream)
		{
			outStream->write((const char*)data, length);
		}
	}
}
//--------------------------------------------------------------------
void ChatPlugin::CHATtab_create(void)
{

	CHATTab = new ui::Menu("CHAT", this);
	reset = new ui::Button(CHATTab, "Reset");
	reset->setText("Reset");
	reset->setCallback([this](bool) {
		
	});
	recVolSlider = new ui::Slider(CHATTab, "RecVolume");
	recVolSlider->setText("RecVolume");
	recVolSlider->setBounds(1, 100);
	recVolSlider->setValue(40.0);
	recVolSlider->setCallback([this](float value, bool) {
		//frequencySurface->radius1 = value;
		//amplitudeSurface->radius1 = value;
	});
	outVolSlider = new ui::Slider(CHATTab, "OutVolume");
	outVolSlider->setText("OutVolume");
	outVolSlider->setBounds(1, 100);
	outVolSlider->setValue(20.0);
	outVolSlider->setCallback([this](float value, bool) {
		//frequencySurface->radius2 = value;
		//amplitudeSurface->radius2 = value;
	});

	infoLabel = new ui::Label(CHATTab, "CHAT Version 1.0");


}


//--------------------------------------------------------------------
void ChatPlugin::CHATtab_delete(void)
{
	if (CHATTab)
	{
		delete infoLabel;
		delete recVolSlider;
		delete outVolSlider;

		delete CHATTab;
	}
}

int ChatPlugin::select_sample_rate(AVCodec* codec)
{
	const int* p;
	int best_samplerate = 0;

	if (!codec->supported_samplerates)
		return 44100;

	p = codec->supported_samplerates;
	while (*p) {
		best_samplerate = FFMAX(*p, best_samplerate);
		p++;

	}
	return best_samplerate;
}
bool ChatPlugin::initialize_encoding_audio()
{
	int ret;
	AVCodecID aud_codec_id = AV_CODEC_ID_OPUS;
	AVSampleFormat sample_fmt = AV_SAMPLE_FMT_S16;

	avcodec_register_all();
	av_register_all();

	aud_codec = avcodec_find_encoder(aud_codec_id);
	avcodec_register(aud_codec);

	if (!aud_codec)
	{
		fprintf(stderr, "could not find OPUS codec\n");
		return false;
	}

	aud_codec_context = avcodec_alloc_context3(aud_codec);
	if (!aud_codec_context)
	{
		fprintf(stderr, "context creation failed\n");
		return false;
	}

	aud_codec_context->bit_rate = 192000;
	aud_codec_context->sample_rate = select_sample_rate(aud_codec);
	aud_codec_context->sample_fmt = sample_fmt;
	aud_codec_context->channel_layout = AV_CH_LAYOUT_MONO;
	aud_codec_context->channels = av_get_channel_layout_nb_channels(aud_codec_context->channel_layout);

	aud_codec_context->codec = aud_codec;
	aud_codec_context->codec_id = aud_codec_id;

	ret = avcodec_open2(aud_codec_context, aud_codec, NULL);

	if (ret < 0)
	{
		fprintf(stderr, "could not open OPUS codec\n");
		return false;
	}
	/*
	outctx = avformat_alloc_context();
	ret = avformat_alloc_output_context2(&outctx, NULL, "mp4", filename);

	outctx->audio_codec = aud_codec;
	outctx->audio_codec_id = aud_codec_id;

	audio_st = avformat_new_stream(outctx, aud_codec);

	audio_st->codecpar->bit_rate = aud_codec_context->bit_rate;
	audio_st->codecpar->sample_rate = aud_codec_context->sample_rate;
	audio_st->codecpar->channels = aud_codec_context->channels;
	audio_st->codecpar->channel_layout = aud_codec_context->channel_layout;
	audio_st->codecpar->codec_id = aud_codec_id;
	audio_st->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
	audio_st->codecpar->format = sample_fmt;
	audio_st->codecpar->frame_size = aud_codec_context->frame_size;
	audio_st->codecpar->block_align = aud_codec_context->block_align;
	audio_st->codecpar->initial_padding = aud_codec_context->initial_padding;

	outctx->streams = new AVStream * [1];
	outctx->streams[0] = audio_st;

	av_dump_format(outctx, 0, filename, 1);

	if (!(outctx->oformat->flags & AVFMT_NOFILE))
	{
		if (avio_open(&outctx->pb, filename, AVIO_FLAG_WRITE) < 0)
			return COULD_NOT_OPEN_FILE;
	}

	ret = avformat_write_header(outctx, NULL);
	*/
	aud_frame = av_frame_alloc();
	aud_frame->nb_samples = aud_codec_context->frame_size;
	aud_frame->format = aud_codec_context->sample_fmt;
	aud_frame->channel_layout = aud_codec_context->channel_layout;

	int buffer_size = av_samples_get_buffer_size(NULL, aud_codec_context->channels, aud_codec_context->frame_size,
		aud_codec_context->sample_fmt, 0);

	av_frame_get_buffer(aud_frame, buffer_size / aud_codec_context->channels);

	if (!aud_frame)
	{
		fprintf(stderr, "frame allocation failed\n");
		return false;
	}

	aud_frame_counter = 0;

	return true;
}
int ChatPlugin::encode_audio_samples(uint8_t** aud_samples)
{
	int ret;
	/*
	int buffer_size = av_samples_get_buffer_size(NULL, aud_codec_context->channels, aud_codec_context->frame_size,
		aud_codec_context->sample_fmt, 0);

	for (size_t i = 0; i < buffer_size / aud_codec_context->channels; i++)
	{
		aud_frame->data[0][i] = aud_samples[0][i];
		aud_frame->data[1][i] = aud_samples[1][i];
	}

	aud_frame->pts = aud_frame_counter++;

	ret = avcodec_send_frame(aud_codec_context, aud_frame);
	if (ret < 0)
		return ERROR_ENCODING_SAMPLES_SEND;

	AVPacket pkt;
	av_init_packet(&pkt);
	pkt.data = NULL;
	pkt.size = 0;

	fflush(stdout);

	while (true)
	{
		ret = avcodec_receive_packet(aud_codec_context, &pkt);
		if (!ret)
		{
			av_packet_rescale_ts(&pkt, aud_codec_context->time_base, audio_st->time_base);

			pkt.stream_index = audio_st->index;
			av_write_frame(outctx, &pkt);
			av_packet_unref(&pkt);
		}
		if (ret == AVERROR(EAGAIN))
			break;
		else if (ret < 0)
			return ERROR_ENCODING_SAMPLES_RECEIVE;
		else
			break;
	}
	*/
	return 0;
}
int ChatPlugin::finish_audio_encoding()
{
	AVPacket pkt;
	av_init_packet(&pkt);
	pkt.data = NULL;
	pkt.size = 0;

	fflush(stdout);
	/*
	int ret = avcodec_send_frame(aud_codec_context, NULL);
	if (ret < 0)
		return ERROR_ENCODING_FRAME_SEND;

	while (true)
	{
		ret = avcodec_receive_packet(aud_codec_context, &pkt);
		if (!ret)
		{
			if (pkt.pts != AV_NOPTS_VALUE)
				pkt.pts = av_rescale_q(pkt.pts, aud_codec_context->time_base, audio_st->time_base);
			if (pkt.dts != AV_NOPTS_VALUE)
				pkt.dts = av_rescale_q(pkt.dts, aud_codec_context->time_base, audio_st->time_base);

			av_write_frame(outctx, &pkt);
			av_packet_unref(&pkt);
		}
		if (ret == -AVERROR(AVERROR_EOF))
			break;
		else if (ret < 0)
			return ERROR_ENCODING_FRAME_RECEIVE;
	}

	av_write_trailer(outctx);
	*/
	return 0;
}

//--------------------------------------------------------------------

COVERPLUGIN(ChatPlugin)
