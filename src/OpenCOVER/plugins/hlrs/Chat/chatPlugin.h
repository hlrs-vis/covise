/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CHAT_PLUGIN_H
#define _CHAT_PLUGIN_H

#include <vrbclient/VRBClient.h>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coPopupHandle.h>
#include <cover/coVRPlugin.h>
#include <vrml97/vrml/Player.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Label.h>
#include <cover/ui/Slider.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Owner.h>
#include <QIODevice>
#include <QtMultimedia/QAudioOutput>
#include <QtMultimedia/QAudioInput>
#include <QtMultimedia/QAudioFormat>

extern "C" {
#ifdef HAVE_FFMPEG_SEPARATE_INCLUDES
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#define HAVE_SWSCALE_H
#else
#include <ffmpeg/avcodec.h>
#include <ffmpeg/avformat.h>
#include <ffmpeg/avutil.h>
#define AV_VERSION_INT(a, b, c) (a << 16 | b << 8 | c)
#ifdef LIBAVCODEC_VERSION_INT
#if (LIBAVCODEC_VERSION_INT > AV_VERSION_INT(51, 9, 0))
#include <ffmpeg/swscale.h>
#define HAVE_SWSCALE_H
#endif
#endif
#endif
};


#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(56, 34, 1)
typedef PixelFormat AVPixelFormat;

#ifndef AV_PIX_FMT_RGB32
#define AV_PIX_FMT_RGB32 PIX_FMT_RGB32
#endif
#define AV_PIX_FMT_RGB24 PIX_FMT_RGB24
#define AV_PIX_FMT_YUV420P PIX_FMT_YUV420P
#define AV_PIX_FMT_YUVJ420P PIX_FMT_YUVJ420P

#define AV_CODEC_ID_NONE CODEC_ID_NONE
#define AV_CODEC_ID_RAWVIDEO CODEC_ID_RAWVIDEO
#define AV_CODEC_ID_FFV1 CODEC_ID_FFV1
#define AV_CODEC_ID_JPEGLS CODEC_ID_JPEGLS
#define AV_CODEC_ID_MJPEG CODEC_ID_MJPEG
#define AV_CODEC_ID_MPEG2VIDEO CODEC_ID_MPEG2VIDEO
#define AV_CODEC_ID_MPEG4 CODEC_ID_MPEG4
#define AV_CODEC_ID_FLV1 CODEC_ID_FLV1
#define AV_CODEC_ID_DVVIDEO CODEC_ID_DVVIDEO

#define AV_CODEC_FLAG_QSCALE CODEC_FLAG_QSCALE
#define AV_CODEC_FLAG_LOW_DELAY CODEC_FLAG_LOW_DELAY

#define av_frame_alloc avcodec_alloc_frame
#endif

#ifndef AV_CODEC_FLAG_GLOBAL_HEADER
#define AV_CODEC_FLAG_GLOBAL_HEADER CODEC_FLAG_GLOBAL_HEADER
#endif


namespace covise
{
    class coSubMenuItem;
    class coRowMenu;
    class coCheckboxMenuItem;
    class coCheckboxGroup;
    class coButtonMenuItem;
    class coSliderMenuItem;
}


using namespace opencover;
using namespace covise;

class ChatPlugin : public coVRPlugin, public ui::Owner
{
private:


public:
    static ChatPlugin *instance();

    // constructor
    ChatPlugin();

    // destructor
    virtual ~ChatPlugin();
    bool init();
    bool destroy();

    // loop
    bool update();
    void preFrame();
    void postFrame();
	virtual void UDPmessage(int type, int length, const void* data);

    void key(int type, int keySym, int mod);

    void CHATtab_create();
    void CHATtab_delete();

	int select_sample_rate(AVCodec* codec);

	bool initialize_encoding_audio();

	int encode_audio_samples(uint8_t** aud_samples);

	int finish_audio_encoding();

    ui::Menu *CHATTab = nullptr;
    ui::Button *reset = nullptr;
    ui::Slider *recVolSlider = nullptr;
    ui::Slider *outVolSlider = nullptr;
    ui::Label *infoLabel = nullptr;
	QIODevice* inStream;
	QIODevice* outStream;
private:
	QAudioInput* input;
	QAudioOutput* output;
    static ChatPlugin *plugin;
	AVCodec* aud_codec;
	AVCodecContext* aud_codec_context;
	AVFrame* aud_frame;
	int aud_frame_counter;

};
#endif
