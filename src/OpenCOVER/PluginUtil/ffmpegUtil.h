#ifndef PLUGIN_UTIL_FFMPEG_UTIL_H
#define PLUGIN_UTIL_FFMPEG_UTIL_H

extern "C"
{
#ifdef HAVE_FFMPEG_SEPARATE_INCLUDES
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/dict.h>
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

//#define USE_CODECPAR

#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <util/coExport.h>
namespace opencover
{
    

void PLUGIN_UTILEXPORT avFree(void *ptr);

template <typename T>
struct avDeleter
{
    void operator()(T *t)
    {
        avFree(t);
    }
};

template <typename T>
using AvPtr = std::unique_ptr<T, avDeleter<T>>;

struct avFrameDeleter
{
    void operator()(AVFrame *);
};
using AvFramePtr = std::unique_ptr<AVFrame, avFrameDeleter>;

class PLUGIN_UTILEXPORT AvWriter2
{
public:

    struct Resolution
    {
        Resolution() = default;
        Resolution(size_t w, size_t h) : w(w), h(h) {}
        size_t w = 0, h = 0;
    };

    struct VideoFormat{
        Resolution resolution;
        std::string codecName;
        AVPixelFormat colorFormat;
    };
    AvWriter2(const VideoFormat &input, const VideoFormat &output, const std::string &outPutFile);
    
    // write frame frameNum to the outPutFile
    // use getPixelBuffer to get pixels and fill them
    // if mirror the pixels are mirrored horizontally 
    void writeVideo(size_t frameNum, uint8_t *pixels, bool mirror);
    uint8_t *getPixelBuffer();
    ~AvWriter2();

private:
    Resolution m_inputRes; //resolution of the source picture
    AVFormatContext *m_oc = nullptr; //io context
    AvPtr<uint8_t> m_pixels; //picture filled by user
    AvFramePtr m_inPicture; //user picture converted to FFmpeg in the input video format
    AvFramePtr m_outPicture; //output picture converted to the output video format
    AvPtr<uint8_t> m_mirroredpixels; //intermediate buffer for reordered pixels
    AVCodecContext *m_outCodecContext = nullptr;
    const AVPixelFormat m_capturePixFmt = AV_PIX_FMT_BGR32;
    SwsContext *m_swsconvertctx = nullptr; //context to convert from raw pixels to FFmpeg
    int m_inSize = 0;
    bool m_error = false;
#ifndef _M_CEE //no future in Managed OpenCOVER
    std::unique_ptr<std::future<bool>> encodeFuture;
#endif
    void SwConvertScale(uint8_t *pixels, bool mirror);
};
}

#endif // PLUGIN_UTIL_FFMPEG_UTIL_H