#ifndef PLUGIN_UTIL_FFMPEG_UTIL_H
#define PLUGIN_UTIL_FFMPEG_UTIL_H

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/dict.h>
};


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