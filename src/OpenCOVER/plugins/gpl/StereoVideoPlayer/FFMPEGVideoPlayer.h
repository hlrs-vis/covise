#include <osgViewer/GraphicsWindow>
#include <cover/coVRConfig.h>

#ifdef HAVE_SDL
#ifdef WIN32
typedef signed __int8 SDLint8_t;
typedef __int8 FFMPEGint8_t;
//#define int8_t SDLint8_t

extern "C" {
#include <SDL.h>
#include <SDL_audio.h>
#include <SDL_events.h>
#include <SDL_thread.h>
};

//#undef int8_t
//#define int8_t FFMPEGint8_t
#else
extern "C" {
#include <SDL.h>
#include <SDL_audio.h>
#include <SDL_events.h>
#include <SDL_thread.h>
};

#endif
#endif

extern "C" {
#ifdef HAVE_FFMPEG_SEPARATE_INCLUDES
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
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

#define AV_PIX_FMT_RGB24 PIX_FMT_RGB24
#define AV_PIX_FMT_BGR24 PIX_FMT_BGR24
#define AV_PIX_FMT_YUV420P PIX_FMT_YUV420P
#define AV_PIX_FMT_YUVJ420P PIX_FMT_YUVJ420P

#define AV_CODEC_ID_RAWVIDEO CODEC_ID_RAWVIDEO
#define AV_CODEC_ID_FFV1 CODEC_ID_FFV1
#define AV_CODEC_ID_JPEGLS CODEC_ID_JPEGLS
#define AV_CODEC_ID_MJPEG CODEC_ID_MJPEG
#define AV_CODEC_ID_MPEG2VIDEO CODEC_ID_MPEG2VIDEO
#define AV_CODEC_ID_MPEG4 CODEC_ID_MPEG4
#define AV_CODEC_ID_FLV1 CODEC_ID_FLV1
#define AV_CODEC_ID_RAWVIDEO CODEC_ID_RAWVIDEO

#define AV_CODEC_FLAG_QSCALE CODEC_FLAG_QSCALE
#define AV_CODEC_FLAG_LOW_DELAY CODEC_FLAG_LOW_DELAY

#define av_frame_alloc avcodec_alloc_frame
#endif

#ifndef AV_CODEC_CAP_TRUNCATED
#define AV_CODEC_CAP_TRUNCATED CODEC_CAP_TRUNCATED
#define AV_CODEC_FLAG_TRUNCATED CODEC_FLAG_TRUNCATED
#endif

class StereoVideoPlayerPlugin;

#ifdef HAVE_SDL
class PacketQueue
{
public:
    PacketQueue();
    ~PacketQueue();

    std::list<AVPacket> *getPktList()
    {
        return &pktList;
    };
    int getSize()
    {
        return size;
    };
    void setSize(int s)
    {
        size = s;
    };
    int getNrElem()
    {
        return pktList.size();
    };
    void clearPktList();
    SDL_mutex *getMutex()
    {
        return mutex;
    };
    SDL_cond *getCond()
    {
        return cond;
    };

private:
    std::list<AVPacket> pktList;
    int size;
    SDL_mutex *mutex;
    SDL_cond *cond;
};
#endif

class DisplayImage
{
public:
    DisplayImage(uint8_t *, double, unsigned int, unsigned int, unsigned int, unsigned int);
    ~DisplayImage();

    uint8_t *getRGBImage()
    {
        return RGBImage;
    };
    void setDisplayImage(uint8_t *, double);
    double getPts()
    {
        return pts;
    };

private:
    uint8_t *RGBImage;
    double pts;
};

class VideoQueue
{
public:
    VideoQueue()
    {
        ;
    };
    ~VideoQueue();

    void putImage(DisplayImage *);
    DisplayImage *getImage();
    DisplayImage *getLastImage();
    void removeImage();
    void clearImgList();
    unsigned int getSize()
    {
        return imgList.size();
    };

private:
    std::list<DisplayImage *> imgList;
};

class VideoStream
{
public:
    static VideoQueue *vq;

    VideoStream();
    ~VideoStream();

    bool openMovieCodec(const std::string, AVPixelFormat *);
    bool allocateFrame();
    bool allocateRGBFrame(AVPixelFormat);
    void setRGBBuffer(uint8_t *);
    uint8_t *getRGBFrame()
    {
        return dispRGBBuffer;
    };
    int readFrame();
    unsigned int getfps()
    {
        return (unsigned int)(codecCtx->time_base.den / codecCtx->time_base.num);
    };
    int getWidth()
    {
        return usedWidth;
    };
    int getHeight()
    {
        return usedHeight;
    };
    AVCodecContext *getCodecContext()
    {
        return codecCtx;
    };
    AVFormatContext *getFormatContext()
    {
        return oc;
    };
    int getVideoStreamID()
    {
        return videoStreamID;
    };
    double synchronizeVideo(double, bool);
    void setMaxVideoBufferSize(unsigned int size)
    {
        maxVideoBufferElements = size;
    };
    unsigned int getMaxVideoBufferSize()
    {
        return maxVideoBufferElements;
    };
    void setVideoClock(double val)
    {
        videoClock = val;
    };
    void setFileTypeParams(int, bool);

#ifdef HAVE_SDL
    static PacketQueue *pq;

    void initAudio();
    static void audio_callback(void *, Uint8 *, int);
    static int audioDecodeFrame(VideoStream *, uint8_t *, int, double *);
    bool putAudio(AVPacket *);
    static int getAudio(AVPacket *, int);
    static void getAudioClock(VideoStream *);
    double getAudioTime()
    {
        return audioClock;
    };
    static int syncAudio(VideoStream *, uint8_t *, int);
    AVCodecContext *getAudioCodecContext()
    {
        return audioCodecCtx;
    };
    int getAudioStreamID()
    {
        return audioStreamID;
    };
    void setAudioPlayback(bool val)
    {
        playAudio = val;
    };
    bool getAudioPlayback()
    {
        return playAudio;
    };
#endif

    StereoVideoPlayerPlugin *myPlugin;

private:
    unsigned int readBufferIndex;
    unsigned int readBufferLinesize;
    unsigned int readBufferLines;
    unsigned int readBufferIncrement;
    int usedWidth, usedHeight;

    AVFormatContext *oc;
    AVCodecContext *codecCtx;
    AVCodec *codec;
    SwsContext *swsConvertCtx;
    AVFrame *pFrame;
    AVFrame *dispFrameRGB;
    uint8_t *dispRGBBuffer;
    int numBytesRGB;
    int videoStreamID;
    double videoClock;
    static unsigned int maxVideoBufferElements;

    bool playAudio;

#ifdef HAVE_SDL
    AVCodecContext *audioCodecCtx;
    AVCodec *audioCodec;
    int audioStreamID;
    double audioClock;
    double audioOffset;
    static double audioDiffThreshold;
    static unsigned int audioBufferSize;
    static unsigned int audioBufferIndex;
    static AVRational audioTimeBase;
    int64_t audioStartTime;
#endif
};

class FFMPEGVideoPlayer
{
public:
    FFMPEGVideoPlayer();
    ~FFMPEGVideoPlayer();

    int getFrame(VideoStream *, osg::Image *, bool, GLenum);
    void setImage(VideoStream *, osg::Image *, uint8_t *, GLenum);
#ifdef HAVE_SDL
    bool openSDL(VideoStream *);
#endif
    enum Status
    {
        Play,
        Pause,
        Stop,
        Loop
    };
    void setStatus(Status);
    bool getStatus(Status);
    void setSpeed(float v)
    {
        speed = v / 100;
    };
    float getSpeed()
    {
        return speed;
    };

    static bool quit;

private:
#ifdef HAVE_SDL
    SDL_AudioSpec movieSpec;
    SDL_AudioSpec spec;
#endif
    bool play;
    bool pause;
    bool stop;
    bool loop;
    float speed;
};
