#define __STDC_CONSTANT_MACROS
#include "StereoVideoPlayer.h"

#include <config/CoviseConfig.h>

#include <cover/VRSceneGraph.h>
#include <cover/coVRMSController.h>

#ifdef WIN32
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif

#include <stdio.h>

bool FFMPEGVideoPlayer::quit = false;
#ifdef HAVE_SDL
#define SDL_AUDIO_BUFFER_SIZE 1024
#define AUDIO_DIFF_AVG_NUMBER 20
#define SAMPLE_CORRECTION_PERCENT_MAX 10
#define AV_NOSYNC_THRESHOLD 10.0
/* averaging filter for audio sync */
#define AUDIO_DIFF_AVG_COEF exp(log(0.01 / AUDIO_DIFF_AVG_NUMBER))
double VideoStream::audioDiffThreshold = 0.0;
unsigned int VideoStream::audioBufferSize = 0;
unsigned int VideoStream::audioBufferIndex = 0;
AVRational VideoStream::audioTimeBase = { 0, 0 };
PacketQueue *VideoStream::pq = NULL;
#endif

VideoQueue *VideoStream::vq = NULL;
unsigned int VideoStream::maxVideoBufferElements = coCoviseConfig::getInt("COVER.Plugin.StereoVideoPlayer.VideoBufferSize", 100);

FFMPEGVideoPlayer::FFMPEGVideoPlayer()
{
    play = false;
    pause = false;
    stop = false;
    loop = false;
    speed = 1.0;

    av_register_all();
}

FFMPEGVideoPlayer::~FFMPEGVideoPlayer()
{
    quit = true;
}

bool FFMPEGVideoPlayer::getStatus(Status s)
{
    switch (s)
    {
    case Play:
        return play;
    case Pause:
        return pause;
    case Stop:
        return stop;
    case Loop:
        return loop;
    }

    return false;
}

void FFMPEGVideoPlayer::setStatus(Status s)
{
    switch (s)
    {
    case Play:
    {
        play = true;
        stop = pause = false;
        break;
    }
    case Pause:
    {
        play = stop = false;
        pause = true;
        break;
    }
    case Stop:
    {
        stop = true;
        pause = true;
        play = false;
        break;
    }
    case Loop:
    {
        if (loop)
            loop = false;
        else
            loop = true;
        break;
    }
    }
}

int FFMPEGVideoPlayer::getFrame(VideoStream *vStream, osg::Image *image, bool show, GLenum format)
{
    double pts = 0.0;

    unsigned int maxSize = vStream->getfps();
    vStream->getfps() < vStream->getMaxVideoBufferSize() ? maxSize = vStream->getfps() : maxSize = vStream->getMaxVideoBufferSize();
    maxSize = vStream->getMaxVideoBufferSize();

    double coverTime = cover->currentTime();
    do
    {
        if (vStream->readFrame() < 0)
        {
            fprintf(stderr, "End of File\n");
            return -1;
        }
        else if (show && (vStream->vq->getSize() == 1))
        {
            image->setImage(vStream->getWidth(), vStream->getHeight(), 1, format, format, GL_UNSIGNED_BYTE, vStream->vq->getImage()->getRGBImage(), osg::Image::NO_DELETE);
            pts = vStream->vq->getImage()->getPts();
        }
    } while (vStream->vq->getSize() < maxSize);

    coverTime = (cover->currentTime() - coverTime) / (maxSize - 1);
    pts = (vStream->vq->getLastImage()->getPts() - pts) / (maxSize - 1);

    return (int)(pts / coverTime);
}

void FFMPEGVideoPlayer::setImage(VideoStream *vStream, osg::Image *image, uint8_t *data, GLenum format)
{
    image->setImage(vStream->getWidth(), vStream->getHeight(), 1, format, format, GL_UNSIGNED_BYTE, data, osg::Image::NO_DELETE);
}

#ifdef HAVE_SDL
bool FFMPEGVideoPlayer::openSDL(VideoStream *vStream)
{
    if (SDL_Init(SDL_INIT_TIMER | SDL_INIT_AUDIO) < 0)
    {
        fprintf(stderr, "Can not initialize SDL: %s\n", SDL_GetError());
    }

    movieSpec.freq = vStream->getAudioCodecContext()->sample_rate;
    movieSpec.format = AUDIO_S16SYS;

    movieSpec.silence = 0;
    movieSpec.channels = vStream->getAudioCodecContext()->channels;
    movieSpec.samples = SDL_AUDIO_BUFFER_SIZE; /* Good values between 512 and 8192 */
    movieSpec.callback = vStream->audio_callback;
    movieSpec.userdata = vStream;

    if (SDL_OpenAudio(&movieSpec, &spec) < 0)
    {
        fprintf(stderr, "Can not open SDL audio: %s\n", SDL_GetError());
        return false;
    }

    return true;
}

void VideoStream::audio_callback(void *usrDat, Uint8 *stream, int len)
{
    VideoStream *vStream = (VideoStream *)usrDat;
    int len1, size;
    double pts;

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55, 52, 102)
    static uint8_t audioBuffer[(AVCODEC_MAX_AUDIO_FRAME_SIZE * 3) / 2];
    static uint8_t stillBuffer[(AVCODEC_MAX_AUDIO_FRAME_SIZE * 3) / 2];
#else
#define MAX_AUDIO_FRAME_SIZE 192000
    static uint8_t audioBuffer[(MAX_AUDIO_FRAME_SIZE * 3) / 2];
    static uint8_t stillBuffer[(MAX_AUDIO_FRAME_SIZE * 3) / 2];
#endif
    while (len > 0)
    {
        if (audioBufferIndex >= audioBufferSize)
        {
            size = audioDecodeFrame(vStream, audioBuffer, sizeof(audioBuffer), &pts);
            if (size < 0)
            {
                audioBufferSize = SDL_AUDIO_BUFFER_SIZE;
                memset(audioBuffer, 0, audioBufferSize); /* silence */
            }
            else
                audioBufferSize = size;
            audioBufferIndex = 0;
        }

        len1 = audioBufferSize - audioBufferIndex;
        if (len1 > len)
            len1 = len;

        int len2 = syncAudio(vStream, stillBuffer, len1);

        if (len2 < 0)
        {
            len2 = -len2;
            memcpy(stream, (uint8_t *)stillBuffer, len2);
            len -= len2;
            stream += len2;
        }
        else if (len2 > 0)
        {

            memcpy(stream, (uint8_t *)audioBuffer + audioBufferIndex, len2);
            if (len2 < len1)
            {
                len -= len2;
                if (len < 256)
                    len = 0;
                audioBufferIndex = audioBufferSize;
            }
            else
            {
                len -= len1;
                audioBufferIndex += len1;
            }
            stream += len2;
        }
        else
        {
            audioBufferIndex = audioBufferSize;
            len -= 128;
            stream += 128;
        }
    }
}

int VideoStream::audioDecodeFrame(VideoStream *vStream, uint8_t *buffer, int size, double *pts)
{
    static AVPacket pkt, audioPkt;
    static uint8_t *audioPktData = NULL;
    static int audioPktSize = 0;
    int len1, bytesDecoded;
    static double diff = 0.0;

    for (;;)
    {
        while (audioPkt.size > 0)
        {
            bytesDecoded = size;
#if LIBAVCODEC_VERSION_MAJOR < 53
            len1 = avcodec_decode_audio2(vStream->audioCodecCtx, (int16_t *)buffer, &bytesDecoded, audioPkt.data, audioPkt.size);
#else
            len1 = avcodec_decode_audio3(vStream->audioCodecCtx, (int16_t *)buffer, &bytesDecoded, &audioPkt);
#endif
            if (len1 < 0) /* Error, skip frame */
            {
                audioPkt.size = 0;
                break;
            }
            audioPkt.data += len1;
            audioPkt.size -= len1;

            if (bytesDecoded <= 0)
                continue;

            double pts1 = vStream->audioClock; /* PTS for packets with multiple frames */
            *pts = pts1;
            int n = 2 * vStream->audioCodecCtx->channels;
            vStream->audioClock += (double)bytesDecoded / (double)(n * vStream->audioCodecCtx->sample_rate);

            return bytesDecoded;
        }

        if (vStream->myPlugin->videoPlayer->getStatus(FFMPEGVideoPlayer::Stop) || vStream->myPlugin->videoPlayer->quit)
        {
            return -1;
        }
        if (getAudio(&pkt, 1) < 0)
            return -1;

        if (pkt.pts != AV_NOPTS_VALUE)
            vStream->audioClock = av_q2d(vStream->audioTimeBase) * (pkt.pts - vStream->audioStartTime);

        audioPkt = pkt;
    }
}

int VideoStream::syncAudio(VideoStream *vStream, uint8_t *stillBuffer, int len1)
{
    int len = len1;

    getAudioClock(vStream);
    double diff = vStream->audioClock + vStream->audioOffset - vStream->myPlugin->getMasterTime(false);

    if (diff > 0.2)
    {
        int bufferSize = diff * vStream->audioCodecCtx->channels * vStream->audioCodecCtx->sample_rate * vStream->audioCodecCtx->bits_per_coded_sample; /* calculate still buffer size */
        if (bufferSize > len1)
            bufferSize = len1;

        return -bufferSize;
    }
    else if (diff < -5.0) /*skip packet */
        len = 0;
    else if (diff < -0.1)
    {
#ifdef WIN32
        int bufferSize = -diff * vStream->audioCodecCtx->channels * vStream->audioCodecCtx->sample_rate * vStream->audioCodecCtx->bits_per_coded_sample;
        len -= bufferSize;
        if (len < 0)
            len = 0;
#else
        len = 0;
#endif
    }

    return len;
}

void VideoStream::getAudioClock(VideoStream *vStream)
{
    static double oldClock = 0.0;
    static double refClock = 0.0;
    static double rate = 1.0 / vStream->audioCodecCtx->sample_rate;

    double diff = vStream->audioClock - refClock;
    if (fabs(diff) < 1E-5 * vStream->audioClock)
    {
        vStream->audioClock = oldClock + rate * (vStream->audioCodecCtx->channels * 2 * vStream->audioCodecCtx->bits_per_coded_sample);
    }
    else
        refClock = vStream->audioClock;

    oldClock = vStream->audioClock;
}

void VideoStream::initAudio()
{
    audioClock = 0.0;
    audioBufferSize = 0;
    audioBufferIndex = 0;
}
#endif

VideoStream::VideoStream()
{
    readBufferIndex = 0;
    readBufferLinesize = 0;
    readBufferLines = 1;
    readBufferIncrement = 0;

    oc = NULL;
    codecCtx = NULL;
    codec = NULL;
    swsConvertCtx = NULL;
    dispFrameRGB = NULL;
    dispRGBBuffer = NULL;
    pFrame = NULL;
    videoStreamID = -1;
    videoClock = 0.0;

#ifdef HAVE_SDL
    audioStreamID = -1;
    initAudio();
    audioOffset = coCoviseConfig::getFloat("COVER.Plugin.StereoVideoPlayer.AudioOffset", 0.0);
    audioCodecCtx = NULL;
    audioCodec = NULL;
    playAudio = coCoviseConfig::isOn("COVER.Plugin.StereoVideoPlayer.Audio", true);
    if (playAudio && !coVRMSController::instance()->isMaster())
    {
        playAudio = false;
    }
    if (!pq)
    {
        pq = new PacketQueue;
    }
#else
    playAudio = false;
#endif
    if (!vq)
    {
        vq = new VideoQueue;
    }
}

VideoStream::~VideoStream()
{
#ifdef HAVE_SDL
    if (playAudio)
    {
        SDL_PauseAudio(1);
        SDL_CondSignal(pq->getCond());
        SDL_CloseAudio();
        SDL_Quit();

        if (audioCodec)
            avcodec_close(audioCodecCtx);
    }
    delete pq;
#endif

    if (codec)
        avcodec_close(codecCtx);
#if FF_API_CLOSE_INPUT_FILE
    if (oc)
        av_close_input_file(oc);
#else
    if (oc)
        avformat_close_input(&oc);
#endif
    if (swsConvertCtx)
        sws_freeContext(swsConvertCtx);
    if (pFrame)
        av_free(pFrame);
    if (dispFrameRGB)
        av_free(dispFrameRGB);
    if (dispRGBBuffer)
        delete[] dispRGBBuffer;

    delete vq;
}

bool VideoStream::openMovieCodec(const std::string filename, PixelFormat *pixFormat)
{
#if LIBAVFORMAT_VERSION_INT <= AV_VERSION_INT(52, 64, 2)
    if (av_open_input_file(&oc, filename.c_str(), NULL, 0, NULL) != 0)
#else
    if (avformat_open_input(&oc, filename.c_str(), NULL, NULL) != 0)
#endif
    {
        fprintf(stderr, "Could not open file\n");
        playAudio = false;
        return false;
    }

#if FF_API_FORMAT_PARAMETERS
    if (av_find_stream_info(oc) < 0)
#else
    if (avformat_find_stream_info(oc, NULL) < 0)
#endif
    {
        fprintf(stderr, "Could not find stream information\n");
        playAudio = false;
        return false;
    }

#if LIBAVFORMAT_VERSION_INT <= AV_VERSION_INT(52, 64, 2)
    dump_format(oc, 0, filename.c_str(), 0);
#else
    av_dump_format(oc, 0, filename.c_str(), 0);
#endif

    for (unsigned int i = 0; i < oc->nb_streams; i++)
    {
#if LIBAVCODEC_VERSION_MAJOR >= 53
        if ((oc->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) && (videoStreamID < 0))
            videoStreamID = i;
#ifdef HAVE_SDL
        else if ((oc->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) && (audioStreamID < 0))
            audioStreamID = i;
#endif
#else

        if ((oc->streams[i]->codec->codec_type == CODEC_TYPE_VIDEO) && (videoStreamID < 0))
            videoStreamID = i;
#ifdef HAVE_SDL
        else if ((oc->streams[i]->codec->codec_type == CODEC_TYPE_AUDIO) && (audioStreamID < 0))
            audioStreamID = i;
#endif
#endif
    }

    if (videoStreamID == -1)
    {
        fprintf(stderr, "Did not find any video stream\n");
        playAudio = false;
        return false;
    }

    codecCtx = oc->streams[videoStreamID]->codec;

    codec = avcodec_find_decoder(codecCtx->codec_id);
    if (codec == NULL)
    {
        fprintf(stderr, "Unsupported video codec\n");
        playAudio = false;
        return false;
    }

    // Inform the codec that we can handle truncated bitstreams -- i.e.,
    // bitstreams where frame boundaries can fall in the middle of packets
    if (codec->capabilities & CODEC_CAP_TRUNCATED)
        codecCtx->flags |= CODEC_FLAG_TRUNCATED;

#if LIBAVCODEC_VERSION_MAJOR < 54
    if (avcodec_open(codecCtx, codec) < 0)
#else
    if (avcodec_open2(codecCtx, codec, NULL) < 0)
#endif
    {
        fprintf(stderr, "Could not open video codec\n");
        playAudio = false;
        return false;
    }

#ifdef HAVE_SDL
    if (audioStreamID == -1)
    {
        fprintf(stderr, "Did not find any audio stream\n");
        playAudio = false;
    }

    else if (playAudio)
    {
        audioCodecCtx = oc->streams[audioStreamID]->codec;

        audioCodec = avcodec_find_decoder(audioCodecCtx->codec_id);
        if (audioCodec == NULL)
        {
            fprintf(stderr, "Unsupported audio codec\n");
            playAudio = false;
            return false;
        }

#if LIBAVCODEC_VERSION_MAJOR < 54
        if (avcodec_open(audioCodecCtx, audioCodec) < 0)
#else
        if (avcodec_open2(audioCodecCtx, audioCodec, NULL) < 0)
#endif
        {
            fprintf(stderr, "Could not open audio codec\n");
            playAudio = false;
            return false;
        }

        /* Correct audio only if larger error than this */
        audioDiffThreshold = 2.0 * SDL_AUDIO_BUFFER_SIZE / audioCodecCtx->sample_rate;

        audioTimeBase = oc->streams[audioStreamID]->time_base;
        audioStartTime = oc->streams[audioStreamID]->start_time;
    }
#endif

    // Hack to correct wrong frame rates that seem to be generated by some codecs
    if (codecCtx->time_base.num > 1000 && codecCtx->time_base.den == 1)
        codecCtx->time_base.den = 1000;

    if (codecCtx->pix_fmt == PIX_FMT_BGR24)
        *pixFormat = codecCtx->pix_fmt;
    else
        *pixFormat = PIX_FMT_RGB24;
    swsConvertCtx = sws_getContext(codecCtx->width, codecCtx->height,
                                   codecCtx->pix_fmt, codecCtx->width, codecCtx->height, *pixFormat, SWS_BICUBIC, NULL, NULL, NULL);
    if (swsConvertCtx == NULL)
    {
        fprintf(stderr, "Cannot initialize the conversion context!\n");
        playAudio = false;
        return false;
    }

    return true;
}

bool VideoStream::allocateFrame()
{
    pFrame = avcodec_alloc_frame();

    if (!pFrame)
    {
        fprintf(stderr, "Can not allocate frame\n");
        return false;
    }

    return true;
}

bool VideoStream::allocateRGBFrame(PixelFormat pixFormat)
{
    dispFrameRGB = avcodec_alloc_frame();

    numBytesRGB = avpicture_get_size(pixFormat, codecCtx->width, codecCtx->height);
    uint8_t *frameRGBBuffer = new uint8_t[numBytesRGB];
    numBytesRGB = avpicture_fill((AVPicture *)dispFrameRGB, frameRGBBuffer, pixFormat, codecCtx->width, codecCtx->height);

    if (numBytesRGB <= 0)
    {
        fprintf(stderr, "Can not allocate frame\n");
        return false;
    }

    dispRGBBuffer = new uint8_t[numBytesRGB];

    readBufferLinesize = numBytesRGB;
    usedWidth = codecCtx->width;
    usedHeight = codecCtx->height;

    return true;
}

void VideoStream::setRGBBuffer(uint8_t *buffer)
{
    memcpy(dispRGBBuffer, buffer, readBufferLines * readBufferLinesize);
}

double VideoStream::synchronizeVideo(double pts, bool init)
{
    double frameDelay;
    static double lastPts = 0.0;

    if (init)
        lastPts = 0.0;
    double delay = pts - lastPts; /* Check if pts makes sense */
    if ((delay <= 0) || (delay >= 1))
    {
        pts = videoClock;
    }
    else
        videoClock = pts;

    frameDelay = av_q2d(codecCtx->time_base);
    frameDelay += pFrame->repeat_pict * (frameDelay * 0.5); /* if frame is repeated */
    videoClock += frameDelay;

    lastPts = pts;

    return pts;
}

int VideoStream::readFrame()
{
    int frameFinished = 0;
    double pts = 0.0;
    AVPacket packet;
    bool first = true;

    while ((frameFinished = av_read_frame(oc, &packet)) >= 0)
    {

        if (packet.stream_index == videoStreamID)
        {
            AVPacket oldPacket = packet;
            do
            {
                int size;
#if LIBAVCODEC_VERSION_MAJOR < 53
                size = avcodec_decode_video(codecCtx, pFrame, &frameFinished, packet.data, packet.size);
#else
                size = avcodec_decode_video2(codecCtx, pFrame, &frameFinished, &packet);
#endif

                if (first)
                {
                    pts = (packet.pts - oc->streams[videoStreamID]->start_time) * av_q2d(oc->streams[videoStreamID]->time_base);
                    first = false;
                }

                packet.data += size;
                packet.size -= size;
            } while ((packet.size > 0) && !frameFinished);

            if (frameFinished)
            {
                if (packet.dts != AV_NOPTS_VALUE)
                    pts = (packet.dts - oc->streams[videoStreamID]->first_dts) * av_q2d(oc->streams[videoStreamID]->time_base);
                if (packet.dts == oc->streams[videoStreamID]->first_dts)
                    pts = synchronizeVideo(pts, true);
                else
                    pts = synchronizeVideo(pts, false);

                sws_scale(swsConvertCtx, pFrame->data, pFrame->linesize, 0, codecCtx->height, dispFrameRGB->data, dispFrameRGB->linesize);
                DisplayImage *dispImg = new DisplayImage(dispFrameRGB->data[0], pts, readBufferIndex, readBufferLinesize, readBufferLines, readBufferIncrement);
                vq->putImage(dispImg);

                first = true;
                av_free_packet(&oldPacket);
                return frameFinished;
            }
            av_free_packet(&oldPacket);
        }
#ifdef HAVE_SDL
        else if (playAudio && (packet.stream_index == audioStreamID))
            putAudio(&packet);
#endif
        else
            av_free_packet(&packet);
    }

    return frameFinished;
}

void VideoStream::setFileTypeParams(int selection, bool switchLR)
{
    delete[] dispRGBBuffer;
    switch (selection)
    {
    case 1:
        readBufferIncrement = numBytesRGB / codecCtx->height;
        readBufferLinesize = readBufferIncrement / 2;
        readBufferLines = codecCtx->height;
        if (((coVRConfig::instance()->channels[0].stereoMode == osg::DisplaySettings::LEFT_EYE) && switchLR) || ((coVRConfig::instance()->channels[0].stereoMode != osg::DisplaySettings::LEFT_EYE) && !switchLR))
            readBufferIndex = readBufferLinesize;
        else
            readBufferIndex = 0;
        dispRGBBuffer = new uint8_t[numBytesRGB / 2];
        usedWidth = codecCtx->width / 2;
        usedHeight = codecCtx->height;
        break;
    case 2:
        readBufferLinesize = numBytesRGB / 2;
        readBufferLines = 1;
        readBufferIncrement = 0;
        if (((coVRConfig::instance()->channels[0].stereoMode == osg::DisplaySettings::LEFT_EYE) && switchLR) || ((coVRConfig::instance()->channels[0].stereoMode != osg::DisplaySettings::LEFT_EYE) && !switchLR))
            readBufferIndex = readBufferLinesize;
        else
            readBufferIndex = 0;
        dispRGBBuffer = new uint8_t[numBytesRGB / 2];
        usedWidth = codecCtx->width;
        usedHeight = codecCtx->height / 2;
        break;
    default:
        readBufferIndex = 0;
        readBufferLinesize = numBytesRGB;
        readBufferLines = 1;
        readBufferIncrement = 0;
        dispRGBBuffer = new uint8_t[numBytesRGB];
        usedWidth = codecCtx->width;
        usedHeight = codecCtx->height;
    }
}

#ifdef HAVE_SDL
PacketQueue::PacketQueue()
{
    size = 0;
    mutex = SDL_CreateMutex();
    cond = SDL_CreateCond();
}

PacketQueue::~PacketQueue()
{
    clearPktList();
    pktList.clear();
}

void PacketQueue::clearPktList()
{

    SDL_LockMutex(getMutex());

    while (!pktList.empty())
    {
        AVPacket pkt = pktList.front();
        size -= pkt.size;
        pktList.pop_front();
    }

    pktList.clear();
    SDL_UnlockMutex(getMutex());
}

bool VideoStream::putAudio(AVPacket *pkt)
{
    AVPacket newpkt;

    if (!pkt->data || (pkt->size <= 0))
    {
        fprintf(stderr, "Invalid packet data\n");
        return false;
    }

    if (av_dup_packet(pkt) < 0)
    {
        fprintf(stderr, "Packet not allocated\n");
        return false;
    }

    uint8_t *buffer = new uint8_t[pkt->size];
    memcpy(buffer, pkt->data, pkt->size);
    newpkt = *pkt;
    newpkt.data = buffer;
    av_free_packet(pkt);
    SDL_LockMutex(pq->getMutex());

    std::list<AVPacket> *pList = pq->getPktList();

    pList->push_back(newpkt);

    pq->setSize(pq->getSize() + pkt->size);

    SDL_CondSignal(pq->getCond());
    SDL_UnlockMutex(pq->getMutex());

    return true;
}

int VideoStream::getAudio(AVPacket *pkt, int block)
{
    AVPacket newPkt;
    int ret;

    SDL_LockMutex(pq->getMutex());

    for (;;)
    {
        if (FFMPEGVideoPlayer::quit)
        {
            ret = -1;
            break;
        }

        std::list<AVPacket> *pList = pq->getPktList();
        if (pList->size() > 0)
        {

            newPkt = pList->front();
            pq->setSize(pq->getSize() - newPkt.size);
            uint8_t *buffer = new uint8_t[newPkt.size];
            memcpy(buffer, newPkt.data, newPkt.size);
            *pkt = newPkt;
            pkt->data = buffer;

            pList->pop_front();
            ret = 1;
            break;
        }
        else if (!block)
        {
            ret = 0;
            break;
        }
        else
        {
            maxVideoBufferElements += 20;
            fprintf(stderr, "Too few audio packets in buffer, resizing buffer %d. To avoid jerking adjust the VideoBufferSize in your config File.\n", maxVideoBufferElements);
            SDL_CondWait(pq->getCond(), pq->getMutex());
        }
    }

    SDL_UnlockMutex(pq->getMutex());

    return ret;
}
#endif

VideoQueue::~VideoQueue()
{
    clearImgList();
}

void VideoQueue::clearImgList()
{
    while (!imgList.empty())
    {
        DisplayImage *newImg = imgList.front();
        imgList.pop_front();
        delete newImg;
    }
    imgList.clear();
}

void VideoQueue::putImage(DisplayImage *img)
{
    imgList.push_back(img);
}

void VideoQueue::removeImage()
{
    imgList.pop_front();
}

DisplayImage::~DisplayImage()
{
    delete RGBImage;
}

DisplayImage::DisplayImage(uint8_t *img, double pts1, unsigned int offset, unsigned int linesize, unsigned int lines, unsigned int increment)
{

    RGBImage = new uint8_t[linesize * lines];
    unsigned int bufferIndex = 0;
    int readIndex = offset;

    for (unsigned int i = 0; i < lines; i++)
    {
        memcpy(RGBImage + bufferIndex, img + readIndex, linesize);
        bufferIndex += linesize;
        readIndex += increment;
    }

    pts = pts1;
}

DisplayImage *VideoQueue::getImage()
{
    if (!imgList.empty())
        return imgList.front();

    return NULL;
}

DisplayImage *VideoQueue::getLastImage()
{
    if (!imgList.empty())
        return imgList.back();

    return NULL;
}
