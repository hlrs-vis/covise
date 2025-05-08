#include "ffmpegEncoder.h"

#include <util/threadname.h>

#include <cassert>
#include <iostream>

constexpr int alignment = 1;
using namespace opencover;

void opencover::avFree(void *ptr)
{
    av_free(ptr);
}

void avFrameDeleter::operator()(AVFrame *f)
{
    av_frame_free(&f);
}

AvFramePtr alloc_picture(AVPixelFormat pix_fmt, const FFmpegEncoder::Resolution &res)
{
    auto picture = AvFramePtr(av_frame_alloc());
    if (!picture)
        return nullptr;

    // int size = av_image_get_buffer_size(pix_fmt, res.w, res.h, alignment);
    int size = av_image_fill_arrays(picture->data, picture->linesize, nullptr, pix_fmt, res.w, res.h, alignment);
    uint8_t *picture_buf = (uint8_t *)av_malloc(size);
    if (!picture_buf)
    {
        return nullptr;
    }
    av_image_fill_arrays(picture->data, picture->linesize, picture_buf, pix_fmt, res.w, res.h, alignment);
    picture->format = pix_fmt;
    picture->width = res.w;
    picture->height = res.h;
    return std::move(picture);
}

int getMatchingSupportedFramerate(const AVCodec *codec, int targetFramerate)
{
    int diff = 1000000;
    int replacement = targetFramerate;
    for (auto framerate = codec->supported_framerates; framerate && framerate->den == 0; framerate++)
    {
        auto d = std::abs(targetFramerate - framerate->num);
        if (d <= diff)
        {
            diff = d;
            replacement = framerate->den;
        }
    }
    if (replacement != targetFramerate)
        std::cerr << "Framerate " << targetFramerate << " not supportded by " << codec->long_name << ": using framerate of " << replacement << " instead." << std::endl;
    return replacement;
}

AVCodecContext *createCodecContext(const AVCodec *codec, const FFmpegEncoder::VideoFormat &format)
{
    auto codecContext = avcodec_alloc_context3(codec);
    // use default color format of not set
    auto ocf = format.colorFormat;
    if (ocf == AV_PIX_FMT_NONE)
        ocf = codec->pix_fmts[0];
    if (ocf == AV_PIX_FMT_NONE)
    {
        std::cerr << "automatic video pixel format determination failed" << std::endl;
        return nullptr;
    }
    codecContext->pix_fmt = ocf;

    /* put sample parameters */
    codecContext->bit_rate = format.bitrate;
    codecContext->rc_max_rate = format.max_bitrate;
    // codecContext->rc_min_rate = codecContext->bit_rate == codecContext->rc_max_rate ? codecContext->rc_max_rate : 0;
    codecContext->rc_min_rate = 0;

    codecContext->width = format.resolution.w;
    codecContext->height = format.resolution.h;
    /* frames per second */
    codecContext->time_base = AVRational{1, getMatchingSupportedFramerate(codec, format.fps)};
    // codecContext->gop_size = 10; /* emit one intra frame every ten frames */
    // codecContext->max_b_frames = 1;

    return codecContext;
}

AVFormatContext *openOutputStream(const FFmpegEncoder::VideoFormat &output, const std::string &outputFile, AVCodecContext *context)
{
    AVFormatContext *oc;
    // cast const away the same way av_guess_format does, might be unneccessary for newer ffmpeg versions
    auto outPutFormat = output.outputFormat ? const_cast<AVOutputFormat *>(output.outputFormat) : av_guess_format(output.codecName.c_str(), nullptr, nullptr);
    int result = avformat_alloc_output_context2(&oc, outPutFormat, output.codecName.c_str(), outputFile.c_str());
    if (!oc)
        return nullptr;
    oc->oformat = outPutFormat;
    oc->audio_preload = (int)(100 * AV_TIME_BASE);
    oc->max_delay = (int)(0.7 * AV_TIME_BASE);
    oc->max_delay = (int)(0.7 * AV_TIME_BASE);
    oc->url = new char[outputFile.length() + 1];
    strcpy(oc->url, outputFile.c_str());
    auto video_st = avformat_new_stream(oc, NULL);
    if (video_st)
        video_st->id = 0;
    if (avio_open(&oc->pb, outputFile.c_str(), AVIO_FLAG_WRITE) < 0)
        std::cerr << "failed to open output file " << outputFile << std::endl;
    video_st->codecpar->codec_id = context->codec_id;
    video_st->codecpar->codec_type = context->codec_type;
    video_st->codecpar->width = output.resolution.w;
    video_st->codecpar->height = output.resolution.h;
    video_st->time_base = context->time_base;
    if (oc->oformat->flags & AVFMT_GLOBALHEADER)
        context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    av_dump_format(oc, 0, outputFile.c_str(), 1);
    if (avformat_write_header(oc, NULL))
    {
        std::cerr << "Could not write header  (incorrect codec parameters ?)" << std::endl;
        return nullptr;
    }
    return oc;
}

FFmpegEncoder::FFmpegEncoder(const VideoFormat &input, const VideoFormat &output, const std::string &outputFile)
    : m_inputRes(input.resolution), m_capturePixFmt(input.colorFormat)
{
    /* find the video encoder */
    auto outCodec = avcodec_find_encoder_by_name(output.codecName.c_str());
    if (!outCodec)
    {
        std::cerr << "codec " << output.codecName << " not found" << std::endl;
        m_error = true;
        return;
    }

    m_outCodecContext = createCodecContext(outCodec, output);
    if (!m_outCodecContext)
    {
        m_error = true;
        std::cerr << "creatopn of output format context failed" << std::endl;
        return;
    }
    m_inPicture.reset(av_frame_alloc());

    m_inSize = av_image_get_buffer_size(m_capturePixFmt, m_inputRes.w, m_inputRes.h, alignment);

    if (m_outCodecContext->pix_fmt != m_capturePixFmt)
    {
        m_swsconvertctx = sws_getContext(m_inputRes.w, m_inputRes.h, m_capturePixFmt,
                                         m_outCodecContext->width, m_outCodecContext->height, m_outCodecContext->pix_fmt,
                                         SWS_FAST_BILINEAR, NULL, NULL, NULL);
    }
    if (!m_swsconvertctx)
        std::cerr << "Did not initialize the conversion context!" << std::endl;

    m_outPicture = alloc_picture(m_outCodecContext->pix_fmt, output.resolution);

    m_oc = openOutputStream(output, outputFile, m_outCodecContext);
    if (!m_oc)
    {
        std::cerr << "could not open output stream" << std::endl;
        m_error = true;
        return;
    }

    if (avcodec_open2(m_outCodecContext, outCodec, nullptr) < 0)
    {
        std::cerr << "could not open codec" << std::endl;
        m_error = true;
        return;
    }
}

bool FFmpegEncoder::isValid() const { return !m_error; }

uint8_t *FFmpegEncoder::getPixelBuffer()
{
    if (!m_pixels)
        m_pixels.reset((uint8_t *)av_malloc(m_inSize));
    return m_pixels.get();
}

void FFmpegEncoder::writeVideo(size_t frameNum, uint8_t *pixels, bool mirror)
{
    if (m_error)
        return;
    auto encodeAndWrite = [this](int frameNum, uint8_t *pixels, bool mirror) -> bool
    {
        if (m_error)
            return false;
        /* encode the image */
        SwConvertScale(pixels, mirror);
        m_inPicture->pts = m_outPicture->pts = av_rescale_q(frameNum, m_outCodecContext->time_base, m_oc->streams[0]->time_base);
        auto out_size = avcodec_send_frame(m_outCodecContext, m_swsconvertctx ? m_outPicture.get() : m_inPicture.get());
        AVPacket *output = av_packet_alloc();
        while (!avcodec_receive_packet(m_outCodecContext, output))
        {
            int ret = av_write_frame(m_oc, output);
            if (ret < 0)
            {
                std::cerr << "error " << ret << " during writing frame for pts=" << output->pts << std::endl;
                return false;
            }
        }
        return true;
    };
    if (m_encodeFuture)
    {
        m_encodeFuture->get();
    }
    else
    {
        m_encodeFuture.reset(new std::future<bool>);
    }
    *m_encodeFuture = std::async(std::launch::async, [this, frameNum, pixels, mirror, encodeAndWrite]()
                                 {
        covise::setThreadName("ffmpeg encoder");
        return encodeAndWrite(frameNum, pixels, mirror); });
}

void FFmpegEncoder::SwConvertScale(uint8_t *pixels, bool mirror)
{
    // OpenGL reads bottom-to-top, encoder expects top-to-bottom
    int linesize = m_inputRes.w * 4; // RGBA?
    if (!m_mirroredpixels)
        m_mirroredpixels.reset((uint8_t *)av_malloc(m_inSize));
    for (int y = m_inputRes.h; y > 0; y--)
        memcpy(m_mirroredpixels.get() + (m_inputRes.h - y) * linesize, pixels + (y - 1) * linesize, linesize);

    auto p = m_mirroredpixels.get();
    if (mirror)
    {
        for (size_t i = 0; i < m_inputRes.h; i++)
        {
            for (size_t j = 0; j < linesize;)
            {
                pixels[i * linesize + j + 0] = m_mirroredpixels.get()[(i + 1) * linesize - j - 4];
                pixels[i * linesize + j + 1] = m_mirroredpixels.get()[(i + 1) * linesize - j - 3];
                pixels[i * linesize + j + 2] = m_mirroredpixels.get()[(i + 1) * linesize - j - 2];
                pixels[i * linesize + j + 3] = m_mirroredpixels.get()[(i + 1) * linesize - j - 1];
                j += 4;
            }
        }
        p = pixels;
    }

    auto err = av_image_fill_arrays(m_inPicture->data, m_inPicture->linesize, p, m_capturePixFmt, m_inputRes.w, m_inputRes.h, alignment);
    if (err < 0)
        std::cerr << "av_image_fill_arrays failed " << err << std::endl;
    // if not m_swsconvertctx in and out are equal and therefor creation of out picture can be omitted
    if (m_swsconvertctx)
        sws_scale(m_swsconvertctx, m_inPicture->data, m_inPicture->linesize, 0, m_inputRes.h, m_outPicture->data, m_outPicture->linesize);
}

FFmpegEncoder::~FFmpegEncoder()
{
    if (m_encodeFuture)
    {
        m_encodeFuture->get();
        m_encodeFuture.reset();
    }

    if (m_oc)
    {
        av_write_trailer(m_oc);
        if (m_oc->pb)
            avio_close(m_oc->pb);
        for (unsigned int i = 0; i < m_oc->nb_streams; i++)
        {
            av_freep(&m_oc->streams[i]);
        }
    }
    avcodec_close(m_outCodecContext);
    avcodec_free_context(&m_outCodecContext);
}
