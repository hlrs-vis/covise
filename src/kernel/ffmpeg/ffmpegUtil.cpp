#include "ffmpegUtil.h"
#include <algorithm>
#include <array>
using namespace covise;

#ifndef AV_PROFILE_UNKNOWN
#define AV_PROFILE_UNKNOWN FF_PROFILE_UNKNOWN
#endif

CodecListEntry::CodecListEntry(const AVCodec *codec)
    : codec(codec)
{
    if (codec && codec->profiles)
    {
        for (const AVProfile *pro = codec->profiles;
             pro->profile != AV_PROFILE_UNKNOWN;
             ++pro)
        {
            profiles.push_back(pro->name);
        }
    }
}

std::map<const AVOutputFormat *, AVCodecList> covise::listFormatsAndCodecs()
{
    std::map<const AVOutputFormat *, AVCodecList> formatList;

    std::vector<const AVCodec *> codecList;
    void *start = nullptr;
    while (auto codec = av_codec_iterate(&start))
    {
        if (codec->type != AVMEDIA_TYPE_VIDEO ||
            !av_codec_is_encoder(codec) ||
            codec->capabilities & AV_CODEC_CAP_AVOID_PROBING ||
            codec->capabilities & AV_CODEC_CAP_EXPERIMENTAL)
            continue;
        codecList.push_back(codec);
        // std::cerr << "Codec " << codec->id << ", " << codec->name << ": " << codec->long_name << std::endl;
    }
    std::sort(codecList.begin(), codecList.end(), [](const AVCodec *c1, const AVCodec *c2)
              { return c1->id < c2->id; });

    start = nullptr;
    while (auto format = av_muxer_iterate(&start))
    {
        if (format->video_codec == 0 ||
#ifdef AVFMT_NODIMENSIONS
            format->flags & AVFMT_NODIMENSIONS ||
#endif
            format->flags & AVFMT_NEEDNUMBER)
            continue;

        constexpr std::array<const char *, 9> supportedFormats{
            "avi",
            "asf",
            "m4v",
            "matroska",
            "mov",
            "mp4",
            "mpeg",
            "ogv",
            "webm"};

        for (auto f : supportedFormats)
        {
            if (strcmp(f, format->name) == 0)
            {
                AVCodecList sublist;
                for (const auto it : codecList)
                {
                    if (!format->codec_tag)
                    {
                        if (it->id == format->video_codec)
                        {
                            sublist.push_back(it);
                            break;
                        }
                    }
                    else if (it->id == AV_CODEC_ID_RAWVIDEO || av_codec_get_tag(format->codec_tag, it->id) > 0)
                    {
                        if (it->id == format->video_codec)
                            sublist.push_front(it);
                        else
                            sublist.push_back(it);
                    }
                }

                if (!sublist.empty())
                    formatList[format] = sublist;
            }
        }
    }
    return formatList;
}
