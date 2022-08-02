#ifndef COVISE_FFMPET_UTIL_H
#define COVISE_FFMPET_UTIL_H

#include <util/coExport.h>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
};
 #include <vector>
 #include <map>
 #include <string>
 #include <list>

namespace covise{
struct FFMPEGEXPORT CodecListEntry
{
    CodecListEntry(const AVCodec *codec);

    const AVCodec *codec = nullptr;
    std::vector<std::string> profiles;
};
typedef std::list<CodecListEntry> AVCodecList;

std::map<const AVOutputFormat *, AVCodecList> FFMPEGEXPORT listFormatsAndCodecs();

} // covise
#endif // COVISE_FFMPET_UTIL_H