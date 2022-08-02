#ifndef FFMPEGVIDEO_H
#define FFMPEGVIDEO_H

#include <osgViewer/GraphicsWindow>
#include <mutex>
#ifndef _M_CEE //no future in Managed OpenCOVER
#include <future>
#endif
#include <memory>

#include <ffmpeg/ffmpegEncoder.h>
#include <ffmpeg/ffmpegUtil.h>

#include <xercesc/dom/DOM.hpp>


#include "Video.h"

//#define USE_CODECPAR



typedef struct
{
    std::string name;
    std::string fps;
    int constFrames;
    int avgBitrate;
    int maxBitrate;
    int width;
    int height;
} VideoParameter;

class FFMPEGPlugin: public SysPlugin
{
public:
    ~FFMPEGPlugin();

    friend class VideoPlugin;

private:
    std::map<const AVOutputFormat *, covise::AVCodecList> formatList;
    std::list<VideoParameter> VPList;

    void tabletEvent(coTUIElement *);
    void Menu(int row);
    void ParamMenu(int row);
    void hideParamMenu(bool hide);
    void ListFormatsAndCodecs(const std::string &);
    void FillComboBoxSetExtension(int selection, int row);
    void changeFormat(coTUIElement *, int row);
    void checkFileFormat(const string &name);
    bool videoCaptureInit(const string &filename, int format, int RGBFormat);
    void videoWrite(int format = 0);
    const AVOutputFormat *getSelectedOutputFormat();
    const AVCodec *getSelectedCodec();

    void init_GLbuffers();
    void unInitialize();
    void close_all(bool stream = false, int format = 0);
    int readParams();
    void loadParams(int);
    void saveParams();
    void addParams();
    void fillParamComboBox(int);
    void sendParams();
    int getParams();

    std::unique_ptr<FFmpegEncoder> m_encoder;


    uint8_t *video_outbuf = nullptr;
    int video_outbuf_size;
    uint8_t *mirroredpixels = nullptr;
    int linesize;

    coTUIEditField *paramNameField = nullptr;
    coTUILabel *paramErrorLabel = nullptr;
    coTUILabel *paramLabel = nullptr;
    coTUILabel *bitrateLabel = nullptr;
    coTUILabel *maxBitrateLabel = nullptr;
    coTUIEditIntField *maxBitrateField = nullptr;
    coTUIToggleButton *saveButton = nullptr;

    xercesc::DOMImplementation *impl = nullptr;
#ifndef _M_CEE //no future in Managed OpenCOVER
    std::unique_ptr<std::future<bool>> encodeFuture;
#endif
};

#endif
