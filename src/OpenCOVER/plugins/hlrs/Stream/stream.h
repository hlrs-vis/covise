#ifndef COVISE_PLUGIN_STREAM_H
#define COVISE_PLUGIN_STREAM_H

#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>


#include <PluginUtil/ffmpegUtil.h>
namespace opencover
{

class Stream : public coVRPlugin, public coTUIListener
{
    public:
        Stream();
        void preSwapBuffers(int windowNumber) override;

    private:
        coTUITab m_tab;
        coTUIToggleButton m_streamButton;
        coTUILabel m_streamNumberLabel;
        coTUIEditIntField m_streamNumber;
        coTUILabel m_outputResolutionLabel;
        coTUIEditIntField m_outputResolutionWidth, m_outputResolutionHeight;
        AvWriter2::VideoFormat m_inputFormat;
        coTUIToggleButton m_mirrorButton; //some conferencing tools mirror the image and sometimes we dont't want this 
        std::unique_ptr<AvWriter2> m_writer;
        void tabletEvent(opencover::coTUIElement *) override;
        size_t m_frameNum = 0;
};
}

#endif // COVISE_PLUGIN_STREAM_H