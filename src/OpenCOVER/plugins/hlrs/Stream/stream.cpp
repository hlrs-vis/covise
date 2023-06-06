#include "stream.h"
#include <cover/coVRTui.h>
#include <cover/coVRConfig.h>

using namespace opencover;

constexpr auto colorFormat = GL_RGBA;
constexpr auto NumbersPerPixel = 4;

Stream::Stream()
: coVRPlugin(COVER_PLUGIN_NAME)
, m_tab("Stream", coVRTui::instance()->mainFolder->getID())
, m_streamButton("stream", m_tab.getID(), false)
, m_streamNumberLabel("video device number", m_tab.getID())
, m_streamNumber("device number", m_tab.getID())
, m_outputResolutionLabel("Virtual webcam resolution", m_tab.getID())
, m_outputResolutionWidth("width", m_tab.getID())
, m_outputResolutionHeight("height", m_tab.getID())
, m_mirrorButton("mirror image", m_tab.getID())
{

    m_streamNumberLabel.setColor(Qt::black);
    m_streamButton.setPos(0, 0);
    m_streamButton.setEventListener(this);
    m_streamNumberLabel.setPos(0, 1);
    m_streamNumber.setPos(1, 1);
    m_streamNumber.setValue(7);

    m_outputResolutionLabel.setPos(0, 3);
    m_outputResolutionWidth.setValue(640);
    m_outputResolutionWidth.setPos(1, 3);
    m_outputResolutionHeight.setValue(480);
    m_outputResolutionHeight.setPos(2, 3);

    m_mirrorButton.setState(false);
    m_mirrorButton.setPos(0, 4);

    int x, y, width, height;
    coVRConfig::instance()->windows[0].window->getWindowRectangle(x, y, width, height);
    m_inputFormat.resolution = FFmpegEncoder::Resolution(width, height);
    m_inputFormat.colorFormat = AV_PIX_FMT_BGR32; // provided by open gl with GL_RGBA
}

void Stream::tabletEvent(opencover::coTUIElement *elem)
{
    if (elem == &m_streamButton && m_streamButton.getState())
    {
        m_frameNum = 0;

        // as configured in akvcam init file
        FFmpegEncoder::VideoFormat output;
        output.colorFormat = AV_PIX_FMT_RGB24;
        output.resolution.w = m_outputResolutionWidth.getValue();
        output.resolution.h = m_outputResolutionHeight.getValue();
        output.codecName = "rawvideo";
        m_writer.reset(new FFmpegEncoder(m_inputFormat, output, "/dev/video" + std::to_string(m_streamNumber.getValue())));
        // AVOutputFormat *f = av_guess_format("rawvideo", nullptr, nullptr);
        preSwapBuffers(0);
    }
}

void Stream::preSwapBuffers(int windowNumber)
{
    if (windowNumber == 0 && m_streamButton.getState())
    {
        glReadPixels(0, 0, m_inputFormat.resolution.w, m_inputFormat.resolution.h, colorFormat, GL_UNSIGNED_BYTE,
                     m_writer->getPixelBuffer());

        m_writer->writeVideo(m_frameNum++, m_writer->getPixelBuffer(), m_mirrorButton.getState());
    }
}

COVERPLUGIN(Stream)