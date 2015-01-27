
#include "VICapture.h"
#include <iostream>
#include <algorithm>

using namespace std;

VICapture::VICapture(void)
{
    m_showSettings = false;
}

VICapture::~VICapture(void)
{
    //Clear objects created in constructor
    delete[] m_vid.getVideoBuffer();
    m_vid.setVideoBuffer(NULL);
    delete m_vid.getVIInstance();
    m_vid.setVIInstance(NULL);
}

bool VICapture::init(string config, bool externalVideoBuffer)
{
    bool returnValue;

    //Parse config string
    //Recognized parameters
    // -width=<int> Width of captured frames in pixel
    // -height=<int> Height of captured frames in pixel
    // -dev=<int> number of device to be used fro capturing
    // -flipImage=<int> 1 ifimage should be flipped, otherwise 0
    // -fps=<int> fps used to capture from camera VI will use closest achievable fps

    parseConfig(config);

    m_vid.setVIInstance((new videoInput()));
    videoInput *vi = m_vid.getVIInstance();
    if ((m_vid.getHeight() == 0) && (m_vid.getWidth() == 0))
    {
        cerr << "VICpature::init(): Width and Height are 0 or not configured in the config!" << endl;
    }
    if (!(returnValue = (vi->setupDevice(m_vid.getDeviceNumber(), m_vid.getWidth(), m_vid.getHeight(), VI_COMPOSITE))))
    {
        cerr << "VICapture::init(): Setup of camera failed!" << endl;
        return returnValue;
    }

    vi->setIdealFramerate(m_vid.getDeviceNumber(), m_vid.getFPS());
    if (m_showSettings)
    {
        vi->showSettingsWindow(m_vid.getDeviceNumber());
    }

    m_vid.setWidth(vi->getWidth(m_vid.getDeviceNumber()));
    m_vid.setHeight(vi->getHeight(m_vid.getDeviceNumber()));

    int imageBufferSize = vi->getSize(m_vid.getDeviceNumber());
    m_vid.setBufferSize(imageBufferSize);

    if (!externalVideoBuffer)
    {
        m_externalVideoBuffer = false;
        m_vid.setVideoBuffer(new unsigned char[imageBufferSize]);
    }
    else
    {
        m_externalVideoBuffer = true;
    }

    return true;
}

void VICapture::setExternalVideoBuffer(unsigned char *buffer)
{
    if (m_externalVideoBuffer)
    {
        m_vid.setVideoBuffer(buffer);
    }
}

int VICapture::getVideoBufferSize()
{
    return m_vid.getBufferSize();
}

void VICapture::parseConfig(string config)
{
    string option = "";
    string optionValue = "";
    string::size_type valueStart;
    string::size_type optionEnd;
    string::size_type optionStart;

    while ((optionStart = config.find("-")) != string::npos)
    {
        //search for next option start
        if ((optionEnd = config.find("-", optionStart + 1)) == string::npos)
        {
            optionEnd = config.length();
        }
        else
        {
            optionEnd--;
        }

        valueStart = config.find("=", optionStart) + 1;
        option = config.substr(1, valueStart - 2);
        optionValue = config.substr(valueStart, optionEnd - valueStart);

        if (optionEnd < config.length())
        {
            config = config.substr(optionEnd + 1, config.length() - (optionEnd + 1));
        }
        else
        {
            config = "";
        }

        //Lower casing all letters in the option
        std::transform(option.begin(), option.end(), option.begin(), (int (*)(int))tolower);

        if (option == "width")
        {
            m_vid.setWidth(atoi(optionValue.c_str()));
        }

        else if (option == "height")
        {
            m_vid.setHeight(atoi(optionValue.c_str()));
        }
        else if (option == "dev")
        {
            m_vid.setDeviceNumber(atoi(optionValue.c_str()));
        }

        else if (option == "fps")
        {
            m_vid.setFPS(atoi(optionValue.c_str()));
        }

        else if (option == "flipimage")
        {
            int value = atoi(optionValue.c_str());
            if (value == 1)
            {
                m_vid.setFlipImage(true);
            }
            else
            {
                m_vid.setFlipImage(false);
            }
        }
        else
        {
            cerr << "Unkown video config option for videoInput: " << option << endl;
        }
    } //end while
}

void VICapture::uninit()
{
    if (m_vid.getVIInstance() == NULL)
    {
        return;
    }
    m_vid.getVIInstance()->stopDevice(m_vid.getDeviceNumber());
    delete m_vid.getVIInstance();
    m_vid.setVIInstance(NULL);
    delete[] m_vid.getVideoBuffer();
}

unsigned char *VICapture::getImage()
{

    if (m_vid.getVIInstance() == NULL)
    {
        return NULL;
    }

    if (m_vid.getVIInstance()->isFrameNew(m_vid.getDeviceNumber()))
    {
        m_vid.getVIInstance()->getPixels(m_vid.getDeviceNumber(), m_vid.getVideoBuffer(), false, m_vid.getFlipImage()); //fills pixels as a BGR (for openCV) unsigned char array - no flipping
        //m_vid.getVIInstance()->getPixels(m_vid.getDeviceNumber(), m_vid.getVideoBuffer(),false, true);	//fills pixels as a BGR (for openCV) unsigned char array - no flipping
    }

    return (m_vid.getVideoBuffer());
}

VideoParameter *VICapture::getVideoParameter()
{
    return &m_vid;
}

Size VICapture::inqCameraImageSize()
{
    Size videoSize;
    videoSize.x = m_vid.getWidth();
    videoSize.y = m_vid.getHeight();
    return videoSize;
}

//*****************************************************************************
// Implementation of VideoParameter
//*****************************************************************************

VideoParameter::VideoParameter()
{
    //Initializing to videoInput default values
    m_width = 320;
    m_height = 240;
    m_bpp = 24;
    m_fps = 60;
    m_deviceNumber = 0;
    m_videoBuffer = NULL;
    m_bufferSize = 0;
    m_flipImage = false;
    m_vi = NULL;
}

int VideoParameter::getWidth()
{
    return m_width;
}
int VideoParameter::getHeight()
{
    return m_height;
}
int VideoParameter::getBPP()
{
    return m_bpp;
}
int VideoParameter::getFPS()
{
    return m_fps;
}
int VideoParameter::getDeviceNumber()
{
    return m_deviceNumber;
}

int VideoParameter::getBufferSize()
{
    return m_bufferSize;
}

bool VideoParameter::getFlipImage()
{
    return m_flipImage;
}

videoInput *VideoParameter::getVIInstance()
{
    return m_vi;
}

unsigned char *VideoParameter::getVideoBuffer()
{
    return m_videoBuffer;
}

void VideoParameter::setWidth(int width)
{
    m_width = width;
}
void VideoParameter::setHeight(int height)
{
    m_height = height;
}
void VideoParameter::setBPP(int bpp)
{
    m_bpp = bpp;
}
void VideoParameter::setFPS(int fps)
{
    m_fps = fps;
}
void VideoParameter::setDeviceNumber(int dev)
{
    m_deviceNumber = dev;
}

void VideoParameter::setBufferSize(int size)
{
    m_bufferSize = size;
}

void VideoParameter::setFlipImage(bool flip)
{
    m_flipImage = flip;
}

void VideoParameter::setVIInstance(videoInput *vi)
{
    if (vi != NULL)
    {
        m_vi = vi;
    }
}

void VideoParameter::setVideoBuffer(unsigned char *buffer)
{
    m_videoBuffer = buffer;
}
