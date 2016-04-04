/**********************************************************************
* @file    VICapture.h
* @brief   __DoxyFileBriefDescr__
* 
* @author  hpcbrait
* @date    05.02.2010
***********************************************************************
* Version History:
* V 0.10  05.02.2010  : First Revision
**********************************************************************/
#ifndef _VICAPTURE_H
#define _VICAPTURE_H

#include "videoInput.h"
#include <string>

class Size
{
public:
    Size()
    {
        x = 0;
        y = 0;
    };

    virtual ~Size(){};

    int x;
    int y;
};

/**
* @class   VideoParameter
* @brief   __DoxyFileBriefDescr__
* 
* @author  hpcbrait
* @date    05.02.2010
* @version 0.10
* @since   0.10
*/
class VideoParameter
{
public:
    /**
    * @brief   __DoxyMethodBriefDescr_VideoParameter__
    */
    VideoParameter();

    /**
    * @brief   __DoxyMethodBriefDescr_~VideoParameter__
    * @return 
    */
    virtual ~VideoParameter(){};

    /**
    * @brief   __DoxyMethodBriefDescr_getWidth__
    * @return int
    */
    int getWidth();

    /**
    * @brief   __DoxyMethodBriefDescr_getHeight__
    * @return int
    */
    int getHeight();

    /**
    * @brief   __DoxyMethodBriefDescr_getBPP__
    * @return int
    */
    int getBPP();

    /**
    * @brief   __DoxyMethodBriefDescr_getFPS__
    * @return int
    */
    int getFPS();

    /**
    * @brief   __DoxyMethodBriefDescr_getDeviceNumber__
    * @return int
    */
    int getDeviceNumber();

    /**
    * @brief   __DoxyMethodBriefDescr_getBufferSize__
    * @return int
    */
    int getBufferSize();

    /**
    * @brief   __DoxyMethodBriefDescr_getFlipH__
    * @return bool
    */
    bool getFlipImage();

    /**
    * @brief   __DoxyMethodBriefDescr_getVIInstance__
    * @return videoInput*
    */
    videoInput *getVIInstance();

    /**
    * @brief   __DoxyMethodBriefDescr_getVideoBuffer__
    * @return unsigned char*
    */
    unsigned char *getVideoBuffer();

    /**
    * @brief   __DoxyMethodBriefDescr_setWidth__
    * @param width 
    */
    void setWidth(int width);

    /**
    * @brief   __DoxyMethodBriefDescr_setHeight__
    * @param height 
    */
    void setHeight(int height);

    /**
    * @brief   __DoxyMethodBriefDescr_setBPP__
    * @param bpp 
    */
    void setBPP(int bpp);

    /**
    * @brief   __DoxyMethodBriefDescr_setFPS__
    * @param fps 
    */
    void setFPS(int fps);

    /**
    * @brief   __DoxyMethodBriefDescr_setDeviceNumber__
    * @param dev 
    */
    void setDeviceNumber(int dev);

    /**
    * @brief   __DoxyMethodBriefDescr_setBufferSize__
    * @param size 
    */
    void setBufferSize(int size);

    /**
    * @brief   __DoxyMethodBriefDescr_setFlip__
    * @param flipV 
    * @param false 
    */
    void setFlipImage(bool flip);

    /**
    * @brief   __DoxyMethodBriefDescr_setVIInstance__
    * @param vi 
    */
    void setVIInstance(videoInput *vi);

    /**
    * @brief   __DoxyMethodBriefDescr_setVideoBuffer__
    * @param buffer 
    */
    void setVideoBuffer(unsigned char *buffer);

private:
    int m_width;
    int m_height;
    int m_bpp;
    int m_fps;
    int m_deviceNumber;
    int m_bufferSize;
    bool m_flipImage;
    unsigned char *m_videoBuffer;
    videoInput *m_vi;
};

/**
* @class   VICapture
* @brief   __DoxyFileBriefDescr__
* 
* @author  hpcbrait
* @date    05.02.2010
* @version 0.10
* @since   0.10
*/
class VICapture
{
public:
    /**
    * @brief   __DoxyMethodBriefDescr_VICapture__
    */
    VICapture();

    /**
    * @brief   __DoxyMethodBriefDescr_~VICapture__
    * @return 
    */
    virtual ~VICapture(void);

    /**
    * @brief   __DoxyMethodBriefDescr_init__
    * @param config 
    */
    bool init(std::string config, bool externalVideoBuffer = false);

    /**
    * @brief   __DoxyMethodBriefDescr_getImage__
    * @return unsigned char*
    */
    unsigned char *getImage();

    /**
    * @brief   __DoxyMethodBriefDescr_inqCameraImageSize__
    * @return Size
    */
    Size inqCameraImageSize();

    /**
    * @brief   __DoxyMethodBriefDescr_uninit__
    */
    void uninit();

    void setExternalVideoBuffer(unsigned char *buffer);

    int getVideoBufferSize();

    VideoParameter *getVideoParameter();

private:
    /**
    * @brief   __DoxyMethodBriefDescr_parseConfig__
    * @param config 
    */
    void parseConfig(std::string config);

    /**
    * 
    */
    VideoParameter m_vid;

    /**
    * 
    */
    bool m_showSettings;

    bool m_externalVideoBuffer;
};

#endif