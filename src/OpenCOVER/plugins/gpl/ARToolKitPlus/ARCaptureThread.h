#ifndef _ARCAPTURETHREAD_H
#define _ARCAPTURETHREAD_H
#include <cover/ARToolKit.h>
#include <ARToolKitPlus/TrackerSingleMarker.h>
#include "VICapture.h"
#include <OpenThreads/Thread>
#include <osg/Matrix>
#include <iostream>
#include "DataBuffer.h"

#ifndef WIN32
#include "ElapseTimer.h"
#endif

int gettimeofday(struct timeval *tp, int * /*tz*/);

class AugmentedRealityData
{
public:
    AugmentedRealityData();
    AugmentedRealityData(const AugmentedRealityData &obj);
    virtual ~AugmentedRealityData();

    bool isBCHEnabled();
    std::string getCameraConfig();
    std::string getVideoConfig();
    std::string getVideoConfigRight();
    bool isRemoteAREnabled();
    bool isARCaptureEnabled();
    int getThreshold();
    bool isAutoThresholdEnabled();
    int getAutoThresholdRetries();
    float getMarkerBorderWidth();
    float getCF();
    Size getImageSize();

    void setBCHEnabled(bool enabled);
    void setCameraConfig(std::string config);
    void setVideoConfig(std::string config);
    void setVideoConfigRight(std::string config);
    void setRemoteAREnabled(bool enabled);
    void setARCaptureEnabled(bool enabled);
    void setThreshold(int value);
    void setAutoThresholdEnabled(bool enabled);
    void setAutoThresholdRetries(int retries);
    void setMarkerBorderWidth(float width);
    void setImageSize(Size value);
    void setCF(float cf);

private:
    bool m_bchEnabled;
    bool m_RemoteAREnabled;
    bool m_captureEnabled;
    bool m_autoThresholdEnabled;
    int m_threshold;
    int m_thresholdRetries;
    float m_markerBorderWidth;
    int m_cf;
    std::string m_cameraConfig;
    std::string m_videoConfig;
    std::string m_videoConfigRight;
    Size m_videoParam;
};

//Provide Logging class for ARToolkitPlus
class MyLogger : public ARToolKitPlus::Logger
{
    void artLog(const char *nStr)
    {
        printf(nStr);
    }
};

class ARCaptureThread : public OpenThreads::Thread, DataBufferListener
{
public:
    ARCaptureThread();
    virtual ~ARCaptureThread();

    ARToolKitPlus::TrackerSingleMarker *setupVideoAndAR(AugmentedRealityData &data);
    virtual void run();
    void stop();

    void updateData(AugmentedRealityData &data);

    void setARInstance(opencover::ARToolKit *instance);

    void dumpImage();

    virtual void update();

private:
    osg::Matrix calculateMatrix(int index);
    int averageFPS();
    unsigned int getMilliseconds();

    ARToolKitPlus::TrackerSingleMarker *m_tracker;
    VICapture m_vid;
    bool m_isARsetup;
    bool m_bQuit;
    bool m_isUpdating;
    opencover::ARToolKit *m_artInstance;

    struct perfDataType
    {
        unsigned int ms;
        int frames;
    };

#ifndef WIN32
    ElapseTimer *mTimer;
#endif

    perfDataType mPerfData;

    unsigned int mLastTime;

    /**
    * Contains the number of detected AR markers in a image.
    * The value is updated with every call of arDetectMarker in
    * ARToolKitPlus::preFrame()
    */
    int m_marker_num;

    /**
    * ARToolKit structure that contains the information of all detected
    * markers after a call of arDetectMarker in ARToolkitPlus::preFrame()
    * Is updated with every call to arDetectMarker
    */
    ARToolKitPlus::ARMarkerInfo *m_marker_info;

    /**
   * ARToolKitPlus logging class instance
   */
    MyLogger m_log;

    bool m_writeOnce;
};

#endif