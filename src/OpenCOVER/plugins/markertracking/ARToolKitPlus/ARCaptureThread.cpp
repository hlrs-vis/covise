#include "ARCaptureThread.h"
#include "DataBuffer.h"
#include <ARToolKitPlus/TrackerSingleMarkerImpl.h>

using namespace std;
using namespace opencover;

ARCaptureThread::ARCaptureThread()
{
    m_bQuit = false;
    m_marker_num = 0;
    m_isARsetup = false;
    m_isUpdating = false;
    m_tracker = NULL;
    m_marker_info = NULL;
    m_writeOnce = true;
    mPerfData.frames = 1;
    mPerfData.ms = 0;
    mLastTime = 0;
}

ARCaptureThread::~ARCaptureThread()
{
    m_tracker->cleanup();
    delete m_tracker;
    m_tracker = NULL;
}

void ARCaptureThread::setARInstance(ARToolKit *instance)
{
    m_artInstance = instance;
}

void ARCaptureThread::dumpImage()
{
    m_writeOnce = false;
}

ARToolKitPlus::TrackerSingleMarker *ARCaptureThread::setupVideoAndAR(AugmentedRealityData &data)
{
    if (!m_vid.init(data.getVideoConfig(), true))
    {
        cerr << "ARCaptureThread::setupVideoAndAR: Setup of capture device failed!" << endl;
        m_isARsetup = false;
        return NULL;
    }

    Size vidSize = m_vid.inqCameraImageSize();
    data.setImageSize(vidSize);

    // Setup ARToolkitPlus Tracker
    m_tracker = new ARToolKitPlus::TrackerSingleMarkerImpl<6, 6, 6, 1, 8>(vidSize.x, vidSize.y);
//m_tracker = new ARToolKitPlus::TrackerSingleMarkerImpl<6,6,32,4,6>(320,240);
#ifdef _DEBUG
    cerr << "ARCaptureThread::setupVideoAndAR(): Capture-size" << endl;
    cerr << "ARCaptureThread::setupVideoAndAR(): W = " << vidSize.x << endl;
    cerr << "ARCaptureThread::setupVideoAndAR(): H = " << vidSize.y << endl;
#endif

    m_tracker->setLogger(&m_log);
    m_tracker->setPixelFormat(ARToolKitPlus::PIXEL_FORMAT_BGR);

    // ARToolKitPlus setup, get a description
    const char *description = m_tracker->getDescription();
    cerr << "ARCaptureThread::setupVideoAndAR(): ARToolKitPlus compile-time information:" << endl;
    cerr << "ARCaptureThread::setupVideoAndAR():" << description << endl;

    // Init ARToolKitPlus with camera data file, set near and far clip plane
    // for calculation of matrices and pass logging class
    m_tracker->init(data.getCameraConfig().c_str(), 1.0f, 1000.0f);

    //Possibly not needed as we pass the pattern size once we call the
    //arGetTransMat function
    m_tracker->setPatternWidth(50);
    m_tracker->setBorderWidth(data.getMarkerBorderWidth());

    //Set image recognition thresholdq
    m_tracker->setThreshold(data.getThreshold());

    /*if (vidSize.x <=1024 && vidSize.y <= 1024)
   {*/
    m_tracker->setUndistortionMode(ARToolKitPlus::UNDIST_LUT);
    //}

    // switch to simple ID based markers
    // use the tool in tools/IdPatGen to generate markers
    m_tracker->setMarkerMode(data.isBCHEnabled() ? ARToolKitPlus::MARKER_ID_BCH : ARToolKitPlus::MARKER_ID_SIMPLE);
    //m_tracker->setMarkerMode(ARToolKitPlus::MARKER_ID_BCH);

    //Init DataBuffer's buffers
    DataBuffer::getInstance()->initBuffers(vidSize.x, vidSize.y, ARToolKitPlus::PIXEL_FORMAT_BGR);
    //m_vid.setExternalVideoBuffer(DataBuffer::getInstance()->getBackBufferPointer());
    DataBuffer::getInstance()->addListener(this);

    /*m_tracker->setNumAutoThresholdRetries(data.getAutoThresholdRetries());
   m_tracker->activateAutoThreshold(data.isAutoThresholdEnabled());*/
    cerr << "ARCaptureThread::setupVideoAndAR(): Auto-Thresholding active: " << m_tracker->isAutoThresholdActivated() << endl;
    cerr << "ARCaptureThread::setupVideoAndAR(): Threshold-Value = " << m_tracker->getThreshold() << endl;

    if (m_tracker != NULL)
    {
        m_isARsetup = true;
    }

    //cerr << "ARCaptureThread::setupVideoAndAR():Image size = " <<  data.getImageSize().x << endl;

    //m_tracker->setImageProcessingMode(ARToolKitPlus::IMAGE_FULL_RES);

    return m_tracker;
}

void ARCaptureThread::update()
{
    //m_vid.setExternalVideoBuffer(DataBuffer::getInstance()->getBackBufferPointer());
}

void ARCaptureThread::run()
{
    // leave thread if AR is not correctly setup
    // or the ARTP object is NULL
    if (!m_isARsetup || m_tracker == NULL)
    {
        return;
    }

    unsigned char *dataPtr = NULL;

    while (!m_bQuit)
    {
        //Do videoInput capture and matrix calculation here
        //Then update the DataBuffer accordingly and swap
        /* grab a vide frame */

        //Swap buffers in DataBuffer
        //while(!(DataBuffer::getInstance()->swapBuffers()))
        while (!m_vid.getVideoParameter()->getVIInstance()->isFrameNew(m_vid.getVideoParameter()->getDeviceNumber()))
        {
            _sleep(5);
        }
        while (!DataBuffer::getInstance()->swapBuffers())
        {
            _sleep(1);
        }
        m_vid.setExternalVideoBuffer(DataBuffer::getInstance()->getBackBufferPointer());

        if ((dataPtr = m_vid.getImage()) == NULL)
        {
            return;
        }

#ifdef WIN32
        unsigned int actualTime = getMilliseconds();
        mPerfData.ms += actualTime - mLastTime;
        mPerfData.frames++;
        mLastTime = actualTime;
#else
        unsigned int actualTime = getMilliseconds();
        mPerfData.ms += actualTime;
        mPerfData.frames++;
#endif
        if (mPerfData.ms > 1000)
        {
            cerr << "Avg camera FPS: " << averageFPS() << endl;
            mPerfData.ms = 0;
            mPerfData.frames = 0;
        }

        if (!m_writeOnce)
        {
            FILE *fh = fopen("f:\\image.raw", "wb");
            fwrite(dataPtr, sizeof(char), m_vid.getVideoBufferSize(), fh);
            fclose(fh);
            m_writeOnce = true;
        }

        //Copy image to DataBuffer here
        // REMARK: No copying needed anymore as we directly pass the
        // DataBuffer image backbuffer pointer to videoInput for capturing
        //DataBuffer::getInstance()->copyImage((const char*)dataPtr);
        DataBuffer::getInstance()->resetCF();

        while (m_isUpdating)
        {
            Sleep(1);
        }

        if (!m_bQuit && m_tracker->arDetectMarker(dataPtr, m_tracker->getThreshold(), &m_marker_info, &m_marker_num) != 0)
        {
            cerr << "ARCaptureThread::run(): arDetectMarker failed!" << endl;
            cerr << "Marker ID = " << m_marker_info->id << endl;
            cerr << "Marker CF = " << m_marker_info->cf << endl;
            cerr << "Marker Pos = (" << m_marker_info->pos[0] << "," << m_marker_info->pos[1] << ")" << endl;

            //There was a failure quit ARPlugin
            m_bQuit = true;
        }
        else
        {
            //#ifdef _DEBUG
            //         cerr << "ARCaptureThread::run(): arDetectMarker succeeded!" << endl;
            //         cerr << "ARCaptureThread::run(): Number of detected markers: " << m_marker_num << endl;
            //         cerr << "ARCaptureThread::run(): ------------------------------- " << endl;
            //#endif
            //Traverse marker_info to update matrix list
            for (int i = 0; i < m_marker_num; i++)
            {
                //#ifdef _DEBUG
                //            cerr << "ARCaptureThread::run(): Marker ID  = " << m_marker_info[i].id << endl;
                //            cerr << "ARCaptureThread::run(): Confidence = " << m_marker_info[i].cf << endl;
                //            cerr << "ARCaptureThread::run(): -------------------" << endl;
                //#endif
                if (!(m_marker_info[i].id == -1))
                {
                    osg::Matrix matrix = calculateMatrix(i);
                    DataBuffer::getInstance()->updateMatrix(m_marker_info[i].id, &matrix, m_marker_info[i].cf);
                }
            }
        } // end if arDetectMarker

    } // end while(!m_bQiut)
}

void ARCaptureThread::stop()
{
    if (isRunning())
    {
        m_bQuit = true;
    }
}

osg::Matrix ARCaptureThread::calculateMatrix(int index)
{
    osg::Matrix camera_transform;
    camera_transform.makeIdentity();

    if (m_bQuit)
    {
        return camera_transform;
    }
    float pattTrans[3][4];

    // For ARToolKitPlus we have to do marker calculation
    // here as we need the pattern id to detect a sepcific pattern

    // TODO: Fix center[] variable here
    // As the marker is visible (detected) we retrieve the AR transformation
    // matrix
    MarkerData *data = DataBuffer::getInstance()->getMarkerData(m_marker_info[index].id);
    if (!data)
    {
        cerr << "ARCaptureThread::calculateMatrix(): MarkerData not found!" << endl;
        return camera_transform;
    }
    ARFloat center[2];
    //double* data_center = data->getPattCenter();
    /*if (data_center == NULL)
   {*/
    center[0] = 0.0f;
    center[1] = 0.0f;
    /*}
   center[0] = data_center[0];
   center[1] = data_center[1];*/
    /*ARFloat arPattTrans[3][4];
   for (int i=0; i <3; i++)
   {
      for(int j=0; j <4; j++)
      {
         arPattTrans[i][j] = pattTrans[i][j];
      }
   }*/

    m_tracker->arGetTransMat(&m_marker_info[index], center, data->getPattSize(), pattTrans);
#if 0
   cerr << "ARToolKitPlusPlugin::getMat(): Confidence of marker detection = " << m_marker_info[index].cf << endl;
   cerr << "ARToolKitPlusPlugin::getMat(): MarkerInfo! ";
   cerr << "ID = " << m_marker_info[index].id << endl;
#endif
    //int detectedPattern = m_tracker->calc(MarkerTracking::instance()->videoData, pattID,true,&loc_markerInfo);
    //float confidenceValue = m_tracker->getConfidence();

    //const ARFloat* oglMatrix = m_tracker->getModelViewMatrix();

    osg::Matrix OpenGLMatrix;
    OpenGLMatrix.makeIdentity();

    // Convert the AR matrix into an OpenGL matrix (row/column order)
    int u, v;
    for (u = 0; u < 3; u++)
    {
        for (v = 0; v < 4; v++)
        {
            OpenGLMatrix(v, u) = pattTrans[u][v];
        }
    }

    //Changed matrix assignment using OpenGL system
    //int u,v;
    //for(u=0;u<4;u++)
    //{
    //   for(v=0;v<4;v++)
    //   {
    //      OpenGLMatrix(u,v)=oglMatrix[v+u];
    //   }
    //}

    osg::Matrix switchMatrix;
    //OpenGLMatrix.print(1,1,"OpenGLMatrix: ",stderr);
    switchMatrix.makeIdentity();
    switchMatrix(0, 0) = -1;
    switchMatrix(1, 1) = 0;
    switchMatrix(2, 2) = 0;
    switchMatrix(1, 2) = 1;
    switchMatrix(2, 1) = -1;
    camera_transform = OpenGLMatrix * switchMatrix;
    return camera_transform;
}

void ARCaptureThread::updateData(AugmentedRealityData &data)
{
    cerr << "ARCaptureThread::updateData(): ----------------------------------" << endl;
    m_isUpdating = true;
    m_tracker->activateAutoThreshold(data.isAutoThresholdEnabled());
    cerr << "ARCaptureThread::updateData(): AutoThreshold = " << data.isAutoThresholdEnabled() << endl;
    m_tracker->setBorderWidth(data.getMarkerBorderWidth());
    cerr << "ARCaptureThread::updateData(): BorderWidth = " << data.getMarkerBorderWidth() << endl;
    m_tracker->setMarkerMode(data.isBCHEnabled() ? ARToolKitPlus::MARKER_ID_BCH : ARToolKitPlus::MARKER_ID_SIMPLE);
    cerr << "ARCaptureThread::updateData(): BCH = " << data.isBCHEnabled() << endl;
    m_tracker->setNumAutoThresholdRetries(data.getAutoThresholdRetries());
    cerr << "ARCaptureThread::updateData(): AutoThresholdRetries = " << data.getAutoThresholdRetries() << endl;
    m_tracker->setThreshold(data.getThreshold());
    cerr << "ARCaptureThread::updateData(): Threshold = " << data.getThreshold() << endl;
    m_isUpdating = false;
    cerr << "ARCaptureThread::updateData(): ----------------------------------" << endl;
}

int ARCaptureThread::averageFPS()
{
    return 1000 / (mPerfData.ms / mPerfData.frames);
}

unsigned int ARCaptureThread::getMilliseconds()
{
#ifdef WIN32
    LARGE_INTEGER cpu_ticks;
    LARGE_INTEGER ticks_p_sec;
    QueryPerformanceCounter(&cpu_ticks);
    QueryPerformanceFrequency(&ticks_p_sec);

    __int64 ms = cpu_ticks.QuadPart / ticks_p_sec.QuadPart * 1000;
    unsigned int milliseconds = (unsigned int)(ms & 0xffffffff);

    return milliseconds;
#else
    double ms = mTimer->getElapsedMilliSeconds(true);
    return (unsigned int)ms;

#endif
}

int gettimeofday(struct timeval *tp, int * /*tz*/)
{
#if defined(_WIN32_WCE)
    /* FILETIME of Jan 1 1970 00:00:00. */
    static const unsigned __int64 epoch = 116444736000000000LL;

    FILETIME file_time;
    SYSTEMTIME system_time;
    ULARGE_INTEGER ularge;

    GetSystemTime(&system_time);
    SystemTimeToFileTime(&system_time, &file_time);
    ularge.LowPart = file_time.dwLowDateTime;
    ularge.HighPart = file_time.dwHighDateTime;

    tp->tv_sec = (long)((ularge.QuadPart - epoch) / 10000000L);
    tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
#else
    static LARGE_INTEGER tickFrequency, epochOffset;

    // For our first call, use "ftime()", so that we get a time with a proper epoch.
    // For subsequent calls, use "QueryPerformanceCount()", because it's more fine-grain.
    static bool isFirstCall = true;

    LARGE_INTEGER tickNow;
    QueryPerformanceCounter(&tickNow);

    if (isFirstCall)
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        tp->tv_sec = ts.tv_sec;
        tp->tv_usec = ts.tv_nsec/1000;

        // Also get our counter frequency:
        QueryPerformanceFrequency(&tickFrequency);

        // And compute an offset to add to subsequent counter times, so we get a proper epoch:
        epochOffset.QuadPart
            = tb.time * tickFrequency.QuadPart + (tb.millitm * tickFrequency.QuadPart) / 1000 - tickNow.QuadPart;

        isFirstCall = false; // for next time
    }
    else
    {
        // Adjust our counter time so that we get a proper epoch:
        tickNow.QuadPart += epochOffset.QuadPart;

        tp->tv_sec = (long)(tickNow.QuadPart / tickFrequency.QuadPart);
        tp->tv_usec = (long)(((tickNow.QuadPart % tickFrequency.QuadPart) * 1000000L) / tickFrequency.QuadPart);
    }
#endif
    return 0;
}

//******************************************************************************
// Implementation of AugmentedRealityData
//******************************************************************************
AugmentedRealityData::AugmentedRealityData()
{
    m_bchEnabled = true;
    m_RemoteAREnabled = false;
    m_captureEnabled = false;
    m_autoThresholdEnabled = true;
    m_thresholdRetries = 40;
    m_markerBorderWidth = 0.125f;
    m_cameraConfig = "";
    m_videoConfig = "";
    m_videoConfigRight = "";
}
AugmentedRealityData::AugmentedRealityData(const AugmentedRealityData &obj)
{
    m_bchEnabled = obj.m_bchEnabled;
    m_RemoteAREnabled = obj.m_RemoteAREnabled;
    m_captureEnabled = obj.m_captureEnabled;
    m_autoThresholdEnabled = obj.m_autoThresholdEnabled;
    m_thresholdRetries = obj.m_thresholdRetries;
    m_markerBorderWidth = obj.m_markerBorderWidth;
    m_cameraConfig = obj.m_cameraConfig;
    m_videoConfig = obj.m_videoConfig;
    m_videoConfigRight = obj.m_videoConfigRight;
    m_videoParam = obj.m_videoParam;
}
AugmentedRealityData::~AugmentedRealityData()
{
}

bool AugmentedRealityData::isBCHEnabled()
{
    return m_bchEnabled;
}
std::string AugmentedRealityData::getCameraConfig()
{
    return m_cameraConfig;
}
std::string AugmentedRealityData::getVideoConfig()
{
    return m_videoConfig;
}
std::string AugmentedRealityData::getVideoConfigRight()
{
    return m_videoConfigRight;
}
bool AugmentedRealityData::isRemoteAREnabled()
{
    return m_RemoteAREnabled;
}
bool AugmentedRealityData::isARCaptureEnabled()
{
    return m_captureEnabled;
}
int AugmentedRealityData::getThreshold()
{
    return m_threshold;
}
bool AugmentedRealityData::isAutoThresholdEnabled()
{
    return m_autoThresholdEnabled;
}
int AugmentedRealityData::getAutoThresholdRetries()
{
    return m_thresholdRetries;
}
float AugmentedRealityData::getMarkerBorderWidth()
{
    return m_markerBorderWidth;
}

float AugmentedRealityData::getCF()
{
    return m_cf;
}

Size AugmentedRealityData::getImageSize()
{
    return m_videoParam;
}

void AugmentedRealityData::setBCHEnabled(bool enabled)
{
    m_bchEnabled = enabled;
}
void AugmentedRealityData::setCameraConfig(std::string config)
{
    m_cameraConfig = config;
}
void AugmentedRealityData::setVideoConfig(std::string config)
{
    m_videoConfig = config;
}
void AugmentedRealityData::setVideoConfigRight(std::string config)
{
    m_videoConfigRight = config;
}
void AugmentedRealityData::setRemoteAREnabled(bool enabled)
{
    m_RemoteAREnabled = enabled;
}
void AugmentedRealityData::setARCaptureEnabled(bool enabled)
{
    m_captureEnabled = enabled;
}
void AugmentedRealityData::setThreshold(int value)
{
    m_threshold = value;
}
void AugmentedRealityData::setAutoThresholdEnabled(bool enabled)
{
    m_autoThresholdEnabled = enabled;
}
void AugmentedRealityData::setAutoThresholdRetries(int retries)
{
    m_thresholdRetries = retries;
}
void AugmentedRealityData::setMarkerBorderWidth(float width)
{
    m_markerBorderWidth = width;
}

void AugmentedRealityData::setCF(float cf)
{
    m_cf = cf;
}

void AugmentedRealityData::setImageSize(Size value)
{
    m_videoParam = value;
}
