#ifndef _DATABUFFER_H
#define _DATABUFFER_H

#include <map>
#include <osg/MatrixTransform>
#include <ARToolKitPlus/ARToolKitPlus.h>
#include <OpenThreads/Mutex>

class ARToolKitPlusPlugin;

class MarkerData
{
public:
    MarkerData();
    virtual ~MarkerData();

    int getPattID();
    double getPattSize();
    double *getPattCenter();
    osg::Matrix &getMatrix();
    float getCF();
    bool isVisible();

    void setPattID(int id);
    void setPattSize(double size);
    void setPattCenter(double center[2]);
    void setMatrix(osg::Matrix &matrix);
    void setCF(float cf);
    void visible(bool visible);

private:
    int m_pattID;
    double m_pattCenter[2];
    double m_pattSize;
    osg::Matrix m_matrix;
    float m_cf;
    bool m_visible;
};

class DataBufferListener
{
public:
    DataBufferListener(){};
    virtual ~DataBufferListener(){};

    virtual void update() = 0;
};

class DataBuffer : public OpenThreads::Mutex
{
    friend class ARCaptureThread;

public:
    static DataBuffer *getInstance();
    virtual ~DataBuffer();
    unsigned char *getImagePointer();
    osg::Matrix *getMarkerMatrix(int pattID);
    void addMarker(MarkerData &data);
    bool isVisible(int pattID);
    void addListener(DataBufferListener *listener);
    void delListener(DataBufferListener *listener);
    void lockFront();
    void unlockFront();

protected:
    DataBuffer();
    bool swapBuffers();
    void initBuffers(int width, int height, ARToolKitPlus::PIXEL_FORMAT format = ARToolKitPlus::PIXEL_FORMAT_BGR);
    void copyImage(const char *image);
    unsigned char *getBackBufferPointer();
    void updateMatrix(int pattID, osg::Matrix *matrix, float cf);
    MarkerData *getMarkerData(int id);
    void resetCF();

private:
    //FrontBuffer objects for external read-only access
    std::map<int, MarkerData *> *m_frontMatrixList;
    std::vector<DataBufferListener *> m_listeners;
    unsigned char *m_frontBuffer;

    bool m_isLocked;
    OpenThreads::Mutex m_mutex;

    //Shadow members for concurrent access of threads
    std::map<int, MarkerData *> *m_backMatrixList;
    unsigned char *m_backBuffer;

    //Data characteristics
    int m_imgWidth;
    int m_imgHeight;
    int m_bpp;
    static DataBuffer *m_instance;
};

#endif