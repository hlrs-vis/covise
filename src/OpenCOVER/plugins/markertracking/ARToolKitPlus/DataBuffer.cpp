#include "DataBuffer.h"
#include <iostream>
#include <assert.h>
#include <OpenThreads/Mutex>

using namespace std;

DataBuffer *DataBuffer::m_instance = NULL;

DataBuffer *DataBuffer::getInstance()
{
    if (m_instance == NULL)
    {
        m_instance = new DataBuffer();
    }
    return m_instance;
}

DataBuffer::DataBuffer()
{
    m_backBuffer = NULL;
    m_frontBuffer = NULL;
    m_isLocked = false;
    m_frontMatrixList = new map<int, MarkerData *>();
    m_backMatrixList = new map<int, MarkerData *>();
}

//DataBuffer::DataBuffer(const DataBuffer &obj)
//{
//   //TODO: This no good code, cause it is only pointer assignment
//   //Actually it should copy the contents behind the pointers
//   //in the local instance's own memory area
//   m_frontBuffer = obj.m_frontBuffer;
//   m_backBuffer = obj.m_backBuffer;
//   m_frontMatrixList = obj.m_frontMatrixList;
//   m_backMatrixList = obj.m_backMatrixList;
//}

DataBuffer::~DataBuffer()
{

    map<int, MarkerData *>::iterator itr;
    map<int, MarkerData *>::iterator itrf;

    itr = m_backMatrixList->begin();
    itrf = m_frontMatrixList->begin();

    //As both lists shoudl be conceptually of the same length we only check
    //bounds for one of them
    while (itr != m_backMatrixList->end() && !m_backMatrixList->empty())
    {
        delete (*itr).second;
        delete (*itrf).second;
        m_backMatrixList->erase(itr);
        m_frontMatrixList->erase(itrf);
        itr++;
        itrf++;
    }

    delete m_frontMatrixList;
    delete m_backMatrixList;

    if (m_backBuffer != NULL)
    {
        delete[] m_backBuffer;
    }

    if (m_frontBuffer != NULL)
    {
        delete[] m_frontBuffer;
    }
}

void DataBuffer::lockFront()
{
    while (m_isLocked)
    {
    };

    m_isLocked = true;
}
void DataBuffer::unlockFront()
{
    m_isLocked = false;
}

unsigned char *DataBuffer::getImagePointer()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(m_mutex);
    return m_frontBuffer;
}

osg::Matrix *DataBuffer::getMarkerMatrix(int pattID)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(m_mutex);
    return &(m_frontMatrixList->find(pattID)->second->getMatrix());
}

bool DataBuffer::swapBuffers()
{
    //THIS CODE SHOULD BE PROTECTED BY A MUTEX
    //Lock static Mutex

    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(m_mutex);

    if (m_isLocked)
    {
        return false;
    }

    //int r = this->lock();
    //assert(r == 0);

    //Swap images
    unsigned char *tempBuffer = m_frontBuffer;
    m_frontBuffer = m_backBuffer;
    m_backBuffer = tempBuffer;

    //Swap marker/matrix lists
    map<int, MarkerData *> *tempList = m_frontMatrixList;
    m_frontMatrixList = m_backMatrixList;
    m_backMatrixList = tempList;

   for
       each(DataBufferListener * d in m_listeners)
       {
           d->update();
       }

   return true;
   /*r = this->unlock();
   assert(r == 0);*/
}

void DataBuffer::addListener(DataBufferListener *listener)
{
    m_listeners.push_back(listener);
}

void DataBuffer::delListener(DataBufferListener *listener)
{
    vector<DataBufferListener *>::iterator itr;
    itr = m_listeners.begin();
    while (itr != m_listeners.end() && !m_listeners.empty())
    {
        if ((*itr) == listener)
        {
            m_listeners.erase(itr);
        }
    }
}

void DataBuffer::initBuffers(int width, int height, ARToolKitPlus::PIXEL_FORMAT format)
{
    m_bpp = 3; //default BGR;

    switch (format)
    {
    case ARToolKitPlus::PIXEL_FORMAT_BGR:
    case ARToolKitPlus::PIXEL_FORMAT_RGB:
    {
        m_bpp = 3;
    }
    break;
    case ARToolKitPlus::PIXEL_FORMAT_BGRA:
    case ARToolKitPlus::PIXEL_FORMAT_ABGR:
    case ARToolKitPlus::PIXEL_FORMAT_RGBA:
    {
        m_bpp = 4;
    }
    break;
    case ARToolKitPlus::PIXEL_FORMAT_LUM:
    {
        m_bpp = 1;
    }
    break;
    }

    m_imgWidth = width;
    m_imgHeight = height;
    int size = m_imgWidth * m_imgHeight * m_bpp;

    m_frontBuffer = new unsigned char[size];
    m_backBuffer = new unsigned char[size];
}
void DataBuffer::copyImage(const char *image)
{
    //memcpy(m_backBuffer,image, m_imgWidth*m_imgHeight*m_bpp);
    memcpy(m_backBuffer, image, m_imgWidth * m_imgHeight * m_bpp);
}

unsigned char *DataBuffer::getBackBufferPointer()
{
    return m_backBuffer;
}

void DataBuffer::updateMatrix(int pattID, osg::Matrix *matrix, float cf)
{
    //Locate matrix entry
    std::map<int, MarkerData *>::iterator itr;
    itr = m_backMatrixList->find(pattID);
    if (itr == m_backMatrixList->end())
    {
        //Not found in list thus not registered by OpenCOVER ==> reject
        std::cerr << "DataBuffer::updateMatrix(): Marker ID unknown --> rejected!" << std::endl;
        return;
    }
    MarkerData *locData = itr->second;

    if (cf >= locData->getCF())
    {
        memcpy(&(locData->getMatrix()), matrix, sizeof(osg::Matrix));
        locData->visible(true);
    }
}

void DataBuffer::addMarker(MarkerData &data)
{
    (*m_backMatrixList)[data.getPattID()] = &data;
    (*m_frontMatrixList)[data.getPattID()] = &data;
}

MarkerData *DataBuffer::getMarkerData(int id)
{
    map<int, MarkerData *>::iterator itr = m_backMatrixList->find(id);
    if (itr != m_backMatrixList->end())
    {
        return itr->second;
    }
    return NULL;
}

void DataBuffer::resetCF()
{
    map<int, MarkerData *>::iterator itr;
    itr = m_backMatrixList->begin();
    while (itr != m_backMatrixList->end() && m_backMatrixList->empty())
    {
        (*itr).second->setCF(0.0f);
        (*itr).second->visible(false);
    }
}

bool DataBuffer::isVisible(int pattID)
{
    map<int, MarkerData *>::iterator itr = m_frontMatrixList->find(pattID);
    if (itr == m_frontMatrixList->end())
    {
        std::cerr << "DataBuffer::isVisible(): pattern ID not found" << std::endl;
        return false;
    }

    return (*itr).second->isVisible();
}

//**************************************************************************

MarkerData::MarkerData()
{
    m_pattID = -1;
    m_pattSize = 50;
    m_pattCenter[0] = 0.0;
    m_pattCenter[1] = 0.0;
    m_visible = false;
    m_cf = 0.0f;
    m_matrix.identity();
}

MarkerData::~MarkerData()
{
}

int MarkerData::getPattID()
{
    return m_pattID;
}
double MarkerData::getPattSize()
{
    return m_pattSize;
}
double *MarkerData::getPattCenter()
{
    return m_pattCenter;
}
osg::Matrix &MarkerData::getMatrix()
{
    return m_matrix;
}
float MarkerData::getCF()
{
    return m_cf;
}

bool MarkerData::isVisible()
{
    return m_visible;
}

void MarkerData::setPattID(int id)
{
    m_pattID = id;
}
void MarkerData::setPattSize(double size)
{
    m_pattSize = size;
}
void MarkerData::setPattCenter(double center[2])
{
    m_pattCenter[0] = center[0];
    m_pattCenter[1] = center[1];
}
void MarkerData::setMatrix(osg::Matrix &matrix)
{
    m_matrix = matrix;
}

void MarkerData::setCF(float cf)
{
    m_cf = cf;
}

void MarkerData::visible(bool visible)
{
    m_visible = visible;
}