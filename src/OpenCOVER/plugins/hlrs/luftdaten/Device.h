/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef _Luft_Device_H
#define _Luft_Device_H
/****************************************************************************\
 **                                                          (C)2020 HLRS  **
 **                                                                        **
 ** Description: OpenCOVER Plug-In for reading Luftdaten sensor data       **
 **                                                                        **
 **                                                                        **
 ** Author: Leyla Kern                                                     **
 **                                                                        **
 ** History:                                                               **
 ** April 2020  v1                                                         **
 **                                                                        **
 **                                                                        **
\****************************************************************************/
#include <util/common.h>
#include <cover/coVRPluginSupport.h>

#include <osgText/Text>
#include <osg/Geode>
#include <osg/Group>
#include <osg/ShapeDrawable>
#include <cover/coBillboard.h>

class DeviceSensor;

class DeviceInfo
{
public:
    DeviceInfo(){};
    ~DeviceInfo(){};
    float lat;
    float lon;
    float height = 0.f;
    float pm10 = -1.f;
    float pm2 = -1.f;
    float humi = -1.f;
    float temp = -100.f;
    std::string ID;
    std::string time;
    int timestep = -1;
    std::string name = "NONE";
};

class Device
{
public:
    Device(DeviceInfo * d, osg::Group *parent);
    ~Device();
    void init(float r, float sH,int c);
    
    DeviceInfo *devInfo;
    DeviceSensor *devSensor;

    void update();
    void activate();
    void disactivate();
    void showInfo();
    void showGraph();
    bool getStatus()
    {
        return InfoVisible;
    }
        
    osg::ref_ptr<osg::Group> myParent;
    osg::ref_ptr<osg::Group> TextGeode;
    osg::ref_ptr<opencover::coBillboard> BBoard;
    osg::ref_ptr<osg::Group> deviceGroup;
    osg::ref_ptr<osg::Geode> geoBars;
private:
    float h = 1.f;
    float w = 2.f;
    float rad;
    bool InfoVisible = false;
};
#endif

