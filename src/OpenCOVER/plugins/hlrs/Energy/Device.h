/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef _Energy_Device_H
#define _Energy_Device_H

#include <cover/coVRPluginSupport.h>
#include <util/common.h>

#include <cover/coBillboard.h>
#include <osg/Geode>
#include <osg/Group>
#include <osg/ShapeDrawable>
#include <osgText/Text>

class DeviceSensor;

class DeviceInfo
{
public:
    DeviceInfo(){};
    DeviceInfo(DeviceInfo &d)
    {
        lat = d.lat;
        lon = d.lon;
        height = d.height;
        strom = d.strom;
        kaelte = d.kaelte;
        waerme = d.waerme;
        baujahr = d.baujahr;
        flaeche = d.flaeche;
        name = d.name;
        ID = d.ID;
        strasse = d.strasse;
    }
    ~DeviceInfo(){};
    float lat;
    float lon;
    float height = 0.f;
    float strom = 0.f;
    float waerme = 0.f;
    float kaelte = 0.f;
    float flaeche = 0;
    float baujahr = 0;
    std::string strasse;
    std::string ID;
    int timestep = -1;
    std::string name = "NONE";
};

class Device
{
public:
    Device(DeviceInfo *d, osg::Group *parent);
    ~Device();
    void init(float r, float sH, int c);

    DeviceInfo *devInfo;
    DeviceSensor *devSensor;

    void update();
    void activate();
    void disactivate();
    void showInfo();
    bool getStatus() { return InfoVisible; }
    osg::Vec4 getColor(float val, float max);

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
