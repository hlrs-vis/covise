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

namespace energy {

struct DeviceInfo {
public:
    typedef std::shared_ptr<DeviceInfo> ptr;
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
    float lat;
    float lon;
    float height = 0.f;
    float strom = 0.f;
    float waerme = 0.f;
    float kaelte = 0.f;
    size_t flaeche = 0;
    size_t baujahr = 0;
    std::string strasse;
    std::string ID;
    int timestep = -1;
    std::string name = "NONE";
};

class Device {
public:
    typedef std::shared_ptr<Device> ptr;
    Device(DeviceInfo::ptr d, osg::ref_ptr<osg::Group> parent);
    ~Device();
    Device(const Device &other) = delete;
    Device &operator=(const Device &) = delete;
    void init(float r, float sH, int c);
    void update();
    void activate();
    void disactivate();
    void showInfo();
    bool getStatus() { return InfoVisible; }
    osg::Vec4 getColor(float val, float max);
    osg::ref_ptr<osg::Group> getGroup() { return deviceGroup; }
    const DeviceInfo::ptr getInfo() { return devInfo; }

private:
    DeviceInfo::ptr devInfo;
    osg::ref_ptr<osg::Group> myParent;
    osg::ref_ptr<osg::Group> TextGeode;
    osg::ref_ptr<opencover::coBillboard> BBoard;
    osg::ref_ptr<osg::Group> deviceGroup;
    osg::ref_ptr<osg::Geode> geoBars;
    float h = 1.f;
    float w = 2.f;
    float rad;
    bool InfoVisible = false;
};
} // namespace energy
#endif
