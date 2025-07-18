/* This file is part of COVISE.

  You can use it under the terms of the GNU Lesser General Public License
  version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef _Energy_Device_H
#define _Energy_Device_H

#include <cover/coBillboard.h>
#include <cover/coVRPluginSupport.h>
#include <util/common.h>

#include <osg/Geode>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osgText/Text>

namespace energy {

struct DeviceInfo {
 public:
  DeviceInfo() {};
  DeviceInfo(const DeviceInfo &info) {
    strasse = info.strasse;
    ID = info.ID;
    name = info.name;
    lat = info.lat;
    lon = info.lon;
    height = info.height;
    strom = info.strom;
    waerme = info.waerme;
    kaelte = info.kaelte;
    flaeche = info.flaeche;
    baujahr = info.baujahr;
  }
  std::string strasse;
  std::string ID;
  std::string name = "NONE";
  float lat = 0.0f;
  float lon = 0.0f;
  float height = 0.f;
  float strom = 0.f;
  float waerme = 0.f;
  float kaelte = 0.f;
  size_t flaeche = 0;
  size_t baujahr = 0;
  int timestep = -1;
};

class Device {
 public:
  Device(osg::ref_ptr<osg::MatrixTransform> node, const DeviceInfo &deviceInfo,
         const std::string &font);
  ~Device();
  Device(Device &&other) {
    m_devInfo = std::move(other.m_devInfo);
    m_font = std::move(other.m_font);
    m_height = other.m_height;
    m_width = other.m_width;
    m_rad = other.m_rad;
    m_txtGroup = other.m_txtGroup.release();
    m_BBoard = other.m_BBoard.release();
    m_node = other.m_node.release();
    m_geoBars = other.m_geoBars.release();
    m_infoVisible = other.m_infoVisible;
  }

  Device &operator=(Device &&other) {
    if (this != &other) {
      m_devInfo = std::move(other.m_devInfo);
      m_font = std::move(other.m_font);
      m_height = other.m_height;
      m_width = other.m_width;
      m_rad = other.m_rad;
      m_txtGroup = other.m_txtGroup.release();
      m_BBoard = other.m_BBoard.release();
      m_node = other.m_node.release();
      m_geoBars = other.m_geoBars.release();
      m_infoVisible = other.m_infoVisible;
    }
    return *this;
  }

  Device(const Device &other) {
    m_devInfo = other.m_devInfo;
    m_font = other.m_font;
    m_height = other.m_height;
    m_width = other.m_width;
    m_rad = other.m_rad;
    m_txtGroup = other.m_txtGroup;
    m_BBoard = other.m_BBoard;
    m_node = other.m_node;
    m_geoBars = other.m_geoBars;
    m_infoVisible = other.m_infoVisible;
  }

  Device &operator=(const Device &other) {
    if (this != &other) {
      m_devInfo = other.m_devInfo;
      m_font = other.m_font;
      m_height = other.m_height;
      m_width = other.m_width;
      m_rad = other.m_rad;
      m_txtGroup = other.m_txtGroup;
      m_BBoard = other.m_BBoard;
      m_node = other.m_node;
      m_geoBars = other.m_geoBars;
      m_infoVisible = other.m_infoVisible;
    }
    return *this;
  }

  void init(float r, float sH, int c);
  void update();
  void activate();
  void disactivate();
  void showInfo();
  bool getStatus() { return m_infoVisible; }
  const DeviceInfo &getInfo() const { return m_devInfo; }

 private:
  DeviceInfo m_devInfo;
  std::string m_font;
  float m_height;
  float m_width;
  float m_rad;
  osg::ref_ptr<osg::Group> m_txtGroup;
  osg::ref_ptr<opencover::coBillboard> m_BBoard;
  osg::ref_ptr<osg::MatrixTransform> m_node;
  osg::ref_ptr<osg::Geode> m_geoBars;
  bool m_infoVisible;
};
}  // namespace energy
#endif
