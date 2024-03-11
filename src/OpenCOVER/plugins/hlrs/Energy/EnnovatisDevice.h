#ifndef _ENNOVATISDEVICE_H
#define _ENNOVATISDEVICE_H

#include "ennovatis/rest.h"
#include "ennovatis/building.h"
#include <cover/coVRPluginSupport.h>
#include <memory>
#include <util/common.h>

#include <cover/coBillboard.h>
#include <osg/Geode>
#include <osg/Group>
#include <osg/ShapeDrawable>
#include <osgText/Text>

class EnnovatisDevice {
public:
    EnnovatisDevice(const ennovatis::Building &building): EnnovatisDevice(building, nullptr){};
    EnnovatisDevice(const ennovatis::Building &building, std::shared_ptr<ennovatis::rest_request> req);
    ~EnnovatisDevice();
    [[nodiscard]] bool getStatus() { return m_InfoVisible; }
    [[nodiscard]] osg::ref_ptr<osg::Group> getDeviceGroup() { return m_deviceGroup; }
    void update();
    void activate();
    void disactivate();
    void setChannelGroup(std::shared_ptr<ennovatis::ChannelGroup> group) { m_channelGroup = group; }

private:
    struct BuildingInfo {
        BuildingInfo(const ennovatis::Building *b): building(b) {}
        const ennovatis::Building *building;
        std::vector<std::string> channelResponse;
    };
    [[nodiscard]] osg::Vec4 getColor(float val, float max);
    void init(float r);
    void showInfo();
    void fetchData();

    osg::ref_ptr<osg::Group> m_TextGeode;
    osg::ref_ptr<osg::Group> m_deviceGroup;
    osg::ref_ptr<osg::Geode> m_geoBars;
    osg::ref_ptr<opencover::coBillboard> m_BBoard;

    float m_rad;
    bool m_InfoVisible = false;
    std::shared_ptr<ennovatis::rest_request> m_request;
    std::shared_ptr<ennovatis::ChannelGroup> m_channelGroup;
    BuildingInfo m_buildingInfo;
};
#endif
