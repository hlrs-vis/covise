#ifndef _ENNOVATISDEVICE_H
#define _ENNOVATISDEVICE_H

#include "cover/coVRMSController.h"
#include "cover/ui/SelectionList.h"
#include "ennovatis/json.h"
#include "ennovatis/rest.h"
#include "ennovatis/building.h"
#include "core/interfaces/IInfoboard.h"
#include <memory>

#include <cover/coBillboard.h>
#include <osg/Geode>
#include <osg/Group>
#include <osg/NodeVisitor>
#include <osg/Shape>
#include <osg/Vec4>
#include <osg/ref_ptr>
#include <osgText/Text>

struct CylinderColormap {
    CylinderColormap(const osg::Vec4 &max, const osg::Vec4 &min, const osg::Vec4 &def)
    : max(max), min(min), defaultColor(def)
    {}
    osg::Vec4 max;
    osg::Vec4 min;
    osg::Vec4 defaultColor;
};

struct CylinderAttributes {
    CylinderAttributes(const float &rad, const float &height, const CylinderColormap &colorMap)
    : radius(rad), height(height), colorMap(colorMap)
    {}

    CylinderAttributes(const float &rad, const float &height, const osg::Vec4 &maxCol, const osg::Vec4 &minCol,
                       const osg::Vec4 &defaultCol)
    : CylinderAttributes(rad, height, CylinderColormap(maxCol, minCol, defaultCol))
    {}
    float radius;
    float height;
    CylinderColormap colorMap;
};

class EnnovatisDevice {
public:
    EnnovatisDevice(const ennovatis::Building &building, std::shared_ptr<opencover::ui::SelectionList> channelList,
                    std::shared_ptr<ennovatis::rest_request> req, std::shared_ptr<ennovatis::ChannelGroup> channelGroup,
                    std::unique_ptr<core::interface::IInfoboard<std::string>> &&infoBoard, const CylinderAttributes &cylinderAttributes);

    void update();
    void activate();
    void disactivate();
    void setChannelGroup(std::shared_ptr<ennovatis::ChannelGroup> group);
    void setTimestep(int timestep) { updateColorByTime(timestep); }
    [[nodiscard]] bool getStatus() const { return m_InfoVisible; }
    [[nodiscard]] const auto &getBuildingInfo() const { return m_buildingInfo; }
    [[nodiscard]] osg::ref_ptr<osg::Group> getDeviceGroup() { return m_deviceGroup; }

private:
    struct BuildingInfo {
        BuildingInfo(const ennovatis::Building *b): building(b) {}
        const ennovatis::Building *building;
        std::vector<std::string> channelResponse;
    };
    typedef osg::Vec4 TimestepColor;
    typedef std::vector<TimestepColor> TimestepColorList;

    void init();
    void fetchData();
    void updateChannelSelectionList();
    void setChannel(int idx);
    void updateColorByTime(int timestep);
    void createTimestepColorList(const ennovatis::json_response_object &j_resp_obj);
    [[nodiscard]] auto getCylinderGeode();
    [[nodiscard]] int getSelectedChannelIdx() const;
    [[nodiscard]] osg::Vec4 getColor(float val, float max) const;
    [[nodiscard]] osgText::Text *createTextBox(const std::string &text, const osg::Vec3 &position, int charSize,
                                               const char *fontFile) const;

    osg::ref_ptr<osg::Group> m_deviceGroup = nullptr;
    std::unique_ptr<core::interface::IInfoboard<std::string>> m_infoBoard;
    std::weak_ptr<ennovatis::rest_request> m_request;
    std::weak_ptr<ennovatis::ChannelGroup> m_channelGroup;
    std::weak_ptr<opencover::ui::SelectionList> m_channelSelectionList;

    bool m_InfoVisible = false;
    BuildingInfo m_buildingInfo;
    ennovatis::rest_request_handler m_rest_worker;
    opencover::coVRMSController *m_opncvr_ctrl; // cannot be const because syncing methods are not const correct
    TimestepColorList m_timestepColors;
    CylinderAttributes m_cylinderAttributes;
};
#endif
