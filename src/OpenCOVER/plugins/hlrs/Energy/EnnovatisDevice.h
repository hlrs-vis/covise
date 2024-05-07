#ifndef _ENNOVATISDEVICE_H
#define _ENNOVATISDEVICE_H

// core
#include "core/interfaces/IInfoboard.h"
#include "core/interfaces/IBuilding.h"

// ennovatis
#include "ennovatis/json.h"
#include "ennovatis/rest.h"
#include "ennovatis/building.h"

// cover
#include <cover/coBillboard.h>
#include "cover/coVRMSController.h"
#include "cover/ui/SelectionList.h"

// osg
#include <osg/Geode>
#include <osg/Group>
#include <osg/NodeVisitor>
#include <osg/Shape>
#include <osg/Vec4>
#include <osg/ref_ptr>
#include <osgText/Text>

// std
#include <memory>

class EnnovatisDevice {
public:
    EnnovatisDevice(const ennovatis::Building &building, std::shared_ptr<opencover::ui::SelectionList> channelList,
                    std::shared_ptr<ennovatis::rest_request> req, std::shared_ptr<ennovatis::ChannelGroup> channelGroup,
                    std::unique_ptr<core::interface::IInfoboard<std::string>> &&infoBoard,
                    std::unique_ptr<core::interface::IBuilding> &&drawableBuilding);

    void update();
    void activate();
    void disactivate();
    void setChannelGroup(std::shared_ptr<ennovatis::ChannelGroup> group);
    void setTimestep(int timestep) { updateColorByTime(timestep); }
    [[nodiscard]] const auto &getBuildingInfo() const { return m_buildingInfo; }
    [[nodiscard]] osg::ref_ptr<osg::Group> getDeviceGroup() { return m_deviceGroup; }

private:
    struct BuildingInfo {
        BuildingInfo(const ennovatis::Building *b): building(b) {}
        const ennovatis::Building *building;
        std::vector<std::string> channelResponse;
    };
    typedef std::unique_ptr<osg::Vec4> TimestepColor;
    typedef std::vector<TimestepColor> TimestepColorList;

    void init();
    void fetchData();
    void updateChannelSelectionList();
    void setChannel(int idx);
    void updateColorByTime(int timestep);
    void createTimestepColorList(const ennovatis::json_response_object &j_resp_obj);
    void updateInfoboard(const std::string &info);
    [[nodiscard]] auto createBillboardTxt();
    [[nodiscard]] int getSelectedChannelIdx() const;

    osg::ref_ptr<osg::Group> m_deviceGroup = nullptr;
    std::unique_ptr<core::interface::IInfoboard<std::string>> m_infoBoard;
    std::unique_ptr<core::interface::IBuilding> m_drawableBuilding;
    std::weak_ptr<ennovatis::rest_request> m_request;
    std::weak_ptr<ennovatis::ChannelGroup> m_channelGroup;
    std::weak_ptr<opencover::ui::SelectionList> m_channelSelectionList;

    bool m_InfoVisible = false;
    BuildingInfo m_buildingInfo;
    ennovatis::rest_request_handler m_restWorker;
    opencover::coVRMSController *m_opncvrCtrl; // cannot be const because syncing methods are not const correct
    TimestepColorList m_timestepColors;
};
#endif
