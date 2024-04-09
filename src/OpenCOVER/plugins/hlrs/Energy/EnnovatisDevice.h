#ifndef _ENNOVATISDEVICE_H
#define _ENNOVATISDEVICE_H

#include "cover/ui/SelectionList.h"
#include "ennovatis/rest.h"
#include "ennovatis/building.h"
#include <memory>

#include <cover/coBillboard.h>
#include <osg/Group>
#include <osgText/Text>

class EnnovatisDevice {
public:
    EnnovatisDevice(const ennovatis::Building &building, std::shared_ptr<opencover::ui::SelectionList> channelList,
                    std::shared_ptr<ennovatis::rest_request> req, std::shared_ptr<ennovatis::ChannelGroup> channelGroup);
    [[nodiscard]] bool getStatus() { return m_InfoVisible; }
    [[nodiscard]] osg::ref_ptr<osg::Group> getDeviceGroup() { return m_deviceGroup; }
    void update();
    void activate();
    void disactivate();
    void setChannelGroup(std::shared_ptr<ennovatis::ChannelGroup> group);

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
    void updateChannelSelectionList();
    int getSelectedChannelIdx();
    void initBillboard();
    void setChannel(int idx);
    osgText::Text* createTextBox(const std::string& text, const osg::Vec3& position, int charSize, const char* fontFile);

    osg::ref_ptr<osg::Group> m_TextGeode = nullptr;
    osg::ref_ptr<osg::Group> m_deviceGroup = nullptr;
    osg::ref_ptr<opencover::coBillboard> m_BBoard = nullptr;
    std::weak_ptr<ennovatis::rest_request> m_request;
    std::weak_ptr<ennovatis::ChannelGroup> m_channelGroup;
    std::weak_ptr<opencover::ui::SelectionList> m_channelSelectionList;

    float m_rad;
    bool m_InfoVisible = false;
    BuildingInfo m_buildingInfo;
    ennovatis::rest_request_handler m_rest_worker;
};
#endif
