#ifndef COVER_TOOLMACHINE_TOOL_H
#define COVER_TOOLMACHINE_TOOL_H

#include <string>
#include <memory>
#include <map>

#include <cover/ui/Button.h>
#include <cover/ui/CovconfigLink.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Group.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Slider.h>

#include <osg/MatrixTransform>
#include <osg/Observer>
#include <PluginUtil/coColorMap.h>
#include <OpcUaClient/opcua.h>
#include <OpcUaClient/variantAccess.h>

struct UpdateValues{
    std::string name;
    std::function<void(double)> func;
};


class SelfDeletingTool;

class Tool{
public:
    friend SelfDeletingTool;
    Tool(opencover::ui::Group* group, opencover::config::File &file, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode);
    virtual ~Tool() = default;
    void update(const opencover::opcua::MultiDimensionalArray<double> &data);
    void pause(bool state);
    const std::vector<UpdateValues> &getUpdateValues();
protected:
    virtual void updateGeo(bool paused, const opencover::opcua::MultiDimensionalArray<double> &data) = 0;
    virtual void clear() = 0;
    virtual void applyShader(const covise::ColorMap& map, float min, float max) = 0;
    virtual std::vector<std::string> getAttributes() = 0;
    osg::Vec3 toolHeadInTableCoords();
    osg::MatrixTransform *m_toolHeadNode = nullptr;
    osg::MatrixTransform *m_tableNode = nullptr;
    std::unique_ptr<opencover::ui::Group> m_group;
    std::unique_ptr<opencover::ui::SliderConfigValue> m_numSectionsSlider;
    covise::ColorMapSelector *m_colorMapSelector;
    std::unique_ptr<opencover::ui::SelectionListConfigValue> m_attributeName;
    std::unique_ptr<opencover::ui::EditFieldConfigValue> m_minAttribute, m_maxAttribute;
    opencover::opcua::Client *m_client;
    opencover::opcua::ObserverHandle m_opcuaAttribId;
    bool m_paused = false;
    std::vector<UpdateValues> m_updateValues;

};

class SelfDeletingTool : public osg::Observer
{
public:
    typedef std::map<std::string, std::unique_ptr<SelfDeletingTool>> Map;
    SelfDeletingTool(Map &toolMap, const std::string &name, std::unique_ptr<Tool> &&tool);
    void objectDeleted(void*) override;
    std::unique_ptr<Tool> value;
private:
    Map::iterator m_iter;
    Map &m_tools;
};

#endif // COVER_TOOLMACHINE_TOOL_H
