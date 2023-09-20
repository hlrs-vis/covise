#ifndef COVER_TOOLMACHINE_TOOL_H
#define COVER_TOOLMACHINE_TOOL_H

#include <string>
#include <memory>
#include <map>

#include <cover/ui/EditField.h>
#include <cover/ui/Group.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Button.h>
#include <osg/MatrixTransform>
#include <osg/Observer>
#include <PluginUtil/coColorMap.h>
#include <OpcUaClient/opcua.h>

class SelfDeletingTool;

class Tool{
public:
    friend SelfDeletingTool;
    Tool(opencover::ui::Group* group, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode);
    virtual ~Tool() = default;
    void update();
    void pause(bool state);
protected:
    virtual void updateGeo(bool paused) = 0;
    virtual void clear() = 0;
    virtual void applyShader(const covise::ColorMap& map, float min, float max) = 0;
    virtual std::vector<std::string> getAttributes() = 0;
    osg::Vec3 toolHeadInTableCoords();
    osg::MatrixTransform *m_toolHeadNode = nullptr;
    osg::MatrixTransform *m_tableNode = nullptr;
    std::unique_ptr<opencover::ui::Group> m_group;
    opencover::ui::Slider *m_numSectionsSlider;
    covise::ColorMapSelector *m_colorMapSelector;
    opencover::ui::SelectionList *m_attributeName;
    opencover::ui::EditField *m_minAttribute, *m_maxAttribute;
    opencover::opcua::Client *m_client;
    opencover::opcua::ObserverHandle m_opcuaAttribId;
    bool m_paused = false;
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
