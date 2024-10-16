#ifndef COVER_TOOLMACHINE_TOOL_H
#define COVER_TOOLMACHINE_TOOL_H

#include "MathExpressions.h"

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

#include <exprtk.hpp>

struct UpdateValues{
    std::string name;
    std::function<void(double)> func;
};

class SelfDeletingTool;

class ToolModel{
public:
    friend SelfDeletingTool;
    ToolModel(opencover::ui::Group* group, opencover::config::File &file, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode);
    virtual ~ToolModel() = default;
    void update(const opencover::opcua::MultiDimensionalArray<double> &data);
    void update();
    void pause(bool state);
    const std::vector<UpdateValues> &getUpdateValues();
    void frameOver();
protected:
    virtual void updateGeo(bool paused, const opencover::opcua::MultiDimensionalArray<double> &data) = 0;
    virtual void clear() = 0;
    virtual void applyShader(const covise::ColorMap& map, float min, float max) = 0;
    virtual std::vector<std::string> getAttributes() = 0;
    virtual void attributeChanged(float value) = 0;
    osg::Vec3 toolHeadInTableCoords();
    float getMinAttribute() const { return m_minAttribute->ui()->number(); }
    float getMaxAttribute() const { return m_maxAttribute->ui()->number(); }
    osg::MatrixTransform *m_toolHeadNode = nullptr;
    osg::MatrixTransform *m_tableNode = nullptr;
    std::unique_ptr<opencover::ui::Group> m_group;
    std::unique_ptr<opencover::ui::SliderConfigValue> m_numSectionsSlider;
    covise::ColorMapSelector *m_colorMapSelector;

    opencover::opcua::Client *m_client;
    bool m_paused = false;
private:
    void observeCustomAttributes();
    void attributeChanged();
    void updateAttributes();
    MathExpressionObserver m_mathExpressionObserver;
    std::vector<UpdateValues> m_updateValues;
    std::unique_ptr<opencover::ui::SelectionListConfigValue> m_attributeName;
    std::unique_ptr<opencover::ui::EditFieldConfigValue> m_minAttribute, m_maxAttribute, m_customAttribute;
    struct CustomAttributeVariable{
        float value = 0;
        bool updated = false;
        
    };
    MathExpressionObserver::ObserverHandle::ptr m_customAttributeHandle;

};


class SelfDeletingTool : public osg::Observer
{
public:
    static void create(std::unique_ptr<SelfDeletingTool> &selfDeletingToolPtr, std::unique_ptr<ToolModel> &&tool);
    void objectDeleted(void*) override;
    std::unique_ptr<ToolModel> value;
private:
    SelfDeletingTool(std::unique_ptr<ToolModel> &&tool);
    std::unique_ptr<SelfDeletingTool>* m_this;
};

#endif // COVER_TOOLMACHINE_TOOL_H
