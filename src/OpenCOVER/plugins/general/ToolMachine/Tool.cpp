#include "Tool.h"
#include <cassert>
#include <cover/ui/Action.h>
#include <cover/coVRPluginSupport.h>

using namespace opencover;

Tool::Tool(ui::Group *group, config::File &file, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode)
: m_toolHeadNode(toolHeadNode)
, m_tableNode(tableNode)
, m_group(group)
{
    auto clearBtn = new ui::Action(group, "clear");
    clearBtn->setCallback([this](){
        clear();
    });
    m_numSectionsSlider = std::make_unique<ui::SliderConfigValue>(group, "numSections", -1, file, "ToolMachinePlugin", config::Flag::PerModel);
    m_numSectionsSlider->ui()->setBounds(-1, 100);

    m_minAttribute = std::make_unique<ui::EditFieldConfigValue>(group, "minAttribute", "0", file, "ToolMachinePlugin", config::Flag::PerModel);
    m_maxAttribute = std::make_unique<ui::EditFieldConfigValue>(group, "maxAttribute", "1", file, "ToolMachinePlugin", config::Flag::PerModel);
    m_maxAttribute->setUpdater([this](){
        applyShader(m_colorMapSelector->selectedMap(), m_minAttribute->ui()->number(), m_maxAttribute->ui()->number());
    });
    m_minAttribute->setUpdater([this](){
        applyShader(m_colorMapSelector->selectedMap(), m_minAttribute->ui()->number(), m_maxAttribute->ui()->number());
    });
    m_colorMapSelector = new covise::ColorMapSelector(*group);
    m_colorMapSelector->setCallback([this](const covise::ColorMap &cm)
    {
        applyShader(m_colorMapSelector->selectedMap(), m_minAttribute->ui()->number(), m_maxAttribute->ui()->number());
    });

    m_attributeName = std::make_unique<ui::SelectionListConfigValue>(group, "attribute", 0, file, "ToolMachinePlugin", config::Flag::PerModel);
    m_client = opcua::getClient(group->name());
    assert(m_client);
}

osg::Vec3 Tool::toolHeadInTableCoords()
{
    osg::Matrix toolHeadToWorld = m_toolHeadNode->getWorldMatrices(cover->getObjectsRoot())[0];
    osg::Matrix tableToWorld = m_tableNode->getWorldMatrices(cover->getObjectsRoot())[0];
    auto worldToTable = osg::Matrix::inverse(tableToWorld);
    auto pointWorld = osg::Vec3() * toolHeadToWorld;
    auto pointWorldOld = toolHeadToWorld.getTrans();
    auto pointTable = pointWorld * worldToTable;
    return pointTable;
}

void Tool::update(const opencover::opcua::MultiDimensionalArray<double> &data)
{
    if(!m_tableNode || !m_toolHeadNode)
        return;

    switch (m_client->statusChanged(this))
    {
    case opcua::Client::Connected:
    case opcua::Client::Disconnected:
    {
        auto attributes = getAttributes();
        attributes.push_back("custom");
        m_attributeName->ui()->setList(attributes);
        m_attributeName->ui()->select(m_attributeName->getValue());
    }
    break;
    
    default:
        break;
    }

    updateGeo(m_paused, data);
}

void Tool::pause(bool state)
{
    m_paused = state;
}

const std::vector<UpdateValues> &Tool::getUpdateValues()
{
    switch (m_client->statusChanged(this))
    {
    case opcua::Client::Connected:
    case opcua::Client::Disconnected:
    {
        auto attributes = getAttributes();
        m_attributeName->ui()->setList(attributes);
        m_attributeName->ui()->select(m_attributeName->getValue());
    }
    break;
    
    default:
        break;
    }
    return m_updateValues;
;
}

SelfDeletingTool::SelfDeletingTool(Map &toolMap, const std::string &name, std::unique_ptr<Tool> &&tool)
: m_tools(toolMap)
, value(std::move(tool))
{
    m_iter = m_tools.insert(std::make_pair(name, this)).first;
    value->m_toolHeadNode->addObserver(this);
    value->m_tableNode->addObserver(this);
}
void SelfDeletingTool::objectDeleted(void* v){
    
    v == value->m_toolHeadNode ? value->m_tableNode->removeObserver(this) : value->m_toolHeadNode->removeObserver(this);
    m_tools.erase(m_iter);
}
