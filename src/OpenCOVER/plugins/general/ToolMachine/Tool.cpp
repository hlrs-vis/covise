#include "Tool.h"
#include <cassert>
#include <cover/ui/Action.h>
#include <cover/coVRPluginSupport.h>

using namespace opencover;

Tool::Tool(ui::Group *group, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode)
: m_toolHeadNode(toolHeadNode)
, m_tableNode(tableNode)
, m_group(group)
{
    auto clearBtn = new ui::Action(group, "clear");
    clearBtn->setCallback([this](){
        clear();
    });
    m_numSectionsSlider = new ui::Slider(group, "numSections");
    m_numSectionsSlider->setBounds(-1, 100);
    m_numSectionsSlider->setValue(-1);

    m_minAttribute = new ui::EditField(group, "minAttribute");
    m_minAttribute->setValue(0);
    m_maxAttribute = new ui::EditField(group, "maxAttribute");
    m_maxAttribute->setValue(1);
    m_maxAttribute->setCallback([this](const std::string &text){
        applyShader(m_colorMapSelector->selectedMap(), m_minAttribute->number(), m_maxAttribute->number());
    });
    m_minAttribute->setCallback([this](const std::string &text){
        applyShader(m_colorMapSelector->selectedMap(), m_minAttribute->number(), m_maxAttribute->number());
    });
    m_colorMapSelector = new covise::ColorMapSelector(*group);
    m_colorMapSelector->setCallback([this](const covise::ColorMap &cm)
    {
        applyShader(m_colorMapSelector->selectedMap(), m_minAttribute->number(), m_maxAttribute->number());
    });

    m_attributeName = new ui::SelectionList(group, "attribute");
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
        m_attributeName->setList(getAttributes());
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
