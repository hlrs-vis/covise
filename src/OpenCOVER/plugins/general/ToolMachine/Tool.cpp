#include "Tool.h"
#include <cassert>
#include <cover/ui/Action.h>
#include <cover/coVRPluginSupport.h>
using namespace opencover;



Tool::Tool(ui::Group *group, config::File &file, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode)
: m_toolHeadNode(toolHeadNode)
, m_tableNode(tableNode)
, m_group(group)
, m_client(opcua::getClient(group->name()))
, m_mathExpressionObserver(m_client)
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
    m_attributeName->setUpdater([this](){
        
        attributeChanged();

    });
    m_customAttribute = std::make_unique<ui::EditFieldConfigValue>(group, "customAttribute", "", file, "ToolMachinePlugin", config::Flag::PerModel);
    m_customAttribute->ui()->setVisible(false);
    m_customAttribute->setUpdater([this](){
        observeCustomAttributes();
    });
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
    updateAttributes();
    updateGeo(m_paused, data);
}

void Tool::update()
{
    if(!m_tableNode || !m_toolHeadNode)
        return;
    updateAttributes();
    m_mathExpressionObserver.update();
    if(m_customAttributeHandle)
        attributeChanged(m_customAttributeHandle->value());
}

void Tool::updateAttributes(){
    switch (m_client->statusChanged(this))
    {
    case opcua::Client::Connected:
    case opcua::Client::Disconnected:
    {
        auto attributes = getAttributes();
        attributes.push_back("custom");
        m_attributeName->ui()->setList(attributes);
        m_attributeName->ui()->select(m_attributeName->getValue());
        attributeChanged();

    }
    break;
    
    default:
        break;
    }
}

void Tool::pause(bool state)
{
    m_paused = state;
}

const std::vector<UpdateValues> &Tool::getUpdateValues()
{
    return m_updateValues;
}

SelfDeletingTool::SelfDeletingTool(std::unique_ptr<Tool> &&tool)
: value(std::move(tool))
{
    value->m_toolHeadNode->addObserver(this);
    value->m_tableNode->addObserver(this);
}

void SelfDeletingTool::objectDeleted(void* v){
    
    v == value->m_toolHeadNode ? value->m_tableNode->removeObserver(this) : value->m_toolHeadNode->removeObserver(this);
    m_this->reset();
}

void SelfDeletingTool::create(std::unique_ptr<SelfDeletingTool> &selfDeletingToolPtr, std::unique_ptr<Tool> &&tool)
{
    selfDeletingToolPtr.reset(new SelfDeletingTool(std::move(tool)));
    selfDeletingToolPtr->m_this = &selfDeletingToolPtr;
}

void Tool::observeCustomAttributes()
{
    m_customAttributeHandle = m_mathExpressionObserver.observe(m_customAttribute->ui()->value());
}

void Tool::attributeChanged()
{
    auto attr = m_attributeName->ui()->selectedItem();
    if(attr == "custom")
    {
        observeCustomAttributes();
        m_customAttribute->ui()->setVisible(true);
    } else{
        m_customAttributeHandle = m_mathExpressionObserver.observe(attr);
        m_customAttribute->ui()->setVisible(false);

    } 
}
