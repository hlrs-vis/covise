#include "Tool.h"
#include <cassert>
#include <cover/ui/Action.h>
#include <cover/coVRPluginSupport.h>

using namespace opencover;



ToolModel::ToolModel(ui::Group *group, config::File &file, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode)
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

osg::Vec3 ToolModel::toolHeadInTableCoords()
{
    osg::Matrix toolHeadToWorld = m_toolHeadNode->getWorldMatrices(cover->getObjectsRoot())[0];
    osg::Matrix tableToWorld = m_tableNode->getWorldMatrices(cover->getObjectsRoot())[0];
    auto worldToTable = osg::Matrix::inverse(tableToWorld);
    auto pointWorld = osg::Vec3() * toolHeadToWorld;
    auto pointWorldOld = toolHeadToWorld.getTrans();
    auto pointTable = pointWorld * worldToTable;
    return pointTable;
}


void ToolModel::update(const opencover::opcua::MultiDimensionalArray<double> &data)
{
    if(!m_tableNode || !m_toolHeadNode)
        return;

    updateAttributes();

    updateGeo(m_paused, data);
}

void ToolModel::updateAttributes(){
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

void ToolModel::pause(bool state)
{
    m_paused = state;
}

const std::vector<UpdateValues> &ToolModel::getUpdateValues()
{
    return m_updateValues;
}

SelfDeletingTool::SelfDeletingTool(std::unique_ptr<ToolModel> &&tool)
: value(std::move(tool))
{
    value->m_toolHeadNode->addObserver(this);
    value->m_tableNode->addObserver(this);
}

void SelfDeletingTool::objectDeleted(void* v){
    
    v == value->m_toolHeadNode ? value->m_tableNode->removeObserver(this) : value->m_toolHeadNode->removeObserver(this);
    m_this->reset();
}

void SelfDeletingTool::create(std::unique_ptr<SelfDeletingTool> &selfDeletingToolPtr, std::unique_ptr<ToolModel> &&tool)
{
    selfDeletingToolPtr.reset(new SelfDeletingTool(std::move(tool)));
    selfDeletingToolPtr->m_this = &selfDeletingToolPtr;
}


bool istReserved(const std::string& symbol)
{
   return exprtk::details::is_reserved_word(symbol) || exprtk::details::is_reserved_symbol(symbol);
}

std::set<std::string> getSymbols(const std::string &expression_string)
{
   exprtk::lexer::parser_helper p;
   p.init(expression_string);
   std::set<std::string> symbols;
   while(p.current_token().type != exprtk::lexer::token::token_type::e_eof)
   {
      if(p.current_token().type == exprtk::lexer::token::token_type::e_symbol && !istReserved(p.current_token().value))
      {
         symbols.insert(p.current_token().value);
      }
      p.next_token();
   }
   return symbols;
}

void ToolModel::observeCustomAttributes()
{
    auto symbols = getSymbols(m_customAttribute->ui()->text());
    //check if all custom attributes are available
    for(const auto &s : symbols)
    {
        const auto &items = m_attributeName->ui()->items();
        if(std::find(items.begin(), items.end(), s) == items.end())
        {
            std::cerr << "ToolMachinePlugin: custom attribute " << s << " not available" << std::endl;
            return;
        }
    }
    //set update functions, only update the expression if the frame is over, triggered by updated values in the next frame
    m_updateValues.clear();
    m_symbolTable.clear();
    for(const auto &s : symbols)
    {
        m_symbolTable.add_variable(s, m_customAttributeData[s].value);
        m_updateValues.push_back({s, [this, s](double value){
            if(m_frameOver)
            {
                attributeChanged(m_expression.value());
                m_frameOver = false;
            }
            m_customAttributeData[s].value = value;
        }});
    }
    m_expression = exprtk::expression<float>();
    m_expression.register_symbol_table(m_symbolTable);
    m_parser.compile(m_customAttribute->ui()->text(), m_expression);
    
    //unsubscribe attributes that are not needed anymore and ignore attribute that are already observed
    for(auto it = m_opcuaAttribId.begin(); it != m_opcuaAttribId.end();)
    {
        if(symbols.find(it->first) == symbols.end())
        {
            it = m_opcuaAttribId.erase(it);
        } else{
            symbols.erase(it->first);
            ++it;
        }
    }
    //observe the rest
    for(const auto &s : symbols)
    {
        m_opcuaAttribId[s] = m_client->observeNode(s);
    }

}

void ToolModel::attributeChanged()
{
    auto attr = m_attributeName->ui()->selectedItem();
    if(attr == "custom")
    {
        observeCustomAttributes();
        m_customAttribute->ui()->setVisible(true);
    } else{
        m_opcuaAttribId.clear();
        m_customAttribute->ui()->setVisible(false);
        m_opcuaAttribId[attr] = m_client->observeNode(attr);

        m_updateValues.clear();
        m_updateValues.push_back({attr, [this](double value){
            attributeChanged(value);
        }});       
    } 
}

void ToolModel::frameOver()
{
    m_frameOver = true;
    updateAttributes();
}
