#include "Currents.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRShader.h>
#include <cover/ui/Action.h>
#include <cover/VRSceneGraph.h>
#include <iostream>
#include <osg/StateSet>
#include <osgDB/StreamOperator>
#include <PluginUtil/ColorMaterials.h>

#include <PluginUtil/coShaderUtil.h>
#include <OpcUaClient/opcua.h>
using namespace opencover;


opencover::coVRShader *applyLineShader(osg::Drawable *drawable, const covise::ColorMap &colorMap, float min, float max)
{
    return applyShader(drawable, colorMap, min, max, "MapColorsAttribUnlit");
}

Currents::Currents(ui::Group *group, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode)
: m_traceLine(new osg::Geometry)
, m_points(new osg::Vec3Array)
, m_drawArrays(new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0, 0, 1))
, m_toolHeadNode(toolHeadNode)
, m_tableNode(tableNode)
, m_group(group)
, m_values(new osg::FloatArray)
{
    auto clearBtn = new ui::Action(group, "clear");
    clearBtn->setCallback([this](){
        m_clear = true;
    });
    m_numPointsSlider = new ui::Slider(group, "numPoints");
    m_numPointsSlider->setBounds(-1, 1000);
    m_numPointsSlider->setValue(-1);

    m_minAttribute = new ui::EditField(group, "minAttribute");
    m_minAttribute->setValue(0);
    m_maxAttribute = new ui::EditField(group, "maxAttribute");
    m_maxAttribute->setValue(1);
    m_maxAttribute->setCallback([this](const std::string &text){
        applyLineShader(m_traceLine, m_colorMapSelector->selectedMap(), m_minAttribute->number(), m_maxAttribute->number());
    });
    m_minAttribute->setCallback([this](const std::string &text){
        applyLineShader(m_traceLine, m_colorMapSelector->selectedMap(), m_minAttribute->number(), m_maxAttribute->number());
    });
    m_colorMapSelector = new covise::ColorMapSelector(*group);
    m_colorMapSelector->setCallback([this](const covise::ColorMap &cm)
    {
        applyLineShader(m_traceLine, m_colorMapSelector->selectedMap(), m_minAttribute->number(), m_maxAttribute->number());
    });

    m_attributeName = new ui::SelectionList(group, "attribute");
    m_client = opcua::getClient(group->name());
    assert(m_client);
    m_client->onConnect([this]()
    {
        m_attributeName->setList(m_client->availableNumericalScalars());
    });
    if(m_client->isConnected())
        m_attributeName->setList(m_client->availableNumericalScalars());



    initGeo();
}

Currents::~Currents()
{
    m_client->onConnect(nullptr);
}

void Currents::initGeo()
{
    osg::ref_ptr<osg::StateSet> stateSet = VRSceneGraph::instance()->loadUnlightedGeostate();
    m_traceLine->setVertexArray(m_points);

    m_traceLine->setUseDisplayList(false);
    m_traceLine->setSupportsDisplayList(false);
    m_traceLine->setUseVertexBufferObjects(true);
    // stateSet->setAttributeAndModes(material::get(material::Red), osg::StateAttribute::ON);
    m_traceLine->setStateSet(stateSet);
    m_drawArrays->setDataVariance(osg::Object::DYNAMIC);
    m_traceLine->addPrimitiveSet(m_drawArrays);
    osg::LineWidth* linewidth = new osg::LineWidth();
    linewidth->setWidth(10.0f);
    stateSet->setAttributeAndModes(linewidth, osg::StateAttribute::ON);
    m_tableNode->addChild(m_traceLine);
    applyLineShader(m_traceLine, m_colorMapSelector->selectedMap(), m_minAttribute->number(), m_maxAttribute->number());

}



void Currents::update()
{
    if(m_clear)
    {
        m_clear = false;
        m_points->clear();
        m_values->clear();
    }
    if(!m_tableNode || !m_toolHeadNode)
        return;
    osg::Matrix toolHeadToWorld = m_toolHeadNode->getWorldMatrices(cover->getObjectsRoot())[0];
    osg::Matrix tableToWorld = m_tableNode->getWorldMatrices(cover->getObjectsRoot())[0];
    auto worldToTable = osg::Matrix::inverse(tableToWorld);
    auto pointWorld = osg::Vec3() * toolHeadToWorld;
    auto pointWorldOld = toolHeadToWorld.getTrans();
    auto pointTable = pointWorld * worldToTable;

    m_points->push_back(pointTable);
    if(m_client->isConnected())
        m_values->push_back(m_client->readNumericValue(m_attributeName->selectedItem()));
    else
    {
        auto size = m_values->size() +1;
        auto diff = m_maxAttribute->number() -  m_minAttribute->number();
        diff = diff/size;
        m_values->clear();
        for (size_t i = 0; i < size; i++)
        {
            auto x = m_minAttribute->number() + i *diff;
            m_values->push_back(x);
        }
    }
    int numElements = m_numPointsSlider->value() < 0? (int)m_points->size() : m_numPointsSlider->value();
    m_drawArrays->setFirst(std::max(0, (int)m_points->size() - numElements));
    m_drawArrays->setCount(std::min(numElements, (int)m_points->size()));
    m_traceLine->setVertexArray(m_points);
    m_traceLine->setVertexAttribArray(DataAttrib, m_values, osg::Array::BIND_PER_VERTEX);

    if(!m_client->isConnected())
        return;
    for(const auto &array: m_client->availableNumericalArrays())
        std::cerr << "arrray " << array << std::endl;
    auto array = m_client->readArray<UA_Float>("fake_octArray");
}

SelfDeletingCurrents::SelfDeletingCurrents(Map &currentsMap, const std::string &name, std::unique_ptr<Currents> &&currents)
: m_currents(currentsMap)
, value(std::move(currents))
{
    m_iter = m_currents.insert(std::make_pair(name, this)).first;
    value->m_toolHeadNode->addObserver(this);
    value->m_tableNode->addObserver(this);
}
void SelfDeletingCurrents::objectDeleted(void* v){
    
    v == value->m_toolHeadNode ? value->m_tableNode->removeObserver(this) : value->m_toolHeadNode->removeObserver(this);
    m_currents.erase(m_iter);
}