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
    // return applyShader(drawable, colorMap, min, max, "MapColorsAttrib");
    return applyShader(drawable, colorMap, min, max, "MapColorsAttribUnlit");
}

Currents::Currents(ui::Group *group, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode)
: Tool(group, toolHeadNode, tableNode)
{
    m_attributeName->setCallback([this](int i){
        m_opcuaAttribId = m_client->observeNode(m_attributeName->selectedItem());
    });
    initGeo();
}

void Currents::clear()
{
    m_vertices->clear();
    m_values->clear();
}

void Currents::applyShader(const covise::ColorMap& map, float min, float max)
{
    applyLineShader(m_traceLine, map, min, max);
}

std::vector<std::string> Currents::getAttributes()
{
    return m_client->availableNumericalScalars();
}


void Currents::initGeo()
{
    m_traceLine = new osg::Geometry;
    m_vertices = new osg::Vec3Array;
    m_values = new osg::FloatArray;
    m_drawArrays = new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP);
    osg::ref_ptr<osg::StateSet> stateSet = VRSceneGraph::instance()->loadUnlightedGeostate();
    // osg::ref_ptr<osg::StateSet> stateSet = VRSceneGraph::instance()->loadDefaultGeostate();
    m_traceLine->setVertexArray(m_vertices);

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

void Currents::updateGeo(bool paused, const opencover::opcua::MultiDimensionalArray<double> &data)
{
   
   auto pointTable = toolHeadInTableCoords();

    auto octMode = true;
    

    if(m_client->isConnected())
    {
        m_values->push_back(m_client->getNumericScalar(m_attributeName->selectedItem()));
        m_vertices->push_back(pointTable);
    }        
    else
    {
        m_vertices->push_back(pointTable);
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

    int numElements = m_numSectionsSlider->value() < 0? (int)m_vertices->size() : m_numSectionsSlider->value();
    m_drawArrays->setFirst(std::max(0, (int)m_vertices->size() - numElements));
    m_drawArrays->setCount(std::min(numElements, (int)m_vertices->size()));
    m_traceLine->setVertexArray(m_vertices);
    m_traceLine->setVertexAttribArray(DataAttrib, m_values, osg::Array::BIND_PER_VERTEX);

    if(!m_client->isConnected())
        return;
    // for(const auto &array: m_client->availableNumericalArrays())
    //     std::cerr << "arrray " << array << std::endl;
    // auto array = m_client->readArray<UA_Float>("fake_octArray");
}
