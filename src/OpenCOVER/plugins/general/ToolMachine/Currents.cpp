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

Currents::Currents(ui::Group *group, config::File &file, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode)
: Tool(group, file, toolHeadNode, tableNode)
{
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
    applyLineShader(m_traceLine, m_colorMapSelector->selectedMap(), getMinAttribute(), getMaxAttribute());

}

void Currents::updateGeo(bool paused, const opencover::opcua::MultiDimensionalArray<double> &data)
{
    std::cerr << "updateGeo not implemented for currents tool" << std::endl;
}


void Currents::attributeChanged(float value)
{
    m_values->push_back(value);
    m_vertices->push_back(toolHeadInTableCoords());
    int numElements = m_numSectionsSlider->getValue() < 0? (int)m_vertices->size() : m_numSectionsSlider->getValue();
    m_drawArrays->setFirst(std::max(0, (int)m_vertices->size() - numElements));
    m_drawArrays->setCount(std::min(numElements, (int)m_vertices->size()));
    m_traceLine->setVertexArray(m_vertices);
    m_traceLine->setVertexAttribArray(DataAttrib, m_values, osg::Array::BIND_PER_VERTEX);
}

