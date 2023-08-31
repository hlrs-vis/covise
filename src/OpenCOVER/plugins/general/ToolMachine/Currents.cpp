#include "Currents.h"

#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <osg/StateSet>
#include <osgDB/StreamOperator>
#include <PluginUtil/ColorMaterials.h>

using namespace opencover;

Currents::Currents()
: m_generalOffset(new osg::MatrixTransform)
, m_aAxis(new osg::MatrixTransform)
, m_cAxis(new osg::MatrixTransform)
, m_traceLine(new osg::Geometry)
, m_points(new osg::Vec3Array)
, m_drawArrays(new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0, 0, 1))
{
    
    cover->getObjectsRoot()->addChild(m_generalOffset);
    m_generalOffset->addChild(m_aAxis);
    m_aAxis->addChild(m_cAxis);

    osg::ref_ptr<osg::StateSet> stateSet = VRSceneGraph::instance()->loadDefaultGeostate();
    m_traceLine->setVertexArray(m_points);

    m_traceLine->setUseDisplayList(false);
    m_traceLine->setSupportsDisplayList(false);
    m_traceLine->setUseVertexBufferObjects(true);
    stateSet->setAttributeAndModes(material::get(material::Red), osg::StateAttribute::ON);
    m_traceLine->setStateSet(stateSet);
    m_drawArrays->setDataVariance(osg::Object::DYNAMIC);
    m_traceLine->addPrimitiveSet(m_drawArrays);
    osg::LineWidth* linewidth = new osg::LineWidth();
    linewidth->setWidth(2.0f);
    stateSet->setAttributeAndModes(linewidth, osg::StateAttribute::ON);
    m_cAxis->addChild(m_traceLine);
    // for (size_t i = 0; i < 20; i++)
    // {
    //     m_points->push_back(osg::Vec3{(float) i, 0, 0});
    // }
    // m_drawArrays->setCount(m_points->size());
}

void Currents::update(const std::array<double, 5> &position, const std::array<double, 5> &currents)
{
    osg::Vec3 point{(float)position[2]/1000, (float)position[3]/1000, (float)position[4]/1000};
    osg::Matrix a = osg::Matrix::rotate(-position[0] / 180 * (float)osg::PI, osg::X_AXIS);
    osg::Matrix b = osg::Matrix::rotate(-position[1] / 180 * (float)osg::PI, osg::Z_AXIS);
    
    m_aAxis->setMatrix(a);
    m_cAxis->setMatrix(b);

    auto worldToC = m_cAxis->getWorldMatrices(m_generalOffset)[0];
    auto pointInC = point * worldToC;
    m_points->push_back(pointInC);
    int numElements = (int)m_points->size();
    m_drawArrays->setFirst(std::max(0, (int)m_points->size() - numElements));
    m_drawArrays->setCount(std::min(numElements, (int)m_points->size()));
    m_traceLine->setVertexArray(m_points);
}

void Currents::setOffset(const std::array<double, 5> &offsets)
{
    m_generalOffset->setMatrix(osg::Matrix::translate(osg::Vec3d{offsets[2], offsets[3], offsets[4]}));
}