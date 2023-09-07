#include "Currents.h"

#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>
#include <cover/ui/Action.h>
#include <osg/StateSet>
#include <osgDB/StreamOperator>
#include <PluginUtil/ColorMaterials.h>
#include <iostream>
using namespace opencover;




Currents::Currents(ui::Group *group, const osg::Node *toolHeadNode, const osg::Node *tableNode)
: m_generalOffset(new osg::MatrixTransform)
, m_tableProxy(new osg::MatrixTransform)
, m_cAxis(new osg::MatrixTransform)
, m_traceLine(new osg::Geometry)
, m_points(new osg::Vec3Array)
, m_drawArrays(new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0, 0, 1))
, m_toolHeadNode(toolHeadNode)
, m_tableNode(tableNode)
, m_group(group)
{
    auto clearBtn = new ui::Action(group, "clear");
    clearBtn->setCallback([this](){
        m_points->clear();
    });
    m_numPointsSlider = new ui::Slider(group, "numPoints");
    m_numPointsSlider->setBounds(-1, 1000);
    init();
}

bool Currents::init()
{
    if(!m_tableNode || !m_toolHeadNode) //need observer to get notified when vrml nodes are deleted
        return false;
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
    linewidth->setWidth(20.0f);
    stateSet->setAttributeAndModes(linewidth, osg::StateAttribute::ON);
    cover->getObjectsRoot()->addChild(m_generalOffset);
    m_generalOffset->addChild(m_tableProxy);
    m_tableProxy->addChild(m_traceLine);
    // m_tableNode->asTransform()->addChild(m_traceLine);

    // for (size_t i = 0; i < 20; i++)
    // {
    //     m_points->push_back(osg::Vec3{(float) i, 0, 0});
    // }
    // m_drawArrays->setCount(m_points->size());
    return true;
}



void Currents::update()
{
    if(!m_tableNode || !m_toolHeadNode)
        return;
    osg::Matrix toolHeadTrans = m_toolHeadNode->getWorldMatrices(cover->getObjectsRoot())[0];
    osg::Matrix tableTrans = m_tableNode->getWorldMatrices(cover->getObjectsRoot())[0];

    auto pointWorld = toolHeadTrans.getTrans();
    auto pointTable = pointWorld * tableTrans;

    m_points->push_back(pointTable);
    int numElements = m_numPointsSlider->value() < 0? (int)m_points->size() : m_numPointsSlider->value();
    m_drawArrays->setFirst(std::max(0, (int)m_points->size() - numElements));
    m_drawArrays->setCount(std::min(numElements, (int)m_points->size()));
    m_traceLine->setVertexArray(m_points);
    m_tableProxy->setMatrix(tableTrans);
}



void Currents::setOffset(const std::array<double, 5> &offsets)
{
    m_generalOffset->setMatrix(osg::Matrix::translate(osg::Vec3d{offsets[0], offsets[1], offsets[2]}));
}