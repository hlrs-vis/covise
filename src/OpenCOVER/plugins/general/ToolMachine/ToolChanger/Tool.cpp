#include "Tool.h"
#include "Utility.h"

#include <osgAnimation/UpdateMatrixTransform>

#include <cassert>



osgAnimation::StackedScaleElement *addScaleElement(osg::Node *node)
{
    auto cb = node->getUpdateCallback();
    if(! cb) return nullptr;
    auto s = dynamic_cast<osgAnimation::UpdateMatrixTransform*>(cb);
    if(! s) return nullptr;
    auto retval = new osgAnimation::StackedScaleElement;
    s->getStackedTransforms().push_back(retval);
    return retval;
}

ToolModel::ToolModel(osg::Node *model)
: m_shaft(findTransform(model, "Schaft"))
, m_tip(findTransform(model, "Spitze"))
, m_tool(findTransform(model, "Werkzeug"))
, m_shaftScale(addScaleElement(m_shaft))
, m_tipScale(addScaleElement(m_tip))
{
    assert(m_shaft && m_tip && m_tool);
    assert(m_shaftScale && m_tipScale);
}

void ToolModel::resize(float length, float radius)
{
    m_tipScale->setScale(osg::Vec3(radius, length, radius) );
}

void ToolModel::setParent(osg::Group *p)
{
    parent()->removeChild(m_tool);
    p->addChild(m_tool);
}

osg::Group* ToolModel::parent()
{
    return m_tool->getParent(0);
}

float ToolModel::getLength() const
{
    return m_tipScale->getScale().y();
}

float ToolModel::getRadius() const
{
    return m_tipScale->getScale().x();
}
