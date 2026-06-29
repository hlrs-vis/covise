#include <osg/Geode>
#include <osg/Geometry>
#include <osg/LineWidth>

#include <cover/coVRPluginSupport.h> // for cover

#include "CoverDrawObject.h"

using namespace opencover;

void CoverDrawObject::clean()
{
    if (m_objectPtr.valid())
    {
        cover->getObjectsRoot()->removeChild(m_objectPtr);
        m_objectPtr = nullptr;
    }
}

void CoverDrawObject::addToCover()
{
    m_objectPtr->setName(m_nodeName);
    cover->getObjectsRoot()->addChild(m_objectPtr);
}

CoverLine::CoverLine()
{
    m_nodeName = "CoverLine";
}

CoverLine::CoverLine(const osg::Vec4 &color, float lineWidth, const std::string &nodeName)
    : m_color(color)
    , m_lineWidth(lineWidth)
{
    m_nodeName = nodeName;
}

void CoverLine::draw()
{
    this->clean();

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
    vertices->push_back(m_origin);
    vertices->push_back(m_end);

    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();
    colors->push_back(m_color);
    colors->push_back(m_color);

    geom->setVertexArray(vertices);
    geom->setColorArray(colors, osg::Array::BIND_PER_VERTEX);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 2));

    osg::ref_ptr<osg::LineWidth> linewidth = new osg::LineWidth(m_lineWidth);
    geom->getOrCreateStateSet()->setAttributeAndModes(linewidth, osg::StateAttribute::ON);
    geom->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom);

    m_objectPtr = new osg::MatrixTransform();
    m_objectPtr->addChild(geode);

    this->addToCover();
}

void CoverLine::draw(const osg::Vec3 &origin, const osg::Vec3 &end)
{
    setOrigin(origin);
    setEnd(end);

    draw();
}

CoverFrame::CoverFrame()
{
    m_nodeName = "CoverFrame";
}

CoverFrame::CoverFrame(float axisLength, float lineWidth, const std::string &nodeName)
    : m_axisLength(axisLength)
    , m_lineWidth(lineWidth)
{
    m_nodeName = nodeName;
}

void CoverFrame::draw()
{
    this->clean();

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();

    osg::Matrix rotationMatrix = m_frame;
    rotationMatrix.setTrans(0, 0, 0);

    osg::Vec3 xAxis = osg::Vec3(1, 0, 0) * rotationMatrix;
    xAxis.normalize();
    vertices->push_back(m_origin);
    vertices->push_back(m_origin + xAxis * m_axisLength);
    colors->push_back(m_xColor);
    colors->push_back(m_xColor);

    osg::Vec3 yAxis = osg::Vec3(0, 1, 0) * rotationMatrix;
    yAxis.normalize();
    vertices->push_back(m_origin);
    vertices->push_back(m_origin + yAxis * m_axisLength);
    colors->push_back(m_yColor);
    colors->push_back(m_yColor);

    osg::Vec3 zAxis = osg::Vec3(0, 0, 1) * rotationMatrix;
    zAxis.normalize();
    vertices->push_back(m_origin);
    vertices->push_back(m_origin + zAxis * m_axisLength);
    colors->push_back(m_zColor);
    colors->push_back(m_zColor);

    geom->setVertexArray(vertices);
    geom->setColorArray(colors, osg::Array::BIND_PER_VERTEX);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 6));
    osg::ref_ptr<osg::LineWidth> linewidth = new osg::LineWidth(m_lineWidth);
    geom->getOrCreateStateSet()->setAttributeAndModes(linewidth, osg::StateAttribute::ON);
    geom->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom);

    m_objectPtr = new osg::MatrixTransform();
    m_objectPtr->addChild(geode);

    this->addToCover();
}

void CoverFrame::draw(const osg::Vec3 &origin, const osg::Matrix &frame)
{
    setOrigin(origin);
    setFrame(frame);

    draw();
}