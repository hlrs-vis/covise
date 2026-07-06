#include "GridPoint.h"

GridPoint::GridPoint(osg::Vec3 pos,osg::Vec4& color, float radius):m_Color(color),m_PreviousColor(color)
{
    osg::Matrix local;
    local.setTrans(pos);
    m_LocalDCS = new osg::MatrixTransform();
    m_LocalDCS->setMatrix(local);
    m_LocalDCS->setName("Translation");
    m_Sphere = new osg::Sphere(osg::Vec3(0,0,0), radius);
    m_SphereDrawable = new osg::ShapeDrawable(m_Sphere);
    m_SphereDrawable->setColor(color);
    m_Geode = new osg::Geode();
    //osg::StateSet *mystateSet = m_Geode->getOrCreateStateSet();
    //setStateSet(mystateSet);
    m_Geode->setName("Point");
    m_Geode->addDrawable(m_SphereDrawable);
    m_LocalDCS->addChild(m_Geode.get());
}
void GridPoint::highlite(const osg::Vec4& color)
{
    m_SphereDrawable->setColor(color);
}

void GridPoint::setColor(const osg::Vec4& color)
{
    m_PreviousColor = color;
    m_SphereDrawable->setColor(color);
}

void GridPoint::setOriginalColor()
{
    m_SphereDrawable->setColor(m_Color);

    m_PreviousColor = m_Color;
}


void GridPoint::setPreviousColor()
{
    m_SphereDrawable->setColor(m_PreviousColor);
}
