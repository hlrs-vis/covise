#pragma once
#include <osg/ref_ptr>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>

class GridPoint
{
public:
    GridPoint(osg::Vec3 position,osg::Vec4& color, float radius);
    osg::ref_ptr<osg::MatrixTransform> m_LocalDCS;

    osg::ref_ptr<osg::MatrixTransform> getPoint()const{return m_LocalDCS.get();} //muss man hier ref_ptr übergeben?
    osg::Vec3 getPosition()const{return m_LocalDCS->getMatrix().getTrans();}

    void highlite(const osg::Vec4& color);
    void setColor(const osg::Vec4& color);
    void setOriginalColor();        //sets the color to m_Color
    void setPreviousColor();

private:
    osg::Vec4 m_Color;
    osg::Vec4 m_PreviousColor; //color: is this zone observed by enough cameras or not

    osg::ref_ptr<osg::Geode> m_Geode;
    osg::ref_ptr<osg::Sphere> m_Sphere;
    osg::ref_ptr<osg::ShapeDrawable> m_SphereDrawable;
};
