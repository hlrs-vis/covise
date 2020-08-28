/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coLine.h"
#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <osg/io_utils>



#include <stdio.h>

using namespace opencover;

coLine::coLine(osg::Vec3 point1, osg::Vec3 point2, bool show, osg::Vec4 color):_point1(point1), _point2(point2),_show(show), _color(color)
{
    if(show)
        drawLine();
    
    update(point1, point2);
}

void coLine::update(osg::Vec3 point1, osg::Vec3 point2)
{
    //fprintf(stderr,"coLine::update\n");
    _dirVec = point2 - point1;
    _dirVec.normalize();
    _point1 = point1;
    _point2 = point2;

    if(_show)
    {
        (*_vertices)[0].set(_point1);
        (*_vertices)[1].set(_point2);
        _vertices->dirty();
        _geom->dirtyBound();
    }
    
}

coLine::~coLine()
{
    hide();
}

bool coLine::getShortestLineDistance(const osg::Vec3& point1, const osg::Vec3 &point2, double &shortestDistance) const
{
    osg::Vec3 u1 = _dirVec;
    osg::Vec3 u2 = point2 - point1;
    osg::Vec3 cross = u1^u2;
    osg::Vec3 s = _point1 - point1;
    if(cross.length() == 0) // parallel
        return false;
    
    shortestDistance  = std::fabs(((s*cross)/cross.length()));

    return true;
}

// math can be found here: https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points
bool coLine::getPointsOfShortestDistance(const osg::Vec3 &lp1, const osg::Vec3 &lp2, osg::Vec3& pointLine1, osg::Vec3& pointLine2) const
{
    osg::Vec3 u1 = _dirVec;
    osg::Vec3 u2 = lp2 - lp1;
    osg::Vec3 cross = u1^u2;
    if(cross.length() == 0) // parallel
        return false;

    pointLine1 = _point1 +  u1.operator*(((lp1-_point1)*(u2^cross))/(u1*(u2^cross)));
    pointLine2 = lp1 +  u1.operator*(((_point1-lp1)*(u1^cross))/(u2*(u1^cross)));

    return true;
}

void coLine::drawLine()
{
    if(!_geode.valid())
    {
        _geode = new osg::Geode();
        _geode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));
        _geom = new osg::Geometry();
        _geom->setDataVariance(osg::Object::DataVariance::DYNAMIC) ;
         _geom->setUseVertexBufferObjects(true);
        _vertices = new osg::Vec3Array(2);

        (*_vertices)[0].set(_point1);
        (*_vertices)[1].set(_point2);
    
        _geom->setVertexArray(_vertices);
        _colors = new osg::Vec4Array;
        _colors->push_back(_color);
        _geom->setColorArray(_colors);
        _geom->setColorBinding(osg::Geometry::BIND_OVERALL);
        _geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,2));

        _geode->addDrawable(_geom);
    }

    (*_vertices)[0].set(_point1);
    (*_vertices)[1].set(_point2);

    _vertices->dirty();
    _geom->dirtyBound();

}

void coLine::setColor(osg::Vec4 color)
{
    _color = color;
    if(!_colors)
        _colors = new osg::Vec4Array();
    
    _colors->clear();
    _colors.get()->push_back(color);
    _colors->dirty();  
}

void coLine::show()
{
    _show = true;
    drawLine();
    cover->getObjectsRoot()->addChild(_geode);

}
void coLine::hide()
{
    _show = false;
    int index = cover->getObjectsRoot()->getChildIndex(_geode);
    cover->getObjectsRoot()->removeChild(index);
}