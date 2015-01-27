/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SHAPE_H
#define SHAPE_H

#include <osg/ShapeDrawable>
#include <osg/Material>

#include "SceneObject.h"
#include "ShapeSegment.h"

class Shape : public SceneObject
{
public:
    enum GeometryType
    {
        GEOMETRY_UNSPECIFIED,
        GEOMETRY_CUBOID,
        GEOMETRY_CYLINDER,
        GEOMETRY_PRISM
    };

    enum CuboidSegments
    {
        CUBOID_TOP,
        CUBOID_BOTTOM,
        CUBOID_FRONT,
        CUBOID_BACK,
        CUBOID_LEFT,
        CUBOID_RIGHT
    };

    enum PrismSegments
    {
        PRISM_TOP,
        PRISM_BOTTOM,
        PRISM_FRONT,
        PRISM_LEFT,
        PRISM_DIAGONAL
    };

    Shape();
    virtual ~Shape();

    virtual EventErrors::Type receiveEvent(Event *e);

    void setShapeDrawable(osg::ShapeDrawable *drawable);
    void setAutomaticHeight(bool ah);

    void setGeometryType(GeometryType geometryType);
    GeometryType getGeometryType();

    void setSize(float width, float height, float length);
    void setWidth(float width);
    void setHeight(float height);
    void setLength(float length);
    float getWidth();
    float getHeight();
    float getLength();

    void getClosestSide(osg::Vec3 queryPosition, osg::Vec3 &result_normal, osg::Vec3 &result_position);

private:
    void _updateGeometry();
    void _updateVisibility();
    void _updatePicking();

    GeometryType _geometryType;

    float _width, _height, _length;
    bool _automaticHeight;

    osg::ref_ptr<osg::Group> _group;

    int _numSegments;
    ShapeSegment **_segments;

    osg::ref_ptr<osg::ShapeDrawable> _cylinderDrawable;

    bool _wireframe;
};

#endif
