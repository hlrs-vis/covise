/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SHAPE_SEGMENT_H
#define SHAPE_SEGMENT_H

#include <osg/Geode>
#include <osg/Geometry>

class ShapeSegment
{
public:
    ShapeSegment(osg::Group *parent, int nV);
    ~ShapeSegment();

    void setCoord(int index, osg::Vec3 pos);
    osg::Vec3 getCoord(int index);
    void setTexCoord(int index, osg::Vec2 tex);
    void setNormal(int index, osg::Vec3 normal);
    osg::Vec3 getNormal(int index);
    void setNormals(osg::Vec3 normal);

    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::Geometry> geometry;
    osg::ref_ptr<osg::Geode> wireframeGeode;
    osg::ref_ptr<osg::Geometry> wireframeGeometry;
    osg::ref_ptr<osg::Vec3Array> coordArray;
    osg::ref_ptr<osg::Vec3Array> normalArray;
    osg::ref_ptr<osg::Vec2Array> texcoordWallpos;

    int numVertices;
};

#endif
