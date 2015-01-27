/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BARRIER_H
#define BARRIER_H

#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Geometry>

class Room;

class Barrier
{
public:
    enum Alignment
    {
        FRONT,
        BACK,
        LEFT,
        RIGHT,
        TOP,
        BOTTOM
    };

    Barrier(Room *room, std::string name, float width, float height, Barrier::Alignment al, osg::Vec3 p = osg::Vec3(0.0, 0.0, 0.0));
    virtual ~Barrier();

    // setter and getter
    float getWidth();
    float getHeight();
    Alignment getAlignment();
    osg::Vec3 getNormal();
    osg::Vec3 getPosition();
    osg::MatrixTransform *getNode();
    Room *getRoom();
    void setSize(float w, float h, float l);
    void setAlignment(Alignment al);
    void setGridVisible(bool visible);

    virtual void preFrame();

    virtual void repaint();

protected:
    // dimension of wall in mm
    float _width;
    float _height;
    // normal of the wall
    Alignment _alignment;
    osg::Vec3 _normal;
    // position of the bottom center point
    osg::Vec3 _position, _startPos;
    Room *_room;
    // geometry
    osg::ref_ptr<osg::MatrixTransform> _transformNode;
    osg::ref_ptr<osg::Geode> _geometryNode;
    osg::ref_ptr<osg::Geometry> _geometry;
    osg::ref_ptr<osg::Vec3Array> _coordArray;
    osg::ref_ptr<osg::Vec3Array> _normalArray;
    osg::ref_ptr<osg::Vec2Array> _texcoordRegular;
    osg::ref_ptr<osg::Vec2Array> _texcoordWallpos;

    // grid
    osg::ref_ptr<osg::Group> _gridGroup;
    osg::ref_ptr<osg::Geode> _gridGeode;
    osg::ref_ptr<osg::Geometry> _gridGeometry;
    osg::ref_ptr<osg::Vec3Array> _gridVertices;
    osg::ref_ptr<osg::Geode> _gridCenterGeode;
    osg::ref_ptr<osg::Geometry> _gridCenterGeometry;
    osg::ref_ptr<osg::Vec3Array> _gridCenterVertices;

    void setPosition(osg::Vec3 v);
    void updateGeometry();
    void updateNormal();

    void initGrid();
    void updateGrid();

    void tessellate();
};

#endif
