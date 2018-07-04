#ifndef NODEVISITORVERTEX_H
#define NODEVISITORVERTEX_H

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/NodeVisitor>
#include <iostream>

#include "raytracer.h"

using namespace osg;

class nodeVisitorVertex : public osg::NodeVisitor
{
private:
    osg::ref_ptr<osg::Vec3Array> _local_coords;

//    void _processPrimitive(unsigned int nv,
//                           osg::Vec3Array::iterator coords,
//                           osg::Vec3Array::iterator normals,
//                           osg::Geometry::AttributeBinding binding,
//                           osg::Matrix &mat, osg::Matrix &nmat);

public:
        nodeVisitorVertex();

        void apply(osg::Node &node);

        osg::Vec3Array *getCoords()
        {
            return _local_coords.get();
        }


};

#endif // NODEVISITORVERTEX_H
