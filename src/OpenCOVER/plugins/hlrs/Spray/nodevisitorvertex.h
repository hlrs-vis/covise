#ifndef NODEVISITORVERTEX_H
#define NODEVISITORVERTEX_H

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/NodeVisitor>
#include <osg/TriangleFunctor>
#include <iostream>
#include <cover/coVRPluginSupport.h>

#include "raytracer.h"

using namespace covise;
using namespace opencover;


using namespace osg;


class nodeVisitorVertex : public osg::NodeVisitor
{
private:
    osg::ref_ptr<osg::Vec3Array> _local_coords;
    osg::Group *localScene;
    osg::Geode *localGeodeTriangle;
    osg::Geode *localGeodeTriangleStrip;


    bool triFunc = true;

public:
        nodeVisitorVertex();

        void apply(osg::Node &node);

        osg::Vec3Array *getCoords()
        {
            return _local_coords.get();
        }

        void createTestFaces(int num, osg::Vec3Array::iterator coords, int type);
        void createTestFaces(const osg::Vec3& v1, const osg::Vec3& v2, const osg::Vec3& v3);

        void _printPrimitiveType(osg::PrimitiveSet *pset);

        osg::Geode* returnGeode()
        {
            return localGeodeTriangle;
        }

};

struct nodeVisitTriangle
{
private:
    osg::Geode *localGeodeTriangle_;
public:
    void operator()(const osg::Vec3& v1, const osg::Vec3& v2, const osg::Vec3& v3, bool)const;
    void setLocalGeode(osg::Geode* localGeodeTriangle)
    {
        localGeodeTriangle_ = localGeodeTriangle;
    }

};

#endif // NODEVISITORVERTEX_H
