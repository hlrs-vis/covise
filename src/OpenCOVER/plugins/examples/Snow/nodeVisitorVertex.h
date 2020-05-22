/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef NODEVISITORVERTEX_H
#define NODEVISITORVERTEX_H

#include <iostream>

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/NodeVisitor>
#include <osg/TriangleFunctor>

#include <cover/coVRPluginSupport.h>

#include <vector>

#include "Raytracer.h"

using namespace covise;
using namespace opencover;
using namespace osg;


class nodeVisitorVertex : public osg::NodeVisitor
{
    friend struct nodeVisitTriangle;

private:
    osg::Group *localScene;
    osg::Geode *rtGeode;
    osg::Matrix childTransform;
    osg::Vec3Array* vertexCoords;
    std::vector<std::string> blacklist;


public:
    nodeVisitorVertex();

    void apply(osg::Node &node);

    void createTestFaces(int num, osg::Vec3Array::iterator coords, int type);
    void createTestFaces(const osg::Vec3& v1, const osg::Vec3& v2, const osg::Vec3& v3);
    void createFaceSet(Vec3Array *coords, int type);
    void fillVertexArray(const osg::Vec3& v1, const osg::Vec3& v2, const osg::Vec3& v3)
    {
        vertexCoords->push_back(v1);
        vertexCoords->push_back(v2);
        vertexCoords->push_back(v3);
    }

    int numOfVertices = 0;

    bool checkBlacklist(osg::Node* node)
    {
        for (auto itr = blacklist.begin(); itr != blacklist.end(); itr++)
        {
            std::string compareString = *itr;
            if (compareString.compare(node->getName()) == 0)
                return true;
        }
        return false;
    }

    void _printPrimitiveType(osg::PrimitiveSet *pset);

    osg::Geode* returnGeode()
    {
        return rtGeode;
    }

    osg::Vec3Array* getVertexArray()
    {
        return vertexCoords;
    }

    std::vector<osg::Node*> coNozzleList;

    osg::Matrix &getChildTransform()
    {
        return childTransform;
    }
};

struct nodeVisitTriangle
{
private:
    nodeVisitorVertex* nvv_;
public:
    void operator()(const osg::Vec3& v1, const osg::Vec3& v2, const osg::Vec3& v3, bool = false)const;
    void setNVV(nodeVisitorVertex* nvv)
    {
        nvv_ = nvv;
    }
};

#endif // NODEVISITORVERTEX_H
