/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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
    friend struct nodeVisitTriangle;

private:    
    osg::Group *localScene;
    osg::Geode *localGeodeTriangle;
    osg::Geode *localGeodeTriangleStrip;
    //osg::ref_ptr<osg::Vec3Array*> vertexCoords;
    osg::Vec3Array* vertexCoords;
    std::vector<std::string> blacklist;


    bool triFunc = true;

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

        bool checkBlacklist(osg::Node* node)
        {
            for(auto itr = blacklist.begin();itr != blacklist.end(); itr++)
            {
                std::string compareString = *itr;
                if(compareString.compare(node->getName()))
                    return true;
            }
            return false;

        }

        void _printPrimitiveType(osg::PrimitiveSet *pset);

        osg::Geode* returnGeode()
        {
            return localGeodeTriangle;
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
