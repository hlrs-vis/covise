/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- OpenSceneGraph - Copyright (C) 1998-2006 Robert Osfield 
 *
 * This library is open source and may be redistributed and/or modified under  
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or 
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * OpenSceneGraph Public License for more details.
*/
#include <osg/TriangleFunctor>

#include "GenNormals.h"

#include <stdio.h>
#include <list>
#include <set>

using namespace osg;

struct LessPtr
{
    inline bool operator()(const osg::Vec3 *lhs, const osg::Vec3 *rhs) const
    {
        return *lhs < *rhs;
    }
};

// triangle functor.
struct mySmoothTriangleFunctor
{

    osg::Vec3 *_coordBase;
    osg::Vec3 *_normalBase;
    float _creaseAngle;

    typedef std::multiset<const osg::Vec3 *, LessPtr> CoordinateSet;
    CoordinateSet _coordSet;

    mySmoothTriangleFunctor()
        : _coordBase(0)
        , _normalBase(0)
    {
    }
    void setCreaseAngle(float ca)
    {
        _creaseAngle = ca;
    }
    void set(osg::Vec3 *cb, int noVertices, osg::Vec3 *nb)
    {
        _coordBase = cb;
        _normalBase = nb;

        osg::Vec3 *vptr = cb;
        for (int i = 0; i < noVertices; ++i)
        {
            _coordSet.insert(vptr++);
        }
    }

    inline void updateNormal(const osg::Vec3 &normal, const osg::Vec3 *vptr)
    {
        std::pair<CoordinateSet::iterator, CoordinateSet::iterator> p = _coordSet.equal_range(vptr);

        for (CoordinateSet::iterator itr = p.first;
             itr != p.second;
             ++itr)
        {
            osg::Vec3 *nptr = _normalBase + (*itr - _coordBase);
            /* if((*nptr)[0]==0.0 &&(*nptr)[1]==0.0 &&(*nptr)[2]==0.0)
            {
               (*nptr) = normal;
            }
            else*/
            {
                osg::Vec3 oldn = (*nptr);
                oldn.normalize();
                if ((oldn * normal) > _creaseAngle)
                {
                    (*nptr) += normal;
                }
            }
        }
    }

    inline void operator()(const osg::Vec3 &v1, const osg::Vec3 &v2, const osg::Vec3 &v3, bool treatVertexDataAsTemporary)
    {
        if (!treatVertexDataAsTemporary)
        {
            // calc orientation of triangle.
            osg::Vec3 normal = (v2 - v1) ^ (v3 - v1);
            normal.normalize();

            updateNormal(normal, &v1);
            updateNormal(normal, &v2);
            updateNormal(normal, &v3);
        }
    }
};

// triangle functor.
struct InitTriangleFunctor
{

    osg::Vec3 *_coordBase;
    osg::Vec3 *_normalBase;
    InitTriangleFunctor()
        : _coordBase(0)
        , _normalBase(0)
    {
    }

    void set(osg::Vec3 *cb, int /*noVertices*/, osg::Vec3 *nb)
    {
        _coordBase = cb;
        _normalBase = nb;
    }

    inline void operator()(const osg::Vec3 &v1, const osg::Vec3 &v2, const osg::Vec3 &v3, bool treatVertexDataAsTemporary)
    {
        if (!treatVertexDataAsTemporary)
        {
            // calc orientation of triangle.
            osg::Vec3 normal = (v2 - v1) ^ (v3 - v1);
            normal.normalize();

            (*(_normalBase + (&v1 - _coordBase))) = normal;
            (*(_normalBase + (&v2 - _coordBase))) = normal;
            (*(_normalBase + (&v3 - _coordBase))) = normal;
        }
    }
};

GenNormalsVisitor::GenNormalsVisitor(float ca)
{
    creaseAngle = cos((ca / 180.0) * M_PI);
    setTraversalMode(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN);
}

GenNormalsVisitor::~GenNormalsVisitor()
{
}
float GenNormalsVisitor::creaseAngle = 0.0;
void GenNormalsVisitor::smooth(osg::Geometry &geom)
{
    Geometry::PrimitiveSetList &primitives = geom.getPrimitiveSetList();
    Geometry::PrimitiveSetList::iterator itr;
    unsigned int numSurfacePrimitives = 0;
    for (itr = primitives.begin();
         itr != primitives.end();
         ++itr)
    {
        switch ((*itr)->getMode())
        {
        case (PrimitiveSet::TRIANGLES):
        case (PrimitiveSet::TRIANGLE_STRIP):
        case (PrimitiveSet::TRIANGLE_FAN):
        case (PrimitiveSet::QUADS):
        case (PrimitiveSet::QUAD_STRIP):
        case (PrimitiveSet::POLYGON):
            ++numSurfacePrimitives;
            break;
        default:
            break;
        }
    }

    if (!numSurfacePrimitives)
        return;

    osg::Vec3Array *coords = dynamic_cast<osg::Vec3Array *>(geom.getVertexArray());
    if (!coords || !coords->size())
        return;

    osg::Vec3Array *normals = new osg::Vec3Array(coords->size());

    osg::Vec3Array::iterator nitr;

    TriangleFunctor<InitTriangleFunctor> itf;
    itf.set(&(coords->front()), coords->size(), &(normals->front()));

    TriangleFunctor<mySmoothTriangleFunctor> stf;
    stf.setCreaseAngle(creaseAngle);
    stf.set(&(coords->front()), coords->size(), &(normals->front()));

    geom.accept(itf);

    geom.accept(stf);

    for (nitr = normals->begin();
         nitr != normals->end();
         ++nitr)
    {
        nitr->normalize();
    }

    geom.setNormalArray(normals);
    geom.setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    geom.dirtyDisplayList();
}

void GenNormalsVisitor::apply(osg::Geode &geode)
{
    for (unsigned int i = 0; i < geode.getNumDrawables(); i++)
    {
        osg::Geometry *geom = dynamic_cast<osg::Geometry *>(geode.getDrawable(i));
        if (geom)
            smooth(*geom);
    }
}
