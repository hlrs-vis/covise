/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

/*! \file
 \brief osgUtil::CullVisitor adapted to change projection and viewing matrices in order to correct environment maps

 \author Robert Osfield
 \author Uwe Woessner <woessner@hlrs.de>

 \date 2004
 */

/* -*-c++-*- OpenSceneGraph - Copyright (C) 1998-2003 Robert Osfield 
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

#include <util/coExport.h>

namespace vive
{
/**
 * Basic NodeVisitor implementation for rendering a scene.
 * This visitor traverses the scene graph, collecting transparent and
 * opaque vsg::Nodes into a depth sorted transparent bin and a state
 * sorted opaque bin.  The opaque bin is rendered first, and then the
 * transparent bin is rendered in order from the furthest vsg::Node
 * from the eye to the one nearest the eye. 
 */
class VVCORE_EXPORT coCullVisitor : public osgUtil::CullVisitor
{
public:
    typedef vsg::dmat4::value_type value_type;

    coCullVisitor();
    virtual ~coCullVisitor();

    virtual coCullVisitor *cloneType() const
    {
        return new coCullVisitor();
    }

    virtual vsg::vec3 getEyePoint() const
    {
        return getEyeLocal();
    }
    virtual float getDistanceToEyePoint(const vsg::vec3 &pos, bool withLODScale) const;
    virtual float getDistanceFromEyePoint(const vsg::vec3 &pos, bool withLODScale) const;
    virtual float getDistanceToViewPoint(const vsg::vec3& pos, bool withLODScale) const;

    // we do need our own version of apply and distance() for depth sorting to work correctly
    virtual void apply(vsg::Node &drawable);
    virtual void apply(osg::Billboard &node);

    value_type computeNearestPointInFrustum(const vsg::dmat4 &matrix, const osg::Polytope::PlaneList &planes, const vsg::Node &drawable);
    value_type computeFurthestPointInFrustum(const vsg::dmat4 &matrix, const osg::Polytope::PlaneList &planes, const vsg::Node &drawable);

    bool updateCalculatedNearFar(const vsg::dmat4 &matrix, const osg::BoundingBox &bb);

    bool updateCalculatedNearFar(const vsg::dmat4 &matrix, const vsg::Node &drawable, bool isBillboard = false);

    void updateCalculatedNearFar(const vsg::vec3 &pos);

protected:
    /** Prevent unwanted copy operator.*/
    coCullVisitor &operator=(const coCullVisitor &)
    {
        return *this;
    }
};
}
#endif
