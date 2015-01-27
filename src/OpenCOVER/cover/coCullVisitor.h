/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCULLVISITOR
#define COCULLVISITOR

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
#include <osgUtil/CullVisitor>

namespace opencover
{
/**
 * Basic NodeVisitor implementation for rendering a scene.
 * This visitor traverses the scene graph, collecting transparent and
 * opaque osg::Drawables into a depth sorted transparent bin and a state
 * sorted opaque bin.  The opaque bin is rendered first, and then the
 * transparent bin is rendered in order from the furthest osg::Drawable
 * from the eye to the one nearest the eye. 
 */
class COVEREXPORT coCullVisitor : public osgUtil::CullVisitor
{
public:
    typedef osg::Matrix::value_type value_type;

    coCullVisitor();
    virtual ~coCullVisitor();

    virtual coCullVisitor *cloneType() const
    {
        return new coCullVisitor();
    }

    virtual osg::Vec3 getEyePoint() const
    {
        return getEyeLocal();
    }
    virtual float getDistanceToEyePoint(const osg::Vec3 &pos, bool withLODScale) const;
    virtual float getDistanceFromEyePoint(const osg::Vec3 &pos, bool withLODScale) const;

    // we do need our own version of apply and distance() for depth sorting to work correctly
    virtual void apply(osg::Drawable &drawable);
    virtual void apply(osg::Billboard &node);

    value_type computeNearestPointInFrustum(const osg::Matrix &matrix, const osg::Polytope::PlaneList &planes, const osg::Drawable &drawable);
    value_type computeFurthestPointInFrustum(const osg::Matrix &matrix, const osg::Polytope::PlaneList &planes, const osg::Drawable &drawable);

    bool updateCalculatedNearFar(const osg::Matrix &matrix, const osg::BoundingBox &bb);

    bool updateCalculatedNearFar(const osg::Matrix &matrix, const osg::Drawable &drawable, bool isBillboard = false);

    void updateCalculatedNearFar(const osg::Vec3 &pos);

protected:
    /** Prevent unwanted copy operator.*/
    coCullVisitor &operator=(const coCullVisitor &)
    {
        return *this;
    }
};
}
#endif
