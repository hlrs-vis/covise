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

#ifndef OSGUTIL_GenNormalsVisitor
#define OSGUTIL_GenNormalsVisitor 1

#include <osg/NodeVisitor>
#include <osg/Geode>
#include <osg/Geometry>

using namespace osg;

/** A smoothing visitor for calculating smoothed normals for
  * osg::GeoSet's which contains surface primitives.
  */
class GenNormalsVisitor : public osg::NodeVisitor
{
public:
    /// default to traversing all children.
    GenNormalsVisitor(float creaseAngle);
    virtual ~GenNormalsVisitor();

    /// smooth geoset by creating per vertex normals.
    static void smooth(osg::Geometry &geoset);

    /// apply smoothing method to all geode geosets.
    virtual void apply(osg::Geode &geode);

private:
    static float creaseAngle;
};

#endif
