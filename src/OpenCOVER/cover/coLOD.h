/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COLOD
#define COLOD

/*
 * LOD node for OpenCOVER
 * 
 * This class is basically the same as osg::LOD but uses the head point
 * instead of the viewpoint for LOD calculation, so the LOD is the same
 * on all walls / eyes.
 * 
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
#include <osg/LOD>
#include <osg/NodeVisitor>

namespace opencover
{

class COVEREXPORT coLOD : public osg::LOD
{
public:
    coLOD();
    virtual ~coLOD();

    virtual void traverse(osg::NodeVisitor &nv);

private:
    float getDistance(osg::NodeVisitor &nv);
};
}
#endif
