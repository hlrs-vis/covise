/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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

#include <cover/coLOD.h>

#include <algorithm>

#include <osg/CullStack>
#include <osgUtil/CullVisitor>
#include "VRViewer.h"

using namespace osg;

namespace opencover
{

coLOD::coLOD()
    : LOD()
{
}

coLOD::~coLOD()
{
}

void coLOD::traverse(NodeVisitor &nv)
{
    switch (nv.getTraversalMode())
    {
    case (NodeVisitor::TRAVERSE_ALL_CHILDREN):
        std::for_each(_children.begin(), _children.end(), NodeAcceptOp(nv));
        break;
    case (NodeVisitor::TRAVERSE_ACTIVE_CHILDREN):
    {
        float required_range = 0;
        if (_rangeMode == DISTANCE_FROM_EYE_POINT)
        {
            required_range = getDistance(nv);
        }
        else
        {
            osg::CullStack *cullStack = dynamic_cast<osg::CullStack *>(&nv);
            if (cullStack && cullStack->getLODScale())
            {
                required_range = cullStack->clampedPixelSize(getBound())
                                 / cullStack->getLODScale();
            }
            else
            {
                // fallback to selecting the highest res tile by
                // finding out the max range
                for (unsigned int i = 0; i < _rangeList.size(); ++i)
                {
                    required_range = osg::maximum(required_range,
                                                  _rangeList[i].first);
                }
            }
        }

        unsigned int numChildren = _children.size();
        if (_rangeList.size() < numChildren)
            numChildren = _rangeList.size();

        for (unsigned int i = 0; i < numChildren; ++i)
        {
            if (_rangeList[i].first <= required_range
                && required_range < _rangeList[i].second)
            {
                _children[i]->accept(nv);
            }
        }
        break;
    }
    default:
        break;
    }
}

float coLOD::getDistance(osg::NodeVisitor &nv)
{
    osg::NodePathList paths = getParentalNodePaths();
    osg::Matrix worldToLocal = osg::computeWorldToLocal(paths.at(0));
    osg::Vec3f localVector = VRViewer::instance()->getViewerPos() * worldToLocal;

    float scale = 1.0f;
    osgUtil::CullVisitor *cv = dynamic_cast<osgUtil::CullVisitor *>(&nv);
    if (cv)
    {
        scale = cv->getLODScale();
    }

    return (getCenter() - localVector).length() * scale;
}
}
