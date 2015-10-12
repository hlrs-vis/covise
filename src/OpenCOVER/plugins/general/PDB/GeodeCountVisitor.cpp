/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   Count num of Geodes in the subtree
//
// Author:        Philip Weber
//
// Creation Date: 2006-02-29
//
// **************************************************************************

#include <iostream>
#include <osg/Geometry>
#include <osg/Geode>
#include "GeodeCountVisitor.h"

using namespace std;

// run from mainNode
GeodeCountVisitor::GeodeCountVisitor()
    : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
{
    _total = 0;
}

void GeodeCountVisitor::printNumGeodes()
{
    cerr << "Total number " << _total << "!!" << endl;
    _total = 0;
}

// only interested in optimizing cones, sphere and cylinders
void GeodeCountVisitor::apply(osg::Geode &node)
{
    for (unsigned int i = 0; i < node.getNumDrawables(); i++)
    {
        osg::Geometry *sd = dynamic_cast<osg::Geometry *>(node.getDrawable(i));
        if (sd)
            _total++;
    }
}
