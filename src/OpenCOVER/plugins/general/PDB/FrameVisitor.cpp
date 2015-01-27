/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   Frame Visitor what will enable the correct frames to be
//		  displayed (set child switches)
//
// Author:        Philip Weber
//
// Creation Date: 2006-02-29
//
// **************************************************************************

#include <iostream>
#include "FrameVisitor.h"

// run from mainNode
FrameVisitor::FrameVisitor()
    : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
{
    _highdetail = false;
    _fade = false;
}

void FrameVisitor::setHighDetailOn(bool detail)
{
    _highdetail = detail;
}

void FrameVisitor::setFadeOn(bool fade)
{
    _fade = fade;
}

void FrameVisitor::apply(osg::Switch &node)
{
    osg::Switch *switchnode = dynamic_cast<osg::Switch *>(node.getParent(0));

    // check if node contains both children
    if (switchnode)
    {
        if (_fade)
        {
            node.setAllChildrenOn();
        }
        else if (_highdetail)
        {
            node.setSingleChildOn(1);
        }
        else
        {
            node.setSingleChildOn(0);
        }
    }
    else
    {
        traverse(node);
    }
}
