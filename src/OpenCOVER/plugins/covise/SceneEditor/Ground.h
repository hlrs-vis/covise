/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef GROUND_H
#define GROUND_H

#include <osg/Node>

#include "SceneObject.h"

class Ground : public SceneObject
{
public:
    Ground();
    virtual ~Ground();

    int setNode(osg::Node *n);
    osg::Node *getNode();

private:
    osg::ref_ptr<osg::Node> _node;
};

#endif
