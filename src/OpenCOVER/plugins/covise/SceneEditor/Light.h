/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef LIGHT_H
#define LIGHT_H

#include <osg/Node>

#include "SceneObject.h"

class Light : public SceneObject
{
public:
    Light();
    virtual ~Light();

    int setNode(osg::Node *n);
    osg::Node *getNode();

private:
    osg::ref_ptr<osg::Node> _node;
};

#endif
