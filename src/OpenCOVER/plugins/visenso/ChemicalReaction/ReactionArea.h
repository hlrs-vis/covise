/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _REACTION_AREA_H
#define _REACTION_AREA_H

#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Material>

#include <iostream>

#define AREA_X_MIN -8.0f
#define AREA_X_MAX 8.0f
#define AREA_Z_MIN -8.0f
#define AREA_Z_MAX 8.0f

#define AREA_BORDER_WIDTH 0.2f

bool isInReactionArea(osg::Vec3 p);

class ReactionArea
{
public:
    ReactionArea();
    ~ReactionArea();

    void setVisible(bool visible);

private:
    osg::ref_ptr<osg::Geode> geode;
};

#endif
