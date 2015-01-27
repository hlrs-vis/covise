/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRROTATOR_H
#define VRROTATOR_H

/*! \file
 \brief  rotating animation of COVISE objects

 \author Lars Frenzel
 \author (C) 1997
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   28.10.1997
 */

// includes
#include <util/DLinkList.h>
#include <osg/Matrix>

namespace osg
{
class Node;
class MatrixTransform;
}

#include <util/coTypes.h>
namespace opencover
{
// class definitions
class Rotator
{
public:
    static Rotator *rot;
    char *feedback_information;

    osg::Node *node;
    osg::MatrixTransform *transform;
    osg::Matrix oldMat;

    float point[3];
    float vector[3];
    float old_dir1[3];
    float old_dir2[3];
    float angle;
    float speed;
    float transformAngle;
    void setRotation(float *, float *);
    void rotate(float);
    void addTransform();
    void removeTransform();

    Rotator();
    ~Rotator();
};

class RotatorList : public covise::DLinkList<Rotator *>
{
public:
    static RotatorList *instance();
    Rotator *find(float, float, float);
    Rotator *find(osg::Node *);
    void update();
};
}

// done
#endif
