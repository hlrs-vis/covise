/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// by Lars Frenzel
// 28.10.1997

#include <util/common.h>

#include <appl/RenderInterface.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include "VRRotator.h"
#include <cover/coVRMSController.h>
#include <osg/Node>
#include <osg/MatrixTransform>
#include <osg/Drawable>
#include <osg/BoundingBox>
#include <osg/Geode>
#include <osg/Version>

using namespace opencover;
using namespace covise;
Rotator::Rotator()
{
    feedback_information = NULL;
    node = NULL;

    point[0] = point[1] = point[2] = 0.0;
    vector[0] = vector[1] = vector[2] = 0.0;
    angle = 0.0;
    speed = 0.0;
    transformAngle = 0.0;
    transform = NULL;
    old_dir1[0] = old_dir1[1] = old_dir1[2] = -2;
    old_dir2[0] = old_dir2[1] = old_dir2[2] = -2;

    return;
}

Rotator::~Rotator()
{
    delete[] feedback_information;
    if (this == rot)
        rot = NULL;
    return;
}

Rotator *Rotator::rot = NULL;

void Rotator::addTransform()
{
    transformAngle = 0.0;
    old_dir1[0] = old_dir1[1] = old_dir1[2] = -2;
    old_dir2[0] = old_dir2[1] = old_dir2[2] = -2;

    transform = dynamic_cast<osg::MatrixTransform *>(node->getParent(0));
    if (transform)
        oldMat = transform->getMatrix();
}

void Rotator::removeTransform()
{
    char buf[600];

    if (node)
    {
        transformAngle = 0.0;
        old_dir1[0] = old_dir1[1] = old_dir1[2] = -2;
        old_dir2[0] = old_dir2[1] = old_dir2[2] = -2;
        if (transform)
        {
            transform->setMatrix(oldMat);
        }
        transform = NULL;
        CoviseRender::set_feedback_info(feedback_information);

        if (coVRMSController::instance()->isMaster())
        {
            fprintf(stdout, "\a");
            fflush(stdout);
            sprintf(buf, "scalar\nFloatScalar\n%f\n", angle);
            CoviseRender::send_feedback_message("PARAM", buf);
            buf[0] = '\0';
            CoviseRender::send_feedback_message("EXEC", buf);
        }
    }
}

void Rotator::setRotation(float *dir1, float *dir2)
{
    float da = 0.0, det;
    if (old_dir1[0] != -2) // do we have a last pair of directions
    {
        da = ((vector[0] * dir1[0]) + (vector[1] * dir1[1]) + (vector[2] * dir1[2]));
        if ((da <= 0.7) || (da >= -0.7))
        { // dir1 is close to rotation axis
            da = (old_dir2[0] * dir2[0]) + (old_dir2[1] * dir2[1]) + (old_dir2[2] * dir2[2]);
            det = (vector[0] * old_dir2[1] * dir2[2]) + (vector[2] * old_dir2[0] * dir2[1]) + (vector[1] * old_dir2[2] * dir2[0]) - (vector[2] * old_dir2[1] * dir2[0]) - (vector[0] * old_dir2[2] * dir2[1]) - (vector[1] * old_dir2[0] * dir2[2]);
        }
        else
        { // dir2 is close to rotation axis
            da = (old_dir1[0] * dir1[0]) + (old_dir1[1] * dir1[1]) + (old_dir1[2] * dir1[2]);
            det = (vector[0] * old_dir2[1] * dir2[2]) + (vector[2] * old_dir2[0] * dir2[1]) + (vector[1] * old_dir2[2] * dir2[0]) - (vector[2] * old_dir2[1] * dir2[0]) - (vector[0] * old_dir2[2] * dir2[1]) - (vector[1] * old_dir2[0] * dir2[2]);
        }
        if (da > 1.0)
            da = 1.0;
        if (da < -1.0)
            da = -1.0;
        da = (acos(da) / M_PI * 180.0);
        //fprintf(stderr,"da det %f %f\n",da,det);
        if (det < 0)
            da = -da;
        angle += da;
        transformAngle += da;
    }
    if (transform)
    {
        osg::Matrix mat = oldMat;

        mat.postMult(osg::Matrix::translate(-point[0], -point[1], -point[2]));
        mat.postMult(osg::Matrix::rotate(transformAngle * M_PI / 180.0, vector[0], vector[1], vector[2]));
        mat.postMult(osg::Matrix::translate(point[0], point[1], point[2]));
        transform->setMatrix(mat);
    }
    //fprintf(stderr,"angle = %f transformAngle = %f\n",angle,transformAngle);
    old_dir1[0] = dir1[0];
    old_dir1[1] = dir1[1];
    old_dir1[2] = dir1[2];
    old_dir2[0] = dir2[0];
    old_dir2[1] = dir2[1];
    old_dir2[2] = dir2[2];
    return;
}

void Rotator::rotate(float da)
{

    angle += da;
    transformAngle += da;
    if (transform)
    {
        osg::Matrix mat = oldMat;

        mat.postMult(osg::Matrix::translate(-point[0], -point[1], -point[2]));
        mat.postMult(osg::Matrix::rotate(transformAngle * M_PI / 180.0, vector[0], vector[1], vector[2]));
        mat.postMult(osg::Matrix::translate(point[0], point[1], point[2]));
        transform->setMatrix(mat);
    }

    return;
}

Rotator *RotatorList::find(float x, float y, float z)
{
    osg::Node *nearest;
    float near_dist, cur_dist;
    float center[3];

    osg::Drawable *gs;
    osg::BoundingBox bb;

    // find the nearest of the Rotator-Geodes
    reset();

    near_dist = -1;
    nearest = NULL;

    while (current())
    {
        if (current()->speed != 0.0)
            break; // not an interactor;
        osg::Geode *geode = (osg::Geode *)current()->node;
        gs = geode->getDrawable(0);

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 3)
        bb = gs->getBoundingBox();
#else
        bb = gs->getBound();
#endif

        center[0] = (bb.xMin() + bb.xMax()) / 2;
        center[1] = (bb.yMin() + bb.yMax()) / 2;
        center[2] = (bb.zMin() + bb.zMax()) / 2;

        // check distance
        cur_dist = (center[0] - x) * (center[0] - x) + (center[1] - y) * (center[1] - y) + (center[2] - z) * (center[2] - z);

        if (cur_dist < near_dist || near_dist == -1)
        {
            near_dist = cur_dist;
            nearest = current()->node;
        }

        next();
    }

    // now return the rotator assigned to the found node

    if (nearest)
        return (find(nearest));

    return (NULL);
}

void RotatorList::update()
{

    reset();

    while (current())
    {
        if (current()->speed != 0.0)
        {
            if (!(current()->transform))
                current()->addTransform();
            current()->rotate(current()->speed);
        }

        next();
    }

    // re implement interaction if needed
    // was in VRSceneGraph, zou can get direction? from hand mat
    /*
   if( rotator && !this->handLocked )
   {
      float position[3];
      float direction[3];
      float direction2[3];
      // retrieve pointer coordinates
      getHandWorldPosition(position, direction, direction2);

      if (((oldHandLocked && (button->getButtonStatus() & ACTION_BUTTON)))||(button->wasPressed()))
      {

         Rotator::rot=rotatorlist.find(position[0],position[1],position[2]);
         if(Rotator::rot)
         {
            Rotator::rot->addTransform();
            Rotator::rot->setRotation(direction,direction2);
         }

      }
      else if(button->getButtonStatus() & ACTION_BUTTON)
      {
         if(Rotator::rot)
         {
            Rotator::rot->setRotation(direction,direction2);
         }

      }
      else if(button->wasReleased()&& (button->oldButtonStatus() & ACTION_BUTTON))
      {
         if(Rotator::rot)
         {
            Rotator::rot->setRotation(direction,direction2);
            Rotator::rot->removeTransform();
         }
      }

   }*/
}

Rotator *RotatorList::find(osg::Node *n)
{
    reset();
    while (current())
    {
        if (current()->node == n)
            return (current());

        next();
    }

    return (NULL);
}

RotatorList *RotatorList::instance()
{
    static RotatorList *rotatorlist = NULL;
    if (rotatorlist == NULL)
        rotatorlist = new RotatorList();
    return rotatorlist;
}
