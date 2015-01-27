/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ElementaryParticleInteractor.h"
#include <osg/ShapeDrawable>
#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPluginSupport.h>
using namespace opencover;
using namespace covise;

ElementaryParticleInteractor::ElementaryParticleInteractor(osg::Vec3 initialPos, osg::Vec3 normal, float size, string geoFileName)
    : coVR2DTransInteractor(initialPos, normal, size, coInteraction::ButtonA, "ElementaryParticle", "Menu", coInteraction::Medium)
{

    normal_ = normal;
    initialPos_ = initialPos;
    if (cover->debugLevel(3))
        fprintf(stderr, "ElementaryParticleInteractor::ElementaryParticleInteractor\n");
    geoFileName_ = geoFileName;
    createGeometry();

    animate_ = false;
    radiusAnimate_ = false;
}

ElementaryParticleInteractor::~ElementaryParticleInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "ElementaryParticleInteractor::~ElementaryParticleInteractor\n");
}

void ElementaryParticleInteractor::createGeometry()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "ElementaryParticleInteractor::createGeometry\n");

    geometryNode = coVRFileManager::instance()->loadIcon(geoFileName_.c_str());
    //fprintf(stderr, "%s = %f\n", geofilename_.c_str(), geometryNode->getBound().radius());

    // remove old geometry
    scaleTransform->removeChild(0, scaleTransform->getNumChildren());

    // add geometry to node
    scaleTransform->addChild(geometryNode.get());
}

void ElementaryParticleInteractor::resetPosition()
{
    updateTransform(initialPos_, normal_);
}

void ElementaryParticleInteractor::preFrame()
{
    coVRIntersectionInteractor::preFrame();
    if (animate_)
    {
        //fprintf(stderr,"ElementaryParticleInteractor::animate\n");
        animTime_ += cover->frameDuration();

        if (animTime_ < 1.0)
        {
            float factor = animTime_ / 1.0 * length_;
            osg::Vec3 pos = animStartPos_ + animVector_ * factor;
            updateTransform(pos, normal_);
        }
        else
        {
            updateTransform(animEndPos_, normal_);
            animate_ = false;
            enableIntersection();
        }
    }
    if (radiusAnimate_)
    {
        //fprintf(stderr,"ElementaryParticleInteractor::radiusAnimate_\n");
        radiusAnimTime_ += cover->frameDuration();

        if (radiusAnimTime_ < 1.0)
        {
            float factor = radiusAnimTime_ / 1.0 * radiusAnimDiffAngle_;
            float ac;
            ac = radiusAnimStartAngle_ + factor;

            if (ac > 2 * M_PI)
                ac = ac - 2 * M_PI;
            //fprintf(stderr,"ac=%f\n", ac*180/M_PI);
            osg::Vec3 pos(radiusAnimRadius_ * sin(ac), 0, radiusAnimRadius_ * cos(ac));
            updateTransform(pos, normal_);
        }
        else
        {
            osg::Vec3 pos(radiusAnimRadius_ * sin(radiusAnimEndAngle_), 0, radiusAnimRadius_ * cos(radiusAnimEndAngle_));
            updateTransform(pos, normal_);
            radiusAnimate_ = false;
            enableIntersection();
        }
    }
}

void ElementaryParticleInteractor::startAnimation(osg::Vec3 to)
{
    animEndPos_ = to;
    animate_ = true;
    animTime_ = 0.0;
    animStartPos_ = getPosition();
    //fprintf(stderr,"moving to initial pos [%f %f %f]\n", initialPos_[0], initialPos_[1], initialPos_[2]);
    animVector_ = animEndPos_ - animStartPos_;
    length_ = animVector_.length();
    animVector_.normalize();
    disableIntersection();
}

void ElementaryParticleInteractor::startRadiusAnimation(float r, float astart, float aend)
{
    if (astart > 2 * M_PI)
        astart = astart - 2 * M_PI;

    if (aend > 2 * M_PI)
        astart = aend - 2 * M_PI;

    if (astart > 180)
        minus_ = false;
    radiusAnimate_ = true;
    radiusAnimTime_ = 0.0;
    radiusAnimStartAngle_ = astart;

    radiusAnimEndAngle_ = aend;
    radiusAnimRadius_ = r;
    float a1 = radiusAnimEndAngle_ - radiusAnimStartAngle_;
    float a2 = 2 * M_PI - (radiusAnimStartAngle_ - radiusAnimEndAngle_);
    if (fabs(a1) < fabs(a2))
        radiusAnimDiffAngle_ = a1;
    else
        radiusAnimDiffAngle_ = a2;

    //fprintf(stderr,"ElementaryParticleInteractor::startRadiusAnimation from %f to %f diff=%f\n", astart*180/M_PI, aend*180/M_PI, radiusAnimDiffAngle_*180/M_PI);

    if (fabs(radiusAnimDiffAngle_) > M_PI)
        radiusAnimDiffAngle_ = 2 * M_PI;
    //fprintf(stderr,"ElementaryParticleInteractor::startRadiusAnimation from %f to %f diff=%f\n", astart*180/M_PI, aend*180/M_PI, radiusAnimDiffAngle_*180/M_PI);

    disableIntersection();
}
