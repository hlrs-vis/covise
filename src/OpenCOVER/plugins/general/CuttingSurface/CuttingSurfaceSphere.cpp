/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "CuttingSurfaceSphere.h"
#include "CuttingSurfaceInteraction.h"
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <PluginUtil/coVR3DTransInteractor.h>
#include <cover/coInteractor.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <config/CoviseConfig.h>

using namespace opencover;
using namespace vrui;
using covise::coCoviseConfig;

CuttingSurfaceSphere::CuttingSurfaceSphere(coInteractor *inter)
{

    if (cover->debugLevel(2))
        fprintf(stderr, "CuttingSurfaceSphere::CuttingSurfaceSphere\n");

    inter_ = inter;
    newModule_ = false;
    wait_ = false;
    showPickInteractor_ = false;
    showDirectInteractor_ = false;

    // default size for all interactors
    float interSize = -1.f;
    // if defined, COVER.IconSize overrides the default
    interSize = coCoviseConfig::getFloat("COVER.IconSize", interSize);
    // if defined, COVERConfigCuttingSurfacePlugin.IconSize overrides both
    interSize = coCoviseConfig::getFloat("COVER.Plugin.Cuttingsurface.IconSize", interSize);

    // extract parameters from covise
    float *p = NULL;
    int dummy; // we know that the vector has 3 elements
    if (inter_->getFloatVectorParam(CuttingSurfaceInteraction::VERTEX, dummy, p) != -1)
    {
        centerPoint_.set(p[0], p[1], p[2]);
    }

    //inter_->getFloatScalarParam(CuttingSurfaceInteraction::SCALAR, radius_);
    //radiusPoint_.set(p[0], p[1], p[2]+radius_);
    if (inter_->getFloatVectorParam(CuttingSurfaceInteraction::POINT, dummy, p) != -1)
    {
        radiusPoint_.set(p[0], p[1], p[2]);
    }

    // create and position pickinteractor
    sphereCenterPickInteractor_ = new coVR3DTransInteractor(centerPoint_, interSize, coInteraction::ButtonA, "hand", "Sphere_Center", coInteraction::Medium);
    sphereCenterPickInteractor_->hide();
    sphereCenterPickInteractor_->disableIntersection();

    sphereRadiusPickInteractor_ = new coVR3DTransInteractor(radiusPoint_, interSize, coInteraction::ButtonA, "hand", "Sphere_Radius", coInteraction::Medium);
    sphereRadiusPickInteractor_->hide();
    sphereRadiusPickInteractor_->disableIntersection();

    // direct interactor
    if (!coVRConfig::instance()->has6DoFInput())
    {
        sphereDirectInteractor_ = NULL;
    }
    else
    {
        sphereDirectInteractor_ = new coTrackerButtonInteraction(coInteraction::ButtonA, "sphere", coInteraction::Medium);
    }
}
CuttingSurfaceSphere::~CuttingSurfaceSphere()
{
    delete sphereCenterPickInteractor_;
    delete sphereRadiusPickInteractor_;
    if (sphereDirectInteractor_)
        delete sphereDirectInteractor_;
}

void
CuttingSurfaceSphere::preFrame()
{

    sphereCenterPickInteractor_->preFrame();
    sphereRadiusPickInteractor_->preFrame();

    // direct interaction mode
    if (showDirectInteractor_ && sphereDirectInteractor_ && sphereDirectInteractor_->wasStarted())
    {
        osg::Matrix pointerMat = cover->getPointerMat();
        //printMatrix("pointer", pointerMat);
        osg::Matrix w_to_o = cover->getInvBaseMat();

        osg::Vec3 centerPoint_w, centerPoint_o;
        centerPoint_w = pointerMat.getTrans();
        centerPoint_o = centerPoint_w * w_to_o;
        centerPoint_ = centerPoint_o;
        sphereCenterPickInteractor_->updateTransform(centerPoint_);
        showGeometry();
        if (showPickInteractor_)
        {
            tmpHidePickInteractor();
        }
    }
    if (showDirectInteractor_ && sphereDirectInteractor_ && sphereDirectInteractor_->isRunning())
    {
        updateGeometry();
    }
    if (showDirectInteractor_ && sphereDirectInteractor_ && sphereDirectInteractor_->wasStopped())
    {
        newModule_ = false;
        if (!wait_)
        {

            osg::Matrix pointerMat = cover->getPointerMat();
            //printMatrix("pointer", pointerMat);
            osg::Matrix w_to_o = cover->getInvBaseMat();

            osg::Vec3 radiusPoint_w, radiusPoint_o;
            radiusPoint_w = pointerMat.getTrans();
            radiusPoint_o = radiusPoint_w * w_to_o;
            radiusPoint_ = radiusPoint_o;
            sphereRadiusPickInteractor_->updateTransform(radiusPoint_);
            hideGeometry();
            if (showPickInteractor_)
            {
                showPickInteractor();
            }
            inter_->setVectorParam(CuttingSurfaceInteraction::VERTEX, centerPoint_[0], centerPoint_[1], centerPoint_[2]);
            inter_->setVectorParam(CuttingSurfaceInteraction::POINT, radiusPoint_[0], radiusPoint_[1], radiusPoint_[2]);
            inter_->executeModule();

            wait_ = true;
        }
    }

    // pick interaction mode with center, radius has to be moved in the same distance

    if (showPickInteractor_ && sphereCenterPickInteractor_->wasStarted())
    {
        centerPoint_ = sphereCenterPickInteractor_->getPos();
        radiusPoint_ = sphereRadiusPickInteractor_->getPos();

        // save diff vector between center and radius point
        diff_ = radiusPoint_ - centerPoint_;
        radius_ = diff_.length();
        diff_.normalize();
        showGeometry();
    }
    if (showPickInteractor_ && sphereCenterPickInteractor_->isRunning())
    {
        // update also radius point interactor
        centerPoint_ = sphereCenterPickInteractor_->getPos();
        radiusPoint_.set(centerPoint_[0] + diff_[0] * radius_, centerPoint_[1] + diff_[1] * radius_, centerPoint_[2] + diff_[2] * radius_);
        sphereRadiusPickInteractor_->updateTransform(radiusPoint_);
        updateGeometry();
    }
    if (showPickInteractor_ && sphereCenterPickInteractor_->wasStopped())
    {
        if (!wait_)
        {
            centerPoint_ = sphereCenterPickInteractor_->getPos();
            radiusPoint_.set(centerPoint_[0] + diff_[0] * radius_, centerPoint_[1] + diff_[1] * radius_, centerPoint_[2] + diff_[2] * radius_);
            sphereRadiusPickInteractor_->updateTransform(radiusPoint_);
            hideGeometry();

            inter_->setVectorParam(CuttingSurfaceInteraction::VERTEX, centerPoint_[0], centerPoint_[1], centerPoint_[2]);
            inter_->setVectorParam(CuttingSurfaceInteraction::POINT, radiusPoint_[0], radiusPoint_[1], radiusPoint_[2]);
            //inter_->setScalarParam(CuttingSurfaceInteraction::SCALAR, radius_);
            inter_->executeModule();

            wait_ = true;
        }
    }

    // pick interaction mode with radius, center keeps position
    if (showPickInteractor_ && sphereRadiusPickInteractor_->wasStarted())
    {
        showGeometry();
    }
    if (showPickInteractor_ && sphereRadiusPickInteractor_->isRunning())
    {
        // update also radius point interactor
        updateGeometry();
    }
    if (showPickInteractor_ && sphereRadiusPickInteractor_->wasStopped())
    {
        if (!wait_)
        {
            radiusPoint_ = sphereRadiusPickInteractor_->getPos();
            hideGeometry();
            //diff = centerPoint_ - radiusPoint_;
            //radius_ = diff.length();
            inter_->setVectorParam(CuttingSurfaceInteraction::VERTEX, centerPoint_[0], centerPoint_[1], centerPoint_[2]);
            inter_->setVectorParam(CuttingSurfaceInteraction::POINT, radiusPoint_[0], radiusPoint_[1], radiusPoint_[2]);
            //inter_->setScalarParam(CuttingSurfaceInteraction::SCALAR, radius_);
            inter_->executeModule();

            wait_ = true;
        }
    }
}
void
CuttingSurfaceSphere::update(coInteractor *inter)
{
    if (wait_)
    {
        wait_ = false;
    }
    inter_ = inter;

    float *p = NULL;
    int dummy; // we know that the vector has 3 elements
    if (inter_->getFloatVectorParam(CuttingSurfaceInteraction::VERTEX, dummy, p) != -1)
    {
        centerPoint_.set(p[0], p[1], p[2]);
    }

    if (inter_->getFloatVectorParam(CuttingSurfaceInteraction::POINT, dummy, p) != -1)
    {
        radiusPoint_.set(p[0], p[1], p[2]);
    }

    //inter_->getFloatScalarParam(CuttingSurfaceInteraction::SCALAR, radius_);
    //radiusPoint_.set(p[0], p[1], p[2]+radius_);

    sphereCenterPickInteractor_->updateTransform(centerPoint_);
    sphereRadiusPickInteractor_->updateTransform(radiusPoint_);
}

void
CuttingSurfaceSphere::showPickInteractor()
{
    if (cover->debugLevel(0))
        fprintf(stderr, "\ncuttingSurfacePlane::showPickInteractor\n");

    showPickInteractor_ = true;
    sphereCenterPickInteractor_->show();
    sphereCenterPickInteractor_->enableIntersection();
    sphereRadiusPickInteractor_->show();
    sphereRadiusPickInteractor_->enableIntersection();
}

void
CuttingSurfaceSphere::hidePickInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\ncuttingSurfacePlane::hide\n");

    showPickInteractor_ = false;
    sphereCenterPickInteractor_->hide();
    sphereRadiusPickInteractor_->hide();
    sphereCenterPickInteractor_->disableIntersection();
    sphereRadiusPickInteractor_->disableIntersection();
}

void
CuttingSurfaceSphere::tmpHidePickInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\ncuttingSurfacePlane::hide\n");

    sphereCenterPickInteractor_->hide();
    sphereRadiusPickInteractor_->hide();
    sphereCenterPickInteractor_->disableIntersection();
    sphereRadiusPickInteractor_->disableIntersection();
}

void
CuttingSurfaceSphere::showDirectInteractor()
{
    showDirectInteractor_ = true;
    if (sphereDirectInteractor_ && !sphereDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(sphereDirectInteractor_);
    }
}

void
CuttingSurfaceSphere::hideDirectInteractor()
{

    showDirectInteractor_ = false;

    if (sphereDirectInteractor_ && sphereDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->unregisterInteraction(sphereDirectInteractor_);
    }
}

void
CuttingSurfaceSphere::setNew()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceSphere::setNew\n");

    newModule_ = true;

    if (sphereDirectInteractor_ && !sphereDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(sphereDirectInteractor_);
    }
}

void
CuttingSurfaceSphere::showGeometry()
{
}
void
CuttingSurfaceSphere::hideGeometry()
{
}
void
CuttingSurfaceSphere::updateGeometry()
{
}
