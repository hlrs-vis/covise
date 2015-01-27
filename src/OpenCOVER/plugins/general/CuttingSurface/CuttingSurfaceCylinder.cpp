/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "CuttingSurfaceCylinder.h"
#include "CuttingSurfaceInteraction.h"
#include <OpenVRUI/coInteraction.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coInteractor.h>
#include <config/CoviseConfig.h>

using namespace vrui;
using namespace opencover;
using covise::coCoviseConfig;

CuttingSurfaceCylinder::CuttingSurfaceCylinder(coInteractor *inter)
{

    if (cover->debugLevel(2))
        fprintf(stderr, "CuttingSurfaceCylinder::CuttingSurfaceCylinder\n");

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
    cylCenterPickInteractor_ = new coVR3DTransInteractor(centerPoint_, interSize, coInteraction::ButtonA, "hand", "Cyl_Center", coInteraction::Medium);
    cylCenterPickInteractor_->hide();
    cylCenterPickInteractor_->disableIntersection();

    cylRadiusPickInteractor_ = new coVR3DTransInteractor(radiusPoint_, interSize, coInteraction::ButtonA, "hand", "Cyl_Radius", coInteraction::Medium);
    cylRadiusPickInteractor_->hide();
    cylRadiusPickInteractor_->disableIntersection();

    // direct interactor
    if (coVRConfig::instance()->mouseTracking())
    {
        cylDirectInteractor_ = NULL;
    }
    else
    {
        cylDirectInteractor_ = new coTrackerButtonInteraction(coInteraction::ButtonA, "sphere", coInteraction::Medium);
    }
}
CuttingSurfaceCylinder::~CuttingSurfaceCylinder()
{
    delete cylCenterPickInteractor_;
    delete cylRadiusPickInteractor_;
    if (cylDirectInteractor_)
        delete cylDirectInteractor_;
}

void
CuttingSurfaceCylinder::preFrame()
{
    cylCenterPickInteractor_->preFrame();
    cylRadiusPickInteractor_->preFrame();

    // direct interaction mode
    if (showDirectInteractor_ && cylDirectInteractor_ && cylDirectInteractor_->wasStarted())
    {
        osg::Matrix pointerMat = cover->getPointerMat();
        //printMatrix("pointer", pointerMat);
        osg::Matrix w_to_o = cover->getInvBaseMat();

        osg::Vec3 centerPoint_w, centerPoint_o;
        centerPoint_w = pointerMat.getTrans();
        centerPoint_o = centerPoint_w * w_to_o;
        centerPoint_ = centerPoint_o;
        cylCenterPickInteractor_->updateTransform(centerPoint_);
        showGeometry();
        if (showPickInteractor_)
        {
            tmpHidePickInteractor();
        }
    }
    if (showDirectInteractor_ && cylDirectInteractor_ && cylDirectInteractor_->isRunning())
    {
        updateGeometry();
    }
    if (showDirectInteractor_ && cylDirectInteractor_ && cylDirectInteractor_->wasStopped())
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
            cylRadiusPickInteractor_->updateTransform(radiusPoint_);
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

    if (showPickInteractor_ && cylCenterPickInteractor_->wasStarted())
    {
        centerPoint_ = cylCenterPickInteractor_->getPos();
        radiusPoint_ = cylRadiusPickInteractor_->getPos();

        // save diff vector between center and radius point
        diff_ = radiusPoint_ - centerPoint_;
        radius_ = diff_.length();
        diff_.normalize();
        showGeometry();
    }
    if (showPickInteractor_ && cylCenterPickInteractor_->isRunning())
    {
        // update also radius point interactor
        centerPoint_ = cylCenterPickInteractor_->getPos();
        radiusPoint_.set(centerPoint_[0] + diff_[0] * radius_, centerPoint_[1] + diff_[1] * radius_, centerPoint_[2] + diff_[2] * radius_);
        cylRadiusPickInteractor_->updateTransform(radiusPoint_);
        updateGeometry();
    }
    if (showPickInteractor_ && cylCenterPickInteractor_->wasStopped())
    {
        if (!wait_)
        {
            centerPoint_ = cylCenterPickInteractor_->getPos();
            radiusPoint_.set(centerPoint_[0] + diff_[0] * radius_, centerPoint_[1] + diff_[1] * radius_, centerPoint_[2] + diff_[2] * radius_);
            cylRadiusPickInteractor_->updateTransform(radiusPoint_);
            hideGeometry();

            inter_->setVectorParam(CuttingSurfaceInteraction::VERTEX, centerPoint_[0], centerPoint_[1], centerPoint_[2]);
            inter_->setVectorParam(CuttingSurfaceInteraction::POINT, radiusPoint_[0], radiusPoint_[1], radiusPoint_[2]);
            //inter_->setScalarParam(CuttingSurfaceInteraction::SCALAR, radius_);
            inter_->executeModule();

            wait_ = true;
        }
    }

    // pick interaction mode with radius, center keeps position
    if (showPickInteractor_ && cylRadiusPickInteractor_->wasStarted())
    {
        showGeometry();
    }
    if (showPickInteractor_ && cylRadiusPickInteractor_->isRunning())
    {
        // update also radius point interactor
        updateGeometry();
    }
    if (showPickInteractor_ && cylRadiusPickInteractor_->wasStopped())
    {
        if (!wait_)
        {
            radiusPoint_ = cylRadiusPickInteractor_->getPos();
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
CuttingSurfaceCylinder::update(coInteractor *inter)
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

    cylCenterPickInteractor_->updateTransform(centerPoint_);
    cylRadiusPickInteractor_->updateTransform(radiusPoint_);
}

void
CuttingSurfaceCylinder::showPickInteractor()
{
    if (cover->debugLevel(0))
        fprintf(stderr, "\ncuttingSurfacePlane::showPickInteractor\n");

    showPickInteractor_ = true;
    cylCenterPickInteractor_->show();
    cylCenterPickInteractor_->enableIntersection();
    cylRadiusPickInteractor_->show();
    cylRadiusPickInteractor_->enableIntersection();
}

void
CuttingSurfaceCylinder::hidePickInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\ncuttingSurfacePlane::hide\n");

    showPickInteractor_ = false;
    cylCenterPickInteractor_->hide();
    cylRadiusPickInteractor_->hide();
    cylCenterPickInteractor_->disableIntersection();
    cylRadiusPickInteractor_->disableIntersection();
}

void
CuttingSurfaceCylinder::tmpHidePickInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\ncuttingSurfacePlane::hide\n");

    cylCenterPickInteractor_->hide();
    cylRadiusPickInteractor_->hide();
    cylCenterPickInteractor_->disableIntersection();
    cylRadiusPickInteractor_->disableIntersection();
}

void
CuttingSurfaceCylinder::showDirectInteractor()
{
    showDirectInteractor_ = true;
    if (cylDirectInteractor_ && !cylDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(cylDirectInteractor_);
    }
}

void
CuttingSurfaceCylinder::hideDirectInteractor()
{

    showDirectInteractor_ = false;

    if (cylDirectInteractor_ && cylDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->unregisterInteraction(cylDirectInteractor_);
    }
}

void
CuttingSurfaceCylinder::setNew()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfacecyl::setNew\n");

    newModule_ = true;

    if (cylDirectInteractor_ && !cylDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(cylDirectInteractor_);
    }
}

void
CuttingSurfaceCylinder::showGeometry()
{
}
void
CuttingSurfaceCylinder::hideGeometry()
{
}
void
CuttingSurfaceCylinder::updateGeometry()
{
}
