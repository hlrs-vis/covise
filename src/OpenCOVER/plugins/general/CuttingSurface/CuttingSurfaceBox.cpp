/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "CuttingSurfaceBox.h"
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

CuttingSurfaceBox::CuttingSurfaceBox(coInteractor *inter)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "CuttingSurfaceBox::CuttingSurfaceBox\n");

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
    cornerBPickInteractor_ = new coVR3DTransInteractor(centerPoint_, interSize, coInteraction::ButtonA, "hand",
                                                       "Corner_A", coInteraction::Medium);
    cornerBPickInteractor_->hide();
    cornerBPickInteractor_->disableIntersection();

    cornerAPickInteractor_ = new coVR3DTransInteractor(radiusPoint_, interSize, coInteraction::ButtonA, "hand",
                                                       "Corner_B", coInteraction::Medium);
    cornerAPickInteractor_->hide();
    cornerAPickInteractor_->disableIntersection();

    // direct interactor
    if (!coVRConfig::instance()->has6DoFInput())
    {
        cornerDirectInteractor_ = NULL;
    }
    else
    {
        cornerDirectInteractor_ = new coTrackerButtonInteraction(coInteraction::ButtonA, "box", coInteraction::Medium);
    }
}
CuttingSurfaceBox::~CuttingSurfaceBox()
{
    delete cornerBPickInteractor_;
    delete cornerAPickInteractor_;
    if (cornerDirectInteractor_)
        delete cornerDirectInteractor_;
}

void CuttingSurfaceBox::preFrame()
{
    cornerBPickInteractor_->preFrame();
    cornerAPickInteractor_->preFrame();

    // direct interaction mode
    if (showDirectInteractor_ && cornerDirectInteractor_ && cornerDirectInteractor_->wasStarted())
    {
        osg::Matrix pointerMat = cover->getPointerMat();
        //printMatrix("pointer", pointerMat);
        osg::Matrix w_to_o = cover->getInvBaseMat();

        osg::Vec3 centerPoint_w, centerPoint_o;
        centerPoint_w = pointerMat.getTrans();
        centerPoint_o = centerPoint_w * w_to_o;
        centerPoint_ = centerPoint_o;
        cornerBPickInteractor_->updateTransform(centerPoint_);
        showGeometry();
        if (showPickInteractor_)
        {
            tmpHidePickInteractor();
        }
    }
    if (showDirectInteractor_ && cornerDirectInteractor_ && cornerDirectInteractor_->isRunning())
    {
        updateGeometry();
    }
    if (showDirectInteractor_ && cornerDirectInteractor_ && cornerDirectInteractor_->wasStopped())
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
            cornerAPickInteractor_->updateTransform(radiusPoint_);
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

    if (showPickInteractor_ && cornerBPickInteractor_->wasStarted())
    {
        centerPoint_ = cornerBPickInteractor_->getPos();
        radiusPoint_ = cornerAPickInteractor_->getPos();

        showGeometry();
    }
    if (showPickInteractor_ && cornerBPickInteractor_->isRunning())
    {
        // update also radius point interactor
        centerPoint_ = cornerBPickInteractor_->getPos();
        updateGeometry();
    }
    if (showPickInteractor_ && cornerBPickInteractor_->wasStopped())
    {
        if (!wait_)
        {
            centerPoint_ = cornerBPickInteractor_->getPos();
            hideGeometry();

            inter_->setVectorParam(CuttingSurfaceInteraction::VERTEX, centerPoint_[0], centerPoint_[1], centerPoint_[2]);
            inter_->setVectorParam(CuttingSurfaceInteraction::POINT, radiusPoint_[0], radiusPoint_[1], radiusPoint_[2]);
            //inter_->setScalarParam(CuttingSurfaceInteraction::SCALAR, radius_);
            inter_->executeModule();

            wait_ = true;
        }
    }

    // pick interaction mode with radius, center keeps position
    if (showPickInteractor_ && cornerAPickInteractor_->wasStarted())
    {
        showGeometry();
    }
    if (showPickInteractor_ && cornerAPickInteractor_->isRunning())
    {
        // update also radius point interactor
        updateGeometry();
    }
    if (showPickInteractor_ && cornerAPickInteractor_->wasStopped())
    {
        if (!wait_)
        {
            radiusPoint_ = cornerAPickInteractor_->getPos();
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
void CuttingSurfaceBox::update(coInteractor *inter)
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

    cornerBPickInteractor_->updateTransform(centerPoint_);
    cornerAPickInteractor_->updateTransform(radiusPoint_);
}

void CuttingSurfaceBox::showPickInteractor()
{
    if (cover->debugLevel(0))
        fprintf(stderr, "\nCuttingSurfaceBox::showPickInteractor\n");

    showPickInteractor_ = true;
    cornerBPickInteractor_->show();
    cornerBPickInteractor_->enableIntersection();
    cornerAPickInteractor_->show();
    cornerAPickInteractor_->enableIntersection();
}

void CuttingSurfaceBox::hidePickInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nCuttingSurfaceBox::hidePickInteractor\n");

    showPickInteractor_ = false;
    tmpHidePickInteractor();
}

void CuttingSurfaceBox::tmpHidePickInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nCuttingSurfaceBox::tmpHidePickInteractor\n");

    cornerBPickInteractor_->hide();
    cornerAPickInteractor_->hide();
    cornerBPickInteractor_->disableIntersection();
    cornerAPickInteractor_->disableIntersection();
}

void CuttingSurfaceBox::showDirectInteractor()
{
    showDirectInteractor_ = true;
    if (cornerDirectInteractor_ && !cornerDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(cornerDirectInteractor_);
    }
}

void CuttingSurfaceBox::hideDirectInteractor()
{
    showDirectInteractor_ = false;

    if (cornerDirectInteractor_ && cornerDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->unregisterInteraction(cornerDirectInteractor_);
    }
}

void CuttingSurfaceBox::setNew()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceBox::setNew\n");

    newModule_ = true;

    if (cornerDirectInteractor_ && !cornerDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(cornerDirectInteractor_);
    }
}

void CuttingSurfaceBox::showGeometry()
{
}
void CuttingSurfaceBox::hideGeometry()
{
}
void CuttingSurfaceBox::updateGeometry()
{
}
