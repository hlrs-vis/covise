/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "IsoSurfacePoint.h"
#include "IsoSurfaceInteraction.h"
#include "IsoSurfacePlugin.h"
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <PluginUtil/coVR3DTransInteractor.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coInteractor.h>
#include <cover/coVRConfig.h>
#include <config/CoviseConfig.h>

using namespace vrui;
using namespace opencover;
using covise::coCoviseConfig;

IsoSurfacePoint::IsoSurfacePoint(coInteractor *inter, IsoSurfacePlugin *pl)
{

    if (cover->debugLevel(2))
        fprintf(stderr, "CuttingSurfaceSphere::CuttingSurfaceSphere\n");
    plugin = pl;
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
    if (inter_->getFloatVectorParam(IsoSurfaceInteraction::ISOPOINT, dummy, p) != -1)
    {
        isoPoint_.set(p[0], p[1], p[2]);
    }

    // create and position pickinteractor
    pointPickInteractor_ = new coVR3DTransInteractor(isoPoint_, interSize, coInteraction::ButtonA, "hand", "Sphere_Center", coInteraction::Medium);
    pointPickInteractor_->hide();
    pointPickInteractor_->disableIntersection();

    // direct interactor
    if (!coVRConfig::instance()->has6DoFInput())
    {
        pointDirectInteractor_ = NULL;
    }
    else
    {
        pointDirectInteractor_ = new coTrackerButtonInteraction(coInteraction::ButtonA, "sphere", coInteraction::Medium);
    }
}
IsoSurfacePoint::~IsoSurfacePoint()
{
    delete pointPickInteractor_;
    if (pointDirectInteractor_)
        delete pointDirectInteractor_;
}

void
IsoSurfacePoint::preFrame()
{

    pointPickInteractor_->preFrame();

    if (showDirectInteractor_ && pointDirectInteractor_ && pointDirectInteractor_->wasStopped())
    {
        newModule_ = false;
        if (!wait_)
        {

            osg::Matrix pointerMat = cover->getPointerMat();
            //printMatrix("pointer", pointerMat);
            osg::Matrix w_to_o = cover->getInvBaseMat();

            osg::Vec3 point_w, point_o;
            point_w = pointerMat.getTrans();
            point_o = point_w * w_to_o;
            isoPoint_ = point_o;
            pointPickInteractor_->updateTransform(isoPoint_);

            plugin->getSyncInteractors(inter_);
            plugin->setVectorParam(IsoSurfaceInteraction::ISOPOINT, isoPoint_[0], isoPoint_[1], isoPoint_[2]);
            plugin->executeModule();

            wait_ = true;
        }
    }

    // pick interaction mode with center, radius has to be moved in the same distance

    if (showPickInteractor_ && pointPickInteractor_->wasStopped())
    {
        if (!wait_)
        {
            isoPoint_ = pointPickInteractor_->getPos();
            plugin->getSyncInteractors(inter_);
            plugin->setVectorParam(IsoSurfaceInteraction::ISOPOINT, isoPoint_[0], isoPoint_[1], isoPoint_[2]);
            plugin->executeModule();

            wait_ = true;
        }
    }
}
void
IsoSurfacePoint::update(coInteractor *inter)
{
    if (wait_)
    {
        wait_ = false;
    }
    inter_ = inter;

    float *p = NULL;
    int dummy; // we know that the vector has 3 elements
    if (inter_->getFloatVectorParam(IsoSurfaceInteraction::ISOPOINT, dummy, p) != -1)
    {
        isoPoint_.set(p[0], p[1], p[2]);
        pointPickInteractor_->updateTransform(isoPoint_);
    }
}

void
IsoSurfacePoint::showPickInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nIsoSurfacePoint::showPickInteractor\n");

    showPickInteractor_ = true;
    pointPickInteractor_->show();
    pointPickInteractor_->enableIntersection();
}

void
IsoSurfacePoint::hidePickInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nIsoSurfacePoint::hide\n");

    showPickInteractor_ = false;
    pointPickInteractor_->hide();
    pointPickInteractor_->disableIntersection();
}

void
IsoSurfacePoint::showDirectInteractor()
{
    showDirectInteractor_ = true;
    if (pointDirectInteractor_ && !pointDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(pointDirectInteractor_);
    }
}

void
IsoSurfacePoint::hideDirectInteractor()
{

    showDirectInteractor_ = false;

    if (pointDirectInteractor_ && pointDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->unregisterInteraction(pointDirectInteractor_);
    }
}

void
IsoSurfacePoint::setNew()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfaceSphere::setNew\n");

    newModule_ = true;

    if (pointDirectInteractor_ && !pointDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(pointDirectInteractor_);
    }
}
