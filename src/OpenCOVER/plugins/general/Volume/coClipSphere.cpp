/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/Matrix>

#include <config/CoviseConfig.h>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>

#include <OpenVRUI/coInteraction.h>

#include "coClipSphere.h"

using namespace boost;
using namespace opencover;
using namespace vrui;
using covise::coCoviseConfig;

coClipSphere::coClipSphere()
    : valid_(false)
    , active_(false)
    , interactorActive_(false)
    , radius_(1.0f)
{
    osg::Matrix m;
    // default size for all interactors
    float interSize = -1.f;
    // if defined, COVER.IconSize overrides the default
    interSize = coCoviseConfig::getFloat("COVER.IconSize", interSize);
    // if defined, COVERConfigCuttingSurfacePlugin.IconSize overrides both
    interSize = coCoviseConfig::getFloat("COVER.Plugin.Cuttingsurface.IconSize", interSize);

    pickInteractor_.reset(new coVR3DTransRotInteractor(m, interSize, coInteraction::ButtonA, "hand", "ClipSphere", coInteraction::Medium));
    pickInteractor_->hide();
}

void coClipSphere::preFrame()
{
    pickInteractor_->preFrame();

    if (interactorActive_)
    {
        pickInteractor_->show();
        pickInteractor_->enableIntersection();
    }
    else
    {
        pickInteractor_->hide();
    }
}

void coClipSphere::setValid(bool valid)
{
    valid_ = valid;
}

bool coClipSphere::valid() const
{
    return valid_;
}

void coClipSphere::setActive(bool active)
{
    active_ = active;
}

bool coClipSphere::active() const
{
    return active_;
}

void coClipSphere::setInteractorActive(bool interactorActive)
{
    interactorActive_ = interactorActive;
}

bool coClipSphere::interactorActive() const
{
    return interactorActive_;
}

void coClipSphere::setRadius(float radius)
{
    radius_ = radius;
}

float coClipSphere::radius() const
{
    return radius_;
}

void coClipSphere::setMatrix(osg::Matrix const& m)
{
    pickInteractor_->updateTransform(m);
}

osg::Vec3 coClipSphere::getPosition() const
{
    osg::Matrix m = pickInteractor_->getMatrix();
    return osg::Vec3(m(3, 0), m(3, 1), m(3, 2));
}
