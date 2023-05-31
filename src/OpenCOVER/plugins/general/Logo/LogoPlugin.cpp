/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/****************************************************************************\ 
**                                                            (C)2009 HLRS  **
**                                                                          **
** Description: Logo Plugin (displays a bitmap logo)                        **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                  **
**                                                                          **
** History:  								                                         **
** Feb-09  v1	    				       		                                   **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "LogoPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <iostream>
#include <config/CoviseConfig.h>
#include <config/coConfig.h>
#include <osg/MatrixTransform>

using namespace covise;

LogoPlugin::LogoPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, camera(NULL)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nLogoPlugin::LogoPlugin\n");
}

bool LogoPlugin::init()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nLogoPlugin::init\n");

    doHide = false;
    hidden = false;

    logoTime = (double)coCoviseConfig::getFloat("time", "COVER.Plugin.Logo", -1.0);
    if (logoTime > 0)
        doHide = true;
    else
        doHide = false;

    camera = new osg::Camera;

    // set the view matrix
    camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    camera->setViewMatrix(osg::Matrix::identity());

    camera->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);

    // only clear the depth buffer
    camera->setClearMask(GL_DEPTH_BUFFER_BIT);

    // no lighting
    camera->getOrCreateStateSet()->setMode(GL_LIGHTING, GL_FALSE);
    camera->getOrCreateStateSet()->setMode(GL_DEPTH_TEST, GL_FALSE);

    // draw subgraph after main camera view.
    camera->setRenderOrder(osg::Camera::NESTED_RENDER); // if POST_RENDER is used, the logo doesn't appear on snapshots
    //stateset->setNestRenderBins(false);

    cover->getScene()->addChild(camera.get());

    defaultLogo = new Logo("", camera);
    recordingLogo = new Logo(".Recording", camera);

    defaultLogo->show();
    hudTime = cover->frameTime();

    return true;
}

// this is called if the plugin is removed at runtime
LogoPlugin::~LogoPlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nLogoPlugin::~LogoPlugin\n");
}

bool LogoPlugin::destroy()
{
    defaultLogo->hide();
    recordingLogo->hide();
    return true;
}

void
LogoPlugin::preFrame()
{
    // set projection matrix
    float xsize;
    float ysize;
    if ((coVRConfig::instance()->viewports[0].viewportXMax - coVRConfig::instance()->viewports[0].viewportXMin) == 0)
    {
        xsize = coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sx;
        ysize = coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sy;
    }
    else
    {
        xsize = coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sx * (coVRConfig::instance()->viewports[0].viewportXMax - coVRConfig::instance()->viewports[0].viewportXMin);
        ysize = coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sy * (coVRConfig::instance()->viewports[0].viewportYMax - coVRConfig::instance()->viewports[0].viewportYMin);
    }
    camera->setProjectionMatrixAsOrtho2D(0, xsize, 0, ysize);

    if (doHide)
    {
        if (cover->frameTime() - hudTime >= logoTime)
        {
            defaultLogo->hide();
            doHide = false;
            hidden = true;
        }
    }
}

void LogoPlugin::message(int toWhom, int, int, const void *data)
{
    if (!camera)
    {
        return;
    }
    const char *chbuf = (const char *)data;
    if (strncmp(chbuf, "startingCapture", strlen("startingCapture")) == 0)
    {
        defaultLogo->hide();
        recordingLogo->show();
    }
    if (strncmp(chbuf, "stoppingCapture", strlen("stoppingCapture")) == 0)
    {
        recordingLogo->hide();
        if (!hidden)
        {
            defaultLogo->show();
        }
    }
}

COVERPLUGIN(LogoPlugin)
