/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2004 HLRS  **
 **                                                                          **
 ** Description: TiledDisplay Plugin                                      **
 **                                                                          **
 **                                                                          **
 ** Author: Andreas Kopecki                                                  **
 **                                                                          **
 ** History:                                                                 **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <config/coConfig.h>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/RenderObject.h>
#include <config/CoviseConfig.h>

#include "TiledDisplay.h"
#include "TiledDisplayClientVV.h"
#include "TiledDisplayServerVV.h"
#include "TiledDisplayClientIBVerbs.h"
#include "TiledDisplayServerIBVerbs.h"

#include "TiledDisplayOGLTexQuadCompositor.h"
#include "TiledDisplayOSGTexQuadCompositor.h"

#include <osg/Camera>

#include <iostream>

using namespace std;
using namespace osg;

using covise::coConfig;

#define TILED_DISPLAY_TEX_SIZE 1024

TiledDisplay::TiledDisplay()
: coVRPlugin(COVER_PLUGIN_NAME)
{

    initPending = true;

    servers = 0;
    client = 0;
    //dimensions = 0;
    compositors = 0;

    coConfig *config = coConfig::getInstance();

    internalCompositor = config->isOn("internalCompositor", "COVER.TiledDisplay", false);

    tileX = config->getInt("x", "COVER.TiledDisplay"); // Doesn't do anything yet
    tileY = config->getInt("y", "COVER.TiledDisplay"); // Doesn't do anything yet
    number = config->getInt("number", "COVER.TiledDisplay");
    compositor = config->getString("compositor", "COVER.TiledDisplay", "viscose.hlrs.de");

    // Check if we are master and internal compositing is on
    if (internalCompositor && number == 0)
    {
        cerr << "TiledDisplay::<init> info: turning on internal compositor" << std::endl;
        if (coVRConfig::instance()->numScreens() != 4)
        {
            cerr << "TiledDisplay::<init> err: missing configuration: need exactly 4 screens and channels" << std::endl;
            exit(-1);
        }

        servers = new TiledDisplayServer *[4];
        compositors = new TiledDisplayCompositor *[4];
        //dimensions = new TiledDisplayDimension[4];

        // Just to make loops easier
        compositors[0] = 0;
        servers[0] = 0;

        for (int ctr = 1; ctr < 4; ++ctr)
        {
#ifdef HAVE_IBVERBS
            servers[ctr] = new TiledDisplayServerIBVerbs(ctr);
#else
            servers[ctr] = new TiledDisplayServerVV(ctr);
#endif
            // hijack channels
            compositors[ctr] = new TiledDisplayOSGTexQuadCompositor(ctr, servers[ctr]);
            compositors[ctr]->initSlaveChannel();
        }
    }
    else
    {
        if (this->number != -1)
        {
            cerr << "TiledDisplay::<init> info: new client " << number << std::endl;
#ifdef HAVE_IBVERBS
            client = new TiledDisplayClientIBVerbs(number, const_cast<char *>(compositor.c_str()));
#else
            client = new TiledDisplayClientVV(number, const_cast<char *>(compositor.c_str()));
#endif
        }
    }
}

TiledDisplay::~TiledDisplay()
{
    if (servers)
        for (int ctr = 0; ctr < 4; ++ctr)
            delete servers[ctr];
    delete[] servers;
    delete client;
}

/**
 * \brief Read the framebuffer after every frame and store it to an image.
 */
void TiledDisplay::operator()(const osg::Camera &cam) const
{
    if (initPending)
        return;

    if (client->isImageAvailable())
        client->readBackImage(cam);
}

void TiledDisplay::preFrame()
{
    // Register the post draw callback
    if (initPending && !internalCompositor)
    {
        cerr << "TiledDisplay::preFrame info: starting client " << number << std::endl;
        client->start();

        camera = coVRConfig::instance()->channels[0].camera;

        camera->setPostDrawCallback(this);

        const osg::Viewport *vp = camera->getViewport();
        width = (unsigned)vp->width();
        height = (unsigned)vp->height();

        cerr << "TiledDisplay::<init> info: created ClusterServer geo = ["
             << tileX << ", " << tileY << " / " << width << ", " << height << "]"
#ifdef TILE_ENCODE_JPEG
             << ", JPEG encoded"
#endif
             << std::endl;

        initPending = false;

        // Don't send data before the first frame
        return;
    }

    if (internalCompositor && number == 0)
    {
        // If internal compositing is on, get the textures from the slaves
        updateTextures();
    }
}

void TiledDisplay::updateTextures()
{

    if (!internalCompositor)
        return;

    if (initPending)
        cerr << "TiledDisplay::updateTextures info: updating textures" << std::endl;

    if (initPending)
    {
        for (int ctr = 1; ctr < 4; ++ctr)
        {
            cerr << "TiledDisplay::updateTextures info: starting server " << ctr << std::endl;
            servers[ctr]->start();
        }
    }

    bool allInSync = false;

    // Wait till all have the same frame
    while (!allInSync)
    {
        allInSync = true;
        for (int ctr = 1; ctr < 4; ++ctr)
        {
            allInSync &= compositors[ctr]->updateTextures();
        }
    }

    if (initPending)
        cerr << "TiledDisplay::updateTextures info: done updating textures" << std::endl;
    initPending = false;
}

COVERPLUGIN(TiledDisplay)
