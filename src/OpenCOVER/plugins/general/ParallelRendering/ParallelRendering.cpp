/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                           (C)2007 HLRS **
 **                                                                        **
 ** Description: ParallelRendering Plugin                                  **
 **                                                                        **
 **                                                                        **
 ** Author: Andreas Kopecki                                                **
 **         Florian Niebling                                               **
 **                                                                        **
 ** History:                                                               **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include <config/coConfig.h>

#include "ParallelRendering.h"
#include "ParallelRenderingClientIBVerbs.h"
#include "ParallelRenderingServerIBVerbs.h"
#include "ParallelRenderingClientSocket.h"
#include "ParallelRenderingServerSocket.h"

#include "ParallelRenderingOGLTexQuadCompositor.h"

#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

#include <osg/Camera>

#include <iostream>

using namespace std;
using namespace osg;

ParallelRendering::ParallelRendering()
{

    server = 0;
    client = 0;
    compositors = 0;

    coConfig *config = coConfig::getInstance();

    compositorRenders = config->isOn("compositorRenders", "COVER.ParallelRendering", true);

    tileX = config->getInt("x", "COVER.ParallelRendering"); // Doesn't do anything yet
    tileY = config->getInt("y", "COVER.ParallelRendering"); // Doesn't do anything yet
    number = config->getInt("number", "COVER.ParallelRendering");

    std::string network = ((QString)config->getString("interconnect", "COVER.ParallelRendering", "socket")).toStdString();

    compositor = ((QString)config->getString("compositor", "COVER.ParallelRendering", "")).toStdString();

    // Check if we are master
    if (number == 0)
    {
        compositors = new ParallelRenderingCompositor *[cover->numScreens];

        if (!network.compare("ibverbs"))
        {
#ifdef HAVE_IBVERBS
            server = new ParallelRenderingServerIBVerbs(cover->numScreens, compositorRenders);
#else
            cerr << "ERROR: you need to compile with IBVerbs support, falling back to socket interface" << endl;
#endif
        }

        if (!server)
            server = new ParallelRenderingServerSocket(cover->numScreens, compositorRenders);

        server->start();

        for (int ctr = 0; ctr < cover->numScreens; ctr++)
        {
            compositors[ctr] = new ParallelRenderingOGLTexQuadCompositor(ctr);
            if (ctr == 0 && compositorRenders)
                compositors[ctr]->initSlaveChannel(false);
            else
                compositors[ctr]->initSlaveChannel(true);

            server->addCompositor(ctr, compositors[ctr]);
        }
    }
    else
    {
        if (this->number != -1)
        {
            cerr << "ParallelRendering::<init> info: new client " << number << endl;

            if (!network.compare("ibverbs"))
            {
#ifdef HAVE_IBVERBS
                client = new ParallelRenderingClientIBVerbs(number, const_cast<char *>(compositor.c_str()));
#else
                cerr << "ERROR: you need to compile with IBVerbs support, falling back to socket interface" << endl;
#endif
            }

            if (!client)
                client = new ParallelRenderingClientSocket(number, const_cast<char *>(compositor.c_str()));

            client->start();
        }
    }
}

ParallelRendering::~ParallelRendering()
{

    if (server)
        server->cancel();
    delete server;
    if (compositors)
    {
        for (int ctr = 0; ctr < cover->numScreens; ctr++)
            delete compositors[ctr];
        delete[] compositors;
    }
    if (client)
        client->cancel();
    delete client;
}

void ParallelRendering::preSwapBuffers()
{

    if (number == 0) /* server */
    {

        if (server)
        {
            if (!server->isConnected())
                server->acceptConnection();

            server->render();
        }
    }
    else /* clients */
    {

        if (client)
        {
            if (!client->isConnected())
                client->connectToServer();

            client->readBackImage();
            client->send();
        }
    }
}

ParallelRendering *plugin = 0;

int coVRInit(coVRPlugin *)
{

    plugin = new ParallelRendering();
    if (plugin)
        return (1);
    else
        return (0);
}

void coVRDelete(coVRPlugin *)
{

    delete plugin;
}

void coVRPreSwapBuffers()
{

    if (plugin)
        plugin->preSwapBuffers();
}
