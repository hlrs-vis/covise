/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TiledDisplayCompositor.h"

#include "TiledDisplayServer.h"

#include <cover/coVRPluginSupport.h>

TiledDisplayCompositor::TiledDisplayCompositor(int channel, TiledDisplayServer *server)
{
    this->channel = channel;
    this->server = server;
    this->lastUpdate = 0.0;
}

TiledDisplayCompositor::~TiledDisplayCompositor()
{
}

bool TiledDisplayCompositor::updateTextures()
{

    double coverFrameTime = opencover::cover->frameTime();
    double serverFrameTime = server->getFrameTime();

    if (lastUpdate == coverFrameTime)
        return true;

#if 0
   if ((serverFrameTime == coverFrameTime)||(serverFrameTime == 0.0))
   {
#endif
    updateTexturesImplementation();
    lastUpdate = serverFrameTime;
    return true;
#if 0
   }
   else
   {
      return false;
   }
#endif
}
