/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef OPENCOVER_PARALLELRENDERING_PLUGIN_H
#define OPENCOVER_PARALLELRENDERING_PLUGIN_H
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
#include "ParallelRenderingDefines.h"

#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <string>

class ParallelRenderingClient;
class ParallelRenderingServer;
class ParallelRenderingDimension;
class ParallelRenderingCompositor;

class ParallelRendering
{

    enum
    {
        SOCKET = 1,
        IBVERBS = 2
    };

public:
    ParallelRendering();
    ~ParallelRendering();

    void preSwapBuffers();

private:
    ParallelRenderingClient *client;
    ParallelRenderingServer *server;
    ParallelRenderingCompositor **compositors;

    int tileX;
    int tileY;
    int number;

    unsigned int width;
    unsigned int height;

    bool initPending;
    bool compositorRenders;
    std::string compositor;
    int interconnect;
};

#endif
