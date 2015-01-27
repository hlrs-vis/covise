/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VRBServer.h"
#include <config/CoviseConfig.h>
#include <util/environment.h>

int main(int argc, char **argv)
{
    covise::setupEnvironment(argc, argv);
    VRBServer server;

    if (server.openServer() < 0)
    {
        return -1;
    }

    server.loop();
    return 0;
}
