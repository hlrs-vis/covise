/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CTRLGlobal.h"
#include "AccessGridDaemon.h"
#include <net/covise_connect.h>
#include <net/covise_host.h>

#include <config/CoviseConfig.h>

using namespace covise;

AccessGridDaemon::AccessGridDaemon(int port) // connect to a local AccessGridDaemon
{
    connect("localhost", port);
}

// connect to a remote AccessGridDaemon to start COVISE
AccessGridDaemon::AccessGridDaemon(const char *host, int port)
{
    connect(host, port);
}

AccessGridDaemon::~AccessGridDaemon()
{
    delete conn;
}

// connect to a remote AccessGridDaemon to start COVISE
void AccessGridDaemon::connect(const char *host, int port)
{
    DaemonPort = coCoviseConfig::getInt("port", "AccessGridDaemon", 31098);
    Host *h = new Host(host);
    conn = new ClientConnection(h, port, 0, (sender_type)0);
    CTRLGlobal::get_handle().controller->addConnection(conn);
    delete h;
}
