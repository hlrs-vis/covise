/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_ACCESS_H
#define CTRL_ACCESS_H

namespace covise
{

class ClientConnection;

class AccessGridDaemon
{
public:
    AccessGridDaemon(int port); // connect to a local AccessGridDaemon
    // connect to a remote AccessGridDaemon to start COVISE
    AccessGridDaemon(const char *host, int port);
    ~AccessGridDaemon();
    ClientConnection *conn;
    int DaemonPort;

private:
    void connect(const char *host, int port); // connect to a AccessGridDaemon
};
}

#endif
