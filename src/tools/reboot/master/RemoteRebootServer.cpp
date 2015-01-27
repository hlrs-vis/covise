/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "RemoteRebootServer.h"
#include "RemoteRebootMaster.h"

#include <qsocket.h>

#include <iostream>
using namespace std;

RemoteRebootServer::RemoteRebootServer(Q_UINT16 port, RemoteRebootMaster *parent = 0)
    : QServerSocket((Q_UINT16)port, (int)1, (QObject *)parent)
{

    this->parent = parent;
    connected = false;
}

RemoteRebootServer::~RemoteRebootServer()
{
}

void RemoteRebootServer::newConnection(int socket)
{

    if (connected)
        return;

    connected = true;

    cerr << "RemoteRebootServer::newConnection info: Connection accepted" << endl;

    QSocket *s = new QSocket(parent);
    s->setSocket(socket);

    parent->setSlaveSocket(s);
}
