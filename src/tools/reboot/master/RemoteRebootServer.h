/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef REMOTEREBOOTSERVER_H
#define REMOTEREBOOTSERVER_H

#include <qserversocket.h>
#include <qstring.h>

class RemoteRebootMaster;

class RemoteRebootServer : public QServerSocket
{

    Q_OBJECT

public:
    RemoteRebootServer(Q_UINT16 port, RemoteRebootMaster *parent);
    virtual ~RemoteRebootServer();

    virtual void newConnection(int socket);

private:
    RemoteRebootMaster *parent;
    bool connected;
};
#endif
