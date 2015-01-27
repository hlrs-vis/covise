/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SPAWNREMOTEDAEMON_H
#define SPAWNREMOTEDAEMON_H

#include <qobject.h>
#include "Spawn.h"

class QSocket;

class SpawnRemoteDaemon : public QObject, public Spawn
{

    Q_OBJECT

public:
    SpawnRemoteDaemon(RemoteRebootMaster *master,
                      const QString &localhostname, int localport,
                      const QString &hostname = "", int port = -1);

    virtual ~SpawnRemoteDaemon();

    virtual bool spawn();
    virtual QString getSpawnName() const;

private slots:
    void socketConnected();
    void socketConnectionClosed();
    void socketReadyRead();
    void socketError(int err);

protected:
    QString hostname;
    int port;

    QSocket *rdaemonSocket;
};
#endif
