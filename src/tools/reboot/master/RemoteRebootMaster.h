/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef REMOTEREBOOTMASTER_H
#define REMOTEREBOOTMASTER_H

#include <qobject.h>
#include <qstring.h>
#include <qvaluevector.h>

#include "RemoteRebootConstants.h"

class QSocket;
class RemoteRebootServer;
class Spawn;
class UI;

class RemoteRebootMaster : public QObject
{

    Q_OBJECT

public:
    RemoteRebootMaster();
    virtual ~RemoteRebootMaster();

    void setSlaveSocket(QSocket *socket);

    UI *getUI();

    QStringList getBootEntries();
    bool setDefaultBoot(int defaultBoot);
    int getDefaultBoot();
    void reboot();

    const QValueVector<QString> &getArgv();

private slots:
    void slaveSocketConnected();
    void slaveSocketConnectionClosed();
    void slaveSocketReadyRead();

    void socketError(int err);

private:
    void spawnSlave(RemoteBootMethod method);

    QSocket *slaveSocket;
    Spawn *spawn;

    RemoteRebootServer *server;

    UI *userInterface;

protected:
    RemoteBootMethod method;
    int defaultBoot;
    char uiType;

    QValueVector<QString> argv;
};
#endif
