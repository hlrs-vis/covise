/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef REMOTEREBOOTSLAVE_H
#define REMOTEREBOOTSLAVE_H

#include <qobject.h>
#include <qstring.h>

class QFile;
class QSocket;

class RemoteRebootSlave : public QObject
{

    Q_OBJECT

public:
    RemoteRebootSlave(const QString &host, int port);
    virtual ~RemoteRebootSlave();

private slots:
    void socketConnected();
    void socketConnectionClosed();
    void socketReadyRead();
    void socketError(int err);

private:
    void sendList();
    void setDefaultBoot(int defaultBoot);
    void getDefaultBoot();
    void reboot();

    QFile *grubConf;
    QSocket *socket;
};
#endif
