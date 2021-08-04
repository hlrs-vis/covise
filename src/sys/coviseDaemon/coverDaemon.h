/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_DAEMON_COVER_DAEMON_H
#define COVISE_DAEMON_COVER_DAEMON_H

#include <QObject>
#include <QSocketNotifier>
#include <net/covise_connect.h>
#include <vector>

//Daemon to start COVER slaves in a multi monitor setup.
//Enabled by setting <Daemon port="my port" /> under the COVER section in the config file
struct CoverDaemon : QObject
{
    Q_OBJECT
public:
    CoverDaemon();

private:
    int m_port = 0;
    std::string m_host;
    std::unique_ptr<QSocketNotifier> m_sn;
    covise::ConnectionList m_connections;
    const covise::SimpleServerConnection *m_serverConn;
    void handleConnections();
    bool startup();
};

#endif // !1