#ifndef COVISE_DAEMON_COVER_DAEMON_H
#define COVISE_DAEMON_COVER_DAEMON_H

#include <QObject>
#include <QSocketNotifier>
#include <net/covise_connect.h>
#include <vector>

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