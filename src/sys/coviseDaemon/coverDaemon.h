#ifndef COVISE_DAEMON_COVER_DAEMON_H
#define COVISE_DAEMON_COVER_DAEMON_H

#include <QObject>
#include <QSocketNotifier>
#include <net/covise_connect.h>
#include <vector>

struct ICover : QObject
{
    ICover(const std::string &masterHost, int masterPort);
    virtual bool startup() = 0;
    virtual bool isMaster() const = 0;

protected:
    int m_port;
    std::string m_host;
    std::unique_ptr<QSocketNotifier> m_sn;
};

struct CoverMaster : ICover
{
    Q_OBJECT

    using ICover::ICover;
    bool startup() override;
    bool isMaster() const override;
private slots:
    void handleMessage();
    void test();

private:
    covise::ConnectionList m_connections;
    const covise::ServerConnection *m_serverConn;
    std::vector<const covise::Connection *> m_slaves;
    std::vector<std::unique_ptr<QSocketNotifier>> m_slavesSNs;

    void handleConnections();
};

struct CoverSlave : ICover
{
    Q_OBJECT
    using ICover::ICover;
    bool startup() override;
    bool isMaster() const override;

private slots:

    void handleMessage();

private:
    std::unique_ptr<covise::ClientConnection> m_conn;
};

class CoverDaemon
{
public:
    CoverDaemon();
    bool isMaster() const;

private:
    std::unique_ptr<ICover> m_cover;
};

#endif // !1