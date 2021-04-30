#include "coverDaemon.h"
#include <boost/algorithm/string.hpp>
#include <config/CoviseConfig.h>
#include <config/coConfig.h>
#include <net/covise_host.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <util/coSpawnProgram.h>
#include <thread>
#include <chrono>
using namespace covise;

ICover::ICover(const std::string &masterHost, int masterPort) : m_host(masterHost), m_port(masterPort) {}

bool CoverSlave::startup()
{
    Host h{m_host.c_str()};
    std::cerr << "connecting to master " << m_host << " on port " << m_port << std::endl;
    do
    {
        m_conn.reset(new covise::ClientConnection{&h, m_port, 0, 0, 1});
        std::this_thread::sleep_for(std::chrono::seconds(1));
    } while (!m_conn->is_connected());
    m_sn.reset(new QSocketNotifier(m_conn->get_id(NULL), QSocketNotifier::Read));
    connect(m_sn.get(), &QSocketNotifier::activated, this, &CoverSlave::handleMessage);
    std::cerr << "connected to master" << std::endl;
    return true;
    std::cerr << "failed to connect" << std::endl;
    return false;
}

bool CoverSlave::isMaster() const
{
    return false;
}

void CoverSlave::handleMessage()
{
    std::cerr << "CoverDaemon::handleMessage()" << std::endl;
    Message msg;
    m_conn->recv_msg(&msg);
    if (msg.type != covise::COVISE_MESSAGE_SOCKET_CLOSED)
    {
        TokenBuffer tb{&msg};
        int id, port;
        tb >> id >> port;
        auto slavesStr = coCoviseConfig::getEntry("slaves", "COVER.MultiPC.CoverDaemon");
        std::vector<std::string> slaves;
        boost::split(slaves, slavesStr, boost::is_any_of(","));

        if (slaves[id - 1] == Host::getHostaddress() || slaves[id - 1] == Host::getHostname())
        {
            std::cerr << "starting cover slave " << id << std::endl;
            std::vector<std::string> args;
            args.push_back("-c");
            args.push_back(std::to_string(id));
            args.push_back(m_host);
            args.push_back(std::to_string(port));
            args.push_back(m_host);
            spawnProgram("opencover", args);
        }
    }
    else
    {
        startup();
    }
}

bool CoverMaster::startup()
{
    m_serverConn = m_connections.tryAddNewListeningConn<covise::ServerConnection>(m_port, 0, 0);
    m_sn.reset(new QSocketNotifier(m_serverConn->get_id(NULL), QSocketNotifier::Read));
    QObject::connect(m_sn.get(), SIGNAL(activated(int)), this, SLOT(test()));

    connect(m_sn.get(), &QSocketNotifier::activated, this, &CoverMaster::handleConnections);
    std::cerr << "waiting for slaves to connect on port " << m_port << std::endl;
    return true;
}

bool CoverMaster::isMaster() const
{
    return true;
}

void CoverMaster::test()
{
    std::cerr << "message received" << std::endl;
}

void CoverMaster::handleConnections()
{
    const auto *conn = m_connections.check_for_input(0.0001f);
    std::unique_ptr<covise::Connection> clientConn;
    if (conn == m_serverConn) //tcp connection to server port
    {
        clientConn = m_serverConn->spawn_connection();
    }
    if (clientConn)
    {
        struct linger linger;
        linger.l_onoff = 0;
        linger.l_linger = 0;
        setsockopt(clientConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));
        auto sn = m_slavesSNs.emplace(m_slavesSNs.end(), new QSocketNotifier(clientConn->get_id(nullptr), QSocketNotifier::Read));
        connect(sn->get(), &QSocketNotifier::activated, this, &CoverMaster::handleMessage);
        m_slaves.emplace_back(m_connections.add(std::move(clientConn)));
        std::cerr << "slave connected" << std::endl;
    }
}

void CoverMaster::handleMessage()
{
    std::cerr << "CoverDaemon::handleMessage()" << std::endl;
    auto conn = m_connections.wait_for_input();
    Message msg;
    conn->recv_msg(&msg);
    if (msg.type != covise::COVISE_MESSAGE_SOCKET_CLOSED)
    {
        for (auto s : m_slaves)
            s->sendMessage(&msg);
    }
    else
    {
        m_slavesSNs.erase(std::remove_if(m_slavesSNs.begin(), m_slavesSNs.end(), [&conn](const std::unique_ptr<QSocketNotifier> &sn) {
            int i = sn->socket();
            return i == conn->get_id(nullptr);
        }));
        m_slaves.erase(std::remove(m_slaves.begin(), m_slaves.end(), conn));
        m_connections.remove(conn);
        std::cerr << "conn closed" << std::endl;
    }
}

CoverDaemon::CoverDaemon()
{
    auto coviseDaemonHost = coCoviseConfig::getEntry("COVER.MultiPC.CoverDaemon");
    if (!coviseDaemonHost.empty())
    {
        auto port = coCoviseConfig::getInt("port", "COVER.MultiPC.CoverDaemon", 0);
        if (coviseDaemonHost == covise::Host::getHostaddress())
        {
            m_cover.reset(new CoverMaster(coviseDaemonHost, port));
        }
        else
        {
            m_cover.reset(new CoverSlave(coviseDaemonHost, port));
        }
        m_cover->startup();
    }
}

bool CoverDaemon::isMaster() const
{
    if (m_cover)
    {
        return m_cover->isMaster();
    }
    return true;
}