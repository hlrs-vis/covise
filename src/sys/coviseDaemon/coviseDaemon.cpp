/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coviseDaemon.h"

#include <config/CoviseConfig.h>
#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <net/message.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <net/tokenbuffer_serializer.h>
#include <net/tokenbuffer_util.h>
#include <util/coSpawnProgram.h>
#include <vrb/PrintClientList.h>
#include <vrb/ProgramType.h>
#include <vrb/SessionID.h>
#include <vrb/VrbSetUserInfoMessage.h>
#include <vrb/client/LaunchRequest.h>
#include <comsg/VRB_PERMIT_LAUNCH.h>
#include <comsg/PROXY.h>

#include <QTextStream>
#include <cassert>
#include <chrono>
#include <cmath>
#include <sstream>
#include <string>
#include <thread>

using namespace vrb;

CoviseDaemon::~CoviseDaemon()
{
    m_terminate = true;
    disconnect();
    if (m_thread)
    {
        m_thread->join();
    }
}

void CoviseDaemon::connect(const vrb::VrbCredentials &credentials)
{
    m_shouldBeConnected = true;
    {
        Guard g{m_mutex};
        m_credentials.reset(new vrb::VrbCredentials(credentials));
        m_newCredentials = true;
        m_clientList.clear();
    }
    if (!m_thread)
    {
        m_thread.reset(new std::thread([this]() { 
            qRegisterMetaType<vrb::Program>();
            qRegisterMetaType<std::vector<std::string>>();

            loop(); }));
    }
}

void CoviseDaemon::disconnect()
{
    if (m_shouldBeConnected)
    {
        m_client->shutdown();
        m_shouldBeConnected = false;
    }
}

void CoviseDaemon::printClientInfo()
{
    Guard g{m_mutex};
    std::vector<const vrb::RemoteClient *> partner;
    for (const auto &cl : m_clientList)
    {
        if (cl.sessionID() == m_client->sessionID())
        {
            partner.push_back(&cl);
        }
    }
    if (partner.empty())
    {
        std::cerr << "No partners available." << std::endl;
    }
    else
    {
        vrb::printClientInfo(partner);
    }
}

void CoviseDaemon::sendPermission(int clientID, bool permit)
{
    covise::VRB_PERMIT_LAUNCH msg(clientID, m_client->ID(), permit);
    covise::sendCoviseMessage(msg, *m_client);
}

void CoviseDaemon::sendLaunchRequest(Program p, int clientID, const std::vector<std::string> &args)
{
    vrb::sendLaunchRequestToRemoteLaunchers(vrb::VRB_MESSAGE{m_client->ID(), p, clientID, std::vector<std::string>{}, args}, m_client.get());
}

void CoviseDaemon::loop()
{
    while (!m_terminate)
    {
        if (m_shouldBeConnected)
        {
            {
                Guard g(m_mutex);
                if (m_newCredentials)
                {
                    m_client.reset(new VRBClient{vrb::Program::coviseDaemon, *m_credentials});
                    m_newCredentials = false;
                }
            }

            while (m_shouldBeConnected && !m_terminate && !m_client->isConnected())
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                m_client->connectToServer();
            }

            while (m_shouldBeConnected && !m_terminate && handleVRB())
            {
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

bool CoviseDaemon::handleVRB()
{
    using namespace covise;
    covise::Message msg;
    m_client->wait(&msg);
    Guard g{m_mutex};
    covise::TokenBuffer tb(&msg);
    //std::cerr << "received message: " << covise_msg_types_array[msg.type] << " type " << msg.type << std::endl;
    switch (msg.type)
    {
    case COVISE_MESSAGE_VRB_GET_ID:
    {
        assert(false);
    }
    break;
    case COVISE_MESSAGE_VRB_SET_USERINFO:
    {

        UserInfoMessage uim(&msg);
        if (uim.hasMyInfo)
        {
            m_client->setID(uim.myClientID);
            m_client->setSession(uim.mySession);
            emit connectedSignal();
        }
        for (auto &cl : uim.otherClients)
        {
            assert(findClient(cl.ID()) == m_clientList.end());
            if (cl.userInfo().userType == vrb::Program::coviseDaemon)
            {
                emit updateClient(cl.ID(), getClientInfo(cl));
            }
            m_clientList.insert(std::move(cl));
        }
    }
    break;
    case COVISE_MESSAGE_PROXY:
    {
        PROXY p{msg};
        if (p.type == PROXY_TYPE::ConnectionTest)
        {
            auto &proxyTest = p.unpackOrCast<PROXY_ConnectionTest>();
            auto toPartner = findClient(proxyTest.toClientID);
            if (toPartner != m_clientList.end())
            {
                auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                std::cerr << " starting conn test with timeout " << proxyTest.timeout << " : " << std::ctime(&now) << std::endl;
                Host testHost{toPartner->userInfo().ipAdress.c_str()};
                ClientConnection testConn{&testHost, proxyTest.port, 0, 0, 0, static_cast<double>(proxyTest.timeout)};

                now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                std::cerr << " ClientConnection is connected : " << testConn.is_connected() << " : " << std::ctime(&now) << std::endl;
                PROXY_ConnectionState stateMsg{proxyTest.fromClientID, proxyTest.toClientID, testConn.is_connected() ? ConnectionCapability::DirectConnectionPossible : ConnectionCapability::ProxyRequired};
                sendCoviseMessage(stateMsg, *m_client);
            }
        }
    }
    break;
    case COVISE_MESSAGE_VRB_QUIT:
    {
        if (!removeOtherClient(tb))
        {
            emit disconnectedSignal();
            m_shouldBeConnected = false;
            m_clientList.clear();
            return false;
        }
    }
    break;
    case COVISE_MESSAGE_VRB_MESSAGE:
    {
        handleVrbLauncherMessage(msg);
    }
    break;
    case COVISE_MESSAGE_SOCKET_CLOSED:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    case COVISE_MESSAGE_VRB_CLOSE_VRB_CONNECTION:
    {
        emit disconnectedSignal();
        return false;
    }
    break;
    default:
        break;
    }
    return true;
}

bool CoviseDaemon::removeOtherClient(covise::TokenBuffer &tb)
{
    int id;
    tb >> id;
    if (id != m_client->ID())
    {
        auto cl = findClient(id);
        if (cl != m_clientList.end())
        {
            if (cl->userInfo().userType == vrb::Program::coviseDaemon)
            {
                emit removeClient(id);
            }
            m_clientList.erase(cl);
        }
        return true;
    }
    return false;
}

std::set<vrb::RemoteClient>::iterator CoviseDaemon::findClient(int id)
{
    return std::find_if(m_clientList.begin(), m_clientList.end(), [id](const vrb::RemoteClient &client) {
        return id == client.ID();
    });
}

void CoviseDaemon::handleVrbLauncherMessage(covise::Message &msg)
{
    vrb::VRB_MESSAGE lrq{msg};
    if (lrq.clientID == m_client->ID())
    {
        QString senderDesc;
        auto cl = findClient(lrq.senderID);
        if (cl != m_clientList.end())
        {
            senderDesc = cl->userInfo().hostName.c_str();
        }
        emit launchSignal(lrq.senderID, senderDesc, lrq.program, lrq.args);
    }
}

QString getClientInfo(const vrb::RemoteClient &cl)
{
    QString s;
    QTextStream ss(&s);
    //ss << "email: " << cl.getEmail().c_str() << ", host: " << cl.getHostname().c_str();
    ss << cl.userInfo().hostName.c_str();
    return s;
}

void spawnProgram(Program p, const std::vector<std::string> &args)
{
    covise::spawnProgram(programNames[p], args);
}
