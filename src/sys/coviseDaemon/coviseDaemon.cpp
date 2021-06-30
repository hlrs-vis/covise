/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coviseDaemon.h"

#include <comsg/PROXY.h>
#include <comsg/VRB_PERMIT_LAUNCH.h>
#include <config/CoviseConfig.h>
#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <net/message.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <net/tokenbuffer_serializer.h>
#include <net/tokenbuffer_util.h>
#include <util/coSignal.h>
#include <util/coSpawnProgram.h>
#include <vrb/PrintClientList.h>
#include <vrb/ProgramType.h>
#include <vrb/SessionID.h>
#include <vrb/VrbSetUserInfoMessage.h>
#include <vrb/client/LaunchRequest.h>

#include <QRandomGenerator>
#include <QTextStream>

#include <cassert>
#include <chrono>
#include <ctime>
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
        qRegisterMetaType<covise::Message>();
        qRegisterMetaType<vrb::Program>();
        QObject::connect(this, &CoviseDaemon::receivedVrbMsg, this, &CoviseDaemon::handleVRB);
        m_thread.reset(new std::thread([this]()
                                       {
                                           qRegisterMetaType<vrb::Program>();
                                           qRegisterMetaType<std::vector<std::string>>();
                                           qRegisterMetaType<covise::Message>();
                                           qRegisterMetaType<vrb::Program>();

                                           loop();
                                       }));
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
        if (cl.userInfo().userType == vrb::Program::coviseDaemon)
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

void CoviseDaemon::spawnProgram(Program p, const std::vector<std::string> &args)
{
    static int numStarts = 0;
    ++numStarts;
    QString name = QString(vrb::programNames[p]) + "_" + QString::number(numStarts);
    auto child = m_children.emplace(programNames[p], args);
    auto &c = *child.first;
    QObject::connect(&*child.first, &ChildProcess::output, this, [name, this](const QString &output)
                     { emit childProgramOutput(name, output); });
    QObject::connect(&*child.first, &ChildProcess::died, this, [name, this, &c]()
                     {
                         m_children.erase(c);
                         emit childTerminated(name);
                     });
}

void CoviseDaemon::sendLaunchRequest(Program p, int clientID, const std::vector<std::string> &args)
{
    covise::VRB_PERMIT_LAUNCH_Ask{m_client->ID(), clientID, p};
    m_sentLaunchRequests.emplace_back(std::unique_ptr<vrb::VRB_MESSAGE>{new vrb::VRB_MESSAGE{m_client->ID(), p, clientID, std::vector<std::string>{}, args, 0}});
}

void CoviseDaemon::answerPermissionRequest(vrb::Program p, int clientID, bool answer)
{
    if (m_receivedLaunchRequest)
    {
        if (answer)
        {
            std::lock_guard<std::mutex> g(m_mutex);
            spawnProgram(m_receivedLaunchRequest->program, m_receivedLaunchRequest->args);
            m_receivedLaunchRequest = nullptr;
        }
    }
    else
    {
        int code = 0;
        if (answer)
            code = m_allowedProgramsToLaunch.emplace(m_allowedProgramsToLaunch.end(), p, clientID)->code();
        covise::VRB_PERMIT_LAUNCH_Answer a{clientID, m_client->ID(), answer, code};
        sendCoviseMessage(a, *m_client.get());
    }
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

            m_client->connectToServer();
            while (m_shouldBeConnected && !m_terminate && !m_client->isConnected())
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            while (m_shouldBeConnected && !m_terminate)
            {
                covise::Message msg;
                m_client->wait(&msg);
                emit receivedVrbMsg(msg);
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

bool CoviseDaemon::handleVRB(const covise::Message &msg)
{
    using namespace covise;
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
    case COVISE_MESSAGE_VRB_PERMIT_LAUNCH:
    {
        VRB_PERMIT_LAUNCH p{msg};
        switch (p.type)
        {
        case VRB_PERMIT_LAUNCH_TYPE::Ask:
        {
            auto &a = p.unpackOrCast<VRB_PERMIT_LAUNCH_Ask>();
            if (a.launcherID == m_client->ID())
            {
                ask(a.program, a.senderID);
            }
        }
        break;
        case VRB_PERMIT_LAUNCH_TYPE::Answer:
        {
            auto &answer = p.unpackOrCast<VRB_PERMIT_LAUNCH_Answer>();
            auto launchRequest = std::find_if(m_sentLaunchRequests.begin(), m_sentLaunchRequests.end(), [this, &answer](const std::unique_ptr<vrb::VRB_MESSAGE> &request)
                                              { return request->clientID == answer.launcherID && answer.requestorID == m_client->ID(); });
            if (launchRequest != m_sentLaunchRequests.end())
            {
                if (answer.permit)
                {
                    vrb::VRB_MESSAGE v{launchRequest->get()->senderID, launchRequest->get()->program, launchRequest->get()->clientID, launchRequest->get()->environment, launchRequest->get()->args, answer.code};
                    sendLaunchRequestToRemoteLaunchers(v, m_client.get());
                }
                m_sentLaunchRequests.erase(launchRequest);
            }
        }
        break;
        case VRB_PERMIT_LAUNCH_TYPE::Abort:
        {
            auto &abort = p.unpackOrCast<VRB_PERMIT_LAUNCH_Abort>();
            if (abort.launcherID == m_client->ID())
            {
                emit askForPermissionAbort(abort.program, abort.requestorID);
            }
        }
        break;
        default:
            break;
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
    return std::find_if(m_clientList.begin(), m_clientList.end(), [id](const vrb::RemoteClient &client)
                        { return id == client.ID(); });
}

void CoviseDaemon::handleVrbLauncherMessage(const covise::Message &msg)
{
    m_receivedLaunchRequest = nullptr;
    auto lrq = std::unique_ptr<vrb::VRB_MESSAGE>{new vrb::VRB_MESSAGE{msg}};
    if (lrq->clientID != m_client->ID())
    {
        return;
    }

    ProgramToLaunch p{lrq->program, lrq->senderID, lrq->code};
    auto permission = std::find(m_allowedProgramsToLaunch.begin(), m_allowedProgramsToLaunch.end(), p) != m_allowedProgramsToLaunch.end();

    if (!permission)
    {
        m_receivedLaunchRequest = std::move(lrq);
        ask(m_receivedLaunchRequest->program, m_receivedLaunchRequest->senderID);
    }
    else
    {
        std::lock_guard<std::mutex> g(m_mutex);
        spawnProgram(lrq->program, lrq->args);
    }
}

void CoviseDaemon::ask(vrb::Program p, int clientID)
{
    auto cl = findClient(clientID);
    QString desc;
    QTextStream ts{&desc};
    ts << cl->userInfo().userName.c_str() << "@" << cl->userInfo().hostName.c_str() << " requests to launch " << vrb::programNames[p] << ":\n";
    emit askForPermission(p, clientID, desc);
}

QString getClientInfo(const vrb::RemoteClient &cl)
{
    QString s;
    QTextStream ss(&s);
    //ss << "email: " << cl.getEmail().c_str() << ", host: " << cl.getHostname().c_str();
    ss << cl.userInfo().hostName.c_str();
    return s;
}

CoviseDaemon::ProgramToLaunch::ProgramToLaunch(vrb::Program p, int requestorId)
    : m_p(p), m_requestorId(requestorId)
{
    m_code = QRandomGenerator::system()->generate();
}

CoviseDaemon::ProgramToLaunch::ProgramToLaunch(vrb::Program p, int requestorId, int code)
    : m_p(p), m_requestorId(requestorId), m_code(code)
{
}

bool CoviseDaemon::ProgramToLaunch::operator==(const ProgramToLaunch &other) const
{
    return m_code == other.m_code && m_p == other.m_p && m_requestorId == other.m_requestorId;
}

int CoviseDaemon::ProgramToLaunch::code() const
{
    return m_code;
}
