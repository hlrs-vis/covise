#include "VrbRemoteLauncher.h"
#include "MessageTypes.h"

#include <net/message.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <net/tokenbuffer_util.h>
#include <vrb/SessionID.h>
#include <vrb/client/PrintClientList.h>

#include <chrono>
#include <thread>

#include <config/CoviseConfig.h>
#include <sstream>
#include <net/covise_host.h>
#include <net/tokenbuffer_serializer.h>
#include <QTextStream>
#include <string>
#include <cmath>

using namespace vrb::launcher;

    const std::array<const char *, static_cast<int>(Program::DUMMY)> ProgramNames::names = {
        "covise",
        "opencover"};
        
VrbRemoteLauncher::~VrbRemoteLauncher()
{
    m_terminate = true;
    disconnect();
    if (m_thread)
    {
        m_thread->join();
    }
}

void VrbRemoteLauncher::connect(const vrb::VrbCredentials &credentials)
{
    m_shouldBeConnected = true;
    {
        Guard g{m_mutex};
        m_credentials.reset(new vrb::VrbCredentials(credentials));
        m_newCredentials = true;
    }
    if (!m_thread)
    {
        m_thread.reset(new std::thread([this]() { 
            qRegisterMetaType<Program>();
            qRegisterMetaType<std::vector<std::string>>();

            loop(); }));
    }
}

void VrbRemoteLauncher::disconnect()
{
    if (m_shouldBeConnected)
    {
        m_client->shutdown();
        m_shouldBeConnected = false;
    }
}

void VrbRemoteLauncher::printClientInfo()
{
    Guard g{m_mutex};
    std::vector<const vrb::RemoteClient *> partner;
    for (const auto &cl : m_clientList)
    {
        if (cl->getSessionID() == m_me.getSessionID())
        {
            partner.push_back(cl.get());
        }
        }
    vrb::printClientInfo(partner);
}

void VrbRemoteLauncher::sendLaunchRequest(Program p, int clientID, const std::vector<std::string> &args)
{
    covise::TokenBuffer tb;
    tb << LaunchType::LAUNCH;
    tb << clientID;
    tb << p;
    covise::serialize(tb, args);
    covise::Message msg(tb);
    msg.type = covise::COVISE_MESSAGE_VRB_MESSAGE;
    m_client->sendMessage(&msg);
}

void VrbRemoteLauncher::loop()
{
    while (!m_terminate)
    {
        if (m_shouldBeConnected)
        {
            {
                Guard g(m_mutex);
                if (m_newCredentials)
                {
                    m_client.reset(new covise::VRBClient{"VrbRemoteLauncher", *m_credentials});
                    m_newCredentials = false;
                }
            }

            while (m_shouldBeConnected && !m_terminate && !m_client->isConnected())
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                m_client->connectToServer("VrbRemoteLauncher");
            }

            while (m_shouldBeConnected && !m_terminate && handleVRB())
            {
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

bool VrbRemoteLauncher::handleVRB()
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
        setMyIDs(tb);
    }
    break;
    case COVISE_MESSAGE_VRB_SET_USERINFO:
    {
        int num;
        tb >> num;
        for (int i = 0; i < num; i++)
        {

            if (!setOtherClientInfo(tb))
            {
                m_me.setInfo(tb);
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
            return false;
        }
    }
    break;
    case COVISE_MESSAGE_VRB_MESSAGE:
    {
        handleVrbLauncherMessage(tb);
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
void VrbRemoteLauncher::setMyIDs(covise::TokenBuffer &tb)
{
    emit connectedSignal();
    int clientID;
    SessionID sessionID;
    tb >> clientID >> sessionID;
    m_me.setID(clientID);
    m_me.setSession(sessionID);
    m_client->setID(clientID);
    m_client->sendMessage(m_me.createHelloMessage().get());
}

bool VrbRemoteLauncher::removeOtherClient(covise::TokenBuffer &tb)
{
    int id;
    tb >> id;
    if (id != m_me.getID())
    {
        auto cl = findClient(id);
        if (cl != m_clientList.end())
        {
            if (cl->get()->getSessionID() == m_me.getSessionID())
            {
                emit removeClient(id);
            }
            m_clientList.erase(cl);
        }
        return true;
    }
    return false;
}

bool VrbRemoteLauncher::setOtherClientInfo(covise::TokenBuffer &tb)
{
    int id;
    tb >> id;
    if (id != m_me.getID())
    {
        auto cl = findClient(id);
        if (cl == m_clientList.end())
        {
            cl = m_clientList.emplace(m_clientList.end(), std::unique_ptr<vrb::RemoteClient>(new vrb::RemoteClient(id)));
        }
        cl->get()->setInfo(tb);
        if (cl->get()->getSessionID().name() == "VrbRemoteLauncher")
        {
            emit updateClient(cl->get()->getID(), getClientInfo(*cl->get()));
        }
        return true;
    }
    return false;
}

std::vector<std::unique_ptr<vrb::RemoteClient>>::iterator VrbRemoteLauncher::findClient(int id)
{
    return std::find_if(m_clientList.begin(), m_clientList.end(), [id](const std::unique_ptr<vrb::RemoteClient> &client) {
        return id == client->getID();
    });
}

void VrbRemoteLauncher::handleVrbLauncherMessage(covise::TokenBuffer &tb)
{
    LaunchType t;
    tb >> t;
    switch (t)
    {
    case LaunchType::LAUNCH:
    {
        int clientID;
        tb >> clientID;
        if (clientID == m_client->getID())
        {
            Program p;
            tb >> p;
            std::vector<std::string> launchOptions;
            covise::deserialize(tb, launchOptions);
            emit launchSignal(p, launchOptions);
        }
    }
    break;
    case LaunchType::TERMINATE:
    {
    }
    break;
    default:
        break;
    }
}

QString vrb::launcher::getClientInfo(const vrb::RemoteClient &cl)
{
    QString s;
    QTextStream ss(&s);
    //ss << "email: " << cl.getEmail().c_str() << ", host: " << cl.getHostname().c_str();
    ss << cl.getHostname().c_str();
    return s;
}
