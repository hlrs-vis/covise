#include "hostManager.h"
#include <vrb/VrbSetUserInfoMessage.h>
#include <vrb/client/VRBClient.h>

#include <messages/CRB_EXEC.h>
#include <messages/NEW_UI.h>
#include <messages/PROXY.h>
#include <messages/VRB_PERMIT_LAUNCH.h>
#include <net/covise_host.h>

#include "crb.h"
#include "exception.h"
#include "global.h"
#include "handler.h"
#include "host.h"
#include "module.h"
#include "renderModule.h"
#include "userinterface.h"
#include "util.h"
#include "proxyConnection.h"

using namespace covise;
using namespace covise::controller;

HostManager::HostManager()
    : m_localHost(m_hosts.insert(HostMap::value_type(0, std::unique_ptr<RemoteHost>{new LocalHost{*this, covise::Program::covise}})).first), m_vrb(new vrb::VRBClient{covise::Program::covise})
{
    m_vrb->connectToServer();
    auto userInfos = getConfiguredHosts();
    std::cerr << userInfos.size() << " hosts are preconfigured" << std::endl;
    int i = clientIdForPreconfigured; 
    for (auto &userInfo : userInfos)
    {
        covise::TokenBuffer tb;
        tb << i << vrb::SessionID{} << userInfo;
        tb.rewind();
        m_hosts.insert(HostMap::value_type{i, std::unique_ptr<RemoteHost>{new RemoteHost{*this, vrb::RemoteClient{tb}}}});
        ++i;
    }
}

HostManager::~HostManager()
{
    m_hosts.clear(); //host that use vrb proxy must be deleted before vrb conn is shutdown
    m_vrb->shutdown();
}

void HostManager::sendPartnerList() const
{
    ClientList clients;

    for (const auto &host : m_hosts)
    {
        if (host.first != m_localHost->first)
        {
            clients.push_back(ClientInfo{host.second->ID(), host.second->userInfo().hostName, host.second->userInfo().userName, host.second->state()});
        }
    }
    NEW_UI_AvailablePartners p{std::move(clients)};
    auto msg = p.createMessage();
    sendAll<Userinterface>(msg);
}

void HostManager::handleActions(const covise::NEW_UI_HandlePartners &msg)
{
    for (auto clID : msg.clients)
    {
        auto hostIt = m_hosts.find(clID);
        if (hostIt != m_hosts.end())
        {
            hostIt->second->setTimeout(msg.timeout);
            handleAction(msg.launchStyle, *hostIt->second);
            if (clID < 0 && msg.launchStyle == LaunchStyle::Disconnect) //the deamon already disconnected
            {
                m_hosts.erase(hostIt);
            }
        }
    }
}

void HostManager::handleAction(LaunchStyle style, RemoteHost &h)
{
    bool proxyRequired = false;
    if (style != LaunchStyle::Disconnect)
    {
        proxyRequired = checkIfProxyRequiered(h.ID(), h.userInfo().hostName);
    }
    if (h.handlePartnerAction(style, proxyRequired))
    {
        if (style == LaunchStyle::Partner)
        {
            const auto &ui = dynamic_cast<const Userinterface &>(h.getProcess(sender_type::USERINTERFACE));

            ui.sendCurrentNetToUI(CTRLHandler::instance()->globalFile());
            // add displays for the existing renderers on the new partner
            for (const auto &renderer : getAllModules<Renderer>())
            {
                if (renderer->isOriginal())
                {
                    renderer->addDisplayAndHandleConnections(ui);
                }
            }
        }

        //inform ui that the connection process is over
        sendPartnerList();
        NEW_UI_ConnectionCompleted cmsg{h.ID()};
        auto m = cmsg.createMessage();
        sendAll<Userinterface>(m);
    }
}

void HostManager::setOnConnectCallBack(std::function<void(void)> cb)
{
    m_onConnectVrbCallBack = cb;
}

int HostManager::vrbClientID() const
{
    return m_vrb->ID();
}

const vrb::VRBClient &HostManager::getVrbClient() const
{
    return *m_vrb;
}

LocalHost &HostManager::getLocalHost()
{
    assert(dynamic_cast<LocalHost *>(m_localHost->second.get()));
    return *dynamic_cast<LocalHost *>(m_localHost->second.get());
}

const LocalHost &HostManager::getLocalHost() const
{
    return const_cast<HostManager *>(this)->getLocalHost();
}

RemoteHost *HostManager::getHost(int clientID)
{
    auto h = m_hosts.find(clientID);
    if (h != m_hosts.end())
    {
        return h->second.get();
    }
    return nullptr;
}

const RemoteHost *HostManager::getHost(int clientID) const
{
    return const_cast<HostManager *>(this)->getHost(clientID);
}

RemoteHost &HostManager::findHost(const std::string &ipAddress)
{
    auto h = std::find_if(m_hosts.begin(), m_hosts.end(), [&ipAddress](HostMap::value_type &host)
                          { return (host.second->state() != LaunchStyle::Disconnect && host.second->userInfo().ipAdress == ipAddress); });
    if (h != m_hosts.end())
    {
        return *h->second.get();
    }
    throw Exception{"HostManager could not find host " + ipAddress};
}

const RemoteHost &HostManager::findHost(const std::string &ipAddress) const
{
    return const_cast<HostManager *>(this)->findHost(ipAddress);
}

std::vector<const SubProcess *> HostManager::getAllModules(sender_type type) const
{
    std::vector<const SubProcess *> modules;
    for (const auto &host : m_hosts)
    {
        for (const auto &module : *host.second)
        {
            if (type == sender_type::ANY || module->type == type)
            {
                modules.push_back(&*module);
            }
        }
    }
    return modules;
}

HostManager::HostMap::const_iterator HostManager::begin() const
{
    return m_hosts.begin();
}

HostManager::HostMap::iterator HostManager::begin()
{
    return m_hosts.begin();
}

HostManager::HostMap::const_iterator HostManager::end() const
{
    return m_hosts.end();
}

HostManager::HostMap::iterator HostManager::end()
{
    return m_hosts.end();
}

SubProcess *HostManager::findModule(int peerID)
{
    for (auto &host : m_hosts)
    {
        for (auto &module : *host.second)
        {
            if (auto render = module->as<Renderer>())
            {
                if (render->getDisplay(peerID) != render->end())
                {
                    return render;
                }
            }
            else
            {
                if (module->processId == peerID)
                {
                    return &*module;
                }
            }
        }
    }
    return nullptr;
}

bool HostManager::slaveUpdate()
{
    if (!m_slaveUpdate)
    {
        return false;
    }
    for (const Renderer *renderer : getAllModules<Renderer>())
    {
        auto &masterUi = getMasterUi();
        auto display = std::find_if(renderer->begin(), renderer->end(), [&masterUi](const Renderer::DisplayList::value_type &disp)
                                    { return &disp->host == &masterUi.host; });
        if (display != renderer->end())
        {
            ostringstream os;
            os << "UPDATE\n"
               << renderer->info().name << "\n"
               << renderer->instance() << "\n"
               << masterUi.host.userInfo().hostName << "\n";
            Message msg{COVISE_MESSAGE_RENDER, os.str()};
            display->get()->send(&msg);
        }
    }
    m_slaveUpdate = false;
    return true;
}

const Userinterface &HostManager::getMasterUi() const
{
    return const_cast<HostManager *>(this)->getMasterUi();
}

Userinterface &HostManager::getMasterUi()
{
    auto uis = getAllModules<Userinterface>();
    auto masterUi = std::find_if(uis.begin(), uis.end(), [](const Userinterface *ui)
                                 { return ui->status() == Userinterface::Status::Master; });
    assert(masterUi != uis.end());
    return **masterUi;
}

std::string HostManager::getHostsInfo() const
{

    std::stringstream buffer;
    int numPartners = 0;
    for (const auto &h : *this)
    {
        if (h.second->state() != LaunchStyle::Disconnect)
        {
            ++numPartners;
            auto &host = *h.second;
            if (&host == &getLocalHost())
            {
                buffer << "LOCAL\nLUSER";
            }
            else
            {
                buffer << host.userInfo().ipAdress << "\n"
                       << host.userInfo().userName;
                try
                {
                    host.getProcess(sender_type::USERINTERFACE);
                    buffer << " Partner";
                }
                catch (const Exception &e)
                {
                    (void)e;
                }
            }
            buffer << "\n";
        }
    }

    return std::to_string(numPartners) + "\n" + buffer.str();
}

const ModuleInfo &HostManager::registerModuleInfo(const std::string &name, const std::string &category) const
{
    return *m_availableModules.insert(ModuleInfo{name, category}).first;
}

void HostManager::resetModuleInstances()
{
    for (const auto &info : m_availableModules)
        info.count = 1;
}

const ControllerProxyConn *HostManager::proxyConn() const
{
    return &*m_proxyConnection;
}

std::unique_ptr<Message> HostManager::receiveProxyMessage()
{
    if (!m_proxyConnection)
        return nullptr;
    std::unique_ptr<Message> m = m_proxyConnection->getCachedMsg();
    if (m)
        return m;

    if (m_proxyConnection->check_for_input())
    {
        m.reset(new Message{});
        m_proxyConnection->recv_msg(m.get());
    }
    return m;
}

bool HostManager::checkIfProxyRequiered(int clID, const std::string &hostName)
{
    if (!m_vrb->isConnected())
        return false;

    PROXY_ConnectionCheck check{clID, m_vrb->ID()};
    sendCoviseMessage(check, *m_vrb);
    Message retval;
    m_vrb->wait(&retval, COVISE_MESSAGE_PROXY);
    PROXY proxyMsg{retval};
    auto conCap = proxyMsg.unpackOrCast<PROXY_ConnectionState>().capability;
    if (conCap == ConnectionCapability::NotChecked)
    {
        int timeout = coCoviseConfig::getInt("System.VRB.CheckConnectionTimeout", 6);
        Message infoMsg{COVISE_MESSAGE_WARNING, "Testing the connection to " + hostName + ", timeout is " + std::to_string(timeout) + " seconds."};
        sendAll<Userinterface>(infoMsg);
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cerr << " creating server conn: " << std::ctime(&now) << std::endl;
        setupServerConnection(0, 0, timeout, [this, timeout, clID](const ServerConnection &c)
                              {
                                  PROXY_ConnectionTest test{clID, m_vrb->ID(), c.get_port(), timeout};
                                  return sendCoviseMessage(test, *m_vrb);
                              });
        now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cerr << " waiting for response from vrb: " << std::ctime(&now) << std::endl;
        m_vrb->wait(&retval, COVISE_MESSAGE_PROXY);
        PROXY proxyMsg2{retval};
        conCap = proxyMsg2.unpackOrCast<PROXY_ConnectionState>().capability;
        now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cerr << " received response from vrb: " << std::ctime(&now) << std::endl;
    }
    assert(conCap != ConnectionCapability::NotChecked);
    std::string msgStr;
    if (conCap == ConnectionCapability::ProxyRequired)
    {
        createProxyConn();
        msgStr = "Connection to " + hostName + " timed out, creating proxy via VRB.";
    }
    else
    {
        msgStr = "Connection to " + hostName + " successful.";
    }

    Message m{COVISE_MESSAGE_WARNING, msgStr};
    sendAll<Userinterface>(m);
    return conCap == ConnectionCapability::ProxyRequired;
}

void HostManager::createProxyConn()
{
    if (!m_proxyConnection)
    {
        //request listening conn on server
        PROXY_CreateControllerProxy p{m_vrb->ID()};
        sendCoviseMessage(p, *m_vrb);
        //connect
        Message retval;
        m_vrb->wait(&retval, COVISE_MESSAGE_PROXY);
        PROXY proxyMsg{retval};
        Host h{m_vrb->getCredentials().ipAddress.c_str()};
        m_proxyConnection = createConnectedConn<ControllerProxyConn>(&h, proxyMsg.unpackOrCast<PROXY_ProxyCreated>().port, 1000, (int)CONTROLLER);
        if (!m_proxyConnection)
            std::cerr << "failed to create proxy connection via VRB" << std::endl;
    }
}

void HostManager::handleVrb()
{
    while (m_vrb->isConnected() && handleVrbMessage())
    {
    }
}

bool HostManager::handleVrbMessage()
{
    covise::Message msg;
    if (!m_vrb->poll(&msg))
        return false;
    switch (msg.type)
    {
    case COVISE_MESSAGE_VRB_SET_USERINFO:
    {

        {
            vrb::UserInfoMessage uim(&msg);
            if (uim.hasMyInfo)
            {
                m_vrb->setID(uim.myClientID);
                m_vrb->setSession(uim.mySession);
                auto old = m_localHost;
                m_localHost = m_hosts.insert(HostMap::value_type{uim.myClientID, std::move(old->second)}).first;
                m_localHost->second->setID(uim.myClientID);
                m_localHost->second->setSession(uim.mySession);
                m_hosts.erase(old);
                moveRendererInNewSessions();

                if (m_onConnectVrbCallBack)
                    m_onConnectVrbCallBack();
            }
            for (auto &cl : uim.otherClients)
            {
                if (cl.userInfo().userType == covise::Program::coviseDaemon)
                {
                    m_hosts.insert(HostMap::value_type{cl.ID(), std::unique_ptr<RemoteHost>{new RemoteHost{*this, std::move(cl)}}});
                }
            }
        }
        sendPartnerList();
    }
    break;
    case COVISE_MESSAGE_VRB_PERMIT_LAUNCH:
    {
        covise::VRB_PERMIT_LAUNCH permission{msg};
        switch (permission.type)
        {
        case VRB_PERMIT_LAUNCH_TYPE::Answer:
        {
            auto &answer = permission.unpackOrCast<VRB_PERMIT_LAUNCH_Answer>();
            if (auto h = getHost(answer.launcherID))
            {
                if (!answer.permit)
                {
                    Message m{COVISE_MESSAGE_WARNING, "Partner " + h->userInfo().userName + "@" + h->userInfo().hostName + " refused to launch COVISE!"};
                    getMasterUi().send(&m);
                    NEW_UI_ConnectionCompleted cmsg{h->ID()};
                    m = cmsg.createMessage();
                    sendAll<Userinterface>(m);
                }
                else if (h->wantsTochangeState())
                {
                    h->permitLaunch(answer.code);
                    handleAction(h->desiredState(), *h);
                }
            }
            else
                std::cerr << "received VRB_PERMIT_LAUNCH_Answer from unknown coviseDaemon with clientID " << answer.launcherID << std::endl;
        }
        break;

        default:
            break;
        }
    }
    break;
    case COVISE_MESSAGE_VRB_QUIT:
    {
        TokenBuffer tb{&msg};
        int id;
        tb >> id;
        if (id != m_vrb->ID())
        {
            auto clIt = m_hosts.find(id);
            if (clIt != m_hosts.end())
            {
                if (clIt->second->state() != LaunchStyle::Disconnect)
                {
                    int i = -1;
                    while (true)
                    {
                        auto hostsToRemove = m_hosts.find(i);
                        if (hostsToRemove != m_hosts.end())
                        {
                            ++i;
                        }
                        else
                        {
                            NEW_UI_ChangeClientId c{clIt->second->ID(), i};
                            clIt->second->setID(i);
                            sendAll<Userinterface>(c.createMessage());
                            m_hosts[i] = std::move(clIt->second);
                            break;
                        }
                    }
                }
                m_hosts.erase(clIt);
            }
            sendPartnerList();
            break;
        }
        //fall through to disconnect
    }
    case COVISE_MESSAGE_SOCKET_CLOSED:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    case COVISE_MESSAGE_VRB_CLOSE_VRB_CONNECTION:
    {
        if (m_hosts.empty()) //we are shutting down
            return false;

        std::unique_ptr<RemoteHost> local{std::move(m_localHost->second)};
        m_hosts.clear();
        m_localHost = m_hosts.insert(HostMap::value_type{0, std::move(local)}).first;
        m_localHost->second->setID(0);
        std::cerr << "lost connection to vrb" << std::endl;
        m_vrb.reset(new vrb::VRBClient{covise::Program::covise});
        m_vrb->connectToServer();
        return false;
    }
    break;
    default:
        break;
    }
    return true;
}

void HostManager::moveRendererInNewSessions()
{
    for (const auto &rend : getAllModules<Renderer>())
    {
        //request new session
        TokenBuffer tb;
        vrb::SessionID sid{m_vrb->ID(), "covise" + std::to_string(m_vrb->ID()) + "_" + std::to_string(rend->instance()), false};
        tb << sid;
        Message sessionRequest(COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION, tb.getData());
        m_vrb->send(&sessionRequest);
        covise::Message sessionUpdate{COVISE_MESSAGE_VRBC_CHANGE_SESSION, tb.getData()};
        rend->send(&sessionUpdate);
    }
}
