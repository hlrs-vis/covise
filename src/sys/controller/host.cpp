/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <algorithm>
#include <cassert>
#include <chrono>

#include <comsg/CRB_EXEC.h>
#include <comsg/NEW_UI.h>
#include <comsg/PROXY.h>
#include <comsg/VRB_PERMIT_LAUNCH.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <net/message_types.h>
#include <util/coSpawnProgram.h>
#include <util/covise_version.h>
#include <vrb/VrbSetUserInfoMessage.h>
#include <vrb/client/LaunchRequest.h>
#include <vrb/client/VRBClient.h>

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

const SubProcess &RemoteHost::getProcess(sender_type type) const
{
    return const_cast<RemoteHost *>(this)->getProcess(type);
}

SubProcess &RemoteHost::getProcess(sender_type type)
{
    auto it = std::find_if(m_processes.begin(), m_processes.end(), [type](const ProcessList::value_type &m)
                           {
                               //when called in destructor of netModule (after m_module.clear()) modules can be null)
                               return (m ? m->type == type : false);
                           });
    if (it != m_processes.end())
    {
        return *it->get();
    }
    throw Exception{"RemoteHost did not find process of type " + std::to_string(type)};
}

const NetModule &RemoteHost::getModule(const std::string &name, int instance) const
{
    return const_cast<RemoteHost *>(this)->getModule(name, instance);
}

NetModule &RemoteHost::getModule(const std::string &name, int instance)
{
    auto app = std::find_if(begin(), end(), [&name, &instance](const std::unique_ptr<SubProcess> &mod)
                            {
                                if (const auto app = dynamic_cast<const NetModule *>(&*mod))
                                {
                                    return app->info().name == name && app->instance() == instance;
                                }
                                return false;
                            });
    if (app != end())
    {
        return dynamic_cast<NetModule &>(**app);
    }
    throw Exception{"RemoteHost " + userInfo().hostName + " did not find application module " + name + "_" + std::to_string(instance)};
}

void RemoteHost::removeModule(NetModule &app, int alreadyDead)
{
    app.setDeadFlag(alreadyDead);
    m_processes.erase(std::remove_if(m_processes.begin(), m_processes.end(), [&app](const std::unique_ptr<SubProcess> &mod)
                                     { return &*mod == &app; }),
                      m_processes.end());
}

RemoteHost::ProcessList::const_iterator RemoteHost::begin() const
{
    return m_processes.begin();
}

RemoteHost::ProcessList::iterator RemoteHost::begin()
{
    return m_processes.begin();
}

RemoteHost::ProcessList::const_iterator RemoteHost::end() const
{
    return m_processes.end();
}

RemoteHost::ProcessList::iterator RemoteHost::end()
{
    return m_processes.end();
}

bool RemoteHost::startCrb()
{
    try
    {
        auto m = m_processes.emplace(m_processes.end(), new CRBModule{*this});
        auto crbModule = m->get()->as<CRBModule>();

        if (!crbModule->setupConn([this, &crbModule](int port, const std::string &ip)
                                  {
                                      std::vector<std::string> args;
                                      args.push_back(std::to_string(port));
                                      args.push_back(ip);
                                      args.push_back(std::to_string(crbModule->processId));
                                      //std::cerr << "Requesting start of crb on host " << hostManager.getLocalHost().userInfo().ipAdress << " port: " << port << " id " << crbModule->processId << std::endl;
                                      return launchCrb(vrb::Program::crb, args);
                                  }))
        {
            std::cerr << "startCrb failed to spawn CRB connection" << std::endl;
            return false;
        }
        if (!crbModule->init())
        {
            return false;
        }
        determineAvailableModules(*crbModule);
        //std::cerr << "sending init message with type " << covise_msg_types_array[crbModule->initMessage.type] << std::endl;
        hostManager.sendAll<Userinterface>(crbModule->initMessage);
        connectShm(*crbModule);

        return true;
    }
    catch (const Exception &e)
    {
        std::cerr << e.what() << '\n';
        std::cerr << "startCrb called on host " << userInfo().hostName << " but crb is already running!" << std::endl;
        return false;
    }
}

void RemoteHost::connectShm(const CRBModule &crbModule)
{
    //only needed for localhost
}

void RemoteHost::askForPermission()
{
        covise::VRB_PERMIT_LAUNCH_Ask ask{ hostManager.getVrbClient().ID(), ID(), vrb::Program::crb };
        sendCoviseMessage(ask, hostManager.getVrbClient());
}

void RemoteHost::determineAvailableModules(const CRBModule &crb)
{
    NEW_UI msg{crb.initMessage};
    auto &pMsg = msg.unpackOrCast<NEW_UI_PartnerInfo>();
    for (size_t i = 0; i < pMsg.modules.size(); i++)
    {
        m_availableModules.emplace_back(&hostManager.registerModuleInfo(pMsg.modules[i], pMsg.categories[i]));
    }
}

bool RemoteHost::startUI(const UIOptions &options)
{
    cerr << "* Starting user interface....                                                 *" << endl;
    std::unique_ptr<Userinterface> ui;
    if (options.usePython)
    {

#ifdef _WIN32
        const char *PythonInterfaceExecutable = "..\\..\\Python\\scriptInterface.bat";
#else
        const char *PythonInterfaceExecutable = "scriptInterface";
#endif
        ui.reset(new PythonInterface{*this, PythonInterfaceExecutable});
        startUI(std::move(ui), options);
    }
    switch (options.type)
    {
    case UIOptions::gui:
    {
        ui.reset(new MapEditor{*this});
    }
    break;

    default:
#ifdef HAVE_GSOAP
        ui.reset(new WsInterface{*this});
#else
        return false;
#endif
        break;
    }
    return startUI(std::move(ui), options);
}

bool RemoteHost::startUI(std::unique_ptr<Userinterface> &&ui, const UIOptions &options)
{
    if (hostManager.getAllModules<Userinterface>().empty() || &hostManager.getMasterUi().host == this)
    {
        ui->setStatus(Userinterface::Master);
    }
    try
    {
        if (ui->start(options, false))
        {
            m_processes.push_back(std::move(ui));
            return true;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        std::cerr << "startUI failed: no CRB running on master" << std::endl;
    }
    return false;
}

void RemoteHost::launchProcess(const CRB_EXEC &exec) const
{
    try
    {
        auto &crb = getProcess(sender_type::CRB);
        sendCoviseMessage(exec, crb);
    }
    catch (const Exception &e)
    {
        std::cerr << e.what() << '\n';
        std::cerr << "failed to start module " << exec.name << " on host " << exec.moduleHostName << ": no crb running on that host" << std::endl;
    }
}

bool RemoteHost::isModuleAvailable(const std::string &moduleName) const
{
    auto it = std::find_if(m_availableModules.begin(), m_availableModules.end(), [&moduleName](const ModuleInfo *info)
                           { return info->name == moduleName; });
    return it != m_availableModules.end();
}

NetModule &RemoteHost::startApplicationModule(const string &name, const string &instanz,
                                              int posx, int posy, int copy, ExecFlag flags, NetModule *mirror)
{
    // check the Category of the Module
    auto moduleInfo = std::find_if(m_availableModules.begin(), m_availableModules.end(), [&name](const ModuleInfo *info)
                                   { return info->name == name; });
    if (moduleInfo == m_availableModules.end())
    {
        throw Exception{"failed to start " + name + "on " + userInfo().hostName + ": module not available!"};
    }
    int nr = std::stoi(instanz);
    std::unique_ptr<NetModule> module;
    if ((*moduleInfo)->category == "Renderer")
        module.reset(new Renderer{*this, **moduleInfo, nr});

    else
        module.reset(new NetModule{*this, **moduleInfo, nr});
    // set initial values
    module->init({posx, posy}, copy, flags, mirror);
    return dynamic_cast<NetModule&>( **m_processes.emplace(m_processes.end(), std::move(module)));
}

bool RemoteHost::launchCrb(vrb::Program exec, const std::vector<std::string> &cmdArgs)
{

    switch (m_exectype)
    {
    case controller::ExecType::VRB:
    {
        vrb::sendLaunchRequestToRemoteLaunchers(vrb::VRB_MESSAGE{hostManager.getVrbClient().ID(), exec, ID(), std::vector<std::string>{}, cmdArgs, static_cast<int>(m_code)}, &hostManager.getVrbClient());
    }
    break;
    case controller::ExecType::Manual:
        launchManual(exec, cmdArgs);
        break;
    case controller::ExecType::Script:
        launchScipt(exec, cmdArgs);
        break;
    default:
        break;
    }
    return true;
}

bool LocalHost::launchCrb(vrb::Program exec, const std::vector<std::string> &cmdArgs)
{
    auto execPath = coviseBinDir() + vrb::programNames[exec];
    spawnProgram(execPath, cmdArgs);
    return true;
}

LocalHost::LocalHost(const HostManager &manager, vrb::Program type, const std::string &sessionName)
    : RemoteHost(manager, type, sessionName)
{
    m_state = LaunchStyle::Local;
}

void LocalHost::connectShm(const CRBModule &crbModule)
{
    CTRLGlobal::getInstance()->controller->get_shared_memory(crbModule.conn());
}

covise::LaunchStyle RemoteHost::state() const
{
    return m_state;
}

void RemoteHost::setTimeout(int seconds)
{
    m_timeout = seconds;
}

void RemoteHost::launchScipt(vrb::Program exec, const std::vector<std::string> &cmdArgs)
{
    std::stringstream start_string;
    std::string script_name;
    start_string << script_name << " " << vrb::programNames[exec];
    for (const auto arg : cmdArgs)
        start_string << " " << arg;
    start_string << " " << userInfo().hostName;
    int retval;
    retval = system(start_string.str().c_str());
    if (retval == -1)
    {
        std::cerr << "Controller::start_datamanager: system failed" << std::endl;
    }
}

void RemoteHost::launchManual(vrb::Program exec, const std::vector<std::string> &cmdArgs)
{
    std::stringstream text;
    text << "please start \"" << vrb::programNames[exec];
    for (const auto arg : cmdArgs)
        text << " " << arg;
    text << "\" on " << userInfo().hostName;
    Message msg{COVISE_MESSAGE_COVISE_ERROR, text.str()};
    hostManager.getMasterUi().send(&msg);
    std::cerr << text.str() << std::endl;
}

RemoteHost::RemoteHost(const HostManager &manager, vrb::Program type, const std::string &sessionName)
    : RemoteClient(type, sessionName), hostManager(manager)
{
}

RemoteHost::RemoteHost(const HostManager &manager, vrb::RemoteClient &&base)
    : RemoteClient(std::move(base)), hostManager(manager)
{
}

bool RemoteHost::handlePartnerAction(covise::LaunchStyle action, bool proxyRequired)
{
    m_isProxy = proxyRequired;
    bool retval = false;
    if (action == covise::LaunchStyle::Disconnect || m_hasPermission || m_exectype != ExecType::VRB) {
        m_hasPermission = false;
    
        switch (action)
        {
        case covise::LaunchStyle::Partner:
            retval = addPartner();
        break;
        case covise::LaunchStyle::Host:
            retval = startCrb();
            break;
        case covise::LaunchStyle::Disconnect:
            retval = removePartner();
            break;
        default:
            retval = false;
        }
        if (retval)
        {
            m_state = action;
        }
        return retval;
    } else {
        m_desiredState = action;
        askForPermission();
        return false;
    }
}

bool RemoteHost::addPartner()
{
    if (startCrb())
        return startUI(CTRLHandler::instance()->uiOptions());
    return false;
}

bool RemoteHost::removePartner()
{
    m_state = LaunchStyle::Disconnect;
    auto &masterUi = hostManager.getMasterUi();
    if (this == &masterUi.host)
    {
        Message msg{COVISE_MESSAGE_WARNING, "Controller\n \n \n REMOVING CONTROLLER OR MASTER HOST IS NOT ALLOWED !!!"};
        masterUi.send(&msg);
        return false;
    }
    try
    {
        auto &crb = dynamic_cast<CRBModule &>(getProcess(sender_type::CRB));
        Message msg{COVISE_MESSAGE_REMOVED_HOST, userInfo().userName + "\n" + userInfo().ipAdress + "\n"};
        for (auto &proc : m_processes)
        {
            if (auto renderer = dynamic_cast<const Renderer *>(proc.get()))
            {
                for (const auto &displ : *renderer)
                {
                    if (&displ->host != this)
                    {
                        displ->send(&msg);
                    }
                }
            }
            if (auto mod = dynamic_cast<NetModule *>(proc.get()))
            {
                mod->setAlive(false);
                CTRLGlobal::getInstance()->modUIList->delete_mod(mod->info().name, std::to_string(mod->instance()), userInfo().ipAdress);
            }
        }
        // remove mapeditor
        std::stringstream modInfo;
        NEW_UI_HandlePartners pMsg{LaunchStyle::Disconnect, 0, std::vector<int>{ID()}};
        auto discMsg = pMsg.createMessage();
        for (const auto &ui : hostManager.getAllModules<Userinterface>())
        {
            if (&ui->host != this)
            {
                ui->send(&discMsg);
            }
        }
        clearProcesses();
        // notify the other CRBs
        msg = Message{COVISE_MESSAGE_CRB_QUIT, userInfo().ipAdress};
        m_state = LaunchStyle::Disconnect;
        for (const auto &host : hostManager)
        {
            if (host.second->state() != LaunchStyle::Disconnect) //this->m_state should already be set to Disconnect
            {
                host.second->getProcess(sender_type::CRB).send(&msg);
            }
        }
    }
    catch (const Exception &e)
    {
        Message msg{COVISE_MESSAGE_WARNING, "Controller\n \n \n" + std::string(e.what())};
        masterUi.send(&msg);
        return false;
    }
    return true;
}

bool RemoteHost::proxyHost() const
{
    return m_isProxy;
}

void RemoteHost::permitLaunch(int code)
{
    m_hasPermission = true;
    m_code = code;
}

covise::LaunchStyle RemoteHost::desiredState() const     {
    return m_desiredState;
}


void RemoteHost::clearProcesses()
{
    while (m_processes.size() > 0)
    {
        m_processes.pop_back();
    }
}

HostManager::HostManager()
    : m_localHost(m_hosts.insert(HostMap::value_type(0, std::unique_ptr<RemoteHost>{new LocalHost{*this, vrb::Program::covise}})).first)
    , m_vrb(new vrb::VRBClient{vrb::Program::covise}) 
{
    m_vrb->connectToServer();
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
            clients.push_back(ClientInfo{host.second->ID(), host.second->userInfo().hostName, host.second->state()});
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
    bool error = false;
    if (h.handlePartnerAction(style, proxyRequired) && style == LaunchStyle::Partner)
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
    PROXY_ConnectionCheck check{clID, m_vrb->ID()};
    sendCoviseMessage(check, *m_vrb);
    Message retval;
    m_vrb->wait(&retval, COVISE_MESSAGE_PROXY);
    PROXY proxyMsg{ retval };
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
        PROXY proxyMsg2{ retval };
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
        PROXY proxyMsg{ retval };
        Host h{m_vrb->getCredentials().ipAddress.c_str()};
        m_proxyConnection = createConnectedConn<ControllerProxyConn>(&h, proxyMsg.unpackOrCast<PROXY_ProxyCreated>().port, 1000, (int)CONTROLLER);
        if (!m_proxyConnection)
            std::cerr << "failed to create proxy connection via VRB" << std::endl;
    }
}

void HostManager::handleVrb()
{
    while (m_vrb->isConnected() && handleVrbMessage()) {
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
                if (cl.userInfo().userType == vrb::Program::coviseDaemon)
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
                if (!answer.permit) {
                    Message m{ COVISE_MESSAGE_WARNING, "Partner " + h->userInfo().userName + "@" + h->userInfo().hostName + " refused to launch COVISE!" };
                    getMasterUi().send(&m);
                }
                else {
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
        m_vrb.reset(new vrb::VRBClient{vrb::Program::covise});
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
