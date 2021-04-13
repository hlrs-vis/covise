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
    auto it = std::find_if(m_processes.begin(), m_processes.end(), [type](const ProcessList::value_type &m) {
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
    auto app = std::find_if(begin(), end(), [&name, &instance](const std::unique_ptr<SubProcess> &mod) {
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
    m_processes.erase(std::remove_if(m_processes.begin(), m_processes.end(), [&app](const std::unique_ptr<SubProcess> &mod) {
                          return &*mod == &app;
                      }),
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
        auto shmMode = CTRLHandler::instance()->Config.getshmMode(userInfo().hostName);
        auto execName = shmMode == ShmMode::Proxie ? vrb::Program::crbProxy : vrb::Program::crb;
        auto m = m_processes.emplace(m_processes.end(), new CRBModule{*this, shmMode == ShmMode::Proxie});
        auto crbModule = m->get()->as<CRBModule>();

        if (!crbModule->setupConn([this, &crbModule, execName](int port, const std::string &ip) {
                std::vector<std::string> args;
                args.push_back(std::to_string(port));
                args.push_back(ip);
                args.push_back(std::to_string(crbModule->processId));
                //std::cerr << "Requesting start of crb on host " << hostManager.getLocalHost().userInfo().ipAdress << " port: " << port << " id " << crbModule->processId << std::endl;
                return launchCrb(execName, args);
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
        Message msg = crbModule->initMessage;
        msg.type = COVISE_MESSAGE_UI;
        hostManager.sendAll<Userinterface>(msg);
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

void RemoteHost::determineAvailableModules(const CRBModule &crb)
{
    // get number of new modules from message
    //LIST      0
    //HOST      1
    //USER      2
    //NUMBER    3
    auto list = splitStringAndRemoveComments(crb.initMessage.data.data(), "\n");
    int mod_count = std::stoi(list[3]);
    int iel = 4;
    for (int i = 0; i < mod_count; i++)
    {
        m_availableModules.emplace_back(&hostManager.registerModuleInfo(list[iel], list[iel + 1]));
        iel = iel + 2;
    }
}

bool RemoteHost::startUI(const UIOptions &options)
{
    cerr << "* Starting user interface....                                                 *" << endl;
    std::unique_ptr<Userinterface> ui;
    switch (options.type)
    {
    case UIOptions::python:
    {

#ifdef _WIN32
        const char *PythonInterfaceExecutable = "..\\..\\Python\\scriptInterface.bat ";
#else
        const char *PythonInterfaceExecutable = "scriptInterface ";
#endif
        ui.reset(new PythonInterface{*this, PythonInterfaceExecutable + options.pyFile});
    }
    break;
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
    auto it = std::find_if(m_availableModules.begin(), m_availableModules.end(), [&moduleName](const ModuleInfo *info) { return info->name == moduleName; });
    return it != m_availableModules.end();
}

NetModule &RemoteHost::startApplicationModule(const string &name, const string &instanz,
                                              int posx, int posy, int copy, ExecFlag flags, NetModule *mirror)
{
    // check the Category of the Module
    auto moduleInfo = std::find_if(m_availableModules.begin(), m_availableModules.end(), [&name](const ModuleInfo *info) {
        return info->name == name;
    });
    if (moduleInfo == m_availableModules.end())
    {
        throw Exception{"failed to start " + name + "on " + userInfo().hostName + ": module not available!"};
    }
    int nr = std::stoi(instanz);
    SubProcess *module = nullptr;
    if ((*moduleInfo)->category == "Renderer")
    {
        module = &**m_processes.emplace(m_processes.end(), new Renderer{*this, **moduleInfo, nr});
    }
    else
    {
        module = &**m_processes.emplace(m_processes.end(), new NetModule{*this, **moduleInfo, nr});
    }
    // set initial values
    auto &app = dynamic_cast<NetModule &>(*module);
    app.init({posx, posy}, copy, flags, mirror);
    return app;
}

bool RemoteHost::get_mark() const
{
    return m_saveInfo;
}

void RemoteHost::reset_mark()
{
    m_saveInfo = false;
}

void RemoteHost::mark_save()
{
    m_saveInfo = true;
}

bool RemoteHost::launchCrb(vrb::Program exec, const std::vector<std::string> &cmdArgs)
{

    switch (m_exectype)
    {
    case controller::ExecType::VRB:
    {
        vrb::sendLaunchRequestToRemoteLaunchers(vrb::VRB_MESSAGE{hostManager.getVrbClient().ID(), exec, ID(), cmdArgs}, &hostManager.getVrbClient());
        if (!hostManager.launchOfCrbPermitted())
        {
            Message m{COVISE_MESSAGE_WARNING, "Partner " + userInfo().userName + "@" + userInfo().hostName + " refused to launch COVISE!"};
            hostManager.getMasterUi().send(&m);
            removePartner();
            return false;
        }
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

bool RemoteHost::handlePartnerAction(covise::LaunchStyle action)
{
    m_state = action;
    switch (action)
    {
    case covise::LaunchStyle::Partner:
        return addPartner();
    case covise::LaunchStyle::Host:
        return startCrb();
    case covise::LaunchStyle::Disconnect:
        return removePartner();
    default:
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
    m_state = covise::LaunchStyle::Disconnect;
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
        if (crb.initMessage.data.data())
        {
            modInfo << "RMV_LIST\n"
                    << userInfo().ipAdress << "\n"
                    << userInfo().userName << "\n"
                    << crb.initMessage.data.data() + 5 << "\n";
            Message msg2{COVISE_MESSAGE_UI, modInfo.str()};
            hostManager.sendAll<Userinterface>(msg2);
        }
        else
        {
            std::cerr << std::endl
                      << "ERROR: rmv_host() initMessage  ==  NULL !!!" << std::endl;
        }
        clearProcesses();
        // notify the other CRBs
        msg = Message{COVISE_MESSAGE_CRB_QUIT, userInfo().ipAdress};
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

void RemoteHost::clearProcesses()
{
    while (m_processes.size() > 0)
    {
        m_processes.pop_back();
    }
}

HostManager::HostManager()
    : m_localHost(m_hosts.insert(HostMap::value_type(0, std::unique_ptr<RemoteHost>{new LocalHost{*this, vrb::Program::covise}})).first), m_vrb(new vrb::VRBClient{vrb::Program::covise}), m_thread([this]() { handleVrb(); })
{
}

HostManager::~HostManager()
{
    m_terminateVrb = true;
    m_hosts.clear(); //host that use vrb proxy must be deleted before vrb conn is shutdown
    m_vrb->shutdown();
    m_thread.join();
}

void HostManager::sendPartnerList()
{
    std::lock_guard<std::mutex> g{m_mutex};
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

std::vector<bool> HostManager::handleAction(const covise::NEW_UI_HandlePartners &msg)
{
    createProxyConnIfNecessary();
    std::vector<bool> retval;
    for (auto clID : msg.clients)
    {
        auto hostIt = m_hosts.find(clID);
        if (hostIt != m_hosts.end())
        {
            hostIt->second->setTimeout(msg.timeout);
            retval.push_back(hostIt->second->handlePartnerAction(msg.launchStyle));
        }
    }
    return retval;
}

void HostManager::setOnConnectCallBack(std::function<void(void)> cb)
{
    std::lock_guard<std::mutex> g{m_mutex};
    m_onConnectVrbCallBack = cb;
}

int HostManager::vrbClientID() const
{
    std::lock_guard<std::mutex> g{m_mutex};
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
    auto h = std::find_if(m_hosts.begin(), m_hosts.end(), [&ipAddress](HostMap::value_type &host) {
        return host.second->userInfo().ipAdress == ipAddress;
    });
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
        auto display = std::find_if(renderer->begin(), renderer->end(), [&masterUi](const Renderer::DisplayList::value_type &disp) {
            return &disp->host == &masterUi.host;
        });
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
    auto masterUi = std::find_if(uis.begin(), uis.end(), [](const Userinterface *ui) {
        return ui->status() == Userinterface::Status::Master;
    });
    assert(masterUi != uis.end());
    return **masterUi;
}

std::string HostManager::getHostsInfo() const
{

    std::stringstream buffer;
    buffer << m_hosts.size() << "\n";
    for (const auto &h : *this)
    {
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

    return buffer.str();
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
    return m_proxyConnection;
}

bool HostManager::launchOfCrbPermitted() const
{
    std::unique_lock<std::mutex> lk(m_launchPermissionMutex);
    m_waitLaunchPermission.wait(lk);
    return m_launchPermission;
}

std::unique_ptr<Message> HostManager::hasProxyMessage()
{
    return m_proxyConnection ? m_proxyConnection->getCachedMsg() : nullptr;
}

void HostManager::createProxyConnIfNecessary()
{
    if (!Host{}.hasRoutableAddress() && !m_proxyConnection)
    {
        //request listening conn on server
        PROXY_CreateControllerProxy p{m_vrb->ID()};
        sendCoviseMessage(p, *m_vrb);
        std::unique_lock<std::mutex> lk(m_proxyMutex);
        m_waitForProxyPort.wait(lk, [this]() { return m_proxyConnPort; });
        //connect
        Host h{m_vrb->getCredentials().ipAddress.c_str()};
        auto conn = createConnectedConn<ControllerProxyConn>(&h, m_proxyConnPort, 1000, (int)CONTROLLER);
        if (!conn)
        {
            std::cerr << "failed to create proxy connection via VRB" << std::endl;
        }

        m_proxyConnection = dynamic_cast<const ControllerProxyConn *>(CTRLGlobal::getInstance()->controller->getConnectionList()->add(std::move(conn)));

        lk.unlock();
    }
}

void HostManager::handleVrb()
{
    using namespace covise;
    while (!m_terminateVrb)
    {
        while (!m_terminateVrb && !m_vrb->isConnected())
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            m_vrb->connectToServer();
        }
        while (!m_terminateVrb && handleVrbMessage())
        {
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

bool HostManager::handleVrbMessage()
{
    covise::Message msg;
    m_vrb->wait(&msg);

    switch (msg.type)
    {
    case COVISE_MESSAGE_VRB_SET_USERINFO:
    {

        std::lock_guard<std::mutex> g{m_mutex};
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
            if (m_onConnectVrbCallBack)
            {
                m_onConnectVrbCallBack();
            }
        }
        for (auto &cl : uim.otherClients)
        {
            if (cl.userInfo().userType == vrb::Program::VrbRemoteLauncher)
            {
                m_hosts.insert(HostMap::value_type{cl.ID(), std::unique_ptr<RemoteHost>{new RemoteHost{*this, std::move(cl)}}});
            }
        }
    }
    break;
    case COVISE_MESSAGE_VRB_PERMIT_LAUNCH:
    {
        covise::VRB_PERMIT_LAUNCH permission{msg};
        {
            std::lock_guard<std::mutex> g{m_launchPermissionMutex};
            m_launchPermission = permission.permit;
        }
        m_waitLaunchPermission.notify_one();
    }
    break;
    case COVISE_MESSAGE_PROXY:
    {
        PROXY proxyMsg{msg};
        assert(proxyMsg.type == PROXY_TYPE::ProxyCreated);
        {
            std::lock_guard<std::mutex> lk{m_proxyMutex};
            m_proxyConnPort = proxyMsg.unpackOrCast<PROXY_ProxyCreated>().port;
        }
        m_waitForProxyPort.notify_one();
    }
    break;
    case COVISE_MESSAGE_VRB_QUIT:
    {
        TokenBuffer tb{&msg};
        int id;
        tb >> id;
        if (id != m_vrb->ID())
        {
            std::lock_guard<std::mutex> g{m_mutex};
            auto clIt = m_hosts.find(id);
            if (clIt != m_hosts.end())
            {
                if (clIt->second->state() == LaunchStyle::Disconnect)
                {
                    m_hosts.erase(clIt);
                }
                else
                {
                    //disconnect partner
                }
            }
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
        return false;
    }
    break;
    default:
        break;
    }
    return true;
}
