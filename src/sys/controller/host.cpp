/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <algorithm>
#include <cassert>
#include <chrono>

#include <messages/CRB_EXEC.h>
#include <messages/NEW_UI.h>
#include <messages/PROXY.h>
#include <messages/VRB_PERMIT_LAUNCH.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <net/message_types.h>
#include <util/coSpawnProgram.h>
#include <util/covise_version.h>
#include <vrb/client/LaunchRequest.h>

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
                                      return launchCrb(covise::Program::crb, args);
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
    covise::VRB_PERMIT_LAUNCH_Ask ask{hostManager.getVrbClient().ID(), ID(), covise::Program::crb};
    sendCoviseMessage(ask, hostManager.getVrbClient());
    m_state = covise::LaunchStyle::Pending;
    m_code = 0;
    m_isProxy = false;
}

void RemoteHost::determineAvailableModules(const CRBModule &crb)
{
    NEW_UI msg{crb.initMessage};
    auto &pMsg = msg.unpackOrCast<NEW_UI_PartnerInfo>();
    for (size_t i = 0; i < pMsg.modules().size(); i++)
    {
        m_availableModules.emplace_back(&hostManager.registerModuleInfo(pMsg.modules()[i], pMsg.categories()[i]));
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
        std::cerr << "failed to start module " << exec.name() << " on host " << exec.moduleHostName() << ": no crb running on that host" << std::endl;
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
    return dynamic_cast<NetModule &>(**m_processes.emplace(m_processes.end(), std::move(module)));
}

bool RemoteHost::launchCrb(covise::Program exec, const std::vector<std::string> &cmdArgs)
{
    auto execType = ID() < clientIdForPreconfigured ? ExecType::VRB : CTRLHandler::instance()->Config.getexectype(userInfo().hostName);
    switch (execType)
    {
    case controller::ExecType::VRB:
        vrb::sendLaunchRequestToRemoteLaunchers(vrb::VRB_MESSAGE{hostManager.getVrbClient().ID(), exec, ID(), std::vector<std::string>{}, cmdArgs, static_cast<int>(m_code)}, &hostManager.getVrbClient());
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

bool LocalHost::launchCrb(covise::Program exec, const std::vector<std::string> &cmdArgs)
{
    auto execPath = coviseBinDir() + covise::programNames[exec];
    spawnProgram(execPath, cmdArgs);
    return true;
}

LocalHost::LocalHost(const HostManager &manager, covise::Program type, const std::string &sessionName)
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

void RemoteHost::setCode(int code)
{
    m_code = code;
}

void RemoteHost::launchScipt(covise::Program exec, const std::vector<std::string> &cmdArgs)
{
    std::stringstream start_string;
    std::string script_name;
    start_string << script_name << " " << covise::programNames[exec];
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

void RemoteHost::launchManual(covise::Program exec, const std::vector<std::string> &cmdArgs)
{
    std::stringstream text;
    text << "please start \"" << covise::programNames[exec];
    for (const auto arg : cmdArgs)
        text << " " << arg;
    text << "\" on " << userInfo().hostName;
    Message msg{COVISE_MESSAGE_COVISE_ERROR, text.str()};
    hostManager.getMasterUi().send(&msg);
    std::cerr << text.str() << std::endl;
}

RemoteHost::RemoteHost(const HostManager &manager, covise::Program type, const std::string &sessionName)
    : RemoteClient(type, sessionName), hostManager(manager)
{
}

RemoteHost::RemoteHost(const HostManager &manager, vrb::RemoteClient &&base)
    : RemoteClient(std::move(base)), hostManager(manager)
{
}

bool RemoteHost::handlePartnerAction(covise::LaunchStyle action)
{
    bool retval = false;
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
        CTRLHandler::instance()->sendCollaborativeState();
    }
    return retval;
}

bool RemoteHost::addPartner()
{
    if (startCrb())
        return startUI(CTRLHandler::instance()->uiOptions());
    return false;
}

bool RemoteHost::removePartner()
{
    if(m_state == LaunchStyle::Pending)
    {
        m_state = LaunchStyle::Disconnect;
        return true;
    }
    if (m_state == LaunchStyle::Disconnect) //already disconnected
    {
        //inform coviseDaemon that launch request is invalid
        VRB_PERMIT_LAUNCH_Abort abort{hostManager.getVrbClient().ID(), ID(), covise::Program::crb};
        sendCoviseMessage(abort, hostManager.getVrbClient());
        if (m_isProxy) //inform vrb to calcel all connection atempts to proxys of this host
        {
            std::vector<int> procs(m_processes.size());
            for (const auto &proc : m_processes)
                procs.push_back(proc->processId);
            PROXY_Abort abort(procs);
            sendCoviseMessage(abort, *hostManager.proxyConn());
        }
        return true;
    }

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
            if (isConnected(host.second->state())) //this->m_state should already be set to Disconnect
            {
                host.second->getProcess(sender_type::CRB).send(&msg);
            } /*  */
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

void RemoteHost::setProxyHost(bool isProxy)
{
    m_isProxy = isProxy;
}

void RemoteHost::clearProcesses()
{
    while (m_processes.size() > 0)
    {
        m_processes.pop_back();
    }
}
