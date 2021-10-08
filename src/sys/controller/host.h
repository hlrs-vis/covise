/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CONTROLLER_HOST_H
#define CONTROLLER_HOST_H

#include <vector>
#include <map>
#include <mutex>

#include <messages/CRB_EXEC.h>
#include <vrb/RemoteClient.h>
#include <vrb/client/VRBClient.h>
#include <messages/coviseLaunchOptions.h>
#include <messages/PROXY.h>

#include "config.h"
#include "subProcess.h"
#include "syncVar.h"

namespace covise
{
struct NEW_UI_HandlePartners;

namespace controller{



constexpr int clientIdForPreconfigured = 1000;//hope that ids greater than this do not conflict with ids given by vrb

class HostManager;
struct UIOptions;
struct ModuleInfo;
class Userinterface;
struct ControllerProxyConn;
struct RemoteHost : vrb::RemoteClient
{
    RemoteHost(const HostManager& manager, covise::Program type, const std::string& sessionName = ""); //constructs a client with local information
    RemoteHost(const HostManager& manager, vrb::RemoteClient &&base); //constructs real remote client
    RemoteHost(RemoteHost &&other) = delete;
    RemoteHost(const RemoteHost &other) = delete;
    
    const HostManager &hostManager;
    bool wantsTochangeState() const;
    bool handlePartnerAction(covise::LaunchStyle action, bool proxyRequired);
    covise::LaunchStyle state() const;
    void setTimeout(int seconds);
    bool startCrb();
    bool startUI(const UIOptions &options);
    const SubProcess &getProcess(sender_type type) const;
    SubProcess &getProcess(sender_type type);
    
    const NetModule &getModule(const std::string &name, int instance) const;
    NetModule &getModule(const std::string&name, int instance);

    void removeModule(NetModule &app, int alreadyDead);
    typedef std::vector<std::unique_ptr<SubProcess>> ProcessList;

    ProcessList::const_iterator begin() const;
    ProcessList::iterator begin();

    ProcessList::const_iterator end() const;
    ProcessList::iterator end();
    void launchProcess(const CRB_EXEC& exec) const;
    bool isModuleAvailable(const std::string &moduleName) const;
    NetModule &startApplicationModule(const string &name, const string &instanz, 
                          int posx, int posy, int copy, ExecFlag flags, NetModule *mirror = nullptr);
    bool removePartner();
    bool proxyHost() const;
    void permitLaunch(int code);
    covise::LaunchStyle desiredState() const;
private:
    bool addPartner();
    void launchScipt(covise::Program exec, const std::vector<std::string> &cmdArgs);  //need to create a new remote host for these
    void launchManual(covise::Program exec, const std::vector<std::string> &cmdArgs); //

    bool startUI(std::unique_ptr<Userinterface> &&ui, const UIOptions &options);
    void determineAvailableModules(const CRBModule &crb);
    void clearProcesses(); //order: modules->ui->crb

    ProcessList m_processes;
    std::vector<const ModuleInfo*> m_availableModules; //contains references to hostmanagers available modules
    int m_shmID;
    int m_timeout = 30;
    bool m_isProxy = false;
    std::atomic_int m_code;

protected:
    covise::LaunchStyle m_state = covise::LaunchStyle::Disconnect;
    covise::LaunchStyle m_desiredState = covise::LaunchStyle::Disconnect;
    bool m_hasPermission = false;
    virtual void connectShm(const CRBModule &crbModule);
    virtual void askForPermission();
    virtual bool launchCrb(covise::Program exec, const std::vector<std::string> &cmdArgs);
};

struct LocalHost : RemoteHost{
    LocalHost(const HostManager& manager, covise::Program type, const std::string& sessionName = "");
    virtual void connectShm(const CRBModule &crbModule) override;
    virtual bool launchCrb(covise::Program exec, const std::vector<std::string> &cmdArgs) override;
};




} //namespace controller
} //namespace covise
#endif
