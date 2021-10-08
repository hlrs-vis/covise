#ifndef CONTROLLER_HOSTMANAGER_H
#define CONTROLLER_HOSTMANAGER_H

#include "host.h"

namespace covise{
struct NEW_UI_HandlePartners;

namespace controller{

class HostManager
{
public:
    typedef std::map<int, std::unique_ptr<RemoteHost>> HostMap;
    HostManager();
    ~HostManager();
    struct UiState{
        int masterRequestSenderId = 0;
    } uiState;

    void sendPartnerList() const; 
    void handleActions(const covise::NEW_UI_HandlePartners &msg);
    void handleAction(LaunchStyle style, RemoteHost &h);

    void setOnConnectCallBack(std::function<void(void)> cb);
    int vrbClientID() const;
    const vrb::VRBClient &getVrbClient() const;
    
    LocalHost &getLocalHost();
    const LocalHost &getLocalHost() const;
    
    RemoteHost *getHost(int clientID);
    const RemoteHost *getHost(int clientID) const;

    RemoteHost &findHost(const std::string &ipAddress); //this will cause collaborative work on a single host to fail
    const RemoteHost &findHost(const std::string &ipAddress) const;

    std::vector<const SubProcess *> getAllModules(sender_type type) const;
    template<typename T>
    std::vector<const T*> getAllModules() const{
            auto v =  const_cast<HostManager *>(this)->getAllModules<T>();
            std::vector<const T *> modules(v.size());
            std::transform(v.begin(), v.end(), modules.begin(), [](T *t) {
                return t;
            });
           return modules;
    }
    template<typename T>
    std::vector<T*> getAllModules(){
        std::vector<T *> modules;
        for (auto &host : m_hosts)
        {
            for (auto &module : *host.second)
            {
                if (auto tmpMod = dynamic_cast<T *>(module.get()))
                {
                    modules.push_back(tmpMod);
                }
            }
        }
        return modules;
    }
    template<typename T>
    void sendAll(const Message &msg) const{
        for(const auto module : getAllModules<T>())
            module->send(&msg);
    }
    
    HostMap::const_iterator begin() const;
    HostMap::iterator begin();
    HostMap::const_iterator end() const;
    HostMap::iterator end();

    SubProcess *findModule(int peerID);
    bool slaveUpdate();
    mutable bool m_slaveUpdate = false;
    const Userinterface &getMasterUi() const;
    Userinterface &getMasterUi();
    std::string getHostsInfo() const; //get info string with all hosts
    const ModuleInfo &registerModuleInfo(const std::string &name, const std::string &category) const;
    void resetModuleInstances();
    const ControllerProxyConn *proxyConn() const;
    std::unique_ptr<Message> receiveProxyMessage();
    void handleVrb();

private:
    mutable std::set<ModuleInfo> m_availableModules; //every module that is available on at leaset one host. This manages the instance ids of the modules.
    std::unique_ptr<vrb::VRBClient> m_vrb;
    HostMap m_hosts;
    HostMap::iterator m_localHost;
    std::function<void(void)> m_onConnectVrbCallBack;
    std::unique_ptr<ControllerProxyConn> m_proxyConnection;
    bool checkIfProxyRequiered(int clID, const std::string &hostName);
    void createProxyConn();
    bool handleVrbMessage();
    void moveRendererInNewSessions();

};

}
}

#endif // CONTROLLER_HOSTMANAGER_H