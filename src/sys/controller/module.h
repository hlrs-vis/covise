/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CONTROLL_MODULE_H
#define CONTROLL_MODULE_H

#include <vector>
#include <string>

#include "subProcess.h"
#include "netLink.h"
#include "moduleInfo.h"
#include <messages/CRB_EXEC.h>


namespace covise{
namespace controller{

struct NumRunning;
//rename Application to module and module to ConnectedProcess
struct NetModule : SubProcess
{
    enum class Status
    {
        Idle,
        executing,
        starting,
        stopping
    };
    static const SubProcess::Type moduleType = SubProcess::Type::App;

    NetModule(const RemoteHost &host, const ModuleInfo &moduleInfo, int instance);
    virtual ~NetModule();
    static void resetId();

    const ModuleInfo &info() const;
    bool isOnTop() const;
    virtual void exec(NumRunning &numRunning);

    std::string fullName() const; //unambiguous name of a module consisting of name_id
    const std::string &title() const;
    void setTitle(const std::string &title);
    /// this flag is set if the module should be started
    bool startflag() const;
    void resetStartFlag();
    void setStartFlag();

    struct MapPosition
    {
        MapPosition() = default;
        MapPosition(int x, int y)
            : x(x), y(y) {}
        int x = 0;
        int y = 0;
    };
    size_t instance() const;

    virtual void init(const MapPosition &pos, int copy, ExecFlag flag, NetModule *mirror);
    int testOriginalcount(const string &intfname) const;
    const ModuleNetConnectivity &connectivity() const;
    ModuleNetConnectivity &connectivity();
    const MapPosition &pos() const;
    void move(const NetModule::MapPosition &pos);

    std::string createBasicModuleDescription() const; //name + instance + host
    std::string createDescription() const;
    bool isOriginal() const;
    bool isExecuting() const;
    void setExecuting(bool state);

    Status status() const;
    void setStatus(Status status);

    void
    setAlive(bool state);
    std::vector<std::string> &errorsSentByModule();
    const std::vector<std::string> &errorsSentByModule() const;

    void set_DO_status(int mode, const string &DO_name);
    std::string get_outparaobj() const;
    std::string get_inparaobj() const;
    bool startModuleWaitingAbove(NumRunning &numRunning); //is_one_waiting_above
    int numRunning() const; //get_num_running
    void setStart();
    void startModulesUnder(NumRunning &numRunning); //start_modules

    bool isOneRunningAbove(bool first) const;
    bool is_one_running_under() const;
    virtual void execute(NumRunning &numRunning);
    int overflowOfNextError() const;
    int numMirrors() const;
    std::vector<NetModule *> &getMirrors();
    void delete_dep_objs();
    std::string get_parameter(controller::Direction direction, bool forSaving) const;
    std::string get_interfaces(controller::Direction direction) const;
    std::string get_moduleinfo() const;
    std::string get_module(bool forSaving) const;
    void setDeadFlag(int flag);
    void writeScript(std::ofstream &of) const;
    virtual void setObjectConn(const string &from_intf, object *obj);
    virtual void delObjectConn(const string &from_intf, object *obj);
    
    const size_t moduleId; //global id, incremented for every module created -> s_nodeID
    
    mutable std::vector<netlink> netLinks;
    mutable std::vector<const NetModule *> to_c_connections, from_c_connections; //unclear what this is good for

private:
    static size_t moduleCount; //global number of modules
    
    std::string m_description;
    mutable std::string m_title; //the name that is shown in the map editor

    bool m_isStarted = false;
    bool m_alive = true;
    int m_instance = 0; //What number Module with this moduleInfo this is (unique in combination with module name)
    std::string getStartMessage();
    void sendWarningMsgToMasterUi(const std::string &msg);
    bool delete_old_objs();
    void new_obj_names();
    void sendFinish();
    void delete_rez_objs();

protected:
    enum
    {
        NOT_MIRR = 0,
        ORG_MIRR,
        CPY_MIRR
    } m_mirror = NOT_MIRR;
    ModuleNetConnectivity m_connectivity;
    std::vector<NetModule *> m_mirrors;
    std::vector<std::string> m_errorsSentByModule;
    int m_numRunning = 0; /// number of startmessages sent to module
    Status m_status = Status::Idle; 
    const ModuleInfo &m_info; 
    MapPosition m_position;
    std::string serialize() const;
    size_t getNumInterfaces(controller::Direction direction) const;
    void mirror(NetModule *original);
    void initConnectivity();
    virtual void copyConnectivity();
    virtual std::string serializeInputInterface(const net_interface &interface) const;
};


} // namespace controller
} // namespace covise


#endif
