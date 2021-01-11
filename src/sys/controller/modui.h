/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CONTROLLER_MODUI_H
#define CONTROLLER_MODUI_H

#include <string>
#include <memory>

#include "userinterface.h"
#include "module.h"
#include "subProcess.h"
namespace covise
{
namespace controller
{
    
class CRBModule;
//************************************************************************/
//
// 				UIF
//
//************************************************************************/
class uif
{
    std::string hostAddress;
    std::string userid;
    std::string passwd;
    Userinterface::Status  status;

    std::unique_ptr<controller::ModuleInfo> appInfo;
    std::unique_ptr<controller::NetModule> applmod;
    int proc_id;

public:
    void set_hostAddress(const std::string &tmp);
    void set_userid(const std::string &tmp);
    void set_passwd(const std::string &tmp);
    void set_status(Userinterface::Status s);

    const std::string &get_hostAddress() const;
    const std::string &get_userid() const;
    Userinterface::Status get_status() const;
    const NetModule *get_app() const;

    int get_procid() const;
    void send_msg(const Message *msg) const;
    void start(const controller::CRBModule &crb, const std::string &execname, const std::string &category, const std::string &key,
               const std::string &name, const std::string &instanz, const std::string &hostAddress);

    void delete_uif();
};

//************************************************************************/
//
// 				UIFLIST
//
//************************************************************************/

class uiflist : public Liste<uif>
{
    int count;

public:
    uiflist();
    void create_uifs(const NetModule& app, const std::string &execname, const std::string &key);
};

//************************************************************************/
//
// 				MODUI
//
//************************************************************************/
class modui
{
    std::string key;
    std::string execname;
    int nodeid;
    const NetModule *app;
    uiflist *uif_list;

public:
    void set_key(const std::string &tmp);
    void set_execname(const std::string &tmp);
    void set_application(const NetModule *tmp);
    void set_nodeid(int id);

    const std::string &get_key() const;
    const std::string &get_execname() const;
    int get_nodeid() const;
    const NetModule *get_application() const;

    void create_uifs();
    void delete_uif();
    void set_new_status();

    void send_msg(const Message *msg);
    void sendapp(const Message *msg);
};

//************************************************************************/
//
//				MODUI_LIST
//
//************************************************************************/
class modui_list : public Liste<modui>
{

public:
    modui_list();
    ~modui_list(){};

    void create_mod(const NetModule& app, const std::string &key, const std::string &executable);

    void delete_mod(const std::string &key);
    void delete_mod(int nodeid);
    void delete_mod(const std::string &name, const std::string &nr, const std::string &hostAddress);

    void set_new_status();

    modui *get(int nodeid);
    modui *get(const std::string &key);
    modui *get(const std::string &name, const std::string &nr, const std::string &hostAddress);
};


} // namespace controller
    
} // namespace covise



#endif // !CONTROLLER_MODUI_H