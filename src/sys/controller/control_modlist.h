/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_MODULELIST_H
#define CTRL_MODULELIST_H

#include <covise/covise.h>
#include <net/message_sender_interface.h>

#include "coHostType.h"

#include "control_list.h"
#include "control_module.h"
#include "control_define.h"
#include "control_def.h"
#ifndef _WIN32
//extern "C" pid_t fork(void);
//extern "C" int execlp (const string &file, const string &arg, ...);
#ifdef __alpha
extern "C" int execvp(const string &file, const string &const argv[]);
#endif
#endif

namespace covise
{

class uif;
class uiflist;
class modui;
class modui_list;

class modulelist;
class DM_interface;
class DM_int_list;
class DM_data;
class DM_list;
class rhost;
class rhost_list;
class userinterface;
class ui_list;

/************************************************************************/
/* 									                                          */
/* 			MODULELIST 					                                    */
/* 									                                          */
/************************************************************************/

class modulelist : public Liste<module>
{
public:
    modulelist();
    module *get(const string &name, const string &host);
    string create_modulelist();
    void add_module_list(const string &user, const string &host, const string &data);
    void rmv_module_list(const string &host);
};

/************************************************************************/
/* 									                                          */
/* 			INTERFACELIST 					                                 */
/* 									                                          */
/************************************************************************/

class DM_interface
{
    string name;

public:
    DM_interface();
    void set_name(const string &tmp);
    string get_name()
    {
        return name;
    };
};

class DM_int_list : public Liste<DM_interface>
{
    int counter;
    int def;

public:
    DM_interface *first;
    DM_int_list();
    void set_count(int a)
    {
        counter = a;
    };
    int get_count()
    {
        return counter;
    };
    void set_default(int a)
    {
        def = a;
    };
    int get_default()
    {
        return def;
    };
};

/************************************************************************/
/* 									                                          */
/* 			DM_DATA 					                                       */
/* 									                                          */
/************************************************************************/

class DM_data
{
    AppModule *dm;
    string user;
    string passwd;
    string hostname;
    string modname;

public:
    Message *list_msg;
    Message *interface_msg;
    DM_data();
    ~DM_data();

    void set_user(const string &str);
    void set_passwd(const string &str);
    void set_hostname(const string &str);
    void set_modname(const string &str);
    void set_DM(AppModule *dmod);

    string get_modname()
    {
        return modname;
    };
    string get_user()
    {
        return user;
    };
    string get_passwd()
    {
        return passwd;
    };
    string get_hostname()
    {
        return hostname;
    };
    AppModule *get_DM();

    int start_crb(int type, const string &host, const string &user, const string &passwd, const string &script_name, coHostType &htype);
    void quit();
    int new_desk();
    void send_msg(Message *msg);
    int get_mod_id()
    {
        return dm->get_id();
    };
};

/************************************************************************/
/* 									                                          */
/* 			DM_LIST 					                                       */
/* 									                                          */
/************************************************************************/

class DM_list : public Liste<DM_data>
{
    DM_data *local;

public:
    DM_list();

    AppModule *start_local(const string &local_user);
    DM_data *get_local();
    int add_crb(int type, const string &host, const string &user, const string &passwd, const string &script_name, coHostType &htype);

    DM_data *get(const string &host);
    DM_data *get(const string &host, const string &user);
    DM_data *get(AppModule *dmod);
    DM_data *get(int id);

    void quit();
    int new_desk();
    void send_msg(Message *msg);
    void connect_all(AppModule *dmod);
};

/************************************************************************/
/* 									                                          */
/* 				RHOST 					                                    */
/* 									                                          */
/************************************************************************/

class rhost
{
    string hostname;
    string user;
    string passwd;
    string htype;
    AppModule *ctrl;
    bool save_info;

#ifdef CONNECT
    DM_int_list *intlist;
#endif

public:
    rhost();
    ~rhost();
    void set_hostname(const string &str);
    void set_user(const string &str);
    void set_passwd(const string &str);
    void set_type(const string &str);
    AppModule *get_crb()
    {
        return ctrl;
    };

    string get_user()
    {
        return user;
    };
    string get_passwd()
    {
        return passwd;
    };
    string get_hostname()
    {
        return hostname;
    };
    string get_type()
    {
        return htype;
    };

    int start_ctrl(int type, const string &script_name, coHostType &htype);

    void send(Message *msg);
    void recv_msg(Message *msg);
    void send_ctrl_quit();
    void send_hostadr(const string &hostname);
    void mark_save();
    void reset_mark();
    bool get_mark();
#ifdef CONNECT
    void set_intflist(const string &intlist);
    string get_DC_list();
#endif
    void print();
};

/************************************************************************/
/* 								            	                              */
/* 			RHOST_LIST 					                                    */
/* 									                                          */
/************************************************************************/

class rhost_list : public Liste<rhost>
{

public:
    rhost_list();
    rhost *get(const string &host);
    rhost *get(const string &host, const string &user);
    string get_hosts(const string &local_name, const string &local_user);
    int add_host(const string &hostname, const string &user_id, const string &passwd, const string &script_name, coHostType &htype);
    int add_local_host(const string &local_user);
    int rmv_host(const string &hostname, const string &user_id);
    void mark_host();
    void mark_all();
    void reset_mark();
    void print();
};

//************************************************************************/
//
// 				UIF
//
//************************************************************************/
class uif
{
    string host;
    string userid;
    string passwd;
    string status;

    AppModule *applmod;
    int proc_id;

public:
    void set_host(const string &tmp);
    void set_userid(const string &tmp);
    void set_passwd(const string &tmp);
    void set_status(const string &tmp);

    string get_host();
    string get_userid();
    string get_status();

    int get_procid();
    void send_msg(Message *msg);
    void start(AppModule *dmod, const string &execname, const string &category, const string &key,
               const string &name, const string &instanz, const string &host);

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
    ~uiflist(){};
    void create_uifs(const string &execname, const string &category, const string &key, const string &name, const string &instanz, const string &host);
};

//************************************************************************/
//
// 				MODUI
//
//************************************************************************/
class modui
{
    string name;
    string instanz;
    string host;
    string category;
    string key;
    string execname;
    int nodeid;
    net_module *netmod;
    uiflist *uif_list;

public:
    void set_name(const string &tmp);
    void set_instanz(const string &tmp);
    void set_host(const string &tmp);
    void set_key(const string &tmp);
    void set_execname(const string &tmp);
    void set_category(const string &tmp);
    void set_netmod(net_module *tmp);
    void set_nodeid(int id);

    string get_name();
    string get_instanz();
    string get_host();
    string get_key();
    string get_category();
    string get_execname();
    int get_nodeid();
    net_module *get_netmod();

    void create_uifs();
    void delete_uif();
    void set_new_status();

    void send_msg(Message *msg);
    void sendapp(Message *msg);
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

    void create_mod(const string &name, const string &instanz, const string &category, const string &host, const string &key, const string &executable);

    void delete_mod(const string &key);
    void delete_mod(int nodeid);
    void delete_mod(const string &name, const string &nr, const string &host);

    void set_new_status();

    modui *get(int nodeid);
    modui *get(const string &key);
    modui *get(const string &name, const string &nr, const string &host);
};

//************************************************************************/
//
// 			USERINTERFACE
//
//************************************************************************/
class userinterface : public MessageSenderInterface
{

protected:
    AppModule *ui;
    string status;
    string hostname;
    string userid;
    string passwd;

    bool rendererIsPossible;
    bool rendererIsActive;

    virtual bool sendMessage(const Message *msg) override;
    virtual bool sendMessage(const UdpMessage *msg) override;

public:
    userinterface();
    virtual ~userinterface()
    {
    }
    void set_host(const string &str);
    string get_host()
    {
        return hostname;
    };
    string get_userid()
    {
        return userid;
    };
    string get_passwd()
    {
        return passwd;
    };
    void set_userid(const string &str);
    void set_passwd(const string &str);
    void set_status(const string &str);
    string get_status()
    {
        return status;
    };
    AppModule *get_mod()
    {
        return ui;
    };
    int get_mod_id()
    {
        return ui->get_id();
    };

    virtual int start(bool restart) = 0;
    int restart();
    int xstart(const string &pyFile);
    void quit();

    void change_status(const string &str);
    void recv_msg(Message *msg)
    {
        ui->recv_msg(msg);
    };
    void change_master(const string &user, const string &host);
};

class UIMapEditor : public userinterface
{
public:
    UIMapEditor()
        : userinterface()
    {
    }
    ~UIMapEditor()
    {
    }
    virtual int start(bool);
};

class UISoap : public userinterface
{
public:
    UISoap()
        : userinterface()
    {
    }
    ~UISoap()
    {
    }
    virtual int start(bool);
};

//************************************************************************/
//
// 				UI_LIST
//
//***********************************************************************/

class ui_list : public Liste<userinterface>
{
private:
    int MR_sender; /* sender des MASTER-REQ */
    string WBhops;
    string WBaddress;
    std::list<userinterface *> locals;
    string status_next; /* status of the next userinterface */
    int iconify;
    int maximize;
    bool m_slaveUpdate;

public:
    ui_list();
    ~ui_list();
    void set_slaveUpdate(int su)
    {
        m_slaveUpdate = (su != 0);
    };
    userinterface *get(const string &hostname, const string &user);
    userinterface *get(const string &hostname); // only host counts anyway :-(
    userinterface *get(int sender_no);
    int start_local_Mapeditor(const string &moduleinfo);
    int start_local_WebService(const string &moduleinfo);
    int start_local_xuif(const string &moduleinfo, const string &pyFile);
    bool add_config(const string &file, const string &mapfile);
    int config_action(const string &mapfile, const string &host, const string &userid, const string &passwd);
    int add_partner(const string &filename, const string &host, const string &userid, const string &passwd, const string &script_name);
    int rmv_partner(const string &host, const string &user_id);
    void send_slave(Message *msg);
    void send_master(Message *msg);
    void sendError(const string &txt);
    void sendWarning(const string &txt);
    void sendError2m(const string &txt);
    void sendWarning2m(const string &txt);
    void send_all(Message *msg);
    void quit_and_del();
    void change_master(int sender_no, const string &user, const string &host);
    void force_master(const string &host);
    void send_new_status(const string &status);
    userinterface *get_master();
    bool testid(int id);
    void set_iconify(int ic);
    bool slave_update();
    void set_maximize(int maxi);
    void update_ui(userinterface *ui);
    int update_all(const string &mod_info, const string &DC_info);
};
}
#endif
