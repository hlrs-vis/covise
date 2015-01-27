/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_OBJECT_H
#define CTRL_OBJECT_H

#include <net/covise_connect.h>
#include <covise/covise_msg.h>
#include "control_process.h"
#include "covise_module.h"
#include "control_def.h"
#include "control_define.h"
#include "control_list.h"
#include "control_netmod.h"

namespace covise
{

class object;
class obj_port;
class obj_port_list;
class object_list;
class net_module;

extern void handleClosedMsg(Message *msg);

#define DO_SAVE 1
#define DO_RELEASE 0

/************************************************************************/
/* 									                                          */
/* 			OBJ_FROM_CONN					                                 */
/* 									                                          */
/************************************************************************/

class obj_from_conn
{
    net_module *module; /* link to the module */
    string mod_intf; /* name of the interface */
    string type; /* type of the interface */
    string state;

public:
    obj_from_conn();

    void conn_module(net_module *mod)
    {
        module = mod;
    };
    net_module *get_mod()
    {
        return module;
    };

    void set_intf(const string &str);
    string get_intf()
    {
        return mod_intf;
    };

    void set_state(const string &str);
    string get_state()
    {
        return state;
    };

    void set_type(const string &str);
    string get_type()
    {
        return type;
    };
};

/************************************************************************/
/* 									                                          */
/* 				DATA					                                       */
/* 									                                          */
/************************************************************************/

class data
{
    string name;
    string status;
    int save;
    int count;

public:
    data();
    ~data();

    void set_name(const string &str);
    string get_name()
    {
        return name;
    };

    void set_status(const string &str);
    string get_status()
    {
        return status;
    };

    void SAVE()
    {
        save = 1;
    };
    void RELEASE()
    {
        save = 0;
    };
    int get_save_status()
    {
        return save;
    };

    void set_count(int i)
    {
        count = i;
    };
    int get_count()
    {
        return count;
    };

    void del_data(AppModule *dmod);
};

/************************************************************************/
/* 									                                          */
/* 				DATA_LIST				                                    */
/* 									                                          */
/************************************************************************/

class data_list : public Liste<data>
{
    int count;

public:
    data_list();
    int get_count()
    {
        return count;
    };
    void set_count(int nc)
    {
        count = nc;
    };
    void inc_count();
    void dec_count();
    string create_new_data(const string &name);
    string create_new_DO(const string &name);
    void new_data(const string &name);
    void set_status(const string &str);
    data *get(const string &name);
    data *get_new();
};

/************************************************************************/
/* 									                                          */
/* 				OBJ_CONN				                                       */
/* 									                                          */
/************************************************************************/

class obj_conn
{
    net_module *mod;
    string mod_intf;
    string old_name;
    data_list *datalist;

public:
    obj_conn();
    ~obj_conn();

    string get_old_name()
    {
        return old_name;
    };
    void set_old_name(const string &name);
    void connect_module(net_module *module);
    net_module *get_mod();

    void set_mod_intf(const string &str);
    string get_mod_intf()
    {
        return mod_intf;
    };

    void new_data(const string &new_name);
    void del_old_DO(const string &name);
    void del_rez_DO(const string &name);

    string get_status();
    void set_status(const string &str);

    void copy_data(data_list *dl);
    data_list *get_datalist()
    {
        return datalist;
    };
};

/************************************************************************/
/* 									                                          */
/* 				OBJ_CONN_LIST				                                 */
/* 									                                          */
/************************************************************************/

class obj_conn_list : public Liste<obj_conn>
{
public:
    obj_conn_list();
    void new_timestep(const string &new_name);
    void new_DO(const string &new_name);
    void del_old_DO(const string &name);
    void del_rez_DO(const string &name);
    obj_conn *get(const string &name, const string &nr, const string &host, const string &intf);
};

/************************************************************************/
/* 								                  	                        */
/* 				OBJECT					                                    */
/* 									                                          */
/************************************************************************/

class ui_list;

class object
{
    string name; /* name of the Object */
    obj_from_conn *from; /* input-connection to a net_module */
    obj_conn_list *to;
    data_list *dataobj; /* referencelist of the dataobjects */

public:
    object();
    ~object();
    void set_name(const string &str);
    string get_name()
    {
        return name;
    };
    string get_type()
    {
        return from->get_type();
    };

    string get_new_name();
    string get_current_name();

    obj_from_conn *get_from()
    {
        return from;
    };
    obj_conn_list *get_to()
    {
        return to;
    };
    data_list *get_dataobj()
    {
        return dataobj;
    };

    obj_conn *connect_to(net_module *module, const string &intfname);
    void connect_from(net_module *module, const string &intfname, const string &type);

    void del_from_connection();
    void del_to_connection(const string &name, const string &nr, const string &host, const string &intf);
    void del_allto_connects();

    void set_start_module();
    void start_modules(ui_list *ul);
    int is_one_running_above();

    void new_timestep();
    void new_DO();
    void del_old_DO();
    void del_rez_DO();
    void del_all_DO(int already_dead);
    void del_old_data();
    void del_dep_data();
    void del_all_data(AppModule *dmod);

    void change_NEW_to_OLD(net_module *mod, const string &intf);
    void set_to_NEW();

    int check_from_connection(net_module *mod, const string &intf);
    string get_conn_state(net_module *mod, const string &port);

    void set_DO_status(const string &DO_name, int mode, net_module *mod, const string &intf_name);
    void print_states();

    bool test(const string &DO_name);

    void set_outputtype(const string &DO_type);
    void resetobj_for_exec(AppModule *dmod);

    void writeScript(ofstream &of, const string &local_name, const string &local_user);
    string get_connection(int *i, const string &local_name, const string &local_user);
    string get_simple_connection(int *i);

    int get_counter();
    bool isEmpty();
};

/************************************************************************/
/* 									                                          */
/* 				OBJECT_LIST				                                    */
/* 									                                          */
/************************************************************************/

class object_list : public Liste<object>
{
    int count;

public:
    object_list();

    object *get(const string &name);
    string get_connections(const string &local_name, const string &local_user);
    void writeScript(ofstream &of, const string &local_name, const string &local_user);
    object *select(const string &name);
};
}
#endif
