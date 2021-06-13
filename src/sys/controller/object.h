/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_OBJECT_H
#define CTRL_OBJECT_H
#include "list.h"
#include <list>
#include <net/covise_connect.h>
#include <covise/covise_msg.h>
namespace covise
{
namespace controller{

struct NetModule;
struct NumRunning;
class CRBModule;
class net_interface;

extern void handleClosedMsg(Message *msg);

/************************************************************************/
/* 									                                          */
/* 			OBJ_FROM_CONN					                                 */
/* 									                                          */
/************************************************************************/

class obj_from_conn
{
    NetModule *module; /* link to the module */
    std::string mod_intf;                 /* name of the interface */
    std::string type;                     /* type of the interface */
    std::string state;

public:
    obj_from_conn();
    void conn_module(NetModule *mod)
    {
        module = mod;
    };

    NetModule *get_mod()
    {
        return module;
    };

    const NetModule *get_mod() const
    {
        return module;
    };

    void set_intf(const std::string &str);
    const std::string &get_intf() const
    {
        return mod_intf;
    };

    void set_state(const std::string &str);
    const std::string &get_state() const
    {
        return state;
    };

    void set_type(const std::string &str);
    const std::string &get_type() const
    {
        return type;
    };
};

/************************************************************************/
/* 									                                          */
/* 				DATA					                                       */
/* 									                                          */
/************************************************************************/

enum
{
    DO_RELEASE,
    DO_SAVE
};

class data
{
    std::string name;
    std::string status;
    int save;
    int count;

public:
    data();

    void set_name(const std::string &str);
    const std::string &get_name() const
    {
        return name;
    };

    void set_status(const std::string &str);
    const std::string &get_status() const
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

    void del_data(const controller::CRBModule &crb);
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
    int get_count() const
    {
        return count;
    };
    void set_count(int nc)
    {
        count = nc;
    };
    void inc_count();
    void dec_count();
    std::string create_new_data(const std::string &name);
    std::string create_new_DO(const std::string &name);
    void new_data(const std::string &name);
    void set_status(const std::string &str);
    data *get(const std::string &name);
    data *get_new();
};

/************************************************************************/
/* 									                                          */
/* 				OBJ_CONN				                                       */
/* 									                                          */
/************************************************************************/

class obj_conn
{
    NetModule *mod;
    std::string mod_intf;
    std::string old_name;
    data_list *datalist;

public:
    obj_conn();
    obj_conn(const obj_conn &) = delete;
    obj_conn(obj_conn &&other);
    obj_conn &operator=(const obj_conn &) = delete;
    obj_conn &operator=(obj_conn &&other);

    ~obj_conn();

    const std::string get_old_name() const
    {
        return old_name;
    };
    void set_old_name(const std::string &name);
    void connect_module(NetModule *module);
    const NetModule *get_mod() const;
    NetModule *get_mod();


    void set_mod_intf(const std::string &str);
    const std::string get_mod_intf() const
    {
        return mod_intf;
    };

    void new_data(const std::string &new_name);
    void del_old_DO(const std::string &name);
    void del_rez_DO(const std::string &name);

    std::string get_status();
    void set_status(const std::string &str);

    void copy_data(data_list *dl);
    data_list *get_datalist()
    {
        return datalist;
    };
    const data_list *get_datalist() const
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
    void new_timestep(const std::string &new_name);
    void new_DO(const std::string &new_name);
    void del_old_DO(const std::string &name);
    void del_rez_DO(const std::string &name);
    obj_conn *get(const std::string &name, const std::string &nr, const std::string &host, const std::string &intf);
};

/************************************************************************/
/* 								                  	                        */
/* 				OBJECT					                                    */
/* 									                                          */
/************************************************************************/

//represents the connections from a single output port to multiple input ports
class object
{
    std::string name; /* name of the Object */
    obj_from_conn from; /* input-connection to a net_module */
    std::list<obj_conn> to;
    data_list *dataobj; /* referencelist of the dataobjects */

public:
    object();
    ~object();
    void set_name(const std::string &str);
    const std::string &get_name() const
    {
        return name;
    };
    const std::string &get_type() const
    {
        return from.get_type();
    };

    std::string get_new_name() const;
    std::string get_current_name() const;

    obj_from_conn &get_from()
    {
        return from;
    };
    const obj_from_conn &get_from() const
    {
        return from;
    };

    std::list<obj_conn> &get_to()
    {
        return to;
    };
    const std::list<obj_conn> &get_to() const
    {
        return to;
    };
    data_list *get_dataobj()
    {
        return dataobj;
    };

    obj_conn *connect_to(controller::NetModule *module, const std::string &intfname);
    void connect_from(controller::NetModule *module, const net_interface& interface);

    void del_from_connection();
    void del_to_connection(const std::string &name, const std::string &nr, const std::string &host, const std::string &intf);
    void del_allto_connects();
    void setStartFlagOnConnectedModules(); //set_start_module
    void start_modules(NumRunning& numRunning);
    int is_one_running_above() const;

    void newDataObject(); //same as new_timestep
    void del_old_DO();
    void del_rez_DO();
    void del_all_DO(int already_dead);
    void del_old_data();
    void del_dep_data();

    void change_NEW_to_OLD(const controller::NetModule *mod, const std::string &intf);
    void set_to_NEW();

    bool check_from_connection(const controller::NetModule * mod, const std::string &interfaceName) const;
    std::string get_conn_state(const controller::NetModule *mod, const std::string &port);

    void set_DO_status(const std::string &DO_name, int mode, const controller::NetModule& mod, const std::string &intf_name);
    void print_states();

    bool test(const std::string &DO_name);

    void set_outputtype(const std::string &DO_type);

    void writeScript(std::ofstream &of, const std::string &local_name, const std::string &local_user);
    std::string get_connection(int *i, const std::string &local_name, const std::string &local_user);
    std::string get_simple_connection(int *i);

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

    object *get(const std::string &name);
    std::string get_connections(const std::string &local_name, const std::string &local_user);
    void writeScript(std::ofstream &of, const std::string &local_name, const std::string &local_user);
    object *select(const std::string &name);
};
} // namespace controller
} // namespace covise

#endif
