/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_PORT_H
#define CTRL_PORT_H

#include "control_def.h"
#include "control_object.h"
#include "control_define.h"
#include "control_netmod.h"

namespace covise
{

class connect_mod;
class connect_mod_list;
class connect_obj;
class connect_obj_list;
class port;
class C_interface;
class net_interface;
class render_interface;
class parameter;
class net_parameter;
class render_module;
class displaylist;

/************************************************************************/
/* 									                                          */
/* 			CONNECT_mod & CONNECT_mod_list		                     	*/
/* 								                                          	*/
/************************************************************************/

class connect_mod
{
    net_module *conn_obj; /* link to connected module */
    string conn_par; /* name of parameter in the connected module */
public:
    connect_mod();

    void link_mod(net_module *mod);
    net_module *get_mod();
    void set_par(const string &par);
    string get_par();
};

class connect_mod_list : public Liste<connect_mod>
{
public:
    connect_mod_list();
    connect_mod *get(net_module *mod, string par);
};

/************************************************************************/
/* 									                                          */
/* 			CONNECT_obj & CONNECT_obj_list		                     	*/
/* 									                                          */
/************************************************************************/

class connect_obj
{
    object *conn_obj;
    string old_name;

public:
    connect_obj();

    void set_conn(object *obj);
    object *get_obj();
    void del_conn();

    void set_oldname(const string &str);
    string get_oldname();
};

class connect_obj_list : public Liste<connect_obj>
{
public:
    connect_obj_list();
};

/************************************************************************/
/* 									                                          */
/* 				PORT				                                       	*/
/* 							                                          		*/
/************************************************************************/

class port
{
    string name;
    string type;
    string text;
    friend class C_interface;
    friend class parameter;

public:
    port();
    virtual ~port()
    {
    }

    void set_name(const string &str);
    void set_type(const string &str);
    void set_text(const string &str);
    string get_name()
    {
        return name;
    };
    string get_type()
    {
        return type;
    };
    string get_text()
    {
        return text;
    };
};

/************************************************************************/
/* 							                                          		*/
/* 			INTERFACE			                                    		*/
/* 							                                          		*/
/************************************************************************/

class C_interface : public port
{
    string demand;
    string direction;
    friend class render_interface;
    friend class net_interface;

public:
    C_interface();

    void set_demand(const string &str);
    string get_demand()
    {
        return demand;
    };
    void set_direction(const string &str);
    string get_direction()
    {
        return direction;
    };
};

/************************************************************************/
/* 								                                          	*/
/* 				VALUE_LIST		                                    		*/
/* 									                                          */
/************************************************************************/
class Value
{
    string s;

public:
    Value();
    void set(string str);
    string get()
    {
        return s;
    };
};

class value_list : public Liste<Value>
{
public:
    int count;
    value_list();
};

/************************************************************************/
/* 									                                          */
/* 			PARAMETER				                                    	*/
/* 									                                          */
/************************************************************************/

class parameter : public port
{
    value_list *values;
    string org_val;

#ifdef PARA_START
    string extension;
#endif

    string panel;

    friend class net_parameter;

public:
    parameter();

    void set_value(int no, string str);
    void set_value_list(string str);
    string get_value(int no);
    int get_count();

    string get_org_val()
    {
        return org_val;
    };
    string get_val_list();
    string get_pyval_list();

#ifdef PARA_START
    void set_extension(const string &ext);
    string get_extension()
    {
        return extension;
    };
#endif

    void set_addvalue(const string &add_para);
    string get_addvalue()
    {
        return panel;
    };
};

/************************************************************************/
/* 									                                          */
/* 			NET_INTERFACE				                                 	*/
/* 								                                          	*/
/************************************************************************/

class net_interface : public C_interface
{
    object *obj;

public:
    net_interface();

    void set_connect(object *conn);
    void del_connect();
    object *get_object();
    int get_state(net_module *mod);
    bool get_conn_state()
    {
        return obj != NULL;
    };
    int check_conn()
    {
        return obj != NULL;
    };

    void set_outputtype(const string &DO_name, const string &DO_type);
};

/************************************************************************/
/* 									                                          */
/* 			RENDER_INTERFACE			                                 	*/
/* 									                                          */
/************************************************************************/

class render_interface : public C_interface
{
    connect_obj_list *connects;
    int conn_count;
    int wait_count;

public:
    render_interface();

    connect_obj_list *get_connects()
    {
        return connects;
    };
    void del_all_connections(render_module *mod);
    void set_connect(object *obj);
    void del_connect(object *obj, displaylist *displays);

    int get_state(net_module *mod);
    bool get_conn_state();
    int check_conn();
    void count_init(render_module *mod);

    connect_obj *get_first_NEW(render_module *mod);
    void reset_wait();
    void decr_wait();
    bool get_wait_status();
    void reset_to_NEW(render_module *mod);

    string get_objlist();
};

/************************************************************************/
/* 								                                          	*/
/* 			NET_PARAMETER		                                 			*/
/* 								                                          	*/
/************************************************************************/

class net_parameter : public parameter
{
    connect_mod_list *connects;

public:
    net_parameter();

    connect_mod_list *get_connectlist();
    void set_P_conn(net_module *mod, string par);
    void del_P_conn(net_module *mod, string par);
};
}
#endif
