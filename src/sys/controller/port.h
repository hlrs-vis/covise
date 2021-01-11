/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_PORT_H
#define CTRL_PORT_H

#include <string>
#include <vector>

#include "object.h"
#include "list.h"
namespace covise
{
namespace controller{

struct NetModule;
struct Renderer;
class Display;
class C_interface;
class net_interface;
class render_interface;
class parameter;

class connect_obj
{
    object *conn_obj;
    std::string old_name;

public:
    connect_obj();

    void set_conn(object *obj);
    object *get_obj();
    void del_conn();

    void set_oldname(const std::string &str);
    const std::string &get_oldname() const;
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
    std::string name;
    std::string type;
    std::string text;
    friend class C_interface;
    friend class parameter;

public:
    port();
    virtual ~port()
    {
    }

    void set_name(const std::string &str);
    void set_type(const std::string &str);
    void set_text(const std::string &str);
    const std::string &get_name() const
    {
        return name;
    };
    const std::string &get_type() const
    {
        return type;
    };
    const std::string &get_text() const
    {
        return text;
    };
};

/************************************************************************/
/* 							                                          		*/
/* 			INTERFACE			                                    		*/
/* 							                                          		*/
/************************************************************************/
enum class Direction
{
    Input,
    Output
};
class C_interface : public port
{
protected:
    std::string demand;
    controller::Direction direction;

public:

    C_interface();
    virtual ~C_interface() = default;
    void set_demand(const std::string &str);
    std::string get_demand() const
    {
        return demand;
    };
    void set_direction(Direction dir);
    Direction get_direction() const
    {
        return direction;
    };
};

/************************************************************************/
/* 									                                          */
/* 			PARAMETER				                                    	*/
/* 									                                          */
/************************************************************************/

class parameter : public port
{
    std::vector<std::string> values;
    std::string org_val;
    std::string extension = "-1";
    std::string panel = "-1";

public:
    void set_value(int no, const std::string &str);
    void set_value_list(const std::string &str);
    int get_count() const;

    const std::string &get_org_val() const
    {
        return org_val;
    };
    std::string get_val_list() const;
    std::string get_pyval_list() const;
    std::string serialize() const;
    std::string getDescription() const;
    void set_extension(const std::string &ext);
    const std::string &get_extension() const
    {
        return extension;
    };
    void set_addvalue(const std::string &add_para);
    const std::string &get_addvalue() const
    {
        return panel;
    };
};

/************************************************************************/
/* 									                                          */
/* 			NET_INTERFACE				                                 	*/
/* 								                                          	*/
/************************************************************************/
enum InterfaceState
{

S_TRUE, // 1
S_INIT, // 2
S_READY, // 3
S_CONN, // 4
S_RUNNING, // 5
S_FINISHED, // 6
S_OLD, // 7
S_NEW, // 8
S_OPT // 9

};

class net_interface : public C_interface
{
    object *obj = nullptr;
    int m_alreadyDead = 0;

public:
    net_interface(const C_interface &other);
    ~net_interface();
    void set_connect(object *conn);
    void del_connect();
    object *get_object();
    const object *get_object() const;
    int get_state(const NetModule *mod) const;

    bool get_conn_state() const
    {
        return obj;
    };
    void setDeadFlag(int flag)
    {
        m_alreadyDead = flag;
    }
    void set_outputtype(const std::string &DO_name, const std::string &DO_type);
};

/************************************************************************/
/* 									                                          */
/* 			RENDER_INTERFACE			                                 	*/
/* 									                                          */
/************************************************************************/

class render_interface : public C_interface
{
    mutable connect_obj_list connects; //mutable because interface of Liste sucks 
    int conn_count = 0;
    int wait_count = 0;

public:
    //render_interface() = default;
    render_interface(const C_interface &other);
    connect_obj_list &get_connects()
    {
        return connects;
    };
    void del_all_connections(const NetModule *mod);
    void set_connect(object *obj);
    void del_connect(object *obj, std::vector<std::unique_ptr<Display>> &displays);

    int get_state(const NetModule *mod) const;
    bool get_conn_state();
    int check_conn();
    void count_init(const Renderer *mod);

    connect_obj *get_first_NEW(const Renderer *mod);
    void reset_wait();
    void decr_wait();
    bool get_wait_status();
    void reset_to_NEW(const Renderer *mod);

    std::string get_objlist() const;
};


} //namespace controller
} //namespace covise
#endif
