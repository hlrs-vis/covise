/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_NETMODULE_H
#define CTRL_NETMODULE_H

#include <net/covise_connect.h>
#include <covise/covise_msg.h>
#include "control_process.h"

#include "control_define.h"
#include "control_def.h"
#include "control_list.h"
#include "control_port.h"
#include "control_modlist.h"
#include "control_object.h"
#include "control_module.h"
#include <string>

//#ifndef __sgi
//#include <sys/times.h>
//#endif

namespace covise
{

class netlink;
class netlink_list;
class net_module;
class net_module_list;
class display;
class displaylist;
class render_module;
class interface_list;
class net_param_list;
class Controller;
enum class ExecFlag : int;

enum
{
    NOT_MIRR = 0,
    ORG_MIRR,
    CPY_MIRR
};
enum
{
    NET_MOD = 0,
    REND_MOD
};

/************************************************************************/
/* 									                                          */
/* 			NET_PARAM_LIST					                                 */
/* 									                                          */
/************************************************************************/

/**
 *  net_param_list class: list of ojects containing infos about
 *  parameters from the net(map) modules
 *
 */
class net_param_list : public Liste<net_parameter>
{
public:
    /// initialization of the list
    net_param_list();
    /**
       *  retrieving the parameter with the specified name
       *  @param   str   the string containing the name of parameter
       */
    net_parameter *get(const string &str);
};

/************************************************************************/
/* 									                                          */
/* 			NET_CONTROL					                                    */
/* 									                                          */
/************************************************************************/

/**
 *  net_control class: contains a reference to the object containing
 *  infos about a module from the net (map)
 */
class net_control
{
    /// reference to the module infos
    net_module *link;

public:
    /// create an empty reference
    net_control();

    /**
       *  linking the internal reference to the module infos
       */
    void set(net_module *mod);

    /**
       *  getting the reference to the module infos
       *  @return   the reference to the module infos
       */
    net_module *get();
};

/************************************************************************/
/* 									                                          */
/* 			NET_CONTROL_LIST				                                 */
/* 									                                          */
/************************************************************************/

/**
 *  net_control class: list of net_control objects holding informations
 *  about the modules from the net
 */
class net_control_list : public Liste<net_control>
{
public:
    /// creating an empty list
    net_control_list();

    /**
       * retrieving the reference to the net_control object stored in the list
       * @param    a reference to the module object whose net_control reference
       *           is searched for
       * @return   the reference to the net_control object
       */
    net_control *get(net_module *mod);
};

/************************************************************************/
/* 									                                          */
/* 			NETLINK						                                    */
/* 									                                          */
/************************************************************************/
class netlink
{
    string name;
    string instanz;
    string host;
    net_module *mptr;
    /* evtl weitere Eintraege bzgl. synchro. */

public:
    netlink();
    string get_name()
    {
        return name;
    };
    string get_instanz()
    {
        return instanz;
    };
    string get_host()
    {
        return host;
    };
    void set_name(const string &str);
    void set_instanz(const string &str);
    void set_host(const string &str);
    void set_mod(net_module *ptr)
    {
        mptr = ptr;
    };
    net_module *get_mod()
    {
        return mptr;
    };
    void del_link(const string &lname, const string &lnr, const string &lhost);
    /* evtl. unnoetig */
    void send_message(Message *msg, const string &loc_nr);
};

/************************************************************************/
/* 									                                          */
/* 			NETLINK_LIST					                                 */
/* 									                                          */
/************************************************************************/

class netlink_list : public Liste<netlink>
{
    net_module *org;

public:
    netlink_list();
    netlink *get(const string &lname, const string &lnr, const string &lhost);
    void set_org(net_module *optr)
    {
        org = optr;
    };
    net_module *get_org()
    {
        return org;
    };
};

/************************************************************************/
/* 									                                          */
/* 			MIRRORED_MODULES				                                 */
/* 									                                          */
/************************************************************************/

/**
 *  class to use Liste template
 */
class Mirrored_Modules
{
private:
    net_module *_mirrored_module;

public:
    void set_module(net_module *mod)
    {
        _mirrored_module = mod;
    };
    net_module *get_module() const
    {
        return _mirrored_module;
    }
    Mirrored_Modules()
    {
        _mirrored_module = NULL;
    };
    Mirrored_Modules(net_module *mod)
    {
        _mirrored_module = mod;
    };
};

/************************************************************************/
/* 									                                          */
/* 			NET_MODULE				                                    	*/
/* 									                                          */
/************************************************************************/

#define ERROR_OVF 25

/**
 *  net_module class: contains infos about a started module
 */

class net_module
{
    friend class render_module;
    friend class kollab_module;
    friend class net_module_list;

    /// pointer to the datamanager
    AppModule *datam;

    /// module number
    string nr;

    /// name of the module
    string name;

    /// name of the host of the module
    string host;

    int nodeid;

    /// title of the module
    string title;

    /// status of execution
    int status;

    /// number of startmessages sent to module
    int numrunning;

    /// this flag is set if the module should be started
    int startflag;

    /// Position of the Module in Mapeditor
    int xkoord, ykoord;

    /// reference to the object from the ModuleList with
    /// of which type this module is
    module *typ;

    /// reference to the list of interfaces of the module
    interface_list *interfaces;

    /// reference to the list of input parameters
    net_param_list *par_in;

    /// reference to the list of output parameters
    net_param_list *par_out;

    /// reference to the list of modules which are connected to the
    /// output ports of this module
    net_control_list *next_control;

    /// reference to the list of modules which are connected to the
    /// input ports of this module
    net_control_list *prev_control;

    /// reference to the object holding the informations about
    /// the connection to the process of this module
    AppModule *applmod;

    /// reference to the list holding infos about the links of the
    /// module
    netlink_list *netlinks;

    //int test;
    bool mark;

    /// number of errors sent by module
    int m_errors;

    /// status of the process associated to the module
    /// 0 - process not running anymore (crashed or in process of deletion)
    int m_alive;

    /// status of the module
    /// NOT_MIRR = 0-not mirrored,ORG_MIRR = 1-original,CPY_MIRR = 2-copy;
    int m_mirror;

    /// list to the mirrored versions of this module
    Liste<Mirrored_Modules> *m_mirror_nodes;

public:
    /// member functions which are different in render_module

    /// creating an empty object
    net_module();

    /// destructor
    virtual ~net_module();

    /**
       *  getting the type of module
       *  @return  REND_MOD for render module
       *           NET_MOD for other types of module
       */
    virtual int is_renderer()
    {
        return (NET_MOD);
    };

    /**
       *  setting the link to the datamanager
       */
    void set_dm(AppModule *dm)
    {
        datam = dm;
    };

    /**
       *  getting the communication link to the datamanager
       *  @return   the oject to the datamanager
       */
    AppModule *get_dm()
    {
        return datam;
    };

    /**
       *  getting the communication link to the application process
       *  @return   the communication link to the application process
       */
    AppModule *get_applmod()
    {
        return applmod;
    };

    /// member functions which are the same in net_module and render_module

    /**
       *  setting the mirror status of the module
       *  @param  status  ORG_MIRR - module used as original for mirroring
       *                  CPY_MIRR - the module is a mirror of another module
       *                  NOT_MIRR - the module is not mirrored
       */
    void set_mirror_status(int status)
    {
        m_mirror = status;
    };
    /**
       *  getting the mirror status of the module
       *  @return   mirror status of the module
       */
    int get_mirror_status()
    {
        return m_mirror;
    };

    /**
       *  setting the reference to the mirrored node of the current module
       *  @param   node  reference to the mirrored node
       */
    void set_mirror_node(net_module *node)
    {
        if (node != NULL)
        {
            Mirrored_Modules *mirr_mod = new Mirrored_Modules(node);
            m_mirror_nodes->add(mirr_mod);
        }
        else
        {
            m_mirror_nodes->empty_list();
        }
    };

    // get number of mirrors
    int get_num_mirrors()
    {
        return m_mirror_nodes->get_nbList();
    };

    /**
       *  resetting the list to the mirrored nodes
       */
    void reset_mirror_list()
    {
        m_mirror_nodes->reset();
    }

    /**
       *  getting the next module in the list to the mirrored nodes
       *  @return    the reference to the next module in the list
       */
    net_module *mirror_list_next()
    {
        Mirrored_Modules *mirr_mod = m_mirror_nodes->next();
        if (mirr_mod == NULL)
        {
            return NULL;
        }
        else
        {
            return mirr_mod->get_module();
        }
    }

    void set_nodeid(int id);
    int get_nodeid()
    {
        return nodeid;
    };

    /// list of errors sent by module
    vector<string> m_errlist;

    /**
       *  setting the status of the corresponding process
       *  @param  al  0-process terminated  1-process running
       */
    void set_alive(int al)
    {
        m_alive = al;
    };

    /**
       *  getting the status of the corresponding process
       *  @return  0-process terminated  1-process running
       */
    int is_alive()
    {
        return m_alive;
    };

    /**
       *  getting the number of the errors above the limit
       *  @return  the number of the errors above the limit ERROR_OVF
       */
    int error_owf()
    {
        if ((++m_errors) > ERROR_OVF)
            return (m_errors - ERROR_OVF);
        else
            return 0;
    };

    /**
       *  adding an error message to the error message list
       *  @param   the new error message
       */
    void add_error(Message *msg);
    ;

    /**
       *  emptying the error message list
       */
    void empty_errlist(void)
    {
        m_errlist.clear();
    };

    /**
       *  getting the number of interfaces
       *  @return    the number of interfaces
       */
    int get_nbInterf();

    /**
       *  getting the number of input parameters
       *  @return    the number of input parameters
       */
    int get_nbParIn()
    {
        return par_in->get_nbList();
    };

    /**
       *  getting the number of output parameters
       *  @return    the number of output parameters
       */
    int get_nbParOut()
    {
        return par_out->get_nbList();
    };

    /**
       *  getting the name, type and description of input parameters
       *  @return    the number of input parameters
       */
    int get_inpars_values(vector<string> *l, vector<string> *t, vector<string> *v, vector<string> *p);

    /**
       *  setting the instance of the module
       *  @param  i  the instance of the parameter
       */
    void set_nr(const string &i);

    /**
       *  getting the instance of the module
       *  @param  i  the instance of the parameter
       */
    string get_nr()
    {
        return nr;
    };

    /**
       *  setting the name of the module
       *  @param  str  the name of the module
       */
    void set_name(const string &str);

    /**
       *  getting the name of the module
       *  @return  the name of the module
       */
    string get_name()
    {
        return name;
    };

    /**
       *  setting the name of the host on which the module
       *             has been started
       *  @param  str     the name of the host
       */
    void set_host(const string &str);

    /**
       *  getting the name of the host on which the module
       *             has been started
       *  @return  the name of the host
       */
    string get_host()
    {
        return host;
    };

    /**
       *  setting the title of the the module
       *
       *  @param  str     title of the module
       */
    void set_title(const string &str);

    /**
       *  setting the standard title modulname_instance the module
       *
       *  @param  str     title of the module
       */
    void set_standard_title()
    {
        set_title(name + "_" + nr);
    };

    /**
       *  getting the title of the module
       *  @return  the title of the module
       */
    string get_title()
    {
        return title;
    };

    /**
       *  getting the status of the module
       *  @return    1 if the module has been started
       */
    int get_start_flag()
    {
        return startflag;
    };

    /**
       *  setting the status of the module to "started" (1)
       */
    void set_start_flag()
    {
        startflag = 1;
    };

    /**
       *  resetting the status of the module to "not started" (0)
       */
    void reset_start_flag()
    {
        startflag = 0;
    };

    /**
       *  increment the number of requests for starting the module
       */
    void inc_running()
    {
        numrunning++;
    };

    /**
       *  decrement the number of requests for starting the module
       */
    void dec_running()
    {
        numrunning--;
    };

    /**
       *  reset the number of requests for starting the module to 0
       */
    void reset_running()
    {
        numrunning = 0;
    };

    /**
       *  get the number of requests for starting the module
       *  @return   the number of requests
       */
    int get_num_running()
    {
        return numrunning;
    };

    /**
       *  checking if a module above the current module has to be started
       *  @param  ul      list of user interfaces
       */
    bool is_one_waiting_above(ui_list *ul);

    /**
       *  checking if a module under the current module is already running
       */
    bool is_one_running_under(void);

    /**
       *  checking if a module above the current module is already running
       */
    bool is_one_running_above(int first);

    /**
       *  set start flag on the connected modules
       */
    void set_start();

    /**
       *  start the connected modules
       */
    void start_modules(ui_list *ul);

    /**
       *  start the current module if there are no conflicting conditions
       */
    void exec_module(ui_list *uilist);

    /**
       *  send a message with an object to be displayed by the renderer
       *       - NOT IMPLEMMENTED IN THIS CLASS
       */
    virtual void send_add(ui_list *, object *, void *){};

    /**
       *  send a message to delete an object displayed by the renderer
       *       - NOT IMPLEMMENTED IN THIS CLASS
       */
    virtual void send_del(const string &){};

    /**
       *   send a message to delete an object displayed by the renderer
       *       - NOT IMPLEMMENTED IN THIS CLASS
       */
    virtual void send_add_obj(const string &, void *){};

    /**
       *   set the status of the module
       *   @param str  the status of the module: MODULE_RUNNING or MODULE_IDLE
       */
    void set_status(int str)
    {
        status = str;
    };

    /**
       *  get the status of the module
       *  @return    status of the module: MODULE_RUNNING or MODULE_IDLE
       */
    int get_status()
    {
        return status;
    };

    /**
       *  check if there are no modules connected on the module input ports
       *  @return   true or false
       */
    bool is_on_top();

    /**
       *  set the informations about the link to another module
       *  @param   l_mod    the module to link to
       */
    void set_netlink(net_module *l_mod);

    /**
       *  removing the link to a module from the link list
       *  @param   m_name   name of the module
       *  @param   m_nr     number of the module
       *  @param   m_host   host of the module
       */
    void del_netlink(const string &m_name, const string &m_nr, const string &m_host);

    /**
       *  removing the link to a module from the link list
       *  @param   m_name   name of the module
       *  @param   m_nr     number of the module
       *  @param   m_host   host of the module
       */
    void r_del_netlink(const string &m_name, const string &m_nr, const string &m_host);

    /**
       *  getting the number of objects created for the port
       *  @param   intfname   name of the port
       *  @return    the number of objects
       */
    int get_count(const string &intfname);

    /**
       *  getting the number of objects created for the port
       *  @param   intfname   name of the port
       *  @return    the number of objects or 1 if the module is a copy
       */
    int test_org_count(const string &intfname);

    /**
       *  test if the module is original
       *  @return    true if it's original false if not
       */
    bool test_copy();

    /**
       *  create the netlink to a module
       *  @param   name     the name of the module
       *  @param   instanz  the nr of the module
       *  @param   host     the hostname of the module
       *  @param   netl     the list of net modules
       */
    void create_netlink(const string &name, const string &instanz, const string &host, net_module_list *netl);

    /**
       *  move the module to a new position in the mapeditor
       *  @param   posx    new x coordinate
       *  @param   posy    new y coordinate
       */
    void move(int posx, int posy)
    {
        xkoord = posx;
        ykoord = posy;
    };

    /**
       *  get the module position
       */
    int get_x_pos()
    {
        return xkoord;
    };
    int get_y_pos()
    {
        return ykoord;
    };
    /**
       *  set a parameter for the module
       *  @param   para    the object containing the infos about the parameter
       *  @param   dir     type of the parameter (input or output)
       */
    void set_parameter(parameter *para, const string &dir);

    /**
       *  getting the list of interfaces
       *  @return    the reference to the list
       */
    interface_list *get_interfacelist();

    /**
       *  setting the reference to the type of the module
       *  @param  tmpname    name of the module
       *  @param  tmphost    name of the host
       *  @param  mod_list   list of module types
       */
    void link_type(const string &tmpname, const string &tmphost, modulelist *mod_list);

    /**
       *  getting the reference to the type of the module
       *  @return   the reference to the type of the module
       */
    module *get_type();

    /**
       *  setting a net_control connection to a module
       *  @param    mod         connected module
       *  @param    direction   direction of the connection "to" or "from" module
       */
    void set_C_conn(net_module *mod, const string &direction);

    /**
       *  deleting a net_control connection to a module
       *  @param    mod         connected module
       *  @param    direction   direction of the connection "to" or "from" module
       */
    void del_C_conn(net_module *mod, const string &direction);

    void set_P_conn(const string &output_par, net_module *mod, const string &input_par, const string &direction);
    void del_P_conn(const string &output_par, net_module *mod, const string &input_par, const string &direction);

    void change_param(const string &param_name, const string &param_value, int i, int count);
    void change_param(const string &param_name, const string &param_value_list);
    string get_one_param(const string &param_name);

    void add_param(const string &param_name, const string &param);

    void set_outputtype(const string &intf_name, const string &DO_name, const string &DO_type);

    void mark_for_reset()
    {
        mark = true;
    };
    int get_mark()
    {
        return mark;
    };

    string get_inparaobj();
    string get_outparaobj();

    void send_finish();

    string get_module(const string &local_name, const string &local_user, bool forSaving = false);
    string get_moduleinfo(const string &local_name, const string &local_user);
    string get_interfaces(const string &direction);
    string get_parameter(const string &direction, bool forSaving = false);
    void writeParamScript(ofstream &of, const string &direction);

    void writeScript(ofstream &of, const string &local_user);

#ifdef PARAM_CONN
    string get_para_connect(int *i, const string &local_name, const string &local_user);
#endif

    // different in net_module and render_module
    virtual int init(int id, const string &name, const string &instanz, const string &host,
                     int posx, int posy, int copy, ExecFlag flags, net_module *mirror_node = NULL);
    virtual void del(int already_dead);

    virtual void set_interface(const string &strn, const string &strt, const string &strtx, const string &strd, const string &strde);
    virtual void set_O_conn(const string &output_name, object *obj);
    virtual void del_O_conn(const string &output_name, object *obj);
    virtual int check_O_conn(const string &interfname);
    virtual string get_intf_type(const string &output_name);
    virtual void set_intf_demand(const string &intf_name, const string &new_type);

    virtual string get_startmessage(ui_list *ul);
    virtual void new_obj_names();
    virtual bool delete_old_objs();
    virtual void delete_rez_objs();
    virtual void delete_dep_objs();
    virtual void delete_all_objs(int already_dead);

    virtual void start_module(ui_list *ul);
    virtual bool is_connected();

    virtual void set_to_OLD();
    virtual void set_to_NEW();

    //virtual void reset_for_exec(DM_list* DMlist);

    virtual void set_to_NEW(const string &intf_name);
    virtual void set_DO_status(int mode, const string &DO_name);

    virtual void send_msg(Message *msg);
    virtual int get_mod_id();
};

/************************************************************************/
/* 									                                          */
/* 			DISPLAY						                                    */
/* 									                                          */
/************************************************************************/

class display
{
    string excovise_status;
    bool DISPLAY_READY;
    bool NEXT_DEL;
    string hostname;
    string userid;
    string passwd;
    string DO_name;
    int m_helper;

public:
    AppModule *applmod;
    display();
    virtual ~display();

    void set_helper(int hlp)
    {
        m_helper = hlp;
    };
    int is_helper(void)
    {
        return m_helper;
    };

    void set_hostname(const string &str);
    void set_userid(const string &str);
    void set_passwd(const string &str);

    string get_hostname()
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

    void set_execstat(const string &str);
    string get_execstat()
    {
        return excovise_status;
    };

    void set_DISPLAY(bool bvar);
    bool get_DISPLAY();
    bool get_NEXT_DEL();

    AppModule *get_mod();
    int get_mod_id();

    int start(AppModule *dmod, const string &info_str, module *mod, ExecFlag flags, const std::vector<std::string> &params = std::vector<std::string>{});
    void quit();

    void send_add(const string &DO_name);
    void send_add();
    void send_del(const string &DO_name, const string &DO_new_name);

    void send_status(const string &info_str);
    void send_message(Message *msg);
};

/************************************************************************/
/* 									                                          */
/* 			DISPLAYLIST					                                    */
/* 									                                          */
/************************************************************************/

class displaylist : public Liste<display>
{
    int count, ready;

public:
    displaylist();
    virtual ~displaylist();

    display *get(int sender);
    display *get(const string &hostname, const string &user);

    int init(const string &excovise_name, const string &info_str, module *mod, int copy, ExecFlag flags, rhost *host);
    void quit();
    void set_DISPLAY_FALSE();

    void send_add(const string &DO_name);
    void send_del(const string &DO_name, const string &DO_new_name);

    void decr_ready()
    {
        ready--;
    };
    void incr_ready();
    void decr_count();
    void incr_count()
    {
        count++;
    };
    int get_ready();
    int get_count();
    void reset_ready();

    bool contains(const std::string &hostname, const std::string &user);

    // add a CRB for a helper, do nothing if already existent, return on fail
    int addHelperCRB(const string &helperHost, const string &host);
};

/************************************************************************/
/* 									                                          */
/* 			RENDER_MODULE					                                 */
/* 									                                          */
/************************************************************************/

class render_module : public net_module
{
    bool DISPLAY_ALL;
    displaylist *displays;
    int m_multisite;

public:
    render_module();
    virtual ~render_module();

    display *get_display(int id);

    void set_multisite(int mst)
    {
        m_multisite = mst;
    };
    int is_multisite(void)
    {
        return m_multisite;
    };

    displaylist *get_displays(void)
    {
        return displays;
    };

    int add_helper(const string &h_host, const string &info_str, const std::vector<std::string> &params = std::vector<std::string>{});

    int init(int id, const string &name, const string &instanz, const string &host,
             int posx, int posy, int copy, ExecFlag flags, net_module *mirror_node = NULL);
    void set_interface(const string &strn, const string &strt, const string &strtx, const string &strd, const string &strde);
    void del(int already_dead);

    void set_O_conn(const string &output_name, object *obj);
    void del_O_conn(const string &output_name, object *obj);

    int is_renderer()
    {
        return (REND_MOD);
    };
    void send_del(const string &);
    void send_add_obj(const string &, void *);

    void start_module(ui_list *ul);
    void send_add(ui_list *ul, object *obj, void *connection);
    int get_mod_id();

    int update(int sender, ui_list *uilist);

    void set_renderstatus(ui_list *uilist);
    void send_msg(Message *msg);

    //void reset_for_exec(DM_list* DMlist);
    void count_init();
    void reset_wait();

    /** Look if this module(I) and the sender module(II) are mirrors of each other
       * @param sender      id of module II
       * @return  true   modules are mirrors
       */
    bool is_mirror_of(int sender);

    bool test_id(int sender);
    bool add_display(userinterface *ui);
    void remove_display(rhost *host);
    void remove_display(int id);
    void remove_display(display *disp);
    int get_count();

    // add a CRB for a helper, do nothing if already existent, return on fail
    int addHelperCRB(const string &helperHost, const string &host);
};

/************************************************************************/
/* 									                                          */
/* 			DOBJECT						                                    */
/* 									                                          */
/************************************************************************/
class dobject
{
    string name;
    int mark;

public:
    dobject();
    void set_name(const string &str);
    string get_name()
    {
        return name;
    };
    void set_mark(int im);
    int get_mark()
    {
        return mark;
    };
    void clear_mark();
};

/************************************************************************/
/* 									                                          */
/* 			DO_LIST						                                    */
/* 									                                          */
/************************************************************************/

class do_list : public Liste<dobject>
{
public:
    do_list();
    dobject *get(const string &str);
};

/************************************************************************/
/* 									                                          */
/* 			NET_MODULE_LIST					                              */
/* 									                                          */
/************************************************************************/

class net_module_list : public Liste<net_module>
{

public:
    net_module_list();
    net_module *get(const string &from_name, const string &from_nr, const string &from_host);
    net_module *get(int nodeid);
    net_module *get(const string &name, const string &nr);
    int getID(const string &name, const string &nr, const string &host);

    net_module *get_mod(int sender);

    void re_move(net_module *mod, int already_dead);
    void re_move(int nodeid, int already_dead);
    void re_move(const string &from_name, const string &from_nr, const string &from_host, int already_dead);
    void set_DO_conn(const string &from_name, const string &from_nr, const string &from_host, const string &output_name,
                     object *obj);
    void del_DO_conn(const string &from_name, const string &from_nr, const string &from_host, const string &output_name,
                     object *obj);
    void *set_DI_conn(const string &from_name, const string &from_nr, const string &from_host, const string &output_name,
                      object *obj);
    void del_DI_conn(const string &from_name, const string &from_nr, const string &from_host, const string &output_name,
                     object *obj);
    void set_C_conn(const string &from_name, const string &from_nr, const string &from_host,
                    const string &to_name, const string &to_nr, const string &to_host);
    void del_C_conn(const string &from_name, const string &from_nr, const string &from_host,
                    const string &to_name, const string &to_nr, const string &to_host);

    void set_P_conn(const string &from_name, const string &from_nr, const string &from_host, const string &output_par,
                    const string &to_name, const string &to_nr, const string &to_host, const string &input_par);
    void del_P_conn(const string &from_name, const string &from_nr, const string &from_host, const string &output_par,
                    const string &to_name, const string &to_nr, const string &to_host, const string &input_par);

    void change_param(const string &name, const string &nr, const string &host, const string &param_name, const string &value, int i);
    void change_param(const string &name, const string &nr, const string &host, const string &param_name, const string &value_list);

    void add_param(const string &name, const string &nr, const string &host, const string &param_name, const string &add_param);

    void set_to_finish(ui_list *uilist);

    void save_config(const string &filename);
    bool load_config(const string &filename);
    char *openNetFile(const string &filename);
    string get_modules(const string &local_name, const string &local_user, bool forSaving = false);

#ifdef PARAM_CONN
    string get_para_connect(const string &local_name, const string &local_user);
#endif
    void move(int id, int posx, int posy);
    bool mirror(net_module *from_module, const string &new_host);
    void mirror_all(const string &new_host);

    /*******************************************/
    /* methods with special renderer behaviour */
    /*******************************************/

    //void reset_for_exec(DM_list* DMlist);
    int init(int id, const string &name, const string &instanz, const string &host,
             int posx, int posy, int copy, ExecFlag flags, net_module *from = NULL);

    /****************************/
    /* special renderer-methods */
    /****************************/
    void set_renderstatus(ui_list *uilist);
    void send_renderer(Message *msg);
    void send_all_renderer(Message *msg); // send msg to every renderer

    /// send generic information( INFO, WARNING, ERROR) to all renderers
    void send_gen_info_renderer(Message *msg);
    int update(int sender, ui_list *uilist);

    void reset_wait(); /* fuer alle renderer */
    int check_connected();
};
}
#endif
