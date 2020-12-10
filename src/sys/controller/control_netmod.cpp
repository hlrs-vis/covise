/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <sys/stat.h>

#include <covise/covise.h>
#include <config/CoviseConfig.h>
#include <util/coTimer.h>
#include <util/covise_version.h>
#include <net/covise_connect.h>
#include <covise/covise_msg.h>
#include <util/coFileUtil.h>

#include "CTRLGlobal.h"
#include "CTRLHandler.h"
#include "control_process.h"
#include "control_def.h"
#include "control_define.h"
#include "control_list.h"
#include "control_port.h"
#include "control_object.h"
#include "control_modlist.h"
#include "control_module.h"
#include "control_netmod.h"
#include "control_coviseconfig.h"

using namespace covise;

extern FILE *msg_prot;
extern string prot_file;

covise::net_interface *unconnectedPort;
covise::net_module *unconnectedModule;

//!
//! send error-message to master userinterface
//!
void send_wmessage(ui_list *uilst, const string &txt)
{
    string data = "Controller\n \n \n" + txt;
    Message *msg = new Message(COVISE_MESSAGE_WARNING, data);
    userinterface *tmp_ui = uilst->get_master();
    tmp_ui->send(msg);
    delete msg;
}

//**********************************************************************
//
// 			NET_PARAM_LIST
//
//**********************************************************************

net_param_list::net_param_list()
    : Liste<net_parameter>()
{
}

net_parameter *net_param_list::get(const string &str)
{
    string tmp_name;
    net_parameter *tmp;

    this->reset();
    do
    {
        tmp = this->next();
        if (!tmp)
            break;

        tmp_name = tmp->get_name();
    } while (tmp_name != str);

    return tmp;
}

//**********************************************************************
//
//       NET_CONTROL
//
//**********************************************************************

net_control::net_control()
{
    link = NULL;
}

void net_control::set(net_module *mod)
{
    link = mod;
}

net_module *net_control::get()
{
    return link;
}

//**********************************************************************
//
//          NET_CONTROL_LIST
//
//**********************************************************************

net_control_list::net_control_list()
    : Liste<net_control>()
{
}

net_control *net_control_list::get(net_module *mod)
{
    string org_name = mod->get_name();
    string org_nr = mod->get_nr();
    string org_host = mod->get_host();

    net_control *tmp;
    string tmp_mod, tmp_nr, tmp_name, tmp_host;
    this->reset();
    do
    {
        tmp = this->next();
        if (!tmp)
            break;
        net_module *tmp_mod = tmp->get();
        tmp_name = tmp_mod->get_name();
        tmp_nr = tmp_mod->get_nr();
        tmp_host = tmp_mod->get_host();
    } while (org_name != tmp_name || org_nr != tmp_nr || org_host != tmp_host);

    return tmp;
}

//**********************************************************************
//
// 			NETLINK
//
//**********************************************************************

netlink::netlink()
{
    mptr = NULL;
}

void netlink::set_name(const string &str)
{
    name = str;
}

void netlink::set_instanz(const string &str)
{
    instanz = str;
}

void netlink::set_host(const string &str)
{
    host = str;
}

void netlink::del_link(const string &lname, const string &lnr, const string &lhost)
{
    mptr->r_del_netlink(lname, lnr, lhost);
}

//**********************************************************************
//
//          NETLINK_LIST
//
//**********************************************************************

netlink_list::netlink_list()
    : Liste<netlink>()
{
    org = NULL;
}

netlink *netlink_list::get(const string &lname, const string &lnr, const string &lhost)
{
    string tmp_nr, tmp_name, tmp_host;
    netlink *tmp;

    this->reset();
    do
    {
        tmp = this->next();
        if (tmp)
        {
            char buf[256];
            sprintf(buf, "netlink_list::get did not find %s::%s_%s", lhost.c_str(), lname.c_str(), lnr.c_str());
            print_comment(__LINE__, __FILE__, buf, 1);
            break;
        }
        tmp_nr = tmp->get_instanz();
        tmp_name = tmp->get_name();
        tmp_host = tmp->get_host();
    } while (tmp_nr != lnr || tmp_name != lname || tmp_host != lhost);

    return tmp;
}

//**********************************************************************
//
//          NET_MODULE
//
//**********************************************************************

net_module::net_module()
{
    status = MODULE_IDLE;
    typ = NULL;
    nodeid = -1;
    interfaces = new interface_list;
    par_in = new net_param_list;
    par_out = new net_param_list;
    next_control = new net_control_list;
    prev_control = new net_control_list;
    netlinks = new netlink_list;
    applmod = NULL;
    datam = NULL;
    mark = false;
    m_errors = 0;
    reset_start_flag();
    numrunning = 0;
    m_alive = 1;
    m_mirror = 0;
    m_mirror_nodes = new Liste<Mirrored_Modules>(1);
}

net_module::~net_module()
{

    if (typ)
    {
        typ = NULL;
    }
    if (interfaces)
    {
        delete interfaces;
    }
    if (par_in)
    {
        delete par_in;
    }
    if (par_out)
    {
        delete par_out;
    }
    if (next_control)
    {
        delete next_control;
    }
    if (prev_control)
    {
        delete prev_control;
    }
    if (netlinks)
    {
        delete netlinks;
    }
    if (applmod)
    {
        delete applmod;
    }
}

void net_module::add_error(Message *msg)
{
    string p_str = "Error log: " + string(msg->data.data());
    m_errlist.push_back(p_str);
}

int net_module::get_nbInterf()
{
    return interfaces->get_nbList();
}

int net_module::get_inpars_values(vector<string> *al, vector<string> *at, vector<string> *av, vector<string> *ap)
{

    int count = par_in->get_nbList();
    if (count == 0)
        return 0;

    par_in->reset();
    net_parameter *tmp_para;
    while ((tmp_para = par_in->next()) != NULL)
    {
        al->push_back(tmp_para->get_name());
        at->push_back(tmp_para->get_type());
        av->push_back(tmp_para->get_val_list());
        ap->push_back(tmp_para->get_addvalue());
    }
    return count;
}

void net_module::set_nodeid(int id)
{
    nodeid = id;
}

void net_module::set_nr(const string &str)
{
    nr = str;
}

void net_module::set_name(const string &str)
{
    name = str;
}

void net_module::set_host(const string &str)
{
    host = str;
}

void net_module::set_title(const string &str)
{
    if (str.empty())
        return;

    title = str;

    ostringstream os;
    os << "MODULE_TITLE\n" << name << "\n" << nr << "\n" << host << "\nSetModuleTitle\nString\n1\n" << title;
    Message *tmpmsg = new Message(COVISE_MESSAGE_UI, os.str());
    this->send_msg(tmpmsg);
    delete tmpmsg;
}

void net_module::set_netlink(net_module *l_mod)
{
    netlink *tmp_link = new netlink;
    tmp_link->set_name(l_mod->get_name());
    tmp_link->set_instanz(l_mod->get_nr());
    tmp_link->set_host(l_mod->get_host());
    tmp_link->set_mod(l_mod);

    netlinks->add(tmp_link);
}

void net_module::del_netlink(const string &m_name, const string &m_nr, const string &m_host)
{
    netlink *tmp_link = netlinks->get(m_name, m_nr, m_host);
    tmp_link->del_link(m_name, m_nr, m_host);
    netlinks->remove(tmp_link);
}

void net_module::new_obj_names()
{
    net_interface *intf;
    interfaces->reset();
    while ((intf = (net_interface *)interfaces->next()) != NULL)
    {
        if (intf->get_direction() == "output")
        {
            if (intf->get_conn_state() == true)
                (intf->get_object())->new_DO();
        }
    }
}

void net_module::set_start()
{
    net_interface *intf;
    interfaces->reset();
    while ((intf = (net_interface *)interfaces->next()) != NULL)
    {
        if (intf->get_direction() == "output")
            (intf->get_object())->set_start_module();
    }
}

void net_module::start_modules(ui_list *ul)
{
    net_interface *intf;
    net_interface *intfs[1000];
    int i = 0;
    interfaces->reset();
    while ((intfs[i] = (net_interface *)interfaces->next()) != NULL)
        i++;
    i = 0;
    while ((intf = intfs[i]) != NULL)
    {
        if (intf->get_direction() == "output")
            (intf->get_object())->start_modules(ul);

        i++;
    }
}

void net_module::exec_module(ui_list *uilist)
{

    if (is_renderer())
    {
        Message *err_msg = new Message(COVISE_MESSAGE_COVISE_ERROR, "Sorry, can't execute a Renderer!\n");
        uilist->send_all(err_msg);
        delete err_msg;
        return;
    }

    if ((CTRLHandler::instance()->m_numRunning == 0) || (!is_one_running_above(1)))
    {
        if (get_status() != MODULE_RUNNING)
        {
            //delete_all Objects if not saved    // moved to start_module()
            //mod->delete_old_objs();
            //give new Names to Output_objects
            //mod->new_obj_names();
            start_module(uilist);
            CTRLHandler::instance()->m_numRunning++;
            if (CTRLHandler::instance()->m_numRunning == 1) // switch to execution mode
            {
                Message *ex_msg = new Message(COVISE_MESSAGE_UI, "INEXEC");
                uilist->send_all(ex_msg);
                CTRLGlobal::getInstance()->netList->send_all_renderer(ex_msg);
                delete ex_msg;
            }

            if (m_mirror == ORG_MIRR)
            {
                Mirrored_Modules *mirrors;
                m_mirror_nodes->reset();
                while ((mirrors = m_mirror_nodes->next()) != NULL)
                {
                    (mirrors->get_module())->start_module(uilist);
                }
            }
        }

        else
        {
            inc_running();
            Message *err_msg = new Message(COVISE_MESSAGE_WARNING, "Controller\n \n \n Sorry: module is already running !");
            uilist->send_all(err_msg);
            delete err_msg;
            return;
        }
    }

    else
    {
        Message *err_msg = new Message(COVISE_MESSAGE_WARNING, "Controller\n \n \n Sorry: Is already one executing above!\n");
        uilist->send_all(err_msg);
        delete err_msg;
        return;
    }
}

bool net_module::is_one_waiting_above(ui_list *ul) // one level up
{
    bool found = false;

    interfaces->reset();
    net_interface *intf;
    while ((intf = (net_interface *)interfaces->next()) != NULL)
    {
        if (intf->get_direction() == "input")
        {
            // ist das Interface verbunden ?
            if (intf->get_conn_state() == true)
            {
                obj_from_conn *fr = (intf->get_object())->get_from();
                net_module *mod = fr->get_mod();
                if (mod->get_start_flag())
                {
                    found = true;
                    mod->reset_start_flag();
                    if (!mod->is_one_running_above(1))
                    {
                        CTRLHandler::instance()->m_numRunning++;
                        mod->start_module(ul);
                    }
                }
            }
        }
    }
    return found;
}

bool net_module::is_one_running_under(void)
{
    if (interfaces == NULL)
        return false;

    net_interface *intf;
    interfaces->reset();
    while ((intf = (net_interface *)interfaces->next()) != NULL)
    {
        if (intf->get_direction() == "output")
        {
            if (intf->get_object())
            {
                obj_conn_list *to = (intf->get_object())->get_to();
                to->reset();
                obj_conn *tmp_conn;
                while ((tmp_conn = to->next()) != NULL)
                {
                    net_module *mod = tmp_conn->get_mod();
                    if (mod && (!mod->is_renderer())
                        && (mod->get_status() == MODULE_RUNNING))
                    {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

bool net_module::is_one_running_above(int first)
{
    net_interface *intfs[1000];
    net_interface *intf;
    int i = 0;

    interfaces->reset();
    while ((intfs[i] = (net_interface *)interfaces->next()) != NULL)
        i++;

    i = 0;
    while ((intf = intfs[i]) != NULL)
    {
        if (intf->get_direction() == "input")
        {
            // is the Interface connected ?
            if (intf->get_conn_state() == true)
            {
                if ((intf->get_object())->is_one_running_above())
                {
                    return true;
                }
            }
        }
        i++;
    }

    if (first == 0)
    {
        if ((get_start_flag()) || (get_status() == MODULE_RUNNING))
        {
            return true;
        }
    }
    return false;
}

bool net_module::delete_old_objs()
{
    net_interface *intf;
    interfaces->reset();
    while ((intf = (net_interface *)interfaces->next()) != NULL)
    {
        if (intf->get_direction() == "output")
        {
            // is the Interface connected ?
            if (intf->get_conn_state() == true)
            {
                object *p_obj = intf->get_object();
                if ((p_obj != NULL) && (!p_obj->isEmpty()))
                {
                    p_obj->del_old_DO();
                    return true;
                }
            }
        }
    }
    return false;
}

void net_module::delete_rez_objs()
{
    net_interface *intf;
    interfaces->reset();
    while ((intf = (net_interface *)interfaces->next()) != NULL)
    {
        if (intf->get_direction() == "output")
        {
            // is the Interface connected ?
            if (intf->get_conn_state() == true)
            {
                object *p_obj = intf->get_object();
                if (p_obj != NULL)
                {
                    p_obj->del_rez_DO();
                    p_obj->del_dep_data();
                }
            }
        }
    }
}

void net_module::delete_dep_objs()
{
    bool objs_deleted = false;
    net_interface *intf;

    interfaces->reset();
    while ((intf = (net_interface *)interfaces->next()) != NULL)
    {
        if (intf->get_direction() == "output")
        {
            // is the Interface connected ?
            if (intf->get_conn_state() == true)
            {
                object *p_obj = intf->get_object();
                if ((p_obj != NULL) && (!p_obj->isEmpty()))
                {
                    p_obj->del_all_DO(0);
                    objs_deleted = true;
                    p_obj->del_dep_data();
                }
            } // conn state =  true
        } // direction =  output
    }
    // update the mapeditor with
    if (objs_deleted)
        send_finish();
}

void net_module::delete_all_objs(int already_dead)
{
    net_interface *intf;
    interfaces->reset();
    while ((intf = (net_interface *)interfaces->next()) != NULL)
    {
        if (intf->get_direction() == "output")
        {
            // is the Interface connected ?
            if (intf->get_conn_state() == true)
                (intf->get_object())->del_all_DO(already_dead);
        }
    }
}

bool net_module::is_on_top()
{
    net_interface *intf;
    interfaces->reset();
    while ((intf = (net_interface *)interfaces->next()) != NULL)
    {
        if (intf->get_direction() == "input" && intf->get_conn_state())
        {
            return false;
        }
    }
    return true;
}

bool net_module::is_connected()
{
    net_interface *intf;
    interfaces->reset();
    while ((intf = (net_interface *)interfaces->next()) != NULL)
    {
        if (intf->get_direction() == "input")
        {
            if (intf->get_conn_state() == false && intf->get_demand() != "opt")
            {
                unconnectedModule = this;
                unconnectedPort = intf;
                return false;
            }
        }
    }
    return true;
}

void net_module::set_to_OLD()
{
    net_interface *tmp_intf;

    // select all input-Interfaces
    interfaces->reset();
    while ((tmp_intf = (net_interface *)interfaces->next()) != NULL)
    {
        if (tmp_intf->get_direction() == "input")
        {
            // get the object and set the state to OLD
            // ACHTUNG nicht alle interfaces muessen verbunden sein !!
            bool state = tmp_intf->get_conn_state();
            if (state == true)
            {
                object *obj = tmp_intf->get_object();
                // change status of Connection in Object from NEW to OLD
                // if the status is OLD or INIT it stops with an error
                obj->change_NEW_to_OLD(this, tmp_intf->get_name());
            } // true
        } // input
    } // while
}

void net_module::set_to_NEW()
{
    net_interface *tmp_intf;

    // select all output-interfaces
    interfaces->reset();
    while ((tmp_intf = (net_interface *)interfaces->next()) != NULL)
    {
        if (tmp_intf->get_direction() == "output")
        {
            // get the object and set the state to NEW
            bool state = tmp_intf->get_conn_state();
            if (state == true)
            {
                object *obj = tmp_intf->get_object();
                // change status in Object: global and substates
                obj->set_to_NEW();
            } // true
        } // output
    } // while
}

void net_module::set_to_NEW(const string &intf_name)
{
    net_interface *tmp = (net_interface *)interfaces->get(intf_name);
    if (tmp)
    {
        object *obj = tmp->get_object();
        obj->set_to_NEW();
    }
}

void net_module::link_type(const string &tmpname, const string &tmphost, modulelist *mod_list)
{
    typ = mod_list->get(tmpname, tmphost);
}

module *net_module::get_type()
{
    return typ;
}

interface_list *net_module::get_interfacelist()
{
    return interfaces;
}

void net_module::set_interface(const string &strn, const string &strt, const string &strtx, const string &strd, const string &strde)
{
    net_interface *tmp = new net_interface;
    tmp->set_name(strn);
    tmp->set_type(strt);
    tmp->set_text(strtx);
    tmp->set_direction(strd);
    tmp->set_demand(strde);
    interfaces->add(tmp);
}

void net_module::set_parameter(parameter *para, const string &dir)
{
    net_parameter *tmp = new net_parameter;

    string tmps = para->get_name();
    tmp->set_name(tmps);

    tmps = para->get_type();
    tmp->set_type(tmps);

    tmps = para->get_text();
    tmp->set_text(tmps);

#ifdef PARA_START
    tmps = para->get_extension();
    tmp->set_extension(tmps);
#endif

    tmps = para->get_org_val();
    tmp->set_value_list(tmps);

    int count = para->get_count();
    for (int i = 1; i <= count; i++)
        tmp->set_value(i, para->get_value(i));

    if (dir == "in")
        par_in->add(tmp);

    else
        par_out->add(tmp);
}

//----------------------------------------------------------------------
// check_O_conn tests, if a net_module has a connection to a Object obj
// at the interface intf_name
// if this connection exists, it returns a 1,
// if this connection exist not, it returns a 0
//----------------------------------------------------------------------
int net_module::check_O_conn(const string &intf_name)
{
    net_interface *tmp = (net_interface *)interfaces->get(intf_name);
    if (tmp)
    {
        int check = tmp->check_conn();
        return check;
    }
    else
        return 0;
}

void net_module::set_O_conn(const string &from_intf, object *obj)
{
    net_interface *tmp = (net_interface *)interfaces->get(from_intf);
    if (tmp)
        tmp->set_connect(obj);
}

void net_module::del_O_conn(const string &output_name, object *)
{
    net_interface *tmp;

    if ((tmp = (net_interface *)interfaces->get(output_name)) != NULL)
    {
        // get local interface
        // delete connection for obj
        tmp->del_connect();
    }
}

string net_module::get_intf_type(const string &output_name)
{
    net_interface *tmp = (net_interface *)interfaces->get(output_name);

    if (tmp)
        return tmp->get_type();

    else
        return "deletedOrOldVersion";
}

void net_module::set_intf_demand(const string &intf_name, const string &new_type)
{
    net_interface *tmp = (net_interface *)interfaces->get(intf_name);
    if (tmp)
        tmp->set_demand(new_type);
}

void net_module::set_C_conn(net_module *mod, const string &direction)
{
    net_control *tmp = new net_control;
    tmp->set(mod);

    if (direction == "to")
        next_control->add(tmp);

    else
        prev_control->add(tmp);
}

void net_module::del_C_conn(net_module *mod, const string &direction)
{
    if (direction == "to")
    {
        net_control *tmp = next_control->get(mod);
        next_control->remove(tmp);
    }

    else
    {
        net_control *tmp = prev_control->get(mod);
        prev_control->remove(tmp);
    }
}

void net_module::set_P_conn(const string &from_para, net_module *mod, const string &to_para, const string &dir)
{
    net_parameter *para;

    if (dir == "in")
        para = par_in->get(from_para);

    else
        para = par_out->get(from_para);

    para->set_P_conn(mod, to_para);
}

void net_module::del_P_conn(const string &from_para, net_module *mod, const string &to_para, const string &dir)
{
    net_parameter *para;

    if (dir == "in")
        para = par_in->get(from_para);

    else
        para = par_out->get(from_para);

    para->del_P_conn(mod, to_para);
}

//----------------------------------------------------------------------
// change_param changes the Parameter para_name to the new Value value
// change_param searches in the list of the in- and output-Parameter
// if the parameter is a output-Parameter, it checks for a connection
// if a connection exists, it copies the new Parameter to the connected
// Module
//----------------------------------------------------------------------

void net_module::change_param(const string &param_name, const string &param_value_list)
{
    // search para
    net_parameter *para = par_in->get(param_name);
    if (para == NULL)
        para = par_out->get(param_name);

    if (para == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: Change_parameter. Parameter not found \n");
        char warn_txt[200];
        sprintf(warn_txt, " ---------- ERROR: parameter %s not found in module %s\n", param_name.c_str(), name.c_str());
        return;
    }
    // parameter found
    para->set_value_list(param_value_list);
}

string net_module::get_one_param(const string &param_name)
{
    // search para
    string value;
    net_parameter *para = par_in->get(param_name);
    if (para != NULL)
        value = para->get_val_list();
    return value;
}

void net_module::change_param(const string &para_name, const string &value, int number, int count)
{

    // search para
    string direction = "input";
    net_parameter *para = par_in->get(para_name);
    if (para == NULL)
    {
        para = par_out->get(para_name);
        direction = "output";
    }
    if (para == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: Change_parameter. Parameter not found \n");
        cerr << "\n ERROR: Change_parameter. Parameter not found \n";
        return;
    }

    // parameter found
    int pnumber = number;
    para->set_value(pnumber, value);
    if (direction == "output")
    {
        // check for connection and change in connected module
        connect_mod_list *c_list = para->get_connectlist();
        connect_mod *par_conn;
        c_list->reset();
        while ((par_conn = c_list->next()) != 0)
        {
            net_module *mod_conn = par_conn->get_mod();
            string mod_par = par_conn->get_par();
            mod_conn->change_param(mod_par, value, number, 1);
        }
    }

    if (count == 1)
    {
        // change param in linked modules
        netlink *tmp_link;
        netlinks->reset();
        while ((tmp_link = netlinks->next()) != NULL)
        {
            net_module *mod_conn = tmp_link->get_mod();
            mod_conn->change_param(para_name, value, number, 0);
        }
    }
}

void net_module::add_param(const string &param_name, const string &add_param)
{
    if (NULL == par_in)
    {
        print_comment(__LINE__, __FILE__, "ERROR: add_param. No par_in \n");
        return;
    }

    net_parameter *para = par_in->get(param_name);
    if (para == NULL)
        para = par_out->get(param_name);

    if (para == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: add_param. Parameter not found \n");
        return;
    }

    para->set_addvalue(add_param);
}

string net_module::get_startmessage(ui_list *ul)
{

    ostringstream buffS;
    buffS << this->get_name() << "\n" << this->get_nr() << "\n" << this->get_host() << "\n";

    int count = 0;
    interfaces->reset();
    net_interface *tmp_intf;
    while ((tmp_intf = (net_interface *)interfaces->next()) != NULL)
    {
        if (tmp_intf->get_direction() == "output")
            count++;

        else
        {
            bool state = tmp_intf->get_conn_state();
            if (state == true)
                count++;
        }
    }
    buffS << count << "\n";

    // get number of input_parameter
    count = 0;
    par_in->reset();
    net_parameter *tmp_para;
    while ((tmp_para = par_in->next()) != NULL)
    {
        count++;
    }
    buffS << count << "\n";

    // get objects
    interfaces->reset();
    while ((tmp_intf = (net_interface *)interfaces->next()) != NULL)
    {
        string intf_name = tmp_intf->get_name();
        bool state = tmp_intf->get_conn_state(); // true or false
        if (state == true)
        {
            object *obj = tmp_intf->get_object();
            if (!obj)
                return "";

            string obj_name = obj->get_current_name();
            if (obj_name.empty())
                return "";

            buffS << intf_name << "\n" << obj_name << "\n" << obj->get_type() << "\n";
            if (obj->get_to() && obj->get_to()->isEmpty())
                buffS << "UNCONNECTED"
                      << "\n";

            else
                buffS << "CONNECTED"
                      << "\n";
        }
        else
        {
            if (tmp_intf->get_direction() == "output")
            {
                buffS << intf_name << "\nwrong_object_name\nwrong_object_type\n";

                string buf = "ERROR old Network file, replace module " + get_name();
                Message *err_msg = new Message(COVISE_MESSAGE_COVISE_ERROR, buf);
                ul->send_all(err_msg);
                delete err_msg;
                return "";
            }

            if (state == false && tmp_intf->get_demand() == "req")
            {
                print_comment(__LINE__, __FILE__, "ERROR: get-startmessage. Interfaces not connected \n");
                ostringstream wtxt;
                wtxt << "Warning: Required input port (" << intf_name << ")" << name << "_" << nr << "@" << host;
                wtxt << " is not connected !! ";
                send_wmessage(ul, wtxt.str());
                return "";
            }
        }
    } // while

    // get parameter
    par_in->reset();
    while ((tmp_para = par_in->next()) != NULL)
    {
        string par_name = tmp_para->get_name();
        string par_type = tmp_para->get_type();
        int par_count = tmp_para->get_count();

        buffS << par_name << "\n" << par_type << "\n" << par_count << "\n";

        for (int i = 1; i <= par_count; i++)
            buffS << tmp_para->get_value(i) << "\n";
    }

    return buffS.str();
    ;
}

void net_module::send_msg(Message *msg)
{
    applmod->send_msg(msg);
}

int net_module::get_mod_id()
{
    return applmod->get_id();
}

void net_module::start_module(ui_list *ul)
{
    if (!is_alive())
        return;

    // set the Inputdataobjects to OLD
    // set status of Module to RUNNING
    reset_running();

    if (is_one_running_under())
    {
        set_start_flag();
        CTRLHandler::instance()->m_numRunning--;
        return;
    }

    set_status(MODULE_RUNNING);

    //delete_all Objects if not saved
    bool ret = delete_old_objs();
    //give new Names to Output_objects
    new_obj_names();

    // reset error list of the module
    if (m_errors)
        empty_errlist();

    string content = get_startmessage(ul);
    if (content.empty())
    {
        delete_rez_objs();
        if (ret)
            send_finish();
        CTRLHandler::instance()->m_numRunning--;
        set_status(MODULE_IDLE);
        string text = "Sorry, can't execute module " + name + "_" + nr + "@" + host;
        Message *err_msg = new Message(COVISE_MESSAGE_WARNING, text);
        ul->send_all(err_msg);
        delete err_msg;
        return;
    }

    Message *msg = new Message(COVISE_MESSAGE_START, content);
    applmod->send_msg(msg);
    delete msg;

    content = this->get_inparaobj();
    if (!content.empty())
    {
        msg = new Message(COVISE_MESSAGE_START, content);
        ul->send_all(msg);
        delete msg;
    }
}

int net_module::init(int nodeid, const string &name, const string &instanz, const string &host,
                     int posx, int posy, int copy, enum Start::Flags flags, net_module *mirror_node)
{


    parameter *tmp;
    module *mod;
    AppModule *dmod;
    DM_data *tmp_data;
    rhost *tmp_host;
    string tmp_user, tmp_passwd;

    set_name(name);
    set_nr(instanz);
    set_host(host);
    set_nodeid(nodeid);
    set_status(MODULE_IDLE);

    xkoord = posx;
    ykoord = posy;

    link_type(name, host, CTRLGlobal::getInstance()->moduleList);

    if (CTRLHandler::instance()->Config->getshminfo(host.c_str()) != COVISE_NOSHM)
    {
        // get Datamanager for host
        tmp_data = CTRLGlobal::getInstance()->dataManagerList->get(host);
        dmod = tmp_data->get_DM();

        // Start a module: send CRB a message to start module.
        //                 !!! Module connects to CTRL only, not to CRB !!!
        applmod = CTRLGlobal::getInstance()->controller->start_applicationmodule(APPLICATIONMODULE, name.c_str(), get_type()->get_category().c_str(), dmod, instanz.c_str(), flags);
        /// aw: applmod may be ==  0
        if (applmod == NULL)
        {
            cerr << "Module accept did not succeed" << endl;
#if 0
         exit(0);
#else
            datam = NULL;
            return 0;
#endif
        }

        /// establish connection CRB(Server) - Module(slave)
        applmod->connect(dmod);
        datam = dmod;
    }

    else
    {
        //------------------------------------------------------------------------
        // These changes for the T3E should be revisited as soon as possible if
        // they are really needed.
        // At least the current version crashes because of an uninitialized dmod!
        //------------------------------------------------------------------------

        // get Datamanager for host
        tmp_data = CTRLGlobal::getInstance()->dataManagerList->get(host);
        dmod = tmp_data->get_DM();
        // END OF CHANGES FOR T3E

        // start applicationmodule
        tmp_host = CTRLGlobal::getInstance()->hostList->get(host);
        tmp_passwd = tmp_host->get_passwd();
        tmp_user = tmp_host->get_user();

        applmod = CTRLGlobal::getInstance()->controller->start_applicationmodule(APPLICATIONMODULE, get_name().c_str(),
                                                             get_type()->get_category().c_str(), dmod, instanz.c_str(), flags);

        /// aw: applmod may be ==  0
        if (applmod == NULL)
        {
            cerr << "Module accept did not succeed : terminating covise session" << endl;
            return 0;
        }

        // add applicationmodule to the DMlist
        tmp_data = new DM_data;
        tmp_data->set_user(tmp_user);
        tmp_data->set_modname(name);
        tmp_data->set_hostname(host);
        tmp_data->set_passwd(tmp_passwd);
        tmp_data->set_DM(applmod);

        CTRLGlobal::getInstance()->dataManagerList->connect_all(applmod);

        CTRLGlobal::getInstance()->dataManagerList->add(tmp_data);

        datam = applmod;
    }

    if (copy == 4)
    {
        // MIRRORING
        mirror_node->set_mirror_status(ORG_MIRR); //original
        mirror_node->set_mirror_node(this);

        set_mirror_status(CPY_MIRR); //mirror
        set_mirror_node(mirror_node);
    }

    // receive the Module description
    mod = get_type();
    Message *msg = new Message;
    applmod->recv_msg(msg);
    if (!msg->data.data())
    {
        cerr << endl << " net_module::init() - NULL module description received\n";
        exit(0);
    }

    mod->read_description(msg->data.data());
    delete msg;

    this->create_netlink(name, instanz, host, CTRLGlobal::getInstance()->netList);

    // copy interfaces from module

    mod->reset_intflist();
    int outNum = 0;
    while (mod->next_interface() != 0)
    {
        set_interface(mod->get_interfname(), mod->get_interftype(), mod->get_interftext(),
                      mod->get_interfdirection(), mod->get_interfdemand());

        // generate data objectname for output port (former DOCONN)
        if (mod->get_interfdirection() == "output")
        {
            ostringstream objName;
            objName << name << "_" << instanz << "(" << nodeid << ")_OUT_" << outNum << "_";
            string outputName = mod->get_interfname();
            object *obj = CTRLGlobal::getInstance()->objectList->select(objName.str());
            CTRLGlobal::getInstance()->netList->set_DO_conn(name, instanz, host, outputName, obj);
            outNum++;
        }
    }

    // copy parameters from module
    mod->reset_paramlist("in");
    while ((tmp = mod->next_param("in")) != 0)
    {
        set_parameter(tmp, "in");
    }

    mod->reset_paramlist("out");
    while ((tmp = mod->next_param("out")) != 0)
    {
        set_parameter(tmp, "out");
    }

    if (mirror_node && copy == 3) // node created by moving
    {
        this->set_title(mirror_node->get_title());
    }

    else
    {
        this->set_standard_title();
    }

    return 1;
}

void net_module::create_netlink(const string &name, const string &instanz, const string &, net_module_list *netl)
{

    // instanz und extension bestimmen
    string sidx_main, sidx_xt;
    int first = (int)instanz.find(".");
    if (first != -1)
    {
        int len = (int)instanz.length();
        sidx_main = instanz.substr(0, first);
        sidx_xt = instanz.substr(first, len - first - 1);
    }

    else
        sidx_main = instanz;

    int idx_xt;
    if (sidx_xt.empty())
        idx_xt = 0;

    else
    {
        istringstream is(sidx_xt);
        is >> idx_xt;
        idx_xt = idx_xt + 1;
    }

    // Module durchgehen und ueberpruefen, ob sie noch existieren
    int i = 0;
    netlink *tmp_link;

    while (i <= (idx_xt - 1))
    {
        // instanz aus  sidx_main und Laufindex bilden
        if (i != 0)
        {
            ostringstream os;
            os << i;
            sidx_main.append(".");
            sidx_main.append(os.str());
        }

        net_module *tmp_mod = netl->get(name, sidx_main);

        if (i == 0)
            netlinks->set_org(tmp_mod);

        if (tmp_mod != NULL)
        {
            // module gibts
            // Format festlegen: name, nr, host, pointer auf net_module
            tmp_link = new netlink;
            tmp_link->set_name(name);
            tmp_link->set_host(tmp_mod->get_host());
            tmp_link->set_instanz(tmp_mod->get_nr());
            tmp_link->set_mod(tmp_mod);

            netlinks->add(tmp_link);

            // Gegeneintrag im Module machen !
            tmp_mod->set_netlink(this);
        }
        i++;
    } // while
}

void net_module::r_del_netlink(const string &m_name, const string &m_nr, const string &m_host)
{
    netlink *tmp_link;

    // delete only the link to m_name, m_nr, m_host
    tmp_link = netlinks->get(m_name, m_nr, m_host);

    netlinks->remove(tmp_link);
}

int net_module::get_count(const string &intfname)
{

    net_interface *tmp_intf = (net_interface *)interfaces->get(intfname);
    if (tmp_intf)
    {
        object *tmp_obj = tmp_intf->get_object();
        return tmp_obj->get_counter();
    }
    else
    {
        return 0;
    }
}

int net_module::test_org_count(const string &intfname)
{
    int org_count;

    // ueberpruefen, ob Module Kopie oder Original (netlinks)
    net_module *tmp_mod = netlinks->get_org();

    if (tmp_mod == NULL)
    {
        // Original -> Zaehler 1 zurueckgeben
        org_count = 1;
    }
    else
    {
        // Kopie
        // Zaehlerstand des Datenobjektes an intfname lesen
        org_count = tmp_mod->get_count(intfname);
    }

    // Zaehler zurueckgeben
    return org_count;
}

bool net_module::test_copy()
{
    if (netlinks)
    {
        net_module *tmp_mod = netlinks->get_org();
        if (tmp_mod)
            return true;
    }
    return false;
}

void net_module::del(int already_dead)
{

    //delete_all Objects if not saved
    delete_all_objs(already_dead);

    // delete the netlinks
    // select all the connected modules
    netlink *tmp_link;
    netlinks->reset();
    while ((tmp_link = netlinks->next()) != NULL)
    {
        // delete mit eigenen Werten zum module senden
        tmp_link->del_link(name, nr, host);
        // Eintrag lokal loeschen
        netlinks->remove(tmp_link);
        // reset der liste machen
        netlinks->reset();
    }

    // alle interfaces durchgehen
    net_interface *C_interface;
    string direction;
    object *obj;
    interfaces->reset();
    while ((C_interface = (net_interface *)interfaces->next()) != NULL)
    {
        // get interface
        // control, if a connection exists
        if ((obj = C_interface->get_object()) != NULL)
        {
            direction = C_interface->get_direction();
            if (direction == "output")
            {
                // delete the object in the shm
                // if itsn't a NEW | OPEN
                if (already_dead >= 0)
                    obj->del_old_data();

                // del all To-Connections in the Object and in the corresponding
                // modules
                obj->del_allto_connects();

                // delete the from-Connection in the object
                obj->del_from_connection();

                // delete local connection
                C_interface->del_connect();

                // delete Object (without any connection)
                CTRLGlobal::getInstance()->objectList->remove(obj);
                CTRLGlobal::getInstance()->objectList->reset();
            }
            else if (direction == "input")
            {
                // del the corresponding to-Connection in the Object
                obj->del_to_connection(name, nr, host, C_interface->get_name());

                // delete the local connection
                C_interface->del_connect();
            } // strcmp
        } // connect-control

        interfaces->remove(C_interface);
        interfaces->reset();
    } // while

    // alle Control-Connections durchgehen
    // prev-Connection
    net_control *controllerconn;
    net_module *mod;
    prev_control->reset();
    while ((controllerconn = prev_control->next()) != NULL)
    {
        // get connected module
        mod = controllerconn->get();
        // remove Connection in Module
        mod->del_C_conn(this, "to");
        // remove Connection
        prev_control->remove(controllerconn);
        prev_control->reset();
    }
    // next-Connection
    next_control->reset();
    while ((controllerconn = next_control->next()) != NULL)
    {
        // get connected module
        mod = controllerconn->get();
        // remove Connection in Module
        mod->del_C_conn(this, "from");
        // remove Connection
        prev_control->remove(controllerconn);
        prev_control->reset();
    }

    // alle Parameter-connections durchgehen
    connect_mod_list *conn_list;
    connect_mod *tmp_conn;
    net_parameter *tmp_par;
    // input-Parameter
    par_in->reset();
    while ((tmp_par = par_in->next()) != NULL)
    {
        // del connections
        conn_list = tmp_par->get_connectlist();
        conn_list->reset();
        while ((tmp_conn = conn_list->next()) != NULL)
        {
            // del the connection in the connected module
            mod = tmp_conn->get_mod();
            mod->del_P_conn(tmp_conn->get_par(), this, tmp_par->get_name(), "out");
            conn_list->remove(tmp_conn);
            conn_list->reset();
        }
        par_in->remove(tmp_par);
        par_in->reset();
    }
    // output-Parameter
    par_out->reset();
    while ((tmp_par = par_out->next()) != NULL)
    {
        // del connections
        conn_list = tmp_par->get_connectlist();
        conn_list->reset();
        while ((tmp_conn = conn_list->next()) != NULL)
        {
            // del the connection in the connected module
            mod = tmp_conn->get_mod();
            mod->del_P_conn(tmp_conn->get_par(), this, tmp_par->get_name(), "in");
            conn_list->remove(tmp_conn);
            conn_list->reset();
        }
        par_out->remove(tmp_par);
        par_out->reset();
    }

    // send the QUIT message to the module
    Message *msg = new Message(COVISE_MESSAGE_QUIT, "");

#ifdef QUITMOD
    if (already_dead <= 0 && applmod)
        applmod->send_msg(msg);
#endif

#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "send %s\n%i %i \n %s \n", this->get_name(), msg->sender, msg->type, msg->data);
//	fflush(msg_prot);
#endif

    // receive QUIT Acknowledge from module
    delete msg;
}

string net_module::get_outparaobj()
{
    ostringstream buffS;
    buffS << name << "\n" << nr << "\n" << host << "\n";

    // get number of out parameters
    par_out->reset();
    net_parameter *tmp_para;
    int par_count = 0;
    while ((tmp_para = par_out->next()) != NULL)
    {
        par_count++;
    }
    buffS << par_count << "\n";

    par_out->reset();
    for (int pi = 1; pi <= par_count; pi++)
    {
        net_parameter *tmp_para = par_out->next();
        int par_nr = tmp_para->get_count();
        buffS << tmp_para->get_name() << "\n" << tmp_para->get_type() << "\n" << tmp_para->get_count() << "\n";

        for (int i = 1; i <= par_nr; i++)
            buffS << tmp_para->get_value(i) << "\n";
    }

    // get number of Interfaces
    int intf_count = 0;
    C_interface *tmp_normintf;
    interfaces->reset();
    while ((tmp_normintf = (net_interface *)interfaces->next()) != NULL)
    {
        if (tmp_normintf->get_direction() == "output")
            intf_count++;
    }

    string cat = typ->get_category();
    if (cat == "Renderer")
        buffS << "0\n";

    else
    {
        buffS << intf_count << "\n";

        // get objects
        net_interface *tmp_intf;
        interfaces->reset();
        while ((tmp_intf = (net_interface *)interfaces->next()) != NULL)
        {
            if (tmp_intf->get_direction() == "output")
            {
                string intf_name = tmp_intf->get_name();
                bool state = tmp_intf->get_conn_state();
                if (state == true)
                {
                    object *obj = tmp_intf->get_object();
                    string obj_name = obj->get_current_name();

                    buffS << intf_name << "\n";
                    if (!obj_name.empty())
                        buffS << obj_name << "\n";

                    else
                        buffS << "NO_OBJ\n"; //DeletedModuleFinished");
                }
                else if (state == false)
                {
                    print_comment(__LINE__, __FILE__, "ERROR: send_finisCTRLGlobal::getInstance()-> Interfaces not connected \n");
                    cerr << "\n ERROR: send_finisCTRLGlobal::getInstance()-> Interfaces not connected !!!\n";
                    string content;
                    return content;
                } // if state
            } // if output
        } // while
    } // else   er
    return buffS.str();
}

string net_module::get_inparaobj()
{
    ostringstream buffS;
    buffS << name << "\n" << nr << "\n" << host << "\n";

    // get number of in parameters
    net_parameter *tmp_para;
    par_in->reset();
    int par_count = 0;
    while ((tmp_para = par_in->next()) != NULL)
    {
        par_count++;
    }
    buffS << par_count << "\n";

    par_in->reset();
    for (int pi = 1; pi <= par_count; pi++)
    {
        tmp_para = par_in->next();
        int par_nr = tmp_para->get_count();
        buffS << tmp_para->get_name() << "\n" << tmp_para->get_type() << "\n" << par_nr << "\n";
        for (int i = 1; i <= par_nr; i++)
            buffS << tmp_para->get_value(i) << "\n";
    }

    // get number of Interfaces
    int intf_count = 0;
    C_interface *tmp_normintf;
    interfaces->reset();
    while ((tmp_normintf = (net_interface *)interfaces->next()) != NULL)
    {
        if (tmp_normintf->get_direction() == "input")
            intf_count++;
    }

    string cat = typ->get_category();
    if (cat == "Renderer")
    {
        buffS << intf_count << "\n";

        // get objects
        net_interface *tmp_intf;
        interfaces->reset();
        while ((tmp_intf = (net_interface *)interfaces->next()) != NULL)
        {
            if (tmp_intf->get_direction() == "input")
            {
                string intf_name = tmp_intf->get_name();
                bool state = tmp_intf->get_conn_state();
                buffS << intf_name << "\n";
                if (state == true)
                {
                    string obj_name = ((render_interface *)tmp_intf)->get_objlist();
                    buffS << obj_name << "\n";
                }
                else
                    buffS << "NOT CONNECTED>\n";
                // if state
            } // if
        } // while
    }
    else
    {
        buffS << intf_count << "\n";

        // get objects
        net_interface *tmp_intf;
        interfaces->reset();
        while ((tmp_intf = (net_interface *)interfaces->next()) != NULL)
        {
            if (tmp_intf->get_direction() == "input")
            {
                string intf_name = tmp_intf->get_name();
                bool state = tmp_intf->get_conn_state();
                if (state == true)
                {
                    object *obj = tmp_intf->get_object();
                    string obj_name = obj->get_current_name();
                    buffS << intf_name << "\n" << obj_name << "\n";
                }
                else if (state == false)
                    buffS << intf_name << "\n<NOT CONNECTED>\n";
                // if state
            } // if input
        } // while

    } // else   er

    return buffS.str();
}

void net_module::send_finish()
{
    string content = this->get_outparaobj();
    if (!content.empty())
    {
        Message *msg = new Message(COVISE_MESSAGE_FINISHED, content);
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);
        delete msg;
    }
}

void net_module::set_DO_status(int mode, const string &DO_name)
{
    net_interface *tmp_intf;
    bool found = false;
    interfaces->reset();
    while (((tmp_intf = (net_interface *)interfaces->next()) != NULL) && (found == false))
    {
        if (tmp_intf->get_direction() == "input")
        {
            bool state = tmp_intf->get_conn_state();
            if (state == true)
            {
                object *obj = tmp_intf->get_object();
                bool test = obj->test(DO_name);
                if (test == true)
                {
                    found = true;
                    obj->set_DO_status(DO_name, mode, this, tmp_intf->get_name());
                }
            }
        }
    }
    if (found == false)
    {
        print_comment(__LINE__, __FILE__, "ERROR: Dataobject not found \n");
        cerr << "\nERROR: Dataobject not found !!!\n";
        return;
    }
}

void net_module::set_outputtype(const string &intf_name, const string &DO_name, const string &DO_type)
{
    net_interface *tmp_intf = (net_interface *)interfaces->get(intf_name);
    if (tmp_intf)
        tmp_intf->set_outputtype(DO_name, DO_type);
}

string net_module::get_moduleinfo(const string &local_name, const string &)
{
    ostringstream os;
    os << get_name() << "\n" << get_nr() << "\n";

    string tmp_host = get_host();
    if (tmp_host == local_name)
        tmp_host = "LOCAL";
    os << tmp_host << "\n" << typ->get_category() << "\n" << get_title() << "\n";

    // get Position x y
    os << xkoord << "\n" << ykoord << "\n";

    return os.str();
}

string net_module::get_interfaces(const string &direction)
{
    int i = 0;
    string buffer;

    C_interface *tmp_intf;
    interfaces->reset();
    while ((tmp_intf = interfaces->next()) != NULL)
    {
        if (tmp_intf->get_direction() == direction)
        {
            i++;
            buffer = buffer + tmp_intf->get_name() + "\n" + tmp_intf->get_type() + "\n";
            buffer = buffer + tmp_intf->get_text() + "\n" + tmp_intf->get_demand() + "\n";

            // object connection is not longer needed
            /*string str5;
         if (tmp_intf->get_direction() == "output")
         {
            object * obj = ((net_interface *)tmp_intf)->get_object();
            if (obj)
               str5 = obj->get_name();
            else
               str5 = "none";
         }
         else
            str5 = "none";*/

            //buffer = buffer + str5 + "\n";
            buffer = buffer + "\n";
        }
    }

    ostringstream os;
    os << i;
    buffer = os.str() + "\n" + buffer;

    return buffer;
}

void net_module::writeScript(ofstream &of, const string &local_user)
{
    (void)local_user;

    of << "#" << endl;
    of << "# MODULE: " << get_name() << endl;
    of << "#" << endl;
    of << get_name() << "_" << get_nr() << " = " << get_name() << "()" << endl;

    /*string tmp_host =  get_host();
   if (tmp_host ==  local_name)
      tmp_host =  "LOCAL";
   os << tmp_host << "\n" <<  typ->get_category() << "\n" << get_title() << "\n";
    */
    of << "network.add( " << get_name() << "_" << get_nr() << " )" << endl;

    // get Position x y
    of << get_name() << "_" << get_nr() << ".setPos( " << xkoord << ", " << ykoord << " )" << endl;

    of << "#" << endl;
    of << "# set parameter values" << endl;
    of << "#" << endl;
    writeParamScript(of, "input");
}

void net_module::writeParamScript(ofstream &of, const string &direction)
{
    net_param_list *tmp_list;
    if (direction == "input")
        tmp_list = par_in;

    else
        tmp_list = par_out;

    tmp_list->reset();
    net_parameter *tmp_para;
    while ((tmp_para = tmp_list->next()) != NULL)
    {
        of << get_name() << "_" << get_nr() << ".set_";
        of << tmp_para->get_name() << "( " << tmp_para->get_pyval_list() << " )" << endl;
    }
}

string net_module::get_module(const string &local_name, const string &local_user, bool forSaving)
{
    ostringstream buffer;

    buffer << "# Module " << name << "\n";
    string data = get_moduleinfo(local_name, local_user);
    buffer << data;

    data = get_interfaces("input");
    buffer << data;

    data = get_interfaces("output");
    buffer << data;

    data = get_parameter("input", forSaving);
    buffer << data;

    data = get_parameter("output", forSaving);
    buffer << data;

    return buffer.str();
}

string net_module::get_parameter(const string &direction, bool forSaving)
{
    net_param_list *tmp_list;
    if (direction == "input")
        tmp_list = par_in;

    else
        tmp_list = par_out;

    int i = 0;
    net_parameter *tmp_para;

    string buffer;
    tmp_list->reset();
    while ((tmp_para = tmp_list->next()) != NULL)
    {
        i++;

        buffer = buffer + tmp_para->get_name() + "\n" + tmp_para->get_type() + "\n";
        buffer = buffer + tmp_para->get_text() + "\n";
        string value = tmp_para->get_val_list();

        // remove leading covise_path (:path1:path2...)
        if (tmp_para->get_type() == "Browser" && forSaving)
        {
            if (get_dm()) // dm is sometimes not initialized, please to this differently anyway
            {
                string path = get_dm()->covise_path;
                string sep = path.substr(0, 1);
                path.erase(0, 1);
                vector<string> pathList = CTRLHandler::instance()->splitString(path, sep);
                for (int i = 0; i < pathList.size(); i++)
                {
                    string path = pathList[i];
                    int find = (int)value.find(path);
                    if (find != std::string::npos)
                    {
                        value = value.substr(path.length());
                        while (value.length() > 0 && value[0] == '/')
                            value.erase(0, 1);
                        break;
                    }
                }
            }
        }

        buffer = buffer + value + "\n";

#ifdef PARA_START
        // not longer needed
        //buffer.append(tmp_para->get_extension() + "\n");
        buffer.append("\n");
#endif
        buffer.append(tmp_para->get_addvalue() + "\n");
    }

    ostringstream os;
    os << i << "\n";
    buffer = os.str() + buffer;
    return buffer;
}

#ifdef PARAM_CONN
string net_module::get_para_connect(int *i, const string &local_name, const string &local_user)
{
    net_parameter *tmp_par;
    connect_mod_list *clist;
    connect_mod *tmp_conn;
    string loc_name, *loc_host, *conn_name, *conn_host, *conn_par, *loc_par;
    // int nr;
    net_module *conn_mod;
    string loc_nr, *conn_nr, *loc_str, *part1, *part2, *res_str, *tmp_user;
    string loc_str2, *part22, *part12, *res_str2;

    res_str = NULL;
    *i = 0;

    // get local infos
    // name, instanz, host, parameter
    loc_name = this->get_name();

    loc_nr = this->get_nr();
    loc_host = this->get_host();

    tmp_user = typ->get_user();

    if ((loc_host, local_name) == 0) &&
      (tmp_user, local_user) == 0) )
        {
            loc_host = "LOCAL";
        }

    loc_str = new char[strlen(loc_name) + 1];
    strcpy(loc_str, loc_name);
    loc_str2 = append2(loc_str, loc_nr, "\n");
    delete[] loc_str;
    loc_str = append2(loc_str2, loc_host, "\n");
    delete[] loc_str2;

    // loop over all PARIN parameter
    par_in->reset();
    while ((tmp_par = par_in->next()) != NULL)
    {
        clist = tmp_par->get_connectlist();
        clist->reset();

        loc_par = tmp_par->get_name();

        // loop over all connections
        while ((tmp_conn = clist->next()) != NULL)
        {
            conn_mod = tmp_conn->get_mod();

            // get infos from connected module
            // name, instanz, host, parameter

            conn_name = conn_mod->get_name();
            conn_nr = conn_mod->get_nr();
            conn_host = conn_mod->get_host();

            tmp_user = typ->get_user();

            if ((conn_host, local_name) == 0) &&
            (tmp_user, local_user) == 0) )
                {
                    conn_host = "LOCAL";
                }

            conn_par = tmp_conn->get_par();

            // order of modules in the file:
            // from-module to-module
            // conn_values loc_values

            part2 = new char[strlen(loc_str) + 1];
            strcpy(part2, loc_str);
            part22 = append2(part2, loc_par, "\n");
            delete[] part2;

            part1 = new char[strlen(conn_name) + 1];
            strcpy(part1, conn_name);
            part12 = append2(part1, conn_nr, "\n");
            delete[] part1;
            part1 = append2(part12, conn_host, "\n");
            delete[] part12;
            part12 = append2(part1, conn_par, "\n");
            delete[] part1;

            // concatenate the two parts
            part1 = append2(part12, part22, "\n");
            delete[] part12;
            delete[] part22;

            if (*i == 0)
            {
                res_str = new char[strlen(part1) + 1];
                strcpy(res_str, part1);
                delete[] part1;
            }
            else
            {
                res_str2 = append2(res_str, part1, "\n");
                delete[] part1;
                delete[] res_str;
                res_str = res_str2;
            }
            (*i)++;
        }
    }

    // built string with result
    return res_str;
}
#endif

//**********************************************************************
//
// 			DISPLAY
//
//**********************************************************************

display::display()
{
    applmod = NULL;
    excovise_status = "INIT";
    DISPLAY_READY = false;
    NEXT_DEL = false;
    m_helper = 0;
}

display::~display()
{
    if (applmod != NULL)
        delete applmod;
}

void display::set_hostname(const string &str)
{
    hostname = str;
}

void display::set_userid(const string &str)
{
    userid = str;
}

void display::set_passwd(const string &str)
{
    passwd = str;
}

void display::set_execstat(const string &str)
{
    excovise_status = str;
}

void display::set_DISPLAY(bool bvar)
{
    DISPLAY_READY = bvar;
}

bool display::get_DISPLAY()
{
    return DISPLAY_READY;
}

AppModule *display::get_mod()
{
    return applmod;
}

int display::get_mod_id()
{
    return applmod->get_id();
}

int display::start(AppModule *dmod, const string &info_str, module *mod, const string &add_param, enum Start::Flags flags)
{

    // parse instanz out of info_str (second token)
    vector<string> list = CTRLHandler::instance()->splitString(info_str, "\n");
    string instanz = list[1];

    const char *addParam = NULL;
    if (!add_param.empty())
        addParam = add_param.c_str();

    applmod = CTRLGlobal::getInstance()->controller->start_applicationmodule(RENDERER, addParam, mod->get_name().c_str(),
                                                         mod->get_category().c_str(), dmod, instanz.c_str(), flags);
    /// aw: applmod may be ==  0
    if (applmod == NULL)
    {
        cerr << "Module accept did not succeed" << endl;
        return 0;
    }

    applmod->connect(dmod);
    // status dem Rendermodule mitteilen
    Message *msg = new Message;
    applmod->recv_msg(msg);
    mod->read_description(msg->data.data());
    delete msg;
    this->send_status(info_str);

    return 1;
}

#ifdef NEW_RENDER_MSG

void display::send_status(const string &info_str)
{
    string text = "STATUS\n" + excovise_status + "\n" + info_str + "\n";
    Message *msg = new Message(COVISE_MESSAGE_RENDER, text);
    applmod->send_msg(msg);

#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "send displaystatus\n%i %i \n %s \n", msg->sender, msg->type, msg->data);
//	fflush(msg_prot);
#endif

    delete msg;
}

#else

void display::send_status(const string &info_str)
{
    string text = excovise_status + "\n" + info_str + "\n";
    Message *msg = new Message(COVISE_MESSAGE_RENDER, text);
    applmod->send_msg(msg);

#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "send displaystatus\n%i %i \n %s \n", msg->sender, msg->type, msg->data);
//	fflush(msg_prot);
#endif

    delete msg;
}

#endif

void display::quit()
{
    Message *msg = new Message(COVISE_MESSAGE_QUIT, "");

#ifdef QUITMOD
    applmod->send_msg(msg);
#endif

#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "send Quit Display\n%i %i \n %s \n", msg->sender, msg->type, msg->data);
//	fflush(msg_prot);
#endif

    delete msg;
}

bool display::get_NEXT_DEL()
{
    return NEXT_DEL;
}

void display::send_add(const string &DO_name)
{
    if (DO_name.empty())
        return;

    CTRLHandler::instance()->m_numRunning++;
    CTRLHandler::instance()->m_numRendererRunning++;

    NEXT_DEL = true;

#ifdef NEW_RENDER_MSG
    string tmp = "ADD\n" + DO_name + "\n";
    Message *msg = new Message(COVISE_MESSAGE_RENDER, tmp);
#else
    string tmp = DO_name;
    Message *msg = new Message(COVISE_MESSAGE_ADD_OBJECT, tmp);
#endif

    if (!is_helper())
        applmod->send_msg(msg);

#ifdef DEBUG
    fprintf(msg_prot, "---------------------------------------------------\n");
    fprintf(msg_prot, "send Add_object to \n%i %s \n", this->get_mod_id(), msg->data);
    fflush(msg_prot);
#endif

    delete msg;
}

void display::send_add()
{
    CTRLHandler::instance()->m_numRunning++;
    CTRLHandler::instance()->m_numRendererRunning++;

    if (!DO_name.empty())
    {
        NEXT_DEL = true;

#ifdef NEW_RENDER_MSG
        string tmp = "ADD\n" + DO_name + "\n";
        Message *msg = new Message(COVISE_MESSAGE_RENDER, tmp);
#else
        string tmp = DO_name;
        Message *msg = new Message(COVISE_MESSAGE_ADD_OBJECT, tmp);
#endif

        if (!is_helper())
            applmod->send_msg(msg);

#ifdef DEBUG
        fprintf(msg_prot, "---------------------------------------------------\n");
        fprintf(msg_prot, "send Add_object to \n%i %s \n", this->get_mod_id(), msg->data);
        fflush(msg_prot);
#endif

        delete msg;
    }
}

void display::send_del(const string &DO_old_name, const string &DO_new_name)
{
    CTRLHandler::instance()->m_numRunning++;
    CTRLHandler::instance()->m_numRendererRunning++;

#ifdef REPLACE_MSG

    NEXT_DEL = true; // DEL or REPLACE possible after REPLACE

    // test, if new-name is an empty string -> only DELETE
    if (DO_new_name.empty())
    {
        NEXT_DEL = true;
        DO_name = DO_new_name;

        if (!DO_old_name.empty())
        {

#ifdef NEW_RENDER_MSG
            string text = "DEL\n" + DO_old_name + "\n";
            Message *msg = new Message(COVISE_MESSAGE_RENDER, text);
#else
            Message *msg = new Message(COVISE_MESSAGE_DELETE_OBJECT, DO_old_name);
#endif
            if (!is_helper())
                applmod->send_msg(msg);

            delete msg;
        }

#ifdef DEBUG
        fprintf(msg_prot, "---------------------------------------------------\n");
        fprintf(msg_prot, "send DEL_Obj to \n%i %s \n", this->get_mod_id(), msg->data);
        fflush(msg_prot);
#endif
    }

    else
    {
        // copy new-name to DO_name
        DO_name = DO_new_name;

        // create Msg-string
        string text = DO_old_name + "\n" + DO_new_name + "\n";
        Message *msg = new Message(COVISE_MESSAGE_REPLACE_OBJECT, text);

        if (!is_helper())
            applmod->send_msg(msg);

#ifdef DEBUG
        fprintf(msg_prot, "---------------------------------------------------\n");
        fprintf(msg_prot, "send DEL_Obj to \n%i %s \n", this->get_mod_id(), msg->data);
        fflush(msg_prot);
#endif
        delete msg;
    }

// else REPLACE
#else
    //

    NEXT_DEL = false;

    DO_name = DO_new_name;

    tmp = new char[strlen(DO_old_name) + 1];
    strcpy(tmp, DO_old_name);

#ifdef NEW_RENDER_MSG
    string text = "DEL\n" + DO_old_name + "\n";
    Message *msg = new Message(COVISE_MESSAGE_RENDER, text);
#else
    Message *msg = new Message(COVISE_MESSAGE_RENDER, DO_old_name);
#endif

    if (!is_helper())
        applmod->send_msg(msg);

#ifdef DEBUG
    fprintf(msg_prot, "---------------------------------------------------\n");
    fprintf(msg_prot, "send DEL_Obj to \n%i %s \n", this->get_mod_id(), msg->data);
    fflush(msg_prot);
#endif

    delete msg;

// endif REPLACE
#endif
}

void display::send_message(Message *msg)
{
    if (!is_helper())
        applmod->send_msg(msg);
#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "send display\n%i %i \n %s \n", msg->sender, msg->type, msg->data);
//	fflush(msg_prot);
#endif
}

//**********************************************************************
//
// 			DISPLAY_LIST
//
//**********************************************************************
displaylist::displaylist()
    : Liste<display>(1)
{
    count = 0;
    ready = 0;
}

displaylist::~displaylist()
{
    display *tmp;
    while ((tmp = next()) != NULL)
        remove(tmp);
}

void displaylist::incr_ready()
{
    ready = ready + 1;
    if (ready > count)
    {
        ready = count;
    }
}

void displaylist::decr_count()
{
    count--;
    if (count < 0)
        count = 0;
}

int displaylist::get_ready()
{
    return ready;
}

int displaylist::get_count()
{
    return count;
}

void displaylist::reset_ready()
{
    ready -= count;
}

int displaylist::init(const string &excovise_name, const string &info_str, module *mod, int copy, enum Start::Flags flags, rhost *host)
{
    (void)excovise_name;

    userinterface *tmp_ui = CTRLGlobal::getInstance()->userinterfaceList->get(host->get_hostname(), host->get_user());
    if (tmp_ui && copy != 4) // !MIRROR
    {
        CTRLGlobal::getInstance()->userinterfaceList->reset();
        while ((tmp_ui = CTRLGlobal::getInstance()->userinterfaceList->next()) != NULL)
        {
            string tmp_hostname = tmp_ui->get_host();
            string tmp_userid = tmp_ui->get_userid();
            string tmp_passwd = tmp_ui->get_passwd();
            string tmp_status = tmp_ui->get_status();

            if (this->contains(tmp_hostname, tmp_userid))
                continue;

            display *tmp_dis = new display;

            /// Do we have to start the Master-Cover of a Cluster version
            //  on a helper host ?
            if (mod->get_name().find("COVER", 0, 5) && coCoviseConfig::getInt("COVER.MultiPC.NumSlaves", 0))
            {
                bool debugCoverStartup = coCoviseConfig::isOn("COVER.MultiPC.Debug", false);
                if (debugCoverStartup)
                {
                    fprintf(stderr, "Controller @ [%s] starting up Cluster-Cover in VISENSO Mode\n", tmp_hostname.c_str());
                }

                string masterName = coCoviseConfig::getEntry("COVER.MultiPC.Master");
                if (!masterName.empty() && masterName == tmp_hostname)
                {
                    string newhost = masterName;
                    if (debugCoverStartup)
                    {
                        fprintf(stderr, "Controller starting up helper CRB @ [%s]\n", newhost.c_str());
                    }
                    addHelperCRB(newhost, tmp_hostname);
                    tmp_hostname = newhost;
                }
            }

            tmp_dis->set_hostname(tmp_hostname);
            tmp_dis->set_userid(tmp_userid);
            tmp_dis->set_passwd(tmp_passwd);
            tmp_dis->set_execstat(tmp_status);

            DM_data *tmp_data = CTRLGlobal::getInstance()->dataManagerList->get(tmp_hostname);
            AppModule *dmod = tmp_data->get_DM();

            // add host to the info_str
            string tmp_info = info_str;
            tmp_info.append(tmp_hostname);

            string dummy;
            int ret = tmp_dis->start(dmod, tmp_info, mod, dummy, flags); // start und send status-message
            if (!ret)
            {
                delete tmp_dis;
                return 0;
            }

            this->add(tmp_dis);
            count++;
        }
    } // end !MIRROR

    string tmp_hostname = host->get_hostname();
    tmp_ui = CTRLGlobal::getInstance()->userinterfaceList->get(tmp_hostname);
    if ((tmp_ui == NULL) || (copy == 4))
    {
        // no partner on this host or it's a mirror node

        string tmp_userid = host->get_user();
        string tmp_passwd = host->get_passwd();

        if (this->contains(tmp_hostname, tmp_userid))
            return 1;

        display *tmp_dis = new display;

        tmp_dis->set_hostname(tmp_hostname);
        tmp_dis->set_userid(tmp_userid);
        tmp_dis->set_passwd(tmp_passwd);
        tmp_dis->set_execstat(string("MIRROR"));

        DM_data *tmp_data = CTRLGlobal::getInstance()->dataManagerList->get(tmp_hostname);
        AppModule *dmod = tmp_data->get_DM();

        // add host to the info_str
        string tmp_info = info_str + tmp_hostname;

        int ret = tmp_dis->start(dmod, tmp_info, mod, "", flags); // start und send status-message
        if (!ret)
        {
            delete tmp_dis;
            return 0;
        }

        this->add(tmp_dis);
        count++;
    }

    return 1;
}

bool displaylist::contains(const std::string &hostname, const std::string &user)
{

    display *dis;

    this->reset();
    while ((dis = this->next()) != 0)
    {
        if (dis->get_hostname() == hostname && dis->get_userid() == user)
            return true;
    }

    return false;
}

void displaylist::quit()
{
    display *tmp_dis;

    this->reset();
    while ((tmp_dis = this->next()) != NULL)
    {
        tmp_dis->quit();
        this->remove(tmp_dis);
        this->reset();
    }
}

void displaylist::set_DISPLAY_FALSE()
{
    display *tmp_dis;

    this->reset();
    while ((tmp_dis = this->next()) != NULL)
    {
        tmp_dis->set_DISPLAY(false);
    }
}

void displaylist::send_add(const string &DO_name)
{
    display *tmp_dis;

    this->reset();
    while ((tmp_dis = this->next()) != NULL)
    {
        tmp_dis->send_add(DO_name);
    }
}

void displaylist::send_del(const string &DO_name, const string &DO_new_name)
{
    display *tmp_dis;

    this->reset();
    while ((tmp_dis = this->next()) != NULL)
    {
        tmp_dis->send_del(DO_name, DO_new_name);
    }
}

display *displaylist::get(int sender)
{
    display *tmp_dis;

    this->reset();
    tmp_dis = this->next();
    while (tmp_dis && ((tmp_dis->get_mod_id()) != sender))
    {
        tmp_dis = this->next();
        if (tmp_dis == NULL)
        {
            break;
        }
    }
    return tmp_dis;
}

display *displaylist::get(const string &name, const string &user)
{
    display *tmp_dis;

    this->reset();
    tmp_dis = this->next();
    while (name != tmp_dis->get_hostname() || user != tmp_dis->get_userid())
    {
        tmp_dis = this->next();
        if (tmp_dis == NULL)
        {
            print_comment(__LINE__, __FILE__, "ERROR: Display for Renderer not found \n");
            return NULL;
        }
    }
    return tmp_dis;
}

//**********************************************************************
//
// 			RENDER_MODULE
//
//**********************************************************************

render_module::render_module()
{
    m_multisite = 0;
    status = MODULE_IDLE;
    DISPLAY_ALL = false;
    typ = NULL;
    displays = new displaylist;
}

render_module::~render_module()
{
    if (displays != NULL)
    {
        delete displays;
    }
}

display *render_module::get_display(int id)
{
    return displays->get(id);
}

void render_module::set_interface(const string &strn, const string &strt, const string &strtx, const string &strd, const string &strde)
{
    if (strd == "input")
    {
        render_interface *tmp_r = new render_interface;
        tmp_r->set_name(strn);
        tmp_r->set_type(strt);
        tmp_r->set_text(strtx);
        tmp_r->set_direction(strd);
        tmp_r->set_demand(strde);
        interfaces->add(tmp_r);
    }
    else
    {
        net_interface *tmp_n = new net_interface;
        tmp_n->set_name(strn);
        tmp_n->set_type(strt);
        tmp_n->set_text(strtx);
        tmp_n->set_direction(strd);
        tmp_n->set_demand(strde);
        interfaces->add(tmp_n);
    }
}

/// start a CRB for a helper
int displaylist::addHelperCRB(const string &helperHost, const string &host)
{


    DM_data *helperDmgr = CTRLGlobal::getInstance()->dataManagerList->get(helperHost);
    DM_data *moduleDmgr = CTRLGlobal::getInstance()->dataManagerList->get(host);

    string user = moduleDmgr->get_user();
    string passwd = moduleDmgr->get_passwd();

    if (helperDmgr == NULL) // no crb on helper host
    {
        // start crb
        int exec_type = CTRLHandler::instance()->Config->getexectype(helperHost.c_str());
        coHostType htype(CO_PARTNER);
        if (CTRLGlobal::getInstance()->dataManagerList->add_crb(exec_type, helperHost, user, passwd, "", htype) == 0)
        {
            return (0);
        }

        // add new host in hostlist
        rhost *tmp_host = new rhost;
        tmp_host->set_hostname(helperHost);
        tmp_host->set_user(user);
        tmp_host->set_passwd(passwd);
        CTRLGlobal::getInstance()->hostList->add(tmp_host);
    }
    return 1;
}

/// start a 'Helper' on another host
int render_module::add_helper(const string &helperHost, // host to start the helper on
                              const string &info_str, // ??
                              const string &param) // parameter for execution
{

    addHelperCRB(helperHost, host);


    DM_data *helperDmgr = CTRLGlobal::getInstance()->dataManagerList->get(helperHost);
    DM_data *moduleDmgr = CTRLGlobal::getInstance()->dataManagerList->get(host);
    string user = moduleDmgr->get_user();
    string passwd = moduleDmgr->get_passwd();

    display *tmp_dis = new display;
    tmp_dis->set_hostname(helperHost);
    tmp_dis->set_userid(user);
    tmp_dis->set_passwd(passwd);
    tmp_dis->set_helper(1);

    helperDmgr = CTRLGlobal::getInstance()->dataManagerList->get(helperHost);
    AppModule *dmod = helperDmgr->get_DM();
    module *mod = get_type();
    // add host to the info_str
    string tmp_info = info_str + helperHost;

    int ret = tmp_dis->start(dmod, tmp_info, mod, param, Start::Normal); // start und send status-message
    if (!ret)
    {
        delete tmp_dis;
        return 0;
    }

    get_displays()->add(tmp_dis);
    return 1;
}

int render_module::init(int nodeid, const string &name, const string &instanz, const string &host,
                        int posx, int posy, int copy, enum Start::Flags flags, net_module *mirror_node)
{

    set_name(name);
    set_nr(instanz);
    set_host(host);
    set_nodeid(nodeid);

    set_standard_title();
    set_status(MODULE_IDLE);

    xkoord = posx;
    ykoord = posy;
    // start applicationmodule
    // liste der MapEditoren durchgehen und  entsprechend den Eintraegen auf
    //  jedem Host einen Renderer mit entsprechendem Status starten

    // create the info_str with 'name \n instanz_nr \n'
    string info_str = name + "\n" + instanz + "\n";

    this->create_netlink(name, instanz, host, CTRLGlobal::getInstance()->netList);

    link_type(name, host, CTRLGlobal::getInstance()->moduleList);
    module *mod = get_type();

    rhost *rh = CTRLGlobal::getInstance()->hostList->get(host);

    userinterface *p_ui = CTRLGlobal::getInstance()->userinterfaceList->get(host);
    if (copy == 4) // MIRROR
    {
        mirror_node->set_mirror_status(ORG_MIRR); //original
        mirror_node->set_mirror_node(this);
        if (p_ui)
            ((render_module *)mirror_node)->remove_display(rh);

        set_mirror_status(CPY_MIRR); //mirror
        set_mirror_node(mirror_node);
    }

    int ret = displays->init(mod->get_exec(), info_str, mod, copy, flags, rh);
    if (!ret)
    {
        return 0;
    }

    // copy interfaces from module
    mod->reset_intflist();
    while (mod->next_interface() != 0)
    {
        set_interface(mod->get_interfname(),
                      mod->get_interftype(),
                      mod->get_interftext(),
                      mod->get_interfdirection(),
                      mod->get_interfdemand());

        // generate data objectname for output port (former DOCONN)
        int outNum = 0;
        if (mod->get_interfdirection() == "output")
        {
            ostringstream os;
            os << name << "_" << instanz << "(" << nodeid << ")_OUT_" << outNum << "_";
            string objName = os.str();
            string outputName = mod->get_interfname();
            object *obj = CTRLGlobal::getInstance()->objectList->select(objName);
            CTRLGlobal::getInstance()->netList->set_DO_conn(name, instanz, host, outputName, obj);
            outNum++;
        }
    }
    // copy parameters from module
    parameter *tmp;
    mod->reset_paramlist("in");
    while ((tmp = mod->next_param("in")) != 0)
    {
        set_parameter(tmp, "in");
    }
    mod->reset_paramlist("out");
    while ((tmp = mod->next_param("out")) != 0)
    {
        set_parameter(tmp, "out");
    }

    return mod->get_counter();
}

void render_module::del(int already_dead)
{


    // alle interfaces durchgehen (nur input)
    C_interface *tmp_intf;
    interfaces->reset();
    while ((tmp_intf = interfaces->next()) != NULL) // get interface
    {

        string direction = tmp_intf->get_direction();
        if (direction == "input")
        {
            render_interface *rinterface = (render_interface *)tmp_intf;

            // delete all connections
            rinterface->del_all_connections(this);
            interfaces->remove(rinterface);
            interfaces->reset();
        }
        else if (direction == "output")
        {

            net_interface *ninterface = (net_interface *)tmp_intf;

            object *obj = ninterface->get_object();
            // delete the object in the shm
            // if itsn't a NEW | OPEN
            if (already_dead >= 0)
                obj->del_old_data();

            // del all To-Connections in the Object and in the corresponding
            // modules
            obj->del_allto_connects();

            // delete the from-Connection in the object
            obj->del_from_connection();

            // delete local connection
            ninterface->del_connect();

            // delete Object (without any connection)
            CTRLGlobal::getInstance()->objectList->remove(obj);
            CTRLGlobal::getInstance()->objectList->reset();

            interfaces->remove(ninterface);
            interfaces->reset();

        } // strcmp

    } // while

    // alle Control-Connections durchgehen
    // prev-Connection
    net_control *controllerconn;
    prev_control->reset();
    while ((controllerconn = prev_control->next()) != NULL)
    {
        // get connected module
        net_module *mod = controllerconn->get();
        // remove Connection in Module
        mod->del_C_conn(this, "to");
        // remove Connection
        prev_control->remove(controllerconn);
        prev_control->reset();
    }

    // next-Connection
    next_control->reset();
    while ((controllerconn = next_control->next()) != NULL)
    {
        // get connected module
        net_module *mod = controllerconn->get();
        // remove Connection in Module
        mod->del_C_conn(this, "from");
        // remove Connection
        prev_control->remove(controllerconn);
        prev_control->reset();
    }

    // alle Parameter-connections durchgehen
    // input-Parameter
    net_parameter *tmp_par;
    par_in->reset();
    while ((tmp_par = par_in->next()) != NULL)
    {
        // del connections
        connect_mod_list *conn_list = tmp_par->get_connectlist();
        connect_mod *tmp_conn;
        conn_list->reset();
        while ((tmp_conn = conn_list->next()) != NULL)
        {
            // del the connection in the connected module
            net_module *mod = tmp_conn->get_mod();
            mod->del_P_conn(tmp_conn->get_par(), this, tmp_par->get_name(), "out");
            conn_list->remove(tmp_conn);
            conn_list->reset();
        }
        par_in->remove(tmp_par);
        par_in->reset();
    }

    // output-Parameter
    par_out->reset();
    while ((tmp_par = par_out->next()) != NULL)
    {
        // del connections
        connect_mod_list *conn_list = tmp_par->get_connectlist();
        connect_mod *tmp_conn;
        conn_list->reset();
        while ((tmp_conn = conn_list->next()) != NULL)
        {
            // del the connection in the connected module
            net_module *mod = tmp_conn->get_mod();
            mod->del_P_conn(tmp_conn->get_par(), this, tmp_par->get_name(), "in");
            conn_list->remove(tmp_conn);
            conn_list->reset();
        }
        par_out->remove(tmp_par);
        par_out->reset();
    }

    // send the QUIT message to the modules
    if (already_dead <= 0)
        displays->quit();
}

void render_module::set_O_conn(const string &from_intf, object *obj)
{
    C_interface *tmp = interfaces->get(from_intf);
    if (tmp)
    {
        if (tmp->get_direction() == "input")
        {
            ((render_interface *)tmp)->set_connect(obj);
        }
        else
        {
            ((net_interface *)tmp)->set_connect(obj);
        }
    }
}

void render_module::del_O_conn(const string &output_name, object *obj)
{
    C_interface *tmp;

    if ((tmp = interfaces->get(output_name)) != NULL) // get local interface
    {
        if (tmp->get_direction() == "input")
        {
            // delete connection for obj
            ((render_interface *)tmp)->del_connect(obj, displays);
        }
        else
            ((net_interface *)tmp)->del_connect();
    }
}

int render_module::get_mod_id()
{
    display *tmp_dis;

    displays->reset();
    while ((tmp_dis = displays->next()) != NULL)
    {
        if (this->host == tmp_dis->get_hostname())
        {
            return tmp_dis->get_mod_id();
        }
    }
    return -1;
}

void render_module::send_del(const string &name)
{
    if (!is_alive())
        return;
    displays->send_del(name, "");
    displays->reset_ready();
}

void render_module::send_add_obj(const string &name, void *connection)
{
    if (!name.empty())
    {
        displays->send_add(name);
        displays->reset_ready();
        ((obj_conn *)connection)->set_old_name(name);
    }
}

void render_module::send_add(ui_list *ul, object *obj, void *connection)
{

    displays->reset_ready();

    // setze Status =  RUNNING
    this->set_status(MODULE_RUNNING);
    inc_running();

    // Namen des alten DO lesen
    string old_name = ((obj_conn *)connection)->get_old_name();

    // Namen des neuen DO lesen
    string DO_name = obj->get_current_name();

    // existent and not empty
    if (!old_name.empty())
        displays->send_del(old_name, DO_name);

    else
        displays->send_add(DO_name);

    // neuen Namen als alten schreiben
    ((obj_conn *)connection)->set_old_name(DO_name);

    string content = this->get_inparaobj();
    if (!content.empty())
    {
        Message *msg = new Message(COVISE_MESSAGE_START, content);
        ul->send_all(msg);
        delete msg;
    }
}

void render_module::start_module(ui_list *ul)
{

    // setze Status =  RUNNING
    this->set_status(MODULE_RUNNING);
    if (m_errors)
        empty_errlist(); // reset error list of the module

    // sende erstes DO mit New an die renderer
    bool found = false;
    connect_obj *connection = NULL;
    C_interface *tmp_intf;
    interfaces->reset();
    while (found == false)
    {
        tmp_intf = interfaces->next();
        string direction = tmp_intf->get_direction();

        if (direction == "input")
        {
            int state = ((render_interface *)tmp_intf)->get_state(this);
            if (state == S_NEW)
            {
                found = true;
                // ((render_interface *)tmp_intf)->reset_wait();
                connection = ((render_interface *)tmp_intf)->get_first_NEW(this);
            } // if state
        } // if direction
    } // while

    // get new Dataobject-name
    object *tmp_obj = connection->get_obj();

    // Namen des alten DO lesen
    string old_name = connection->get_oldname();

    // Namen des neuen DO lesen
    string DO_name = tmp_obj->get_current_name();

    if (!old_name.empty()) /// existent and not empty
        displays->send_del(old_name, DO_name);

    else
    {
        // send name to the displays
        displays->send_add(DO_name);
    }

    // neuen Namen als alten schreiben
    connection->set_oldname(DO_name);

    // decrement wait-counter
    ((render_interface *)tmp_intf)->decr_wait();

    // setze DO auf OLD
    tmp_obj->change_NEW_to_OLD((net_module *)this, tmp_intf->get_name());

    // setze DISPLAY_ALL auf false
    DISPLAY_ALL = false;

    // setze DISPLAY_READY in den Renderen auf false
    displays->set_DISPLAY_FALSE();
    displays->reset_ready();

    string content = this->get_inparaobj();
    if (!content.empty())
    {
        Message *msg = new Message(COVISE_MESSAGE_START, content);
        ul->send_all(msg);
        delete msg;
    }
}

int render_module::update(int sender, ui_list *uilist)
{
#ifdef DEBUG
    fprintf(msg_prot, "---------------------------------------------------\n");
    fprintf(msg_prot, "FINISHED recv from\n%i \n", sender);
    fflush(msg_prot);
#endif

    // suche das rendermodule mit mod_id == sender
    display *tmp_dis = displays->get(sender);
    // feststellen, ob das FINISHED fuer ein DEL oder ein ADD kam
    //NEXT_DEL = tmp_dis->get_NEXT_DEL();

    // last FINISHED was a ADD

    tmp_dis->set_DISPLAY(true);

    displays->incr_ready();

    CTRLHandler::instance()->m_numRunning--;
    CTRLHandler::instance()->m_numRendererRunning--;

#ifdef DEBUG
    fprintf(stderr, "get_ready: %d\n", displays->get_ready());
#endif

    // alle rendermodule fertig ?
    if ((displays->get_ready()) == 0)
    {
        // nein: status auf FINISHED setzen und Werte initialisieren
        status = MODULE_IDLE;
        displays->set_DISPLAY_FALSE();

        // send finish-message to the UIFs
        stringstream os;
        os << name << "\n" << nr << "\n" << host << "\n" << 0 << "\n" << 0 << "\n";

        Message *msg = new Message(COVISE_MESSAGE_FINISHED, os.str());
        uilist->send_all(msg);
        delete msg;

        return (1);
    }
    return (0);
}

void render_module::set_renderstatus(ui_list *uilist)
{

    // userinterfacelist durchgehen
    userinterface *tmp_ui;
    uilist->reset();
    while ((tmp_ui = uilist->next()) != NULL)
    {

        // hostname und user lesen
        string host = tmp_ui->get_host();

        string user = tmp_ui->get_userid();
        string ui_status = tmp_ui->get_status();

        // entsprechendes display holen
        display *tmp_dis = displays->get(host, user);

        if (tmp_dis != NULL)
        {
            // status von userinterface kopieren
            tmp_dis->set_execstat(ui_status);

            // status verschicken

            // create the info_str with 'name \n instanz_nr \n host \n'
            ostringstream info_str;
            info_str << this->get_name() << "\n" << this->get_nr() << "\n" << host << "\n";
            tmp_dis->send_status(info_str.str());
        }
    }
}

void render_module::send_msg(Message *msg)
{
    displays->reset();
    while (display *tmp_dis = displays->next())
    {
        int dis_id = tmp_dis->get_mod_id();
        if (dis_id != msg->sender)
        {
            tmp_dis->send_message(msg);
        }
    }
}

bool render_module::test_id(int sender)
{
    display *tmp_dis;
    bool found = false;

    displays->reset();
    while ((tmp_dis = displays->next()) != NULL)
    {
        int dis_id = tmp_dis->get_mod_id();
        if (dis_id == sender)
        {
            found = true;
            break;
        }
    }

    return found;
}

bool render_module::is_mirror_of(int sender)
{

    if (get_mod_id() == sender)
        return false;

    switch (get_mirror_status())
    {

    case NOT_MIRR:
        return false;

    case ORG_MIRR:
    {
        render_module *tmp_mod;
        reset_mirror_list();
        while ((tmp_mod = (render_module *)mirror_list_next()) != NULL)
        {
            if (tmp_mod->get_mod_id() == sender)
                return true;
        }
        break;
    }
    case CPY_MIRR:
    {
        render_module *tmp_mod;
        reset_mirror_list();
        render_module *org = (render_module *)mirror_list_next();
        if (org->get_mod_id() == sender)
            return true;

        org->reset_mirror_list();
        while ((tmp_mod = (render_module *)org->mirror_list_next()) != NULL)
        {
            if (tmp_mod->get_mod_id() == sender)
                return true;
        }
        break;
    }
    default:
    {
        break;
    }
    }

    return false;
}

void render_module::count_init()
{
    // count the connections with INIT-Dataobjects
    // problem: INIT means sometimes, that there is no connected Object
    C_interface *intf;
    interfaces->reset();
    while ((intf = interfaces->next()) != NULL)
    {
        string direction = intf->get_direction();
        if (direction == "input")
        {
            ((render_interface *)intf)->count_init(this);
        }
    }
}

void render_module::reset_wait()
{
    C_interface *intf;
    interfaces->reset();
    while ((intf = interfaces->next()) != NULL)
    {
        string direction = intf->get_direction();
        if (direction == "input")
            ((render_interface *)intf)->reset_wait();
    }
}

bool render_module::add_display(userinterface *ui)
{

    bool exec = false;

    display *tmp_dis = new display;
    string tmp_hostname = ui->get_host();
    string tmp_userid = ui->get_userid();
    string tmp_passwd = ui->get_passwd();
    string tmp_status = ui->get_status();

    tmp_dis->set_hostname(tmp_hostname);
    tmp_dis->set_userid(tmp_userid);
    tmp_dis->set_passwd(tmp_passwd);
    tmp_dis->set_execstat(tmp_status);

    DM_data *tmp_data = CTRLGlobal::getInstance()->dataManagerList->get(tmp_hostname);
    AppModule *dmod = tmp_data->get_DM();

    // add host to the info_str
    ostringstream tmp_info;
    tmp_info << name << "\n" << nr << "\n" << tmp_hostname << "\n";

    int ret = tmp_dis->start(dmod, tmp_info.str(), typ, "", Start::Normal); // start und send status-message
    if (!ret)
    {
        delete tmp_dis;
        return false;
    }

    displays->add(tmp_dis);
    displays->incr_count();

    render_interface *p_intf;
    connect_obj *p_co;
    interfaces->reset();
    while ((p_intf = (render_interface *)interfaces->next()) != NULL)
    {
        connect_obj_list *conn_list = p_intf->get_connects();
        conn_list->reset();
        while ((p_co = conn_list->next()) != NULL)
        {
            object *p_obj;
            string DO_name;
            p_obj = p_co->get_obj();
            if ((p_obj != NULL)
                && (!p_obj->isEmpty()))
            {
                DO_name = p_obj->get_current_name();
                displays->decr_ready(); //reset_ready();

                // setze Status =  RUNNING
                this->set_status(MODULE_RUNNING);
                inc_running();
                tmp_dis->send_add(DO_name);
                exec = true;
            }
        }
    }

    if (exec)
    {
        string content = this->get_inparaobj();
        if (!content.empty())
        {
            Message *msg = new Message(COVISE_MESSAGE_START, content);
            CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);
            delete msg;
        }
    }

    return exec;
}

void render_module::remove_display(rhost *host)
{
    display *tmp_disp = displays->get(host->get_hostname(), host->get_user());
    if (tmp_disp)
        remove_display(tmp_disp);

    return;
}

void render_module::remove_display(int id)
{
    display *tmp_disp = displays->get(id);
    if (tmp_disp)
        remove_display(tmp_disp);

    return;
}

void render_module::remove_display(display *disp)
{

    if (!disp)
        return;

    ConnectionList *con_list = CTRLGlobal::getInstance()->controller->getConnectionList();
    Connection *p_conn = (disp->get_mod())->get_conn();
    con_list->remove(p_conn);
    disp->quit();
    displays->remove(disp);
    displays->decr_count();
}

int render_module::get_count()
{
    return displays->get_count();
}

//**********************************************************************
//
// 			DOBJECT
//
//**********************************************************************

dobject::dobject()
{
}

void dobject::set_name(const string &str)
{
    name = str;
}

void dobject::set_mark(int im)
{
    mark = im;
}

void dobject::clear_mark()
{

    mark = 0;
}

//**********************************************************************
//
// 			DO_LIST
//
//**********************************************************************

do_list::do_list()
    : Liste<dobject>()
{
}

dobject *do_list::get(const string &str)
{
    dobject *tmp;
    string tmp_name;

    this->reset();
    do
    {
        tmp = this->next();
        if (tmp == NULL)
            print_exit(__LINE__, __FILE__, 1);
        tmp_name = tmp->get_name();

    } while (tmp_name != str);

    return tmp;
}

//**********************************************************************
//
// 			NET_MODULE_LIST
//
//**********************************************************************

net_module_list::net_module_list()
    : Liste<net_module>(1)
{
}

/// start a CRB for a helper
int render_module::addHelperCRB(const string &helperHost, const string &host)
{

    DM_data *helperDmgr = CTRLGlobal::getInstance()->dataManagerList->get(helperHost);
    DM_data *moduleDmgr = CTRLGlobal::getInstance()->dataManagerList->get(host);

    string user = moduleDmgr->get_user();
    string passwd = moduleDmgr->get_passwd();

    if (helperDmgr == NULL) // no crb on helper host
    {
        // start crb
        int exec_type = CTRLHandler::instance()->Config->getexectype(helperHost.c_str());
        coHostType htype(CO_PARTNER);
        if (CTRLGlobal::getInstance()->dataManagerList->add_crb(exec_type, helperHost, user, passwd, "", htype) == 0)
        {
            return (0);
        }

        // add new host in hostlist
        rhost *tmp_host = new rhost;
        tmp_host->set_hostname(helperHost);
        tmp_host->set_user(user);
        tmp_host->set_passwd(passwd);
        CTRLGlobal::getInstance()->hostList->add(tmp_host);
    }
    else
    {
        if (coCoviseConfig::isOn("COVER.MultiPC.Debug", false))
            fprintf(stderr, "there is already a CRB @ [%s] - helper CRB not started\n", helperHost.c_str());
    }
    return 1;
}

int net_module_list::init(int nodeid, const string &name, const string &instanz, const string &host,
                          int posx, int posy, int copy, enum Start::Flags flags, net_module *mirror)
{

    // check the Category of the Module
    module *mod = CTRLGlobal::getInstance()->moduleList->get(name, host);
    if (mod == NULL)
        return 0;

    string category = mod->get_category();

    // use only a new instanz number if current is -1
    istringstream is(instanz);
    int nr, count;
    is >> nr;
    if (nr != -1)
    {
        mod->set_counter(nr);
        count = nr;
    }
    else
    {
        count = mod->get_counter();
    }

    ostringstream os;
    os << count;

    int ret = 0;
    if (category == "Renderer")
    {
        render_module *tmp_rend = new render_module;
        this->add(tmp_rend);

        // set initial values
        ret = tmp_rend->init(nodeid, name, os.str(), host, posx, posy, copy, flags, mirror);
    }

    else
    {
        // create new net_module and add it to the list
        net_module *tmp_mod = new net_module;
        this->add(tmp_mod);

        // set initial values
        ret = tmp_mod->init(nodeid, name, os.str(), host, posx, posy, copy, flags, mirror);
    }

    if (ret == 0)
    {
        re_move(name, instanz, host, 1);
        return 0;
    }

    else
        return count;
}

void net_module_list::re_move(const string &from_name, const string &from_nr, const string &from_host, int already_dead)
{
    re_move(get(from_name, from_nr, from_host), already_dead);
}

void net_module_list::re_move(int nodeid, int already_dead)
{
    re_move(get(nodeid), already_dead);
}

void net_module_list::re_move(net_module *mod, int already_dead)
{
    if (mod == NULL)
        return;

    ConnectionList *con_list = CTRLGlobal::getInstance()->controller->getConnectionList();
    bool first_run = true;
    net_module *p_mirror;
    mod->reset_mirror_list();
    while ((p_mirror = mod->mirror_list_next()) != NULL || first_run)
    {
        first_run = false;

        int mod_type = mod->is_renderer();
        switch (mod_type)
        {
        case NET_MOD:
        {
            if (mod->get_applmod())
            {
                Connection *p_conn = (mod->get_applmod())->get_conn();
                con_list->remove(p_conn);
            }
            int status = mod->get_mirror_status();
            if ((status == ORG_MIRR)
                || (status == CPY_MIRR))
            {
                p_mirror->set_mirror_status(NOT_MIRR);
                p_mirror->set_mirror_node(NULL);
            }
            break;
        }
        case REND_MOD:
        {
            render_module *p_rend = (render_module *)mod;
            displaylist *p_displist = p_rend->get_displays();
            if (p_displist)
            {
                display *p_disp;
                p_displist->reset();
                while ((p_disp = p_displist->next()) != NULL)
                {
                    if (p_disp->get_mod())
                    {
                        Connection *p_conn = (p_disp->get_mod())->get_conn();
						CTRLHandler::instance()->removeVrbConnection(p_conn);
                        con_list->remove(p_conn);
                    }
                }
            }

            switch (mod->get_mirror_status())
            {
            case ORG_MIRR: //original
            {
                p_rend = (render_module *)p_mirror;
                p_rend->set_mirror_status(NOT_MIRR);
                p_rend->set_mirror_node(NULL);
                break;
            }

            case CPY_MIRR: //mirror
            {
                p_rend = (render_module *)p_mirror;
                p_rend->set_mirror_status(NOT_MIRR);
                p_rend->set_mirror_node(NULL);
                string tmp_host = mod->get_host();
                userinterface *p_ui = CTRLGlobal::getInstance()->userinterfaceList->get(tmp_host);
                if (p_ui)
                {
                    int upd = p_rend->add_display(p_ui);
                    CTRLGlobal::getInstance()->userinterfaceList->set_slaveUpdate(upd);
                }
                break;
            }

            default:
            {
                break;
            }
            }
            break;
        }

        default:
            cerr << "\nController ERROR: wrong module type !!!\n";

        } // switch mod_type
    }

    int save_nbr = CTRLHandler::instance()->m_numRendererRunning; // deleting conn = > incr of nbr
    int save_nb = CTRLHandler::instance()->m_numRunning;

    mod->del(already_dead);

    //correction of CTRLHandler::instance()->m_numRunning & CTRLHandler::instance()->m_numRendererRunning
    if ((save_nbr != CTRLHandler::instance()->m_numRendererRunning) && (mod->is_renderer()))
    {
        CTRLHandler::instance()->m_numRendererRunning = save_nbr;
        CTRLHandler::instance()->m_numRunning = save_nb;
    }

    this->remove(mod);
}

void net_module_list::move(int nodeid, int posx, int posy)
{
    net_module *tmp = get(nodeid);
    if (NULL == tmp)
        return;

    tmp->move(posx, posy);
}

bool net_module_list::mirror(net_module *from_mod, const string &new_host)
{

    string name = from_mod->get_name();
    char nr[64];
    sprintf(nr, "%d", atoi(from_mod->get_nr().c_str()) + 1000 * (from_mod->get_num_mirrors() + 1));

    int posX = from_mod->get_x_pos() + (from_mod->get_num_mirrors() + 1) * 400;
    int posY = from_mod->get_y_pos();

    CTRLGlobal::getInstance()->s_nodeID++;
    string nrr(nr);
    int iret = init(CTRLGlobal::getInstance()->s_nodeID, from_mod->get_name(), nrr, new_host, posX + (from_mod->get_num_mirrors() + 1) * 400, posY, 4, Start::Normal, from_mod);
    if (iret)
    {
        //--------------------------//
        // get moduledescr          //
        //--------------------------//
        module *mod = (get(name, nrr, new_host))->get_type();

        ostringstream oss;
        oss << "COPY2\n" << name << "\n" << nr << "\n" << new_host << "\n" << posX << "\n" << posY << "\n";
        oss << from_mod->get_name() << "\n" << from_mod->get_nr() << "\n" << from_mod->get_host() << "\n";
        Message *msg = new Message(COVISE_MESSAGE_UI, oss.str());
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);
        delete msg;

        string text = "DESC\n" + mod->create_descr();
        msg = new Message(COVISE_MESSAGE_UI, text);
        CTRLGlobal::getInstance()->userinterfaceList->send_all(msg);
        delete msg;
        return true;
    }

    else
    {
        ostringstream os;
        os << "Failing to start " << name << "_" << nr << "@" << new_host << "!!!";
        CTRLGlobal::getInstance()->userinterfaceList->sendError(os.str());
        return false;
    }
}

void net_module_list::mirror_all(const string &new_host)
{
    net_module *mod;

    net_module **list = new net_module *[this->get_nbList()];
    int i, num = 0;
    this->reset();
    while ((mod = this->next()))
    {
        list[num++] = mod;
    }

    for (i = 0; i < num; i++)
    {
        if (!mirror(list[i], new_host))
            return;
    }
    delete[] list;
}

net_module *net_module_list::get(const string &from_name, const string &from_nr, const string &from_host)
{
    this->reset();
    net_module *tmp = this->next();

    while (tmp != NULL)
    {
        string tmp_nr = tmp->get_nr();
        string tmp_name = tmp->get_name();
        string tmp_host = tmp->get_host();
        if (tmp_name == from_name && tmp_nr == from_nr && tmp_host == from_host)
        {
            break;
        }
        tmp = this->next();
    }
    return tmp;
}

net_module *net_module_list::get(int nodeid)
{
    this->reset();
    while (1)
    {
        net_module *tmp = this->next();
        if (tmp == NULL)
        {
            // set COVISE_DEBUG_ALL 2 or more to receive this message
            char buf[256];
            sprintf(buf, "netlink_list::get did not find module with id %d", nodeid);
            print_comment(__LINE__, __FILE__, buf, 1);
            break;
        }

        if (nodeid == tmp->get_nodeid())
            return tmp;
    }
    return NULL;
}

int net_module_list::getID(const string &name, const string &nr, const string &host)
{
    this->reset();
    net_module *tmp = this->next();

    while (tmp != NULL)
    {
        string tmp_nr = tmp->get_nr();
        string tmp_name = tmp->get_name();
        string tmp_host = tmp->get_host();
        if (tmp_name == name && tmp_nr == nr && tmp_host == host)
        {
            break;
        }
        tmp = this->next();
    }
    if (tmp != NULL)
        return tmp->get_nodeid();

    else
        return -1;
}

net_module *net_module_list::get(const string &name, const string &nr)
{
    bool ende = false;
    net_module *tmp;
    this->reset();
    do
    {
        // hole naechstes module
        tmp = this->next();

        // ist module ==  NULL Pointer?
        if (!tmp)
            ende = true;

        else
        {
            // nein:
            // ist aktuelles module =  gesuchtes ?
            string tmp_nr = tmp->get_nr();
            string tmp_name = tmp->get_name();
            if (tmp_nr == nr && tmp_name == name)
            {
                // ja -> ende = true
                ende = true;
            }
        } // if tmp

    } while (ende == false);

    return tmp;
}

net_module *net_module_list::get_mod(int sender)
{
    net_module *tmp;
    int applid = 0;

    this->reset();
    do
    {
        tmp = this->next();
        if (tmp == NULL)
        {
            // set COVISE_DEBUG_ALL 2 or more to receive this message
            char buf[256];
            sprintf(buf, "net_module_list::get_mod get did not find sender %d", sender);
            print_comment(__LINE__, __FILE__, buf, 1);
            break;
        }
        if (tmp->is_renderer() == NET_MOD)
            applid = tmp->get_mod_id();

        else if (tmp->is_renderer() == REND_MOD)
        {
            render_module *p_rend = (render_module *)tmp;
            if (p_rend->get_display(sender) != NULL)
                break;
        }

    } while (sender != applid);

    return tmp;
}

void net_module_list::set_DO_conn(const string &from_name, const string &from_nr, const string &from_host, const string &output_name,
                                  object *obj)
{

    int check_mod = -1, check_obj = -1;

    net_module *from = get(from_name, from_nr, from_host);

    // check_O_conn returns 1 if the Connection to the Object exists
    // checK_O_conn returns 0 if the Connection not exists

    if (from)
    {
        check_mod = from->check_O_conn(output_name);
        check_obj = obj->check_from_connection(from, output_name);
    }

    if ((check_mod == 0) && (check_obj == 0))
    {

        from->set_O_conn(output_name, obj);
        string type = from->get_intf_type(output_name);
        obj->connect_from(from, output_name, type);
    }

    else if (((check_mod == 0) && (check_obj == 1)) || ((check_mod == 1) && (check_obj == 0)))
    {
        print_comment(__LINE__, __FILE__, " ERROR: Connection between object and module destroyed!\n");
    }

    else if ((check_mod == 1) && (check_obj == 1))
    {
        // connection exists
    }

    else
    {
        print_comment(__LINE__, __FILE__, " ERROR: Don't ask me what happend!\n");
    }
}

void net_module_list::del_DO_conn(const string &from_name, const string &from_nr, const string &from_host, const string &output_name,
                                  object *obj)
{
    net_module *from = get(from_name, from_nr, from_host);
    if (from)
    {
        from->del_O_conn(output_name, obj);
        obj->del_from_connection();
    }
}

void *net_module_list::set_DI_conn(const string &from_name, const string &from_nr,
                                   const string &from_host, const string &output_name, object *obj)
{
    obj_conn *connection = NULL;

    net_module *from = get(from_name, from_nr, from_host);
    if (from)
    {
        C_interface *intf = from->get_interfacelist()->get(output_name);
        if (intf)
        {
            net_interface *netintf = dynamic_cast<net_interface *>(intf);
            if (!netintf || !netintf->get_conn_state())
            {
                from->set_O_conn(output_name, obj);
                connection = obj->connect_to(from, output_name);
            }
            else
            {
                fprintf(stderr, "attempted duplicate connection to %s_%s:%s\n",
                        from_name.c_str(), from_nr.c_str(), output_name.c_str());
            }
        }
        else
        {
            fprintf(stderr, "attempted connection to non-existing port %s_%s:%s\n",
                    from_name.c_str(), from_nr.c_str(), output_name.c_str());
        }
    }
    return (connection);
}

void net_module_list::del_DI_conn(const string &from_name, const string &from_nr, const string &from_host,
                                  const string &output_name, object *obj)
{
    net_module *from = get(from_name, from_nr, from_host);
    if (from)
    {
        from->del_O_conn(output_name, obj);
        obj->del_to_connection(from_name, from_nr, from_host, output_name);
    }
}

void net_module_list::set_C_conn(const string &from_name, const string &from_nr, const string &from_host,
                                 const string &to_name, const string &to_nr, const string &to_host)
{
    net_module *from = get(from_name, from_nr, from_host);
    net_module *to = get(to_name, to_nr, to_host);
    if ((from) && (to))
    {
        from->set_C_conn(to, "to");
        to->set_C_conn(from, "from");
    }
}

void net_module_list::del_C_conn(const string &from_name, const string &from_nr, const string &from_host,
                                 const string &to_name, const string &to_nr, const string &to_host)
{
    net_module *from = get(from_name, from_nr, from_host);
    net_module *to = get(to_name, to_nr, to_host);
    if ((from) && (to))
    {
        from->del_C_conn(to, "to");
        to->del_C_conn(from, "from");
    }
}

void net_module_list::set_P_conn(const string &from_name, const string &from_nr, const string &from_host, const string &output_par,
                                 const string &to_name, const string &to_nr, const string &to_host, const string &input_par)
{
    net_module *from = get(from_name, from_nr, from_host);
    net_module *to = get(to_name, to_nr, to_host);
    if ((from) && (to))
    {
        from->set_P_conn(output_par, to, input_par, "out");
        to->set_P_conn(input_par, from, output_par, "in");
    }
}

void net_module_list::del_P_conn(const string &from_name, const string &from_nr, const string &from_host, const string &output_par,
                                 const string &to_name, const string &to_nr, const string &to_host, const string &input_par)
{

    net_module *from = get(from_name, from_nr, from_host);
    net_module *to = get(to_name, to_nr, to_host);
    if ((from) && (to))
    {
        from->del_P_conn(output_par, to, input_par, "out");
        to->del_P_conn(input_par, from, output_par, "in");
    }
}

void net_module_list::change_param(const string &name, const string &nr, const string &host, const string &param_name, const string &value, int i)
{
    net_module *mod = get(name, nr, host);
    if (mod)
        mod->change_param(param_name, value, i, 1);
}

void net_module_list::change_param(const string &name, const string &nr, const string &host, const string &param_name, const string &value_list)
{
    net_module *mod = get(name, nr, host);
    if (mod)
        mod->change_param(param_name, value_list);
}

void net_module_list::add_param(const string &name, const string &nr, const string &host, const string &param_name, const string &add_param)
{
    net_module *mod = get(name, nr, host);
    if (mod)
        mod->add_param(param_name, add_param);
}

void net_module_list::set_renderstatus(ui_list *uilist)
{

    net_module *tmp_mod;
    this->reset();
    while ((tmp_mod = this->next()) != NULL)
    {
        module *type = tmp_mod->get_type();
        string category = type->get_category();
        if (category == "Renderer")
            ((render_module *)tmp_mod)->set_renderstatus(uilist);
    }
}

void net_module_list::send_renderer(Message *msg)
{
    net_module *tmp_mod;
    this->reset();
    while ((tmp_mod = this->next()) != NULL)
    {
        module *type = tmp_mod->get_type();
        string category = type->get_category();
        if (category == "Renderer")
        {

            // sende an die entsprechenden partnermodule
            bool test = ((render_module *)tmp_mod)->test_id(msg->sender);

            if (test == true ||
                // send every message to a mirrored                                                                                                                       //renderer
                ((render_module *)tmp_mod)->is_mirror_of(msg->sender))
            {
                ((render_module *)tmp_mod)->send_msg(msg);
            }
            else // sending the VRML msg to VRMLRenderer
            {
                if ((strncmp(msg->data.data(), "VRML", 4) == 0) && tmp_mod->get_name() == "VRMLRenderer")
                {
                    ((render_module *)tmp_mod)->send_msg(msg);
                }
                else if (strncmp(msg->data.data(), "GRMSG", 5) == 0)
                {
                    ((render_module *)tmp_mod)->send_msg(msg);
                }
            }
        }
    }
}

/// send generic information( INFO, WARNING, ERROR) to all renderers
void net_module_list::send_gen_info_renderer(Message* msg)
{
    Message* rmsg = new Message(*msg);

    DataHandle new_data{(size_t)(msg->data.length() + 10)};
    switch (rmsg->type)
    {
    case COVISE_MESSAGE_INFO:
    {
        sprintf(new_data.accessData(), "INFO\n%s", msg->data.data());
        break;
    }

    case COVISE_MESSAGE_WARNING:
    {
        sprintf(new_data.accessData(), "WARNING\n%s", msg->data.data());
        break;
    }

    case COVISE_MESSAGE_COVISE_ERROR:
    {
        sprintf(new_data.accessData(), "ERROR\n%s", msg->data.data());
        break;
    }
    default:
    {
        break;
    }
    }

    new_data.setLength((int)strlen(new_data.data()) + 1);
    rmsg->data = new_data;
    rmsg->type = COVISE_MESSAGE_COVISE_ERROR;

    this->send_all_renderer(rmsg);
}

void net_module_list::send_all_renderer(Message *msg)
{
    if (this->isEmpty())
        return;

    ListeIter<net_module> iter(this);
    while (iter)
    {
        net_module *mod = iter();
        module *type = mod->get_type();
        string category = type->get_category();

        if (category == "Renderer")
            ((render_module *)mod)->send_msg(msg);

        ++iter;
    }
}

int net_module_list::update(int sender, ui_list *uilist)
{
    net_module *tmp_mod;
    this->reset();
    while ((tmp_mod = this->next()) != NULL)
    {
        module *type = tmp_mod->get_type();
        string category = type->get_category();
        if (category == "Renderer")
        {
            // search for the render-module with pid = sender
            bool test = ((render_module *)tmp_mod)->test_id(sender);
            if (test == true)
                return (((render_module *)tmp_mod)->update(sender, uilist));
        }
    }
    return (1);
}

void net_module_list::reset_wait()
{
    net_module *tmp;
    this->reset();
    while ((tmp = this->next()) != NULL)
    {
        module *type = tmp->get_type();
        string category = type->get_category();
        if (category == "Renderer")
            ((render_module *)tmp)->reset_wait();
    }
}

string net_module_list::get_modules(const string &local_name, const string &local_user, bool forSaving)
{
    int i = 0;
    string buffer;

    this->reset();
    net_module *tmp_mod;
    while ((tmp_mod = this->next()) != NULL)
    {
        bool test = tmp_mod->test_copy();
        if (test != true)
        {
            string data = tmp_mod->get_module(local_name, local_user, forSaving);
            i++;
            buffer = buffer + data;
        }
    }

    if (!buffer.empty())
    {
        ostringstream os;
        os << "#numModules\n";
        os << i;
        buffer = os.str() + "\n" + buffer;
        return buffer;
    }
    else
        return "";
}

#ifdef PARAM_CONN
string net_module_list::get_para_connect(const string &local_name, const string &local_user)
{
    net_module *tmp_mod;
    int i;
    int ges;
    string tmp_str, *para_str = NULL, *para_str2 = NULL;

    ges = 0;

    // loop over all modules
    this->reset();
    while ((tmp_mod = this->next()) != NULL)
    {
        i = 0;
        tmp_str = tmp_mod->get_para_connect(&i, local_name, local_user);

        if (i > 0)
        {
            if (ges > 0)
            {
                para_str2 = append2(para_str, tmp_str, "\n");
                delete[] para_str;
                para_str = para_str2;
            }
            else
            {
                para_str = new char[strlen(tmp_str) + 1];
                strcpy(para_str, tmp_str);
            }
        }
        delete[] tmp_str;

        ges = ges + i;
    }

    // convert *ges to ascii and built param_string
    tmp_str = new char[10];
    itoa(tmp_str, ges);
    if (para_str != NULL)
    {
        para_str2 = append2(tmp_str, para_str, "\n");
        delete[] para_str;
    }
    else
        para_str2 = append2(tmp_str, "", "\n");
    delete[] tmp_str;

    return para_str2;
}
#endif

//!
//! save current network to file
//!
void net_module_list::save_config(const string &filename)
{
    CTRLGlobal *global = CTRLGlobal::getInstance();

    DM_data *dm_local = CTRLGlobal::getInstance()->dataManagerList->get_local();
    string local_name = dm_local->get_hostname();
    string local_user = dm_local->get_user();

    std::filebuf *pbuf = NULL;

    Message err_msg(COVISE_MESSAGE_COVISE_ERROR, std::string("Error saving file " + filename));

#ifdef _WIN32
    ofstream outFile(filename.c_str(), std::ios::binary);
#else
    ofstream outFile(filename.c_str());
#endif
    if (!outFile.good())
    {
        string covisepath = getenv("COVISEDIR");
        if (filename == "UNDO.NET" && !covisepath.empty())
        {
            string filestr = covisepath + "/" + filename;
#ifdef _WIN32
            ofstream outFile2(filestr.c_str(), std::ios::binary);
#else
            ofstream outFile2(filestr.c_str());
#endif
            if (!outFile2.good())
            {
                string tmp = "Error saving file " + filename;
                Message *err_msg = new Message(COVISE_MESSAGE_COVISE_ERROR, tmp);
                std::cerr << "ERROR: " << tmp << std::endl;
                CTRLGlobal::getInstance()->userinterfaceList->send_all(err_msg);
                delete err_msg;
                return;
            }

            else
                pbuf = outFile2.rdbuf();
        }
        else
        {
            std::cerr << "ERROR: Error saving file " + filename << std::endl;
            CTRLGlobal::getInstance()->userinterfaceList->send_all(&err_msg);
            return;
        }
    }

    else
        pbuf = outFile.rdbuf();

    if (filename.length() > 3 && filename.substr(filename.length() - 3, 3) == ".py")
    {
        // write a python script

        outFile << "#" << endl;
        outFile << "# create global net" << endl;
        outFile << "#" << endl;
        outFile << "network = net()" << endl;

        // store all modules
        this->reset();
        net_module *tmp_mod;
        while ((tmp_mod = this->next()) != NULL)
        {
            bool test = tmp_mod->test_copy();
            if (test != true)
            {
                tmp_mod->writeScript(outFile, local_user);
            }
        }

        // store all connections
        outFile << "#" << endl;
        outFile << "# CONNECTIONS" << endl;
        outFile << "#" << endl;
        CTRLGlobal::getInstance()->objectList->writeScript(outFile, local_name, local_user);

        // same ending as python files from map_converter
        outFile << "#" << endl;
        outFile << "# uncomment the following line if you want your script to be executed after loading" << endl;
        outFile << "#" << endl;
        outFile << "#runMap()" << endl;
        outFile << "#" << endl;
        outFile << "# uncomment the following line if you want exit the COVISE-Python interface" << endl;
        outFile << "#" << endl;
        outFile << "#sys.exit()" << endl;
    }

    else // write a normal .net
    {

        // write content
        // get hosts
        outFile << "#" << NET_FILE_VERSION << endl;
        string hdata = CTRLGlobal::getInstance()->hostList->get_hosts(local_name, local_user);

        // get module descrptions
        string buffer;
        string mdata = get_modules(local_name, local_user, true);
        if (!mdata.empty())
        {
            // get connections
            string cdata = CTRLGlobal::getInstance()->objectList->get_connections(local_name, local_user);
            buffer = hdata + mdata + cdata;

#ifdef PARAM_CONN
            // get parameter connections
            str = get_para_connect(local_name, local_user);
            buffer = buffer + str;
#endif
        }

        if (pbuf)
        {
            pbuf->sputn(buffer.c_str(), buffer.length());
            if (outFile.good())
                outFile.close();
        }
    }

    return;
}

char *net_module_list::openNetFile(const string &filename)
{
    char *buffer = NULL;
    std::filebuf *pbuf = NULL;

    // try to read given file or the UNDO>NET file
    // otherwise send an error message

    ifstream inFile(filename.c_str());
    if (!inFile.good())
    {
        string covisepath = getenv("COVISEDIR");
        if (filename == "UNDO.NET" && !covisepath.empty())
        {
            string filestr = covisepath + "/" + filename;
            ifstream inFile2(filestr.c_str(), ifstream::in);
            if (!inFile2.good())
            {
                Message *err_msg = new Message(COVISE_MESSAGE_COVISE_ERROR, "Can't open file " + filename);
                CTRLGlobal::getInstance()->userinterfaceList->send_all(err_msg);
                delete err_msg;
                return NULL;
            }
            else
                pbuf = inFile2.rdbuf();
        }
    }
    else
        pbuf = inFile.rdbuf();

    try
    {
        if (pbuf)
        {
            std::streamsize size = pbuf->pubseekoff(0, ios::end, ios::in);
            pbuf->pubseekpos(0, ios::in);
            buffer = new char[size + 1];
            std::streamsize got = pbuf->sgetn(buffer, size);
            buffer[got] = '\0';
        }
    }
    catch (...)
    {
        std::cerr << "net_module_list::openNetFile: error reading " << filename << std::endl;
        delete[] buffer;
        buffer = NULL;
    }

    if (inFile.good())
        inFile.close();

    return buffer;
}

//!
//! load a network file
//!
bool net_module_list::load_config(const string &filename)
{
    char *buffer = openNetFile(filename);
    // read content

    if (buffer != NULL)
    {
        bool oldFile = true;
        if (buffer[0] == '#')
        {
            //we have a version information
            int version;
            int n = sscanf(buffer + 1, "%d", &version);
            if (n == 1 && version >= NET_FILE_VERSION)
            {
                oldFile = false; // this is a new .net file, convert all other files
            }
        }

        if (oldFile)
        {
            // convert
            string path = filename;
#ifdef WIN32
            for (int i = 0; i < path.length(); i++)
            {
                if (path[i] == '/')
                    path[i] = '\\';
            }
            string name = path;
            for (int i = (int)path.length() - 1; i >= 0; i--)
            {
                if (path[i] == '\\')
                {
                    name = string(path, i + 1, path.length() - i);
                    break;
                }
            }

            std::string command = "map_converter -f -o " + path + ".new " + path;
#else
            std::string command = "map_converter -f -o \"" + path + ".new\" \"" + path + "\"";
#endif
            if (system(command.c_str()) == 0)
            {
#ifdef WIN32
                command = "rename " + path + " " + name + ".bak";
#else
                command = "mv \"" + path + "\" \"" + path + ".bak\"";
#endif
                if (system(command.c_str()) == 0)
                {
#ifdef WIN32
                    command = "rename " + path + ".new " + name;
#else
                    command = "mv \"" + path + ".new\" \"" + path + "\"";
#endif
                    if (system(command.c_str()) == 0)
                    {
                        delete[] buffer;
                        // read again
                        buffer = openNetFile(filename);
                        oldFile = false;

                        Message *tmpmsg = new Message(COVISE_MESSAGE_UI, "CONVERTED_NET_FILE\n" + filename);
                        CTRLGlobal::getInstance()->userinterfaceList->send_all(tmpmsg);
                        delete tmpmsg;
                    }
                }
            }
        }

        if (oldFile)
        {
            // conversion failed
            Message *tmpmsg = new Message(COVISE_MESSAGE_UI, "FAILED_NET_FILE_CONVERSION\n" + filename);
            CTRLGlobal::getInstance()->userinterfaceList->send_all(tmpmsg);
            delete tmpmsg;
        }
        else
        {
            bool ready = CTRLHandler::instance()->recreate(buffer, CTRLHandler::NETWORKMAP);
            delete[] buffer;
            return ready;
        }
    }
    delete[] buffer;
    return false;
}
