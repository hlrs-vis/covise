/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <covise/covise.h>
#include "covise_module.h"
#include <net/covise_connect.h>
#include <covise/covise_msg.h>

#include "CTRLHandler.h"
#include "control_process.h"
#include "control_def.h"
#include "control_define.h"
#include "control_list.h"
#include "control_object.h"
#include "control_netmod.h"

using namespace covise;

extern FILE *msg_prot;
extern string prot_file;

//**********************************************************************
//
// 			OBJ_FROM_CONN
//
//**********************************************************************

obj_from_conn::obj_from_conn()
{
    module = NULL;
    state = "INIT";
}

void obj_from_conn::set_intf(const string &str)
{
    mod_intf = str;
}

void obj_from_conn::set_state(const string &str)
{
    state = str;
}

void obj_from_conn::set_type(const string &str)
{
    type = str;
}

//**********************************************************************
//
// 				DATA
//
//**********************************************************************

data::data()
{
    save = DO_RELEASE;
    count = 0;
}

data::~data()
{
}

void data::set_name(const string &str)
{
    name = str;
}

void data::set_status(const string &str)
{
    status = str;
}

void data::del_data(AppModule *dmod)
{
    this->set_status("DEL");

    Message *msg = new Message(COVISE_MESSAGE_CTRL_DESTROY_OBJECT, name);
    dmod->send(msg);
    delete msg;

#ifdef DEBUG
//	fprintf(msg_prot,"---------------------------------------------------\n");
//	fprintf(msg_prot,"send DEST OBJ\n%i %i \n %s \n",msg->sender,msg->type,msg->data);
//	fflush(msg_prot);
#endif

    msg = new Message;
    dmod->recv_msg(msg);
    switch (msg->type)
    {
    case COVISE_MESSAGE_EMPTY:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    case COVISE_MESSAGE_SOCKET_CLOSED:
        CTRLHandler::instance()->handleClosedMsg(msg);
        break;

    case COVISE_MESSAGE_MSG_OK:
        break;

    default:
        break;
    }

#ifdef DEBUG
//	fprintf(msg_prot,"---------------------------------------------------\n");
//	fprintf(msg_prot,"receive DEST OBJ\n%i %i \n %s \n",msg->sender,msg->type,msg->data);
//	fflush(msg_prot);
#endif

    delete msg;
}

//**********************************************************************
//
// 				DATA_LIST
//
//**********************************************************************

data_list::data_list()
    : Liste<data>()
{
    count = 0;
}

void data_list::inc_count()
{
    count++;
}

void data_list::dec_count()
{
    if (count > 0)
        count--;
}

string data_list::create_new_data(const string &name)
{
    count = count + 1;
    ostringstream os;
    os << count;
    string new_name = name + os.str();

    data *tmp_data = new data; // Save = off
    tmp_data->set_name(new_name);
    tmp_data->set_status("INIT");
    tmp_data->set_count(count);
    this->add(tmp_data);

    ostringstream oss;
    oss << count - 1;
    new_name = name + os.str();
    this->reset();
    while ((tmp_data = this->next()) != NULL)
    {
        if (tmp_data->get_name() == new_name)
        {
            this->remove(tmp_data);
            break;
        }
    }

    return new_name;
}

string data_list::create_new_DO(const string &name)
{
    count = count + 1;
    ostringstream os;
    os << count;
    string new_name = name + os.str();

    data *tmp_data = new data; // Save = off
    tmp_data->set_name(new_name);
    tmp_data->set_status("INIT");
    tmp_data->set_count(count);
    this->add(tmp_data);
    return new_name;
}

void data_list::new_data(const string &name)
{
    count = count + 1;
    data *tmp_data = new data; // Save = off
    tmp_data->set_name(name);
    tmp_data->set_status("INIT");
    tmp_data->set_count(count);
    this->add(tmp_data);
}

void data_list::set_status(const string &str)
{
    this->reset();
    data *tmp = this->next();
    while ((tmp->get_count()) != count)
    {
        tmp = this->next();
    }

    tmp->set_status(str);
}

data *data_list::get(const string &name)
{
    this->reset();
    data *tmp = this->next();
    while (tmp != NULL && tmp->get_name() != name)
    {
        tmp = this->next();
    }

    return tmp;
}

data *data_list::get_new()
{
    this->reset();
    data *tmp = this->next();
    while ((tmp != NULL) && (tmp->get_count() != count))
    {
        tmp = this->next();
    };

    return tmp;
}

//**********************************************************************
//
// 				OBJ_CONN
//
//**********************************************************************

obj_conn::obj_conn()
{
    mod = NULL;
    datalist = new data_list;
}

obj_conn::~obj_conn()
{
    if (!old_name.empty())
        mod->send_del(old_name);
    delete datalist;
}

void obj_conn::set_old_name(const string &str)
{
    old_name = str;
}

void obj_conn::connect_module(net_module *module)
{
    mod = module;
}

net_module *obj_conn::get_mod()
{
    return mod;
}

void obj_conn::set_mod_intf(const string &str)
{
    mod_intf = str;
}

void obj_conn::copy_data(data_list *dl)
{
    data *ref_data;
    dl->reset();
    while ((ref_data = dl->next()) != NULL)
    {
        data *new_data = new data;
        new_data->set_name(ref_data->get_name());
        new_data->set_status(ref_data->get_status());
        new_data->set_count(ref_data->get_count());

        datalist->add(new_data);
    };
    datalist->set_count(dl->get_count());
}

//-----------------------------------------------------------------------
// get_status returns the status of the last data_object in the datalist
//-----------------------------------------------------------------------

string obj_conn::get_status()
{
    string str;

    if (datalist->get_count() == 0)
    {
        str = "INIT";
    }
    else
    {
        datalist->reset();
        data *tmp = datalist->next();
        while ((tmp->get_count()) != datalist->get_count())
        {
            tmp = datalist->next();
        }
        str = tmp->get_status();
    };
    return str;
}

//-----------------------------------------------------------------------
// set_status changes the status of the last data_object to str
//-----------------------------------------------------------------------

void obj_conn::set_status(const string &str)
{
    if (datalist->get_count())
    {
        datalist->reset();
        data *tmp = datalist->next();
        while ((tmp->get_count()) != datalist->get_count())
        {
            tmp = datalist->next();
        };

        tmp->set_status(str);
    }
}

void obj_conn::new_data(const string &new_name)
{
    datalist->new_data(new_name);
}

void obj_conn::del_old_DO(const string &name)
{
    data *tmp = datalist->get(name);
    if (tmp)
    {
        datalist->remove(tmp);
    }
}

void obj_conn::del_rez_DO(const string &name)
{
    data *tmp = datalist->get(name);
    if (tmp)
    {
        datalist->remove(tmp);
        datalist->dec_count();
    }
}

//**********************************************************************
//
// 			OBJ_CONN_LIST
//
//**********************************************************************

obj_conn_list::obj_conn_list()
    : Liste<obj_conn>()
{
}

void obj_conn_list::new_timestep(const string &new_name)
{
    obj_conn *tmp;

    this->reset();
    while ((tmp = this->next()) != NULL)
    {
        tmp->new_data(new_name);
    };
}

void obj_conn_list::del_old_DO(const string &name)
{
    obj_conn *tmp;

    this->reset();
    while ((tmp = this->next()) != NULL)
    {
        tmp->del_old_DO(name);
    }
}

void obj_conn_list::del_rez_DO(const string &name)
{
    obj_conn *tmp;

    this->reset();
    while ((tmp = this->next()) != NULL)
    {
        tmp->del_rez_DO(name);
    }
}

void obj_conn_list::new_DO(const string &new_name)
{
    obj_conn *tmp;

    this->reset();
    while ((tmp = this->next()) != NULL)
    {
        tmp->new_data(new_name);
    }
}

obj_conn *obj_conn_list::get(const string &name, const string &nr, const string &host, const string &intf)
{
    obj_conn *tmp;
    string tmp_name, tmp_nr, tmp_host, tmp_intf;
    this->reset();
    do
    {
        tmp = this->next();
        if (tmp == NULL)
            break; // not found
        net_module *mod = tmp->get_mod();
        tmp_name = mod->get_name();
        tmp_nr = mod->get_nr();
        tmp_host = mod->get_host();
        tmp_intf = tmp->get_mod_intf();
    } while (tmp_name != name || tmp_host != host || tmp_intf != intf || tmp_nr != nr);
    return tmp;
}

//**********************************************************************
//
// 				OBJECT
//
//**********************************************************************

object::object()
{
    from = new obj_from_conn;
    to = new obj_conn_list;
    dataobj = new data_list;
}

object::~object()
{
    delete from;
    delete to;
    delete dataobj;
}

void object::set_name(const string &str)
{
    name = str;
}

string object::get_new_name()
{
    string str;

    if ((dataobj->get_count()) != 0)
    {
        dataobj->reset();
        data *tmp = dataobj->next();
        while ((tmp->get_count()) != dataobj->get_count())
        {
            tmp = dataobj->next();
        };
        str = tmp->get_name();
    }
    else
    {
        str = this->get_name();
    };
    return str;
}

string object::get_current_name()
{
    data *datao;
    string tmp;

    dataobj->reset();
    while ((datao = dataobj->next()))
    {
        tmp = datao->get_name();
    }
    return (tmp);
}

//----------------------------------------------------------------------
// connect_to adds a new to-connection into the to-list
//----------------------------------------------------------------------

obj_conn *object::connect_to(net_module *module, const string &intfname)
{

    if (to->get(module->get_name(), module->get_nr(), module->get_host(), intfname) != NULL)
    {
        cerr << endl << "Error !!! duplicate connections !!!!!";
        return NULL;
    }
    obj_conn *tmp = new obj_conn;

    tmp->connect_module(module);
    tmp->set_mod_intf(intfname);

    // copy the dataobjects from the reflist
    tmp->copy_data(dataobj);
    to->add(tmp);
    // send connect message to the module
    string text(intfname);
    Message *msg = new Message(COVISE_MESSAGE_INFO, text);
    module->send_msg(msg);
    return (tmp);
}

//----------------------------------------------------------------------
// connect_from connects the object with the writing module
//----------------------------------------------------------------------

void object::connect_from(net_module *module, const string &intfname, const string &type)
{
    int i, org_count;

    // testen, ob Module Kopie oder Org
    // liefert den Zaehler des Originals, falls es eine Kopie ist, 1 sonst
    org_count = module->test_org_count(intfname);

    from->conn_module(module);
    from->set_intf(intfname);
    from->set_type(type);

    // evtl. Zaehlerstand hochsetzen
    if (org_count > 1)
    {
        for (i = 1; i <= org_count; i++)
        {
            this->new_timestep();
        };
    };
}

//----------------------------------------------------------------------
// del_to_connection deletes one to connection
//----------------------------------------------------------------------

void object::del_to_connection(const string &name, const string &nr, const string &host, const string &intf)
{

    // search the connection to the net_module in the conn_list
    if (to == NULL)
        return;

    obj_conn *objConn = to->get(name, nr, host, intf);

    // this was a crash - now test it
    if (!objConn)
        return;

    // send disconnect message to the module
    Message *msg = new Message(COVISE_MESSAGE_INFO, intf);
    (objConn->get_mod())->send_msg(msg);
    delete msg;

    // delete the obj_conn
    to->remove(objConn);
    delete objConn;
}

void object::del_from_connection()
{
    from->conn_module(NULL);
    from->set_intf("");
    from->set_state("INIT");
}

//----------------------------------------------------------------------
// del_allto_connects: removes all TO-connections in the Object
// and the corresponding connections in the modules
//----------------------------------------------------------------------

void object::del_allto_connects()
{
    obj_conn *tmp_conn;

    to->reset();
    while ((tmp_conn = to->next()) != NULL)
    {
        // remove the connection in the module
        net_module *mod = tmp_conn->get_mod();
        string intf_name = tmp_conn->get_mod_intf();
        mod->del_O_conn(intf_name, this);

        // remove the local connection
        to->remove(tmp_conn);
        delete tmp_conn;
        to->reset();
    };
}

int object::is_one_running_above()
{
    if ((from->get_mod())->is_one_running_above(0))
        return (true);
    return (false);
}

// set start flag on all connected modules
void object::set_start_module()
{
    obj_conn *tmp_conn;

    to->reset();
    while ((tmp_conn = to->next()) != NULL)
    {
        net_module *mod = tmp_conn->get_mod();
        mod->set_start_flag();
    }
}

void object::start_modules(ui_list *ul)
{
    to->reset();
    obj_conn *tmp_conn;
    while ((tmp_conn = to->next()) != NULL)
    {
        net_module *mod = tmp_conn->get_mod();
        if (mod->is_renderer())
        {
            mod->send_add(ul, this, tmp_conn);
        }
        else if (mod->get_start_flag())
        {
            mod->reset_start_flag();
            if (!(mod->is_one_running_above(1)))
            {
                mod->start_module(ul);
                CTRLHandler::instance()->m_numRunning++;
            }
        }
    }
}

void object::new_timestep()
{
    string new_name = dataobj->create_new_data(this->get_name());
    to->new_timestep(new_name);
}

void object::new_DO()
{
    string new_name = dataobj->create_new_DO(this->get_name());
    to->new_DO(new_name);
}

void object::del_old_DO()
{
    obj_from_conn *tmp_from = this->get_from();
    net_module *tmp_mod = tmp_from->get_mod();
    AppModule *dmod = tmp_mod->get_dm();

    dataobj->reset();
    data *tmp_data;
    while ((tmp_data = dataobj->next()) != NULL)
    {
        if (tmp_data->get_save_status() == 0)
        {
            tmp_data->del_data(dmod);
            to->del_old_DO(tmp_data->get_name());
            dataobj->remove(tmp_data);
        }
    }
}

void object::del_rez_DO()
{
    data *tmp_data;

    dataobj->reset();
    while ((tmp_data = dataobj->next()) != NULL)
    {
        if (tmp_data->get_save_status() == 0)
        {
            to->del_rez_DO(tmp_data->get_name());
            dataobj->remove(tmp_data);
            dataobj->dec_count();
        }
    }
}

void object::del_all_DO(int already_dead)
{

    obj_from_conn *tmp_from = this->get_from();
    net_module *tmp_mod = tmp_from->get_mod();
    AppModule *dmod = tmp_mod->get_dm();

    dataobj->reset();
    data *tmp_data;
    while ((tmp_data = dataobj->next()) != NULL)
    {
        if (already_dead >= 0)
            tmp_data->del_data(dmod);
        to->del_old_DO(tmp_data->get_name());
        dataobj->remove(tmp_data);
    }
}

void object::del_old_data()
{

    obj_from_conn *tmp_from = this->get_from();
    net_module *tmp_mod = tmp_from->get_mod();
    if (tmp_mod)
    {
        AppModule *dmod = tmp_mod->get_dm();
        int nr_of_objects = dataobj->get_count();

        dataobj->reset();
        data *tmp_data;
        while ((tmp_data = dataobj->next()) != NULL)
        {
            if (nr_of_objects != tmp_data->get_count())
            {
                if (tmp_data->get_save_status() == 0)
                {
                    string del_status = tmp_data->get_status();
                    if (del_status != "DEL")
                    {
                        tmp_data->del_data(dmod);
                    }
                }
            }
        }
    }
}

void object::del_dep_data()
{
    to->reset();
    obj_conn *tmp_conn;
    while ((tmp_conn = to->next()) != NULL)
    {
        net_module *mod = tmp_conn->get_mod();
        if (mod->is_renderer())
        {
            string old_name = tmp_conn->get_old_name();
            if (!old_name.empty())
                mod->send_del(old_name);
        }
        else
        {
            mod->delete_dep_objs();
        }
    }
}

void object::del_all_data(AppModule *dmod)
{
    data *tmp_data;

    dataobj->reset();
    while ((tmp_data = dataobj->next()) != NULL)
    {
        tmp_data->del_data(dmod);
    };
}

//------------------------------------------------------------------------
// change_NEW_to_OLD selects the connection to Module mod Interface intf
// if the state of this connection ist INIT or OLD it stops with an Error
// if in all connections the state is OLD, the state in the referencelist
// is set to OLD
//------------------------------------------------------------------------

void object::change_NEW_to_OLD(net_module *mod, const string &intf)
{
    // change substate from NEW to OLD
    obj_conn *tmp_conn = to->get(mod->get_name(), mod->get_nr(), mod->get_host(), intf);
    if (tmp_conn == NULL)
    {
        print_comment(__LINE__, __FILE__, "ERROR: Objectport doesn't exist!\n");
        print_exit(__LINE__, __FILE__, 1);
    }

    string tmp_state = tmp_conn->get_status();
    if (tmp_state == "INIT" || tmp_state == "OLD")
    {
        print_comment(__LINE__, __FILE__, "ERROR: Object: There is no Data to read\n");
        print_exit(__LINE__, __FILE__, 1);
    }

    tmp_conn->set_status("OLD");

    string ref_state = "OLD";
    // loop over all To-Connections
    to->reset();
    while ((tmp_conn = to->next()) != 0)
    {
        tmp_state = tmp_conn->get_status();
        if (tmp_state == "NEW")
            ref_state = "NEW";
    };

    dataobj->set_status(ref_state);
}

//----------------------------------------------------------------------
// set_to_NEW changes the status of the dataobjects in the reflist and
// the status of the dataobjects in the connections to NEW
//----------------------------------------------------------------------

void object::set_to_NEW()
{

    // change status in reference list
    dataobj->set_status("NEW");

    // change status in the connections
    to->reset();
    obj_conn *tmp_conn;
    while ((tmp_conn = to->next()) != 0)
    {
        tmp_conn->set_status("NEW");
    }
}

//----------------------------------------------------------------------
// check_from_connection examines if the object has a connection to the
// Module mod at the port intf_name
// if the connection exists, it returns 1, else 0
//----------------------------------------------------------------------

int object::check_from_connection(net_module *mod, const string &intf)
{
    int check;

    if (mod == from->get_mod() && intf == from->get_intf())
    {
        check = 1;
    }
    else
    {
        check = 0;
    };

    return check;
}

//----------------------------------------------------------------------
// get_conn_state checks the status of a object for a special
// net_module and its connected interface
// if the connection doesn't exist, it returns INIT
// if the connections exits, it returns NEW or OLD or INIT
//----------------------------------------------------------------------
string object::get_conn_state(net_module *mod, const string &intf)
{
    string tmp_state;

    // get substate for connected modules
    obj_conn *tmp_conn = to->get(mod->get_name(), mod->get_nr(), mod->get_host(), intf);
    if (tmp_conn == NULL)
    {
        tmp_state = "INIT";
    }
    else
    {
        tmp_state = tmp_conn->get_status();
    }
    return tmp_state;
}

void object::set_DO_status(const string &DO_name, int mode, net_module *mod, const string &intf_name)
{
    obj_conn *tmp_conn = to->get(mod->get_name(), mod->get_nr(), mod->get_host(), intf_name);
    data_list *tmp_dlist = tmp_conn->get_datalist();
    data *tmp_data = tmp_dlist->get(DO_name);

    // SAVE or RELEASE
    if (mode == DO_SAVE)
    {
        if (tmp_dlist->get_count() == tmp_data->get_count())
        {
            // local SAVE in the connection
            tmp_data->SAVE();
            // global SAVE in the referencelist
            tmp_data = dataobj->get(DO_name);
            tmp_data->SAVE();
        }
        else
        {
            print_comment(__LINE__, __FILE__, " ERROR: SAVE for an old Dataobject \n");
            print_exit(__LINE__, __FILE__, 1);
        }
    }

    else // DO_RELEASE
    {
        // set local RELEASE in the connection
        tmp_data->RELEASE();
        bool release_all = true;
        to->reset();
        while ((tmp_conn = to->next()) != NULL)
        {
            tmp_dlist = tmp_conn->get_datalist();
            tmp_data = tmp_dlist->get(DO_name);
            if (tmp_data->get_save_status())
            {
                release_all = false;
            }
        }

        if (release_all == true)
        {
            tmp_data = dataobj->get(DO_name);
            // set global RELEASE in the referencelist
            tmp_data->RELEASE();
        };
    };
}

void object::print_states()
{
    obj_conn *tmp_conn;

    to->reset();
    while ((tmp_conn = to->next()) != NULL)
    {
        cerr << "Object: "
             << this->get_new_name()
             << " Interface: "
             << tmp_conn->get_mod_intf()
             << " Status: "
             << tmp_conn->get_status() << "\n";
    };
}

bool object::test(const string &DO_name)
{
    bool test;
    data *tmp_data;

    test = false;
    dataobj->reset();
    while (((tmp_data = dataobj->next()) != NULL) && (test == false))
    {
        if (DO_name == tmp_data->get_name())
        {
            test = true;
        }
    }

    return test;
}

void object::set_outputtype(const string &DO_type)
{
    // Sobald ein durchlaufen der Objektlisten realisiert wird, muss
    // der Objekttype mit dem jeweils aktuellen Datenobjekt verknuepft
    // werden. Problem: Wird in jedem Zeitschritt die Connection veraendert
    // gibt es in jedem Zeitschritt Daten eines anderen Types !!

    from->set_type(DO_type);
}

void object::resetobj_for_exec(AppModule *dmod)
{
    // get new dataobjectname
    data *tmp_data = dataobj->get_new();
    // destroy dataobject
    if (tmp_data != NULL)
    {
        tmp_data->del_data(dmod);
        // set status in all connections to INIT
        tmp_data->set_status("INIT");
    }

    to->reset();
    obj_conn *tmp_conn;
    while ((tmp_conn = to->next()) != NULL)
    {
        data_list *tmp_dlist = tmp_conn->get_datalist();
        tmp_data = tmp_dlist->get_new();
        if (tmp_data)
        {
            tmp_data->set_status("INIT");
        }
        net_module *tmp_mod = tmp_conn->get_mod();
        // select module from each connection and
        // do a reset_for_exec on the module
        tmp_mod->mark_for_reset();
    }
}

void object::writeScript(ofstream &of, const string &local_name, const string &local_user)
{
    // test des from-Modules auf Kopie

    // to-Module ebenfalls auf Kopie testen. Die Connections die zu einer Kopie
    // gehen, werden uebersprungen !

    // get from-connection-info
    net_module *tmp_mod = from->get_mod();
    bool copy = true;
    if (tmp_mod)
        copy = tmp_mod->test_copy();

    if (!copy)
    {
        string tmp_host = tmp_mod->get_host();
        module *tmp_typ = tmp_mod->get_type();
        string tmp_user = tmp_typ->get_user();

        if (tmp_host == local_name && tmp_user == local_user)
        {
            tmp_host = "LOCAL";
            tmp_user = "LOCAL";
        }

        // get all to-connections
        to->reset();
        obj_conn *conn_tmp;
        while ((conn_tmp = to->next()) != NULL)
        {
            tmp_mod = conn_tmp->get_mod();
            copy = tmp_mod->test_copy();

            if (!copy)
            {
                tmp_host = tmp_mod->get_host();
                tmp_typ = tmp_mod->get_type();
                tmp_user = tmp_typ->get_user();

                if (tmp_host == local_name && tmp_user == local_user)
                {
                    tmp_host = "LOCAL";
                    tmp_user = "LOCAL";
                }
                of << "network.connect( " << from->get_mod()->get_name() << "_" << from->get_mod()->get_nr() << ", \"" << from->get_intf();
                of << "\", " << tmp_mod->get_name() << "_" << tmp_mod->get_nr() << ", \"" << conn_tmp->get_mod_intf() << "\" )" << endl;
            } // if test
        }
    }
}

string object::get_connection(int *i, const string &local_name, const string &local_user)
{
    // test des from-Modules auf Kopie

    // to-Module ebenfalls auf Kopie testen. Die Connections die zu einer Kopie
    // gehen, werden uebersprungen !

    // get from-connection-info
    net_module *tmp_mod = from->get_mod();
    bool copy = true;
    if (tmp_mod)
        copy = tmp_mod->test_copy();

    ostringstream res_str;
    if (!copy)
    {
        ostringstream from_str;
        string tmp_host = tmp_mod->get_host();
        module *tmp_typ = tmp_mod->get_type();
        string tmp_user = tmp_typ->get_user();

        if (tmp_host == local_name && tmp_user == local_user)
        {
            tmp_host = "LOCAL";
            tmp_user = "LOCAL";
        }

        from_str << tmp_mod->get_name() << "\n" << tmp_mod->get_nr() << "\n" << tmp_host << "\n" << from->get_intf() << "\n";

        // get all to-connections
        to->reset();
        obj_conn *conn_tmp;
        while ((conn_tmp = to->next()) != NULL)
        {
            tmp_mod = conn_tmp->get_mod();
            copy = tmp_mod->test_copy();

            if (!copy)
            {
                (*i)++;
                tmp_host = tmp_mod->get_host();
                tmp_typ = tmp_mod->get_type();
                tmp_user = tmp_typ->get_user();

                if (tmp_host == local_name && tmp_user == local_user)
                {
                    tmp_host = "LOCAL";
                    tmp_user = "LOCAL";
                }

                res_str << from_str.str() << "\n" << tmp_mod->get_name() << "\n" << tmp_mod->get_nr();
                res_str << "\n" << tmp_host << "\n" << conn_tmp->get_mod_intf() << "\n";
            }; // if test
        };
    };
    return res_str.str();
}

//
// return all connection lines for a given module
//
string object::get_simple_connection(int *i)
{

    // test des from-Modules auf Kopie

    // to-Module ebenfalls auf Kopie testen. Die Connections die zu einer Kopie
    // gehen, werden uebersprungen !

    // get from-connection-info
    net_module *tmp_mod = from->get_mod();
    bool test = true;
    if (tmp_mod)
        test = tmp_mod->test_copy();

    string sg;
    if (test == false)
    {
        string s1 = tmp_mod->get_name();
        string s2 = tmp_mod->get_nr();
        string s3 = tmp_mod->get_host();
        string s4 = from->get_intf();

        // get all to-connections
        to->reset();
        obj_conn *conn_tmp;
        while ((conn_tmp = to->next()) != NULL)
        {
            tmp_mod = conn_tmp->get_mod();
            test = tmp_mod->test_copy();
            if (test == false)
            {
                (*i)++;
                string ss1 = tmp_mod->get_name();
                string ss2 = tmp_mod->get_nr();
                string ss3 = tmp_mod->get_host();
                string ss4 = conn_tmp->get_mod_intf();
                sg = sg + s1 + "\n" + s2 + "\n" + s3 + "\n" + s4 + "\n" + ss1 + "\n" + ss2 + "\n" + ss3 + "\n" + ss4 + "\n";
            }
        }
    }
    return sg;
}

int object::get_counter()
{
    return dataobj->get_count();
}

bool object::isEmpty()
{
    bool ret = (dataobj == NULL) || (dataobj->isEmpty());
    return ret;
}

//**********************************************************************
//
// 			OBJECT_LIST
//
//**********************************************************************

object_list::object_list()
    : Liste<object>()
{
    count = 0;
}

void object_list::writeScript(ofstream &of, const string &local_name, const string &local_user)
{

    object *tmp_obj;
    reset();
    while ((tmp_obj = next()) != NULL)
    {
        tmp_obj->writeScript(of, local_name, local_user);
    }
}

string object_list::get_connections(const string &local_name, const string &local_user)
{

    object *tmp_obj;
    string buffer;
    int i = 0;

    reset();
    while ((tmp_obj = next()) != NULL)
    {
        string conn = tmp_obj->get_connection(&i, local_name, local_user);
        if (!conn.empty())
            buffer.append(conn);
    }

    if (buffer.empty())
        buffer.append("0");

    else
    {
        ostringstream os;
        os << i;
        buffer = os.str() + "\n" + buffer;
    }
    return buffer;
}

object *object_list::get(const string &name)
{
    string tmp_name;
    object *tmp;

    this->reset(); // die Liste muss von hinten durchlaufen werden wegen Ports > 10, nein, funktioniert auch nicht...!!!
    do
    {
        tmp = this->next();
        if (tmp == NULL)
            break; // not in the list
        tmp_name = tmp->get_name();
    } while (tmp_name != name);

    return tmp;
}

object *object_list::select(const string &name)
{
    object *tmp = this->get(name);
    if (tmp == NULL)
    {
        // create new object
        count = count + 1;
        tmp = new object;
        tmp->set_name(name);
        this->add(tmp);
    };
    return tmp;
}
