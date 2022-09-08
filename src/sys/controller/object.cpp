/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "controlProcess.h"
#include "crb.h"
#include "global.h"
#include "handler.h"
#include "list.h"
#include "module.h"
#include "object.h"
#include "renderModule.h"

#include <covise/covise.h>
#include <covise/covise_msg.h>
#include <net/covise_connect.h>
#include <net/message.h>
#include <net/message_types.h>

using namespace covise;
using namespace covise::controller;

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

void data::set_name(const string &str)
{
    name = str;
}

void data::set_status(const string &str)
{
    status = str;
}

void data::del_data(const controller::CRBModule &crb)
{
    this->set_status("DEL");

    Message msg{COVISE_MESSAGE_CTRL_DESTROY_OBJECT, name};
    crb.send(&msg);

#ifdef DEBUG
//	fprintf(msg_prot,"---------------------------------------------------\n");
//	fprintf(msg_prot,"send DEST OBJ\n%i %i \n %s \n",msg->sender,msg->type,msg->data);
//	fflush(msg_prot);
#endif

    Message recvMsg;
    crb.recv_msg(&recvMsg);
    switch (recvMsg.type)
    {
    case COVISE_MESSAGE_EMPTY:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    case COVISE_MESSAGE_SOCKET_CLOSED:
        CTRLGlobal::getInstance()->controller->getConnectionList()->remove(recvMsg.conn);
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

obj_conn::obj_conn(obj_conn &&other)
{
    *this = std::move(other);
}
obj_conn &obj_conn::operator=(obj_conn &&other)
{
    mod = other.mod;
    mod_intf = std::move(other.mod_intf);
    old_name = std::move(other.old_name);
    datalist = other.datalist;
    other.mod = nullptr;
    other.datalist = nullptr;

    return *this;
}

obj_conn::~obj_conn()
{
    if (!mod)
    {
        return;
    }

    //remove this connection from the connected to-module
    const string &s = mod_intf;
    try
    {
        auto &inter = mod->connectivity().getInterface<C_interface>(s);
        if (auto netInter = dynamic_cast<net_interface *>(&inter))
        {
            netInter->del_connect();
        }
    }
    catch (const Exception &e)
    {
        (void)e;
    }
    if (!old_name.empty())
    {
        if (auto renderer = dynamic_cast<Renderer *>(mod))
        {
            renderer->send_del(old_name);
        }
    }
    delete datalist;
}

void obj_conn::set_old_name(const string &str)
{
    old_name = str;
}

void obj_conn::connect_module(controller::NetModule *module)
{
    mod = module;
}

const controller::NetModule *obj_conn::get_mod() const
{
    return mod;
}

controller::NetModule *obj_conn::get_mod()
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
    this->reset();
    while (obj_conn *tmp = this->next())
    {
        controller::NetModule *mod = tmp->get_mod();
        if (mod->info().name == name && mod->getHost() == host && tmp->get_mod_intf() == intf && mod->instance() == std::stoi(nr))
        {
            return tmp;
        }
    }
    return nullptr;
}

//**********************************************************************
//
// 				OBJECT
//
//**********************************************************************

object::object()
{
    dataobj = new data_list;
}

object::~object()
{
    delete dataobj;
}

void object::set_name(const string &str)
{
    name = str;
}

string object::get_new_name() const
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

string object::get_current_name() const
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

obj_conn *object::connect_to(controller::NetModule *module, const string &intfname)
{

    auto it = std::find_if(to.begin(), to.end(), [&module, &intfname](const obj_conn &conn) {
        return conn.get_mod() == module && conn.get_mod_intf() == intfname;
    });
    if (it != to.end())
    {
        cerr << endl
             << "Error !!! duplicate connections !!!!!";
        return nullptr;
    }
    obj_conn &tmp = *to.emplace(to.end());

    tmp.connect_module(module);
    tmp.set_mod_intf(intfname);

    // copy the dataobjects from the reflist
    tmp.copy_data(dataobj);

    // send connect message to the module
    string text(intfname);
    Message msg{COVISE_MESSAGE_INFO, text};
    module->send(&msg);
    return &tmp;
}

//----------------------------------------------------------------------
// connect_from connects the object with the writing module
//----------------------------------------------------------------------

void object::connect_from(controller::NetModule *module, const net_interface &interface)
{

    from.conn_module(module);
    from.set_intf(interface.get_name());
    from.set_type(interface.get_type());

    int originalCount = module->testOriginalcount(interface.get_name());
    if (originalCount > 1) //this is the original
    {
        for (int i = 1; i <= originalCount; i++)
        {
            this->newDataObject();
        }
    }
}

//----------------------------------------------------------------------
// del_to_connection deletes one to connection
//----------------------------------------------------------------------

void object::del_to_connection(const string &name, const string &nr, const string &host, const string &intf)
{

    // search the connection to the net_module in the conn_list
    auto objConn = std::find_if(to.begin(), to.end(), [&name, &nr, &host, &intf](const obj_conn &conn) {
        return conn.get_mod()->info().name == name &&
               conn.get_mod()->instance() == std::stoi(nr) &&
               conn.get_mod()->getHost() == host &&
               conn.get_mod_intf() == intf;
    });
    if (objConn == to.end())
    {
        return;
    }

    // send disconnect message to the module
    Message msg{COVISE_MESSAGE_INFO, intf};
    objConn->get_mod()->send(&msg);

    // delete the obj_conn
    to.erase(objConn);
}

//----------------------------------------------------------------------
// del_allto_connects: removes all TO-connections in the Object
// and the corresponding connections in the modules
//----------------------------------------------------------------------
void object::del_allto_connects()
{
    for(auto &conn : to)
    {
        conn.get_mod()->delObjectConn(conn.get_mod_intf(), this);
    }
    to.clear();
}

void object::del_from_connection()
{
    from.conn_module(NULL);
    from.set_intf("");
    from.set_state("INIT");
}

int object::is_one_running_above() const
{
    if (from.get_mod()->isOneRunningAbove(0))
        return (true);
    return (false);
}

// set start flag on all connected modules
void object::setStartFlagOnConnectedModules()
{
    std::for_each(to.begin(), to.end(), [](obj_conn &conn) {
        controller::NetModule *app = conn.get_mod();
        app->setStartFlag();
    });
}

void object::start_modules(controller::NumRunning &numRunning)
{
    for (auto &conn : to)
    {
        auto app = conn.get_mod();
        if (auto renderer = dynamic_cast<controller::Renderer *>(app))
        {
            renderer->send_add(*this, conn);
        }
        else if (app->startflag())
        {
            app->resetStartFlag();
            if (!app->isOneRunningAbove(true))
            {
                app->execute(numRunning);
                ++numRunning.apps;
            }
        }
    }
}

void object::newDataObject()
{
    const string new_name = dataobj->create_new_DO(this->get_name());
    std::for_each(to.begin(), to.end(), [&new_name](obj_conn &conn) { conn.new_data(new_name); });
}

void object::del_old_DO()
{
    auto &crb = dynamic_cast<const controller::CRBModule &>(get_from().get_mod()->host.getProcess(sender_type::CRB));
    dataobj->reset();
    data *tmp_data;
    while ((tmp_data = dataobj->next()) != NULL)
    {
        if (tmp_data->get_save_status() == 0)
        {
            tmp_data->del_data(crb);
            for (auto &c : to)
            {
                c.del_old_DO(tmp_data->get_name());
            }
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
            std::for_each(to.begin(), to.end(), [tmp_data](obj_conn &conn) {
                conn.del_rez_DO(tmp_data->get_name());
            });
            dataobj->remove(tmp_data);
            dataobj->dec_count();
        }
    }
}

void object::del_all_DO(int already_dead)
{

    auto tmp_mod = from.get_mod();
	try {
		const controller::CRBModule& crb = dynamic_cast<const controller::CRBModule&>(tmp_mod->host.getProcess(sender_type::CRB));
		dataobj->reset();
		data* tmp_data;
		while ((tmp_data = dataobj->next()) != NULL)
		{
			if (already_dead >= 0)
				tmp_data->del_data(crb);
			std::for_each(to.begin(), to.end(), [&tmp_data](obj_conn& conn) { conn.del_old_DO(tmp_data->get_name()); });
			dataobj->remove(tmp_data);
		}
    }
    catch (covise::controller::Exception e)
    {
        return; // no CRB, no need to do anything any more
    }
}

void object::del_old_data()
{
    const auto tmp_mod = from.get_mod();
    if (tmp_mod)
    {
        try {
			const controller::CRBModule& crb = dynamic_cast<const controller::CRBModule&>(tmp_mod->host.getProcess(sender_type::CRB));

			int nr_of_objects = dataobj->get_count();

			dataobj->reset();
			data* tmp_data;
			while ((tmp_data = dataobj->next()) != NULL)
			{
				if (nr_of_objects != tmp_data->get_count())
				{
					if (tmp_data->get_save_status() == 0)
					{
						string del_status = tmp_data->get_status();
						if (del_status != "DEL")
						{
							tmp_data->del_data(crb);
						}
					}
				}
			}
		}
        catch (covise::controller::Exception e)
        {
            return; // no CRB, no need to do anything any more
        }
    }
}

void object::del_dep_data()
{
    for (auto &conn : to)
    {
        const auto mod = conn.get_mod();
        if (auto renderer = dynamic_cast<controller::Renderer *>(mod))
        {
            if (!conn.get_old_name().empty())
                renderer->send_del(conn.get_old_name());
        }
        else
        {
            mod->delete_dep_objs();
        }
    }
}

//------------------------------------------------------------------------
// change_NEW_to_OLD selects the connection to Module mod Interface intf
// if the state of this connection ist INIT or OLD it stops with an Error
// if in all connections the state is OLD, the state in the referencelist
// is set to OLD
//------------------------------------------------------------------------

void object::change_NEW_to_OLD(const controller::NetModule *mod, const string &intf)
{
    // change substate from NEW to OLD

    auto conn = std::find_if(to.begin(), to.end(), [&mod](const obj_conn &conn) {
        return conn.get_mod() == mod;
    });
    if (conn == to.end())
    {
        print_comment(__LINE__, __FILE__, "ERROR: Objectport doesn't exist!\n");
        print_exit(__LINE__, __FILE__, 1);
    }

    if (conn->get_status() == "INIT" || conn->get_status() == "OLD")
    {
        print_comment(__LINE__, __FILE__, "ERROR: Object: There is no Data to read\n");
        print_exit(__LINE__, __FILE__, 1);
    }

    conn->set_status("OLD");

    string ref_state = "OLD";
    for (auto &c : to)
    {
        if (c.get_status() == "NEW")
        {
            ref_state = c.get_status();
            break;
        }
    }
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
    for (auto &c : to)
        c.set_status("NEW");
}

//----------------------------------------------------------------------
// check_from_connection examines if the object has a connection to the
// Module mod at the port intf_name
// if the connection exists, it returns 1, else 0
//----------------------------------------------------------------------

bool object::check_from_connection(const controller::NetModule *mod, const string &interfaceName) const
{
    return (mod == from.get_mod() && interfaceName == from.get_intf());
}

//----------------------------------------------------------------------
// get_conn_state checks the status of a object for a special
// net_module and its connected interface
// if the connection doesn't exist, it returns INIT
// if the connections exits, it returns NEW or OLD or INIT
//----------------------------------------------------------------------
string object::get_conn_state(const controller::NetModule *mod, const string &intf)
{
    // get substate for connected modules
    auto conn = std::find_if(to.begin(), to.end(), [&mod, &intf](const obj_conn &c) {
        return c.get_mod() == mod && c.get_mod_intf() == intf;
    });
    if (conn == to.end())
        return "INIT";
    else
        return conn->get_status();
}

void object::set_DO_status(const string &DO_name, int mode, const controller::NetModule &mod, const string &intf_name)
{
    auto conn = std::find_if(to.begin(), to.end(), [&mod](const obj_conn &conn) {
        return conn.get_mod() == &mod;
    });
    data_list *tmp_dlist = conn->get_datalist();
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
        for (auto &c : to)
        {
            tmp_dlist = c.get_datalist();
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
    for (auto &c : to)
    {
        cerr << "Object: "
             << this->get_new_name()
             << " Interface: "
             << c.get_mod_intf()
             << " Status: "
             << c.get_status() << "\n";
    }
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

    from.set_type(DO_type);
}

std::string remoteHostOrLOCAL(const std::string &localApAddress, const std::string &localUserName, const controller::RemoteHost &host)
{
    if (host.userInfo().ipAdress == localApAddress && host.userInfo().userName == localUserName)
    {
        return "LOCAL";
    }
    return host.userInfo().ipAdress;
}

void object::writeScript(ofstream &of, const string &local_name, const string &local_user)
{
    if (from.get_mod() && from.get_mod()->isOriginal())
    {
        for (const obj_conn &conn : to)
        {
            if (conn.get_mod() && conn.get_mod()->isOriginal())
            {
                of << "network.connect( " << from.get_mod()->info().name << "_" << from.get_mod()->instance() << ", \"" << from.get_intf();
                of << "\", " << conn.get_mod()->info().name << "_" << conn.get_mod()->instance() << ", \"" << conn.get_mod_intf() << "\" )" << endl;
            } // if test
        }
    }
}

string object::get_connection(int *i, const string &local_name, const string &local_user)
{
    //skip connections to copies
    if (from.get_mod() && from.get_mod()->isOriginal())
    {
        std::stringstream from_str, retval;
        from_str << from.get_mod()->info().name << "\n"
                 << from.get_mod()->instance() << "\n"
                 << remoteHostOrLOCAL(local_name, local_user, from.get_mod()->host) << "\n"
                 << from.get_intf() << "\n";
        // get all to-connections
        for (const obj_conn &conn : to)
        {
            if (conn.get_mod() && conn.get_mod()->isOriginal())
            {
                (*i)++;
                retval << from_str.str() << "\n"
                       << conn.get_mod()->info().name << "\n"
                       << conn.get_mod()->instance() << "\n"
                       << remoteHostOrLOCAL(local_name, local_user, conn.get_mod()->host) << "\n"
                       << conn.get_mod_intf() << "\n";
            }
        }
        return retval.str();
    }
    return std::string{};
}

//
// return all connection lines for a given module
//
string object::get_simple_connection(int *i)
{
    //skip connections to copies
    if (from.get_mod() && from.get_mod()->isOriginal())
    {
        std::string retval;
        for (const obj_conn &conn : to)
        {
            const auto mod = conn.get_mod();
            if (mod && mod->isOriginal())
            {
                (*i)++;
                retval += from.get_mod()->createBasicModuleDescription() + from.get_intf() + "\n";
                retval += conn.get_mod()->createBasicModuleDescription() + conn.get_mod_intf() + "\n";
            }
        }
        return retval;
    }
    *i = 0;
    return std::string{};
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
