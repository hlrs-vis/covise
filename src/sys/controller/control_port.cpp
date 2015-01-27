/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include "CTRLHandler.h"
#include "control_define.h"
#include "control_list.h"
#include "control_object.h"
#include "control_port.h"

//**********************************************************************
//
// 				PORT
//
//**********************************************************************

using namespace covise;

port::port()
{
}

void port::set_name(const string &str)
{
    name = str;
}

void port::set_type(const string &str)
{
    type = str;
}

void port::set_text(const string &str)
{
    text = str;
}

//**********************************************************************
//
// 				INTERFACE
//
//**********************************************************************

C_interface::C_interface()
    : port()
{
}

void C_interface::set_demand(const string &str)
{
    demand = str;
}

void C_interface::set_direction(const string &str)
{
    direction = str;
}

//**********************************************************************
//
// 				VALUE_LIST
//
//**********************************************************************
Value::Value()
{
}

void Value::set(string str)
{
    string tmp = str;
    int len = (int)str.length();
    if (len > 1 && (tmp[0] == '"') && (tmp[len - 1] == '"')) // we have a string with quotes, so remove them
    {
        tmp.erase(len, 1);
        tmp.erase(0, 1);
    }
    s = str;
}

value_list::value_list()
    : Liste<Value>()
{
    count = 0;
}

//**********************************************************************
//
// 				PARAMETER
//
//**********************************************************************

parameter::parameter()
    : port()
{
    values = new value_list;

#ifdef PARA_START
    extension = "-1";
#endif
    panel = "-1";
}

int parameter::get_count()
{
    return values->count;
}

void parameter::set_addvalue(const string &add_para)
{
    panel = add_para;
}

#ifdef PARA_START
void parameter::set_extension(const string &ext)
{
    extension = ext;
}
#endif

//Wird aufgerufen, wenn ein Parameter verändert wird
void parameter::set_value(int no, string str)
{
    int i;
    Value *v_tmp = NULL;

    if (values->count < no)
    {
        v_tmp = new Value;
        v_tmp->set(str);
        values->add(v_tmp);
        values->count++;
    }
    else
    {
        values->reset();
        i = 0;
        while (i < no)
        {
            v_tmp = values->next();
            i++;
        }
        v_tmp->set(str);
    }
}

//Wird evtl. auch aufgerufen, wenn ein Parameter verändert wird
void parameter::set_value_list(string strVal)
{
    Value *tmp_val;

    // copy original string into org_val
    org_val = strVal;

    // test, if a values-list exits
    if (values->count > 0) // remove all values
    {
        while (values->count > 0)
        {
            values->reset();
            tmp_val = values->next();
            values->remove(tmp_val);
            values->count--;
        }
    }

    // parse values
    string partype = this->get_type();
    if (partype == "String" || partype == "STRING" || partype == "Text" || partype == "TEXT" || partype == "text")
    {
        Value *tmp_val = new Value();
        tmp_val->set(strVal);
        values->add(tmp_val);
        values->count++;
    }

    else if (partype == "Browser")
    {
        Value *tmp_val = new Value();
        tmp_val->set(strVal);
        values->add(tmp_val);
        values->count++;
    }

    else
    {
        if (strVal.empty())
        {
            cerr << "parameter::set_value_list error: parameter value is NULL" << endl;
            return;
        }

        vector<string> list = CTRLHandler::instance()->splitString(strVal, " ");
        for (int i = 0; i < list.size(); i++)
        {
            Value *tmp_val = new Value;
            tmp_val->set(list[i]);
            values->add(tmp_val);
            values->count++;
        }
    }
}

string parameter::get_value(int no)
{
    Value *v_tmp;
    string tmp;

    if (no <= values->count)
    {
        values->reset();
        int i = 0;
        while (i < no)
        {
            i++;
            v_tmp = values->next();
            tmp = v_tmp->get();
        }
    }

    return tmp;
}

string parameter::get_val_list()
{
    string retVal;
    values->reset();
    for (int i = 0; i < values->count; i++)
    {
        if (i == 0 && get_type() == "Browser")
        {
            retVal.append(this->get_value(i + 1));
        }

        else
        {
            retVal.append(this->get_value(i + 1));
        }

        if (i < values->count - 1)
        {
            retVal.append(" ");
        }
    }

    return retVal;
}

string parameter::get_pyval_list()
{

    string retVal;
    values->reset();

    if (get_type() == "Browser" || get_type() == "Browser-Filter" || get_type() == "Boolean" || get_type() == "String")
    {
        retVal.append("\"");
        retVal.append(get_value(1));
        retVal.append("\"");
    }

    else if (get_type() == "FloatVector" || get_type() == "FloatSlider" || get_type() == "IntVector" || get_type() == "IntSlider")
    {
        // python uses a fix vector length of 3
        int end = values->count;
        for (int i = 0; i < end; i++)
        {
            retVal.append(get_value(i + 1));
            if (i != end - 1)
                retVal.append(", ");
        }
    }

    else if (get_type() == "Color")
    {
        int end = values->count;
        for (int i = 0; i < end; i++)
        {
            retVal.append(get_value(i + 1));
            if (i != values->count - 1)
                retVal.append(", ");
        }
    }

    else if (get_type() == "Colormap" || get_type() == "BrowserFilter")
    {
        retVal.append("\"");
        ;
        for (int i = 0; i < values->count; i++)
        {
            retVal.append(get_value(i + 1));
            retVal.append(" ");
        }
        retVal.append("\"");
    }

    else
    {
        retVal.append(get_value(1));
    }

    return retVal;
}

//**********************************************************************
//
// 		     CONNECT & CONNECT_OBJ_LIST
//
//**********************************************************************
connect_obj::connect_obj()
{
    conn_obj = NULL;
}

void connect_obj::set_conn(object *obj)
{
    conn_obj = obj;
}

object *connect_obj::get_obj()
{
    return conn_obj;
}

void connect_obj::del_conn()
{
    conn_obj = NULL;
}

void connect_obj::set_oldname(const string &str)
{
    old_name = str;
}

string connect_obj::get_oldname()
{
    return old_name;
}

connect_obj_list::connect_obj_list()
    : Liste<connect_obj>()
{
}

//**********************************************************************
//
// 		     CONNECT & CONNECT_MOD_LIST
//
//**********************************************************************

connect_mod::connect_mod()
{
    conn_obj = NULL;
    conn_par = "0";
}

void connect_mod::link_mod(net_module *mod) { conn_obj = mod; }

net_module *connect_mod::get_mod() { return conn_obj; }

void connect_mod::set_par(const string &par)
{
    conn_par = par;
}

string connect_mod::get_par()
{
    return conn_par;
}

connect_mod_list::connect_mod_list()
    : Liste<connect_mod>()
{
}

connect_mod *connect_mod_list::get(net_module *mod, string par)
{
    string org_name = mod->get_name();
    string org_nr = mod->get_nr();
    string org_host = mod->get_host();

    string tmp_par, tmp_name, tmp_nr, tmp_host;

    connect_mod *tmp_conn;
    this->reset();
    do
    {
        tmp_conn = this->next();
        if (tmp_conn == NULL)
            break; // not found
        net_module *tmp_mod = tmp_conn->get_mod(); // get connected module
        tmp_par = tmp_conn->get_par();
        tmp_name = tmp_mod->get_name();
        tmp_nr = tmp_mod->get_nr();
        tmp_host = tmp_mod->get_host();
    } while (tmp_name != org_name || tmp_host != org_host || tmp_nr != org_nr || tmp_par != par);

    return tmp_conn;
}

//**********************************************************************
//
// 			NET_INTERFACE
//
//**********************************************************************

net_interface::net_interface()
    : C_interface()
{
    obj = NULL;
}

void net_interface::set_connect(object *conn)
{
    obj = conn;
}

void net_interface::del_connect()
{
    obj = NULL;
}

object *net_interface::get_object()
{
    return obj;
}

//----------------------------------------------------------------------
// get_state checks the status of a net_interface
// if the demand is opt and no connection exists, it returns OPT
// if the demand is not opt and no connection exists, it return INIT
// if a connection to the object exists and no Dataobject exists
//    or no connection from the object to another module,
//    it returns INIT,
// if in the connected objects exists new Data, it returns NEW
// if in the connected objects exists Data, which was read before, it
//    returns OLD
//----------------------------------------------------------------------

int net_interface::get_state(net_module *mod)
{
    int tmp;

    // get object-state:
    if (obj != NULL)
    {
        // INIT, NEW, OLD
        string tmp_state = obj->get_conn_state(mod, this->get_name());
        if (tmp_state == "INIT")
        {
            tmp = S_INIT;
        }
        else if (tmp_state == "NEW")
        {
            tmp = S_NEW;
        }
        else
        {
            tmp = S_OLD;
        }
    }
    else
    {
        // no connection specified
        if (this->demand == "opt")
        {
            // no connections required
            tmp = S_OPT;
        }
        else
        {
            tmp = S_INIT;
        }
    }

#ifdef DEBUG
    cerr << "Module " << mod->get_name() << " Interface " << this->get_name() << " status " << tmp_state << "\n";
#endif

    return tmp;
}

void net_interface::set_outputtype(const string & /*unused*/, const string &DO_type)
{
    if (obj != 0)
    {
        obj->set_outputtype(DO_type);
    }
}

//**********************************************************************
//
// 			RENDER_INTERFACE
//
//**********************************************************************

render_interface::render_interface()
    : C_interface()
{
    connects = new connect_obj_list;
    conn_count = 0;
    wait_count = 0;
}

void render_interface::set_connect(object *obj)
{
    connect_obj *tmp;

    tmp = new connect_obj;
    tmp->set_conn(obj);
    connects->add(tmp);
    conn_count = conn_count + 1;
    // wait_count = wait_count+1;
    wait_count = conn_count;
}

void render_interface::del_connect(object *obj, displaylist *displays)
{

    connects->reset();
    connect_obj *tmp = connects->next();

    while (tmp && obj != tmp->get_obj())
        tmp = connects->next();

    if (tmp)
    {
        string old_name = tmp->get_oldname();
        if (!old_name.empty())
            displays->send_del(old_name, "");
        tmp->del_conn();
        connects->remove(tmp);
        conn_count = conn_count - 1;
        wait_count = wait_count - 1;
    }
}

void render_interface::reset_wait()
{
    wait_count = conn_count;
}

void render_interface::decr_wait()
{
    wait_count = wait_count - 1;
}

bool render_interface::get_wait_status()
{
    bool tmp = false;
    if (wait_count >= 1)
        tmp = true;
    return tmp;
}

void render_interface::del_all_connections(render_module *mod)
{
    connect_obj *tmp;
    connects->reset();
    while ((tmp = connects->next()) != NULL)
    {
        object *obj = tmp->get_obj();
        obj->del_to_connection(mod->get_name(), mod->get_nr(), mod->get_host(), this->get_name());
        connects->remove(tmp);
        connects->reset();
    }
    delete connects;
}

bool render_interface::get_conn_state()
{
    bool tmp_state;
    connect_obj *tmp;

    connects->reset();
    tmp = connects->next();
    if (tmp == NULL)
    {
        tmp_state = false;
    }
    else
    {
        tmp_state = true;
    }
    return tmp_state;
}

int render_interface::check_conn()
{
    int check;
    connect_obj *tmp;

    connects->reset();
    tmp = connects->next();
    if (tmp == NULL)
    {
        check = 0;
    }
    else
    {
        check = 1;
    }
    return check;
}

//----------------------------------------------------------------------
// get_state checks the status of a render_interface
// if the demand is opt and no connection exists, it returns OPT
// if the demand is req and no connection exists, it return INIT
// if a connection to the object exists and no Dataobject exists
//    or no connection from the object to another module,
//    it returns INIT,
// if in the connected objects exists new Data, it returns NEW
// if in the connected objects exists Data, which was read before, it
//    returns OLD
// falls mindestens in einer Connection neue Daten vorhanden sind, NEW
//----------------------------------------------------------------------
int render_interface::get_state(net_module *mod)
{
    int state;
    string tmp_state;
    connect_obj *tmp;
    object *obj;

    connects->reset();
    bool new_data = false;
    bool connected = false;

    // search, if in any connection new data exists
    while ((tmp = connects->next()) != NULL)
    {
        connected = true;
        obj = tmp->get_obj();
        //INIT, NEW, OLD
        tmp_state = obj->get_conn_state(mod, this->get_name());

        if (tmp_state == "INIT")
        {
            state = S_INIT;
        }
        else if (tmp_state == "NEW")
        {
            state = S_NEW;
        }
        else
        {
            state = S_OLD;
        }

        // new data found. set signal NEW_DATA
        if (state == S_NEW)
            new_data = true;
    }

    if (connected == false)
    {
        if (this->demand == "opt")
        {
            // no connections required
            state = S_OPT;
        }
        else
        {
            state = S_INIT;
        }
    }
    else if (new_data == true)
    {
        state = S_NEW;
    }
    else
    {
        state = S_OLD;
    }

    return state;
}

connect_obj *render_interface::get_first_NEW(render_module *mod)
{
    connect_obj *tmp_conn;
    connects->reset();
    bool new_data = false;

    while (new_data == false)
    {
        tmp_conn = connects->next();
        object *obj = tmp_conn->get_obj();
        string tmp_state = obj->get_conn_state((net_module *)mod, this->get_name());
        if (tmp_state == "NEW")
            new_data = true;
    }
    return tmp_conn;
}

//**************************************************************************
//
//	get_objlist returns a string with the new names of the dataobjects
//	of the renderer interface. The dataobjects are seperated by \n
//	At the end and the beginning of the string is no \n!
//**************************************************************************

string render_interface::get_objlist()
{
    bool first = true;
    string tmpbuffer;

    connects->reset();
    connect_obj *tmp_conn;
    while ((tmp_conn = connects->next()) != NULL)
    {
        object *obj = tmp_conn->get_obj();
        string tmp = obj->get_current_name();
        if (tmp.empty())
            tmp = "Not jet created";
        if (first == true)
        {
            tmpbuffer = tmp;
            first = false;
        }

        else
        {
            tmpbuffer = tmp;
#ifdef SWITCH
            tmpbuffer.append("|");
#else
            tmpbuffer.append("\n");
#endif
            tmpbuffer.append(tmp);
        }
    }
    return tmpbuffer;
}

void render_interface::count_init(render_module *mod)
{
    int tmp_count;

    connects->reset();
    connect_obj *tmp;
    tmp_count = 0;
    while ((tmp = connects->next()) != NULL)
    {
        object *obj = tmp->get_obj();
        string tmp_state = obj->get_conn_state((net_module *)mod, this->get_name());
        if (tmp_state == "INIT")
            tmp_count = tmp_count + 1;
    }
    wait_count = tmp_count;
}

void render_interface::reset_to_NEW(render_module *mod)
{
    connects->reset();
    connect_obj *tmp_conn;
    while ((tmp_conn = connects->next()) != NULL)
    {
        object *obj = tmp_conn->get_obj();
        string tmp_state = obj->get_conn_state((net_module *)mod, this->get_name());
        if (tmp_state == "OLD")
            obj->set_to_NEW();
    }
}

//**********************************************************************
//
// 			NET_PARAMETER
//
//**********************************************************************

net_parameter::net_parameter()
    : parameter()
{
    connects = new connect_mod_list;
}

connect_mod_list *net_parameter::get_connectlist()
{
    return connects;
}

void net_parameter::set_P_conn(net_module *mod, string par)
{
    connect_mod *tmp = new connect_mod;
    tmp->link_mod(mod);
    tmp->set_par(par);
    connects->add(tmp);
}

void net_parameter::del_P_conn(net_module *mod, string par)
{
    connect_mod *to = connects->get(mod, par);
    connects->remove(to);
}
