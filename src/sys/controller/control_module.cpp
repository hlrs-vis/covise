/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <covise/covise.h>

#include "CTRLHandler.h"
#include "control_object.h"
#include "control_port.h"
#include "control_module.h"
#include "control_define.h"

using namespace covise;
/*!
    \class control_module
    \brief Interface list    
*/

interface_list::interface_list()
    : Liste<C_interface>()
{
}

C_interface *interface_list::get(const string &str)
{
    C_interface *tmp;

    this->reset();
    do
    {
        tmp = this->next();
    } while (tmp && (tmp->get_name() != str));

    return tmp;
}

//**********************************************************************
//
// 				PARAM_LIST
//
//**********************************************************************

paramlist::paramlist()
    : Liste<parameter>()
{
}

parameter *paramlist::get(const string &name)
{
    parameter *tmp;

    this->reset();
    do
    {
        tmp = this->next();
        if (tmp == NULL)
            break; // not in list
    } while (tmp->get_name() != name);

    return tmp;
}

//**********************************************************************
//
// 				MODULE
//
//**********************************************************************

module::module()
{
    interfaces = new interface_list;
    para_in = new paramlist;
    para_out = new paramlist;
    counter = 0;
}

int module::get_counter()
{
    counter++;
    return counter;
}

void module::reset_counter()
{
    counter = 0;
}

void module::set_counter(const int &nr)
{
    counter = nr;
}

void module::set_name(const string &str)
{
    name = str;
}

void module::set_exec(const string &str)
{
    exec = str;
}

void module::set_user(const string &str)
{
    user = str;
}

void module::set_host(const string &str)
{
    host = str;
}

void module::set_moduledescr(const string &str)
{
    if (!str.empty())
        moduledescr = str;
}

void module::set_category(const string &str)
{
    if (!str.empty())
        category = str;
}

string module::get_interfname()
{
    return act_interface->get_name();
}

string module::get_interftype()
{
    return act_interface->get_type();
}

string module::get_interfdirection()
{
    return act_interface->get_direction();
}

string module::get_interftext()
{
    return act_interface->get_text();
}

string module::get_interfdemand()
{
    return act_interface->get_demand();
}

void module::reset_intflist()
{
    interfaces->reset();
}

void module::clear_intflist()
{
    delete interfaces;
    delete para_in;
    delete para_out;
    interfaces = new interface_list;
    para_in = new paramlist;
    para_out = new paramlist;
}

int module::next_interface()
{
    act_interface = interfaces->next();

    if (act_interface == NULL)
        return 0;

    else
        return 1;
}

void module::set_interface(const string &strn, const string &strt, const string &strd, const string &strpd, const string &strde)
{
    C_interface *tmp = new C_interface;
    interfaces->add(tmp);
    tmp->set_name(strn);
    tmp->set_type(strt);
    tmp->set_direction(strd);
    tmp->set_text(strpd);
    tmp->set_demand(strde);
}

void module::reset_paramlist(string dir)
{
    if (dir == "in")
        para_in->reset();

    else if (dir == "out")
        para_out->reset();

    else
    {
        print_comment(__LINE__, __FILE__, "ERROR: Wrong Parameterdirection in reset_parameter\n");
        print_exit(__LINE__, __FILE__, 1);
    }
}

parameter *module::next_param(string dir)
{
    if (dir == "in")
    {
        return para_in->next();
    }
    else if (dir == "out")
    {
        return para_out->next();
    }
    else
    {
        cerr << "ERROR: Wrong Parameterdirection in next_param \n";
        print_comment(__LINE__, __FILE__, "ERROR: Wrong Parameterdirection in next_param \n");
    }
    return NULL;
}

void module::set_parameter(const string &strn, const string &strt, const string &strx, const string &strv, const string &dir)
{
    parameter *tmp = new parameter;
    tmp->set_name(strn);
    tmp->set_type(strt);
    tmp->set_text(strx);
    tmp->set_value_list(strv);

    if (dir == "in")
        para_in->add(tmp);

    else if (dir == "out")
        para_out->add(tmp);

    else
    {
        print_comment(__LINE__, __FILE__, "ERROR: Wrong Direction for Parameter !\n");
        print_exit(__LINE__, __FILE__, 1);
    }
}

#ifdef PARA_START

void module::set_parameter(const string &strn, const string &strt, const string &strx, const string &strv, const string &ext, const string &dir)
{
    parameter *tmp = new parameter;
    tmp->set_name(strn);
    tmp->set_type(strt);
    tmp->set_text(strx);
    tmp->set_extension(ext);
    tmp->set_value_list(strv);

    if (dir == "in")
        para_in->add(tmp);

    else if (dir == "out")
        para_out->add(tmp);

    else
    {
        print_comment(__LINE__, __FILE__, "ERROR: Wrong Direction for Parameter !\n");
        print_exit(__LINE__, __FILE__, 1);
    }
}
#endif

parameter *module::get_parameter(const string &dir, const string &name)
{

    if (dir.empty() || name.empty())
        return NULL;

    reset_paramlist(dir);
    parameter *par;
    while ((par = next_param(dir)) != NULL)
    {
        if (par->get_name() == name)
            return par;
    }
    return NULL;
}

C_interface *module::get_inpIntf(const string &name)
{
    if (name.empty())
        return NULL;

    reset_intflist();
    while (next_interface())
    {
        if (get_interfname() == name && get_interfdirection() == "input")
            return act_interface;
    }
    return NULL;
}

C_interface *module::get_outIntf(const string &name)
{
    if (name.empty())
        return NULL;

    reset_intflist();
    while (next_interface())
    {
        if (get_interfname() == name && get_interfdirection() == "output")
            return act_interface;
    }
    return NULL;
}

string module::create_descr()
{
    if (get_name().empty() || get_category().empty() || get_host().empty() || get_moduledescr().empty())
        return "";

    ostringstream os;
    os << get_name() << "\n" << get_category() << "\n" << get_host() << "\n" << get_moduledescr() << "\n";

    // get number of interfaces
    int count = 0;
    int input = 0;
    int output = 0;
    this->reset_intflist();
    this->next_interface();
    while (act_interface != NULL)
    {
        count++;
        if (act_interface->get_direction() == "input")
            input++;
        if (act_interface->get_direction() == "output")
            output++;
        this->next_interface();
    }
    os << input << "\n" << output << "\n";

    // get number of in parameters
    parameter *tmp_para;
    int para_count = 0;
    this->reset_paramlist("in");
    while ((tmp_para = next_param("in")) != NULL)
    {
        para_count++;
    }
    os << para_count << "\n";

    // get number of out parameters
    para_count = 0;
    this->reset_paramlist("out");
    while ((tmp_para = next_param("out")) != NULL)
    {
        para_count++;
    }
    os << para_count << "\n";

    // append the interfaceinformations
    this->reset_intflist();
    this->next_interface();
    while (act_interface != NULL)
    {
        os << act_interface->get_name() << "\n" << act_interface->get_type() << "\n";
        os << act_interface->get_text() << "\n" << act_interface->get_demand() << "\n";
        this->next_interface();
    }

    // append the "in" parameterinformations
    this->reset_paramlist("in");
    while ((tmp_para = next_param("in")) != NULL)
    {
        os << tmp_para->get_name() << "\n" << tmp_para->get_type() << "\n";
        os << tmp_para->get_text() << "\n" << tmp_para->get_org_val() << "\n";

#ifdef PARA_START
        os << tmp_para->get_extension() << "\n";
#endif
    }

    // append the "out" parameterinformations
    this->reset_paramlist("out");
    while ((tmp_para = next_param("out")) != NULL)
    {
        os << tmp_para->get_name() << "\n" << tmp_para->get_type() << "\n";
        os << tmp_para->get_text() << "\n" << tmp_para->get_org_val() << "\n";
    }

    return os.str();
}

void module::read_description(string str)
{
    this->clear_intflist();

    vector<string> list = CTRLHandler::instance()->splitString(str, "\n");

    int iel = 3;
    this->set_moduledescr(list[iel]);
    iel++;

    // get the 4 counters for interfaces and parameter
    istringstream s1(list[iel]);
    iel++;
    int in_interf;
    s1 >> in_interf;

    istringstream s2(list[iel]);
    iel++;
    int out_interf;
    s2 >> out_interf;

    istringstream s3(list[iel]);
    iel++;
    int in_param;
    s3 >> in_param;

    istringstream s4(list[iel]);
    iel++;
    int out_param;
    s4 >> out_param;

    // read the input -interfaces
    for (int i = 1; i <= in_interf; i++)
    {
        this->set_interface(list[iel], list[iel + 1], "input", list[iel + 2], list[iel + 3]);
        iel = iel + 4;
    }

    // read the output -interfaces
    for (int i = 1; i <= out_interf; i++)
    {
        this->set_interface(list[iel], list[iel + 1], "output", list[iel + 2], list[iel + 3]);
        iel = iel + 4;
    }

    // read the input-parameter
    for (int i = 1; i <= in_param; i++)
    {
#ifdef PARA_START
        this->set_parameter(list[iel], list[iel + 1], list[iel + 2], list[iel + 3], list[iel + 4], string("in"));
        iel = iel + 5;
#else
        this->set_parameter(list[iel], list[iel + 1], list[iel + 2], list[iel + 3], "in");
        iel = iel + 4;
#endif
    }

    // read the output-parameter
    for (int i = 1; i <= out_param; i++)
    {
        this->set_parameter(list[iel], list[iel + 1], list[iel + 2], list[iel + 3], string("out"));
        iel = iel + 4;
    }
}
