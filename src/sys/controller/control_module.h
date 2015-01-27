/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CTRL_MODULE_H
#define CTRL_MODULE_H

#include "control_define.h"
#include "control_list.h"
#include "control_port.h"

namespace covise
{

class interface_list;
class parameter;
class paramlist;
class module;

class interface_list : public Liste<C_interface>
{
public:
    interface_list();
    C_interface *get(const string &str);
};

class paramlist : public Liste<parameter>
{
public:
    paramlist();
    parameter *get(const string &name);
};

class module
{
    string name;
    string exec;
    string host;
    string user;
    string category;
    string moduledescr;
    int counter;
    C_interface *act_interface;
    interface_list *interfaces;
    paramlist *para_in;
    paramlist *para_out;

public:
    module();

    void set_name(const string &str);
    void set_exec(const string &str);
    void set_host(const string &str);
    void set_user(const string &str);

    void set_counter(const int &nr);
    int get_counter();
    void reset_counter();

    void set_category(const string &str);
    void set_moduledescr(const string &str);
    void set_interface(const string &strn, const string &strt, const string &strd, const string &strpd, const string &strde);
    void set_parameter(const string &strn, const string &strt, const string &strx, const string &strv, const string &dir);

#ifdef PARA_START
    void set_parameter(const string &strn, const string &strt, const string &strx, const string &strv, const string &ext, const string &dir);
#endif

    string get_name()
    {
        return name;
    };
    string get_exec()
    {
        return exec;
    };
    string get_host()
    {
        return host;
    };

    string get_user()
    {
        return user;
    };
    string get_category()
    {
        return category;
    };
    string get_moduledescr()
    {
        return moduledescr;
    };
    void reset_intflist();
    void clear_intflist();
    void reset_paramlist(string dir);
    parameter *next_param(string dir);
    int next_interface();
    string get_interfname();
    string get_interftype();
    string get_interfdirection();
    string get_interftext();
    string get_interfdemand();
    parameter *get_parameter(const string &dir, const string &name);
    C_interface *get_inpIntf(const string &name);
    C_interface *get_outIntf(const string &name);

    string create_descr();
    void read_description(string s);

private:
    int nodeid;
};

void read_message(string str);
}
#endif
