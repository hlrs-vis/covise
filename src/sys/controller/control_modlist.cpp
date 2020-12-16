/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <covise/covise.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <pwd.h>
#include <dirent.h>
#endif
#include <util/covise_version.h>
#include "CTRLHandler.h"
#include "CTRLGlobal.h"
#include "control_modlist.h"
#include "control_process.h"
#include "covise_module.h"
#include <covise/covise_process.h>
#include "control_def.h"
#include "control_define.h"
#include "control_module.h"
#include "control_coviseconfig.h"
#include <config/CoviseConfig.h>
#include <net/covise_host.h>
#include <config/coConfig.h>

#define MAXMODULES 1000

using namespace covise;

extern FILE *msg_prot;
extern string prot_file;

//**********************************************************************
//
// 			MODULELIST
//
//**********************************************************************

modulelist::modulelist()
    : Liste<module>()
{
}

module *modulelist::get(const string &tmpname, const string &tmphost)
{
    module *tmp;
    string list_name, list_host;

    this->reset();
    do
    {
        tmp = this->next();
        if (tmp == NULL)
        {
            print_comment(__LINE__, __FILE__, "ERROR: Module not found\n");
            return NULL;
        }
        list_name = tmp->get_name();
        list_host = tmp->get_host();
    } while (tmpname != list_name || tmphost != list_host);

    return tmp;
}

string modulelist::create_modulelist()
{

    int count = 0;
    module *mod;
    this->reset();
    while ((mod = this->next()) != NULL)
    {
        count++;
    }

    // read one hostname
    this->reset();
    mod = this->next();

    ostringstream os;
    os << "LIST\n" << mod->get_host() << "\n" << mod->get_user() << "\n" << count << "\n";

    this->reset();
    while ((mod = this->next()) != NULL)
    {
        os << mod->get_name() << "\n" << mod->get_category() << "\n";
    }

    string buffer = os.str();
    return buffer;
}

void modulelist::add_module_list(const string &host, const string &user, const string &data)
{
    // get number of new modules from message
    //LIST      0
    //HOST      1
    //USER      2
    //NUMBER    3

    vector<string> list = CTRLHandler::instance()->splitString(data, "\n");
    int mod_count;
    istringstream inStream(list[3]);
    inStream >> mod_count;

    int iel = 4;
    for (int i = 0; i < mod_count; i++)
    {
        module *tmp = new module;
        tmp->set_name(list[iel]);
        tmp->set_category(list[iel + 1]);
        tmp->set_host(host);
        tmp->set_user(user);
        this->add(tmp);
        iel = iel + 2;
    }
}

void modulelist::rmv_module_list(const string &host)
{
    module *tmp;

    this->reset();
    while ((tmp = this->next()))
    {
        string tmp_host = tmp->get_host();
        if (tmp_host == host)
        {
            this->remove(tmp);
            delete tmp;
        }
    }
}

//**********************************************************************
//
// 			INTERFACELIST
//
//**********************************************************************

DM_interface::DM_interface()
{
}

void DM_interface::set_name(const string &tmp)
{
    name = tmp;
}

DM_int_list::DM_int_list()
    : Liste<DM_interface>()
{
    first = NULL;
}

//**********************************************************************
//
// 			DM_DATA
//
//**********************************************************************

DM_data::DM_data()
{
    dm = NULL;
    list_msg = NULL;
    interface_msg = NULL;
}

DM_data::~DM_data()
{
    if (dm)
        delete dm;
    if (list_msg)
    {
        delete list_msg;
        list_msg = NULL;
    }
    if (interface_msg)
    {
        delete interface_msg;
        interface_msg = NULL;
    }
}

void DM_data::set_hostname(const string &str)
{
    hostname = str;
}

void DM_data::set_user(const string &str)
{
    user = str;
}

void DM_data::set_modname(const string &str)
{
    modname = str;
}

void DM_data::set_passwd(const string &str)
{
    passwd = str;
}

void DM_data::set_DM(AppModule *dmod)
{
    dm = dmod;
}

AppModule *DM_data::get_DM()
{
    return dm;
}

int DM_data::start_crb(int type, const string& host, const string& user, const string& passwd, const string& script_name, coHostType& /*htype*/)
{

    CTRLGlobal* global = CTRLGlobal::getInstance();

    string executable = "crb";
    if (CTRLHandler::instance()->Config->getshminfo(host.c_str()) == COVISE_PROXIE)
        executable = "crbProxy";

    Host* p_host = new Host(host.c_str());
    switch (type)
    {
    case COVISE_LOCAL:
    {
        dm = CTRLGlobal::getInstance()->controller->start_datamanager("crb");
        break;
    }
    case COVISE_REXEC:
    {
        dm = CTRLGlobal::getInstance()->controller->start_datamanager(p_host, user.c_str(), passwd.c_str(), executable.c_str());
        break;
    }
    case COVISE_SSH:
    case COVISE_RSH:
    case COVISE_NQS:
    case COVISE_MANUAL:
    case COVISE_SSLDAEMON:
    case COVISE_SCRIPT:
    case COVISE_ACCESSGRID:
    case COVISE_REMOTE_DAEMON:
    {
        dm = CTRLGlobal::getInstance()->controller->start_datamanager(p_host, user.c_str(), executable.c_str(), type, script_name.c_str());
        break;
    }
    default:
    {
        print_comment(__LINE__, __FILE__, " ERROR: unknown EXEC_TYPE\n");
        //print_exit(__LINE__, __FILE__, 1);
        return 0;
        break;
    }
    }
    if (dm == NULL)
        return (0); // starting datamanager failed

    //comparing main / partner versions

    list_msg = new Message;
    dm->recv_msg(list_msg);

    switch (list_msg->type)
    {
    case COVISE_MESSAGE_EMPTY:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    case COVISE_MESSAGE_SOCKET_CLOSED:
        CTRLHandler::instance()->handleClosedMsg(list_msg);
        break;

    default:
        break;
    }
    // check if we have information about partner version
    string main_version = CoviseVersion::shortVersion();
    if (!list_msg->data.data())
        return 0;

    string version_info = strchr(list_msg->data.data(), '@');
    version_info.erase(0, 1);
    if (!version_info.empty())
    {
        string partner_version = version_info;
        if (main_version != partner_version)
        {
            string text = "Controller WARNING : main covise version = " + main_version + " and the partner version = ";
            text = text + partner_version + " from host " + host + " are different !!!";
            CTRLGlobal::getInstance()->userinterfaceList->sendWarning2m(text);
        }
    } else
    {
        string text = "Controller WARNING : main covise version = " + main_version;
        text = text + " and the partner version = \"unknown\" from host " + host + " are different !!!";
        CTRLGlobal::getInstance()->userinterfaceList->sendWarning2m(text);
    }

    // patch Message to include hostname & user !!
    string module_info;
    module_info.append(list_msg->data.data());
    module_info.insert(5, host + "\n" + user + "\n");
    char* txt = new char[module_info.length() + 1];
    strcpy(txt, module_info.c_str());

    list_msg->data = DataHandle{txt, module_info.length() + 1};

    interface_msg = new Message;
    dm->recv_msg(interface_msg);
    switch (interface_msg->type)
    {
    case COVISE_MESSAGE_EMPTY:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    case COVISE_MESSAGE_SOCKET_CLOSED:
        CTRLHandler::instance()->handleClosedMsg(interface_msg);
        break;
    default:
        break;
    }

    Message *msg = new Message(COVISE_MESSAGE_QUERY_DATA_PATH, "");
    dm->send(msg);
    msg->data = DataHandle{};

    dm->recv_msg(msg);
    if (msg->type == COVISE_MESSAGE_SEND_DATA_PATH)
    {
        dm->covise_path = msg->data.data();
    }
    switch (msg->type)
    {
    case COVISE_MESSAGE_EMPTY:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    case COVISE_MESSAGE_SOCKET_CLOSED:
        CTRLHandler::instance()->handleClosedMsg(msg);
        break;
    default:
        break;
    }
    delete msg;

    set_hostname(host);
    set_user(user);
    set_passwd(passwd);
    return (1);
}

void DM_data::quit()
{
    if (CTRLHandler::instance()->Config->getshminfo(hostname.c_str()) != COVISE_NOSHM)
    {
        Message *msg = new Message(COVISE_MESSAGE_QUIT, "");
        dm->send(msg);

#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "send DM\n%i %i \n %s \n", msg->sender, msg->type, msg->data);
//	fflush(msg_prot);
#endif

#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "received DM\n%i %i \n %s \n", recvmsg->sender, recvmsg->type, recvmsg->data);
//	fflush(msg_prot);
#endif
        delete msg;
    }
}

int DM_data::new_desk()
{
    if (CTRLHandler::instance()->Config->getshminfo(hostname.c_str()) != COVISE_NOSHM)
    {
        Message *msg = new Message(COVISE_MESSAGE_NEW_DESK, "");
        dm->send(msg);
        delete msg;
        return 1;
    }
    return 0;
}

void DM_data::send_msg(Message *msg)
{
    if (CTRLHandler::instance()->CTRLHandler::instance()->Config->getshminfo(hostname.c_str()) != COVISE_NOSHM)
    {
        dm->send(msg);

#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "send DM\n%i %i \n %s \n", msg->sender, msg->type, msg->data);
//	fflush(msg_prot);
#endif
    }
}

//**********************************************************************
//
// 			DM_LIST
//
//**********************************************************************
DM_list::DM_list()
    : Liste<DM_data>()
{
    local = NULL;
}

DM_data *DM_list::get_local()
{
    return local;
}

// start_remote startet einen datamanager auf dem Rechner host, wenn es dort
// noch keinen gibt.

int DM_list::add_crb(int type, const string &host, const string &user, const string &passwd, const string &script_name, coHostType &htype)
{
    DM_data *tmp_data = get(host, user);

    if (tmp_data == NULL)
    {
        tmp_data = new DM_data;
        if (tmp_data->start_crb(type, host, user, passwd, script_name, htype) == 0)
        {
            delete tmp_data;
            return (0);
        }

        AppModule *local_dm = tmp_data->get_DM();
        DM_data *conn_data;
        reset();
        while ((conn_data = next()) != NULL)
        {
            AppModule *dmod = conn_data->get_DM();
            local_dm->connect_datamanager(dmod);
        }
        add(tmp_data);
        if (type == COVISE_LOCAL)
            local = tmp_data;
        return 1;
    }
    else
    {
        return 0;
    }
}

void DM_list::connect_all(AppModule *conn_dmod)
{
    DM_data *tmp_data;

    this->reset();
    while ((tmp_data = this->next()) != NULL)
    {
        AppModule *dmod = tmp_data->get_DM();
        conn_dmod->connect_datamanager(dmod);
    }
}

DM_data *DM_list::get(const string &host)
{
    DM_data *tmp_data = NULL;

    this->reset();
    while ((tmp_data = this->next()) != NULL)
    {
        string tmp_host = tmp_data->get_hostname();
        if (tmp_host == host)
            break;
    }

    return tmp_data;
}

DM_data *DM_list::get(const string &host, const string &user)
{
    DM_data *tmp_data;

    this->reset();
    while ((tmp_data = this->next()) != NULL)
    {
        string tmp_host = tmp_data->get_hostname();
        string tmp_user = tmp_data->get_user();
        if (tmp_host == host && tmp_user == user)
            break;
    }

    return tmp_data;
}

DM_data *DM_list::get(int id)
{
    DM_data *tmp_data = NULL;

    this->reset();
    while ((tmp_data = this->next()) != NULL)
    {
        if (tmp_data->get_mod_id() == id)
            break;
    }

    return tmp_data;
}

void DM_list::quit()
{
    DM_data *tmp_data;

    this->reset();
    while ((tmp_data = this->next()) != NULL)
    {
        tmp_data->quit();
        this->remove(tmp_data);
    }
}

int DM_list::new_desk()
{
    Message *msg = new Message(COVISE_MESSAGE_NEW_DESK, "");
    this->send_msg(msg);
    delete msg;
    return 1;
}

void DM_list::send_msg(Message *msg)
{
    DM_data *tmp_data;

    this->reset();
    while ((tmp_data = this->next()) != NULL)
        tmp_data->send_msg(msg);
}

//**********************************************************************
//
// 			RHOST
//
//**********************************************************************

rhost::rhost()
{
    ctrl = NULL;
    save_info = false;
    htype = "COPARTNER";

#ifdef CONNECT
    intlist = new DM_int_list;
#endif
}

rhost::~rhost()
{
#ifdef CONNECT
    if (intlist)
    {
        delete intlist;
        intlist = NULL;
    }
#endif
}

#ifdef CONNECT

void rhost::set_intflist(const string &data)
{
    int count = 0;
    string dcop = data;
    vector<string> list = CTRLHandler::instance()->splitString(dcop, "\n");

    for (int i = 0; i < list.size(); i++)
    {
        count++;
        M_interface *tmpint = new DM_interface();
        tmpint->set_name(list[i]);
        intlist->add(tmpint);
    }
    intlist->set_count(count);
    intlist->set_default(1);
}

string rhost::get_DC_list()
{

    /*
    *   alle interface auslesen
    *   get hostname
    *   get default
    *   get #nr_of_conn
    *   get interfaces
    */

    ostringstream os;
    os << "DC\n" << get_hostname() << "\n";

    int tmpi = intlist->get_default();
    os << tmpi << "\n";

    tmpi = intlist->get_count();
    os << tmpi << "\n";

    DM_interface *tmpint;
    intlist->reset();
    while ((tmpint = intlist->next()) != NULL)
    {
        os << tmpint->get_name() << "\n";
    }

    return os.str();
}
#endif

void rhost::set_hostname(const string &str)
{
    hostname = str;
}

void rhost::set_user(const string &str)
{
    user = str;
}

void rhost::set_passwd(const string &str)
{
    passwd = str;
}

void rhost::set_type(const string &str)
{
    htype = str;
}

void rhost::mark_save()
{
    save_info = true;
}

void rhost::reset_mark()
{
    save_info = false;
}

bool rhost::get_mark()
{
    return save_info;
}

int rhost::start_ctrl(int type, const string &script_name, coHostType &htype)
{

    string DC_info;

    if (CTRLGlobal::getInstance()->dataManagerList->add_crb(type, hostname, user, passwd, script_name, htype))
    {
        DM_data *tmp_data = CTRLGlobal::getInstance()->dataManagerList->get(hostname, user);
        ctrl = tmp_data->get_DM();

        //  add Modules
        if (tmp_data->list_msg == NULL)
        {
            cerr << endl << " ERROR: list_msg = NULL !!!\n";
            return 0;
        }
        string module_info = tmp_data->list_msg->data.data();
        CTRLGlobal::getInstance()->moduleList->add_module_list(hostname, user, module_info);

//  DC-info received from remote host
#ifdef CONNECT
        Message *p_msg = CTRLGlobal::getInstance()->dataManagerList->get(hostname, user)->interface_msg;
        set_intflist(p_msg->data);
        DC_info = get_DC_list();
#endif
        //    Send Message with current modulelist to all userinterfaces
        CTRLGlobal::getInstance()->userinterfaceList->update_all(module_info, DC_info);

        return (1);
    }
    else
        return (0);
}

void rhost::send_hostadr(const string &hostname)
{
    string tmp(hostname);
    tmp.append("\n");
    Message *msg = new Message(COVISE_MESSAGE_INIT, tmp);
    ctrl->send(msg);
    delete msg;
}

void rhost::recv_msg(Message *msg)
{
    ctrl->recv_msg(msg);
}

void rhost::send(Message *msg)
{
    ctrl->send(msg);
}

void rhost::send_ctrl_quit()
{
    Message *msg = new Message(COVISE_MESSAGE_QUIT, "");
    Message *recvmsg = new Message;

    if (ctrl != NULL)
    {
        ctrl->send(msg);
#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "send rhost\n%i %i \n %s \n", msg->sender, msg->type, msg->data);
//	fflush(msg_prot);
#endif
        msg->data = DataHandle{};
        ctrl->recv_msg(recvmsg);
        switch (recvmsg->type)
        {
        case COVISE_MESSAGE_EMPTY:
        case COVISE_MESSAGE_CLOSE_SOCKET:
        case COVISE_MESSAGE_SOCKET_CLOSED:
            CTRLHandler::instance()->handleClosedMsg(recvmsg);
            break;
        default:
            break;
        }
#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "receive rhost\n%i %i \n %s \n", recvmsg->sender, recvmsg->type, recvmsg->data);
//	fflush(msg_prot);
#endif
        delete ctrl;
    }

    delete msg;
    delete recvmsg;
}

void rhost::print(void)
{
    cerr << endl << " Host " << user << "@" << hostname << endl;
}

//**********************************************************************
//
// 			RHOST_LIST
//
//**********************************************************************

rhost_list::rhost_list()
    : Liste<rhost>()
{
    ;
}

rhost *rhost_list::get(const string &host, const string &user)
{
    rhost *tmp;

    this->reset();
    do
    {
        tmp = this->next();
        if (!tmp)
            break;
    } while (tmp->get_hostname() != host || tmp->get_user() != user);

    return tmp;
}

rhost *rhost_list::get(const string &host)
{
    rhost *tmp;

    this->reset();
    do
    {
        tmp = this->next();
        if (!tmp)
            break;
    } while (tmp->get_hostname() != host);

    return tmp;
}

string rhost_list::get_hosts(const string &local_name, const string &local_user)
{

    int i = 0;

    // select each marked host
    reset();
    rhost *host;
    ostringstream buffer;
    while ((host = next()) != NULL)
    {
        bool mark = host->get_mark();
        if (mark)
        {
            string tmp_host = host->get_hostname();
            string tmp_user = host->get_user();

            if (tmp_host == local_name)
                tmp_host = "LOCAL";

            if (tmp_user == local_user)
                tmp_user = "LUSER";

            i++;

            buffer << tmp_host << "\n" << tmp_user;

            // if this host has a Userinterface
            if (tmp_host != "LOCAL" && CTRLGlobal::getInstance()->userinterfaceList->get(host->get_hostname()))
                buffer << " "
                       << "Partner";
            buffer << "\n";
        }
    }

    string result;
    if (buffer.str().empty())
        result = "1\nLOCAL\nLUSER\n";

    else
    {
        ostringstream os;
        os << i;
        result = os.str() + "\n" + buffer.str();
    }
    return result;
}

int rhost_list::add_host(const string &hostname, const string &user_id, const string &passwd, const string &script_name, coHostType &htype)
{
    Message *ui_msg = new Message;
    (void)ui_msg;


    // add new host in hostlist

    rhost *tmp_host = get(hostname); //, user_id); // restrict to only one user/host
    if (tmp_host == NULL)
    {
        // the host is new
        tmp_host = new rhost;
        tmp_host->set_hostname(hostname);
        tmp_host->set_user(user_id);
        tmp_host->set_passwd(passwd);
        if (htype.get_type() == CO_HOST)
            tmp_host->set_type("COHOST");
    }
    else
    {
        ostringstream os;
        os << "Controller\n \n \n user " << tmp_host->get_user() << " is already started on host " << hostname;
        CTRLGlobal::getInstance()->userinterfaceList->sendWarning2m(os.str());
        return 0;
    }

    // start new datamanager
    int exec_type = CTRLHandler::instance()->Config->getexectype(hostname.c_str());

    if (tmp_host->start_ctrl(exec_type, script_name, htype) == 0)
    {
        string text = "Controller\n \n \n CRB could not be started on host " + hostname + " !!!";
        CTRLGlobal::getInstance()->userinterfaceList->sendWarning2m(text);
        delete tmp_host;
        return 0;
    }

    this->add(tmp_host);

    return (1);
}

int rhost_list::add_local_host(const string &local_user)
{

#ifndef _WIN32
    struct passwd *pwd;
#endif

    Host host;

    if (!host.hasRoutableAddress())
    {
        cerr << "Your IP address is not routable. Collaborative sessions will not work." << endl;
        cerr << "Set COVISE_HOST to the address of your external NIC." << endl;
    }
    if (!host.hasValidAddress())
    {
        cerr << "Unresolvable IP address of host (" << host.getAddress() << ")" << endl;
        cerr << "Please check the network configuration!" << endl;
        return 0;
    }
    cerr << "* Local IP address: " << host.getAddress() << endl;

    string user, passwd;
    if (!local_user.empty())
    {
#ifdef _WIN32
        user = local_user;
        passwd = "none";
#else
        pwd = getpwnam(local_user.c_str());
        if (pwd)
        {
            user = local_user;
            passwd = pwd->pw_passwd;
        }
#endif
    }

    if (user.empty())
    {
#ifdef _WIN32
        user = getenv("USERNAME");
        passwd = "none";
#else
        pwd = getpwuid(getuid());
        user = pwd->pw_name;
        passwd = pwd->pw_passwd;
#endif
    }

    rhost *tmp_host = new rhost;
    tmp_host->set_hostname(host.getAddress());
    tmp_host->set_user(user);
    tmp_host->set_passwd(passwd);
    this->add(tmp_host);

    coHostType htype(CO_PARTNER);
    int ret = tmp_host->start_ctrl(COVISE_LOCAL, "", htype);

    return ret;
}

int rhost_list::rmv_host(const string &hostname, const string &user_id)
{

    Host localhost;
    userinterface *p_ui = CTRLGlobal::getInstance()->userinterfaceList->get_master();
    string master_hostname = p_ui->get_host();
    if (hostname == localhost.getAddress() || hostname == master_hostname)
    {
        CTRLGlobal::getInstance()->userinterfaceList->sendWarning2m("Controller\n \n \n REMOVING CONTROLLER OR MASTER HOST IS NOT ALLOWED !!!");
        return 0;
    }

    rhost *tmp_host = this->get(hostname, user_id);
    if (tmp_host == NULL)
    {
        string text = "Controller\n \n \n host " + user_id + "@" + hostname + "  not found  !!!";
        CTRLGlobal::getInstance()->userinterfaceList->sendWarning2m(text);
        return 0;
    }

    DM_data *tmp_data = CTRLGlobal::getInstance()->dataManagerList->get(hostname);
    if (tmp_data == NULL)
    {
        string text = "Controller\n \n \n  crb not found for the host " + hostname + "  !!!";
        CTRLGlobal::getInstance()->userinterfaceList->sendWarning2m(text);
        return 0;
    }

    //
    // remove all modules on target host
    //
    tmp_data->new_desk();

    Message *tmpmsg = new Message(COVISE_MESSAGE_REMOVED_HOST, user_id + "\n" + hostname + "\n");
    CTRLGlobal::getInstance()->netList->reset();
    net_module *p_netmod = CTRLGlobal::getInstance()->netList->next();
    while (p_netmod)
    {
        int mod_type = p_netmod->is_renderer();
        if (mod_type == REND_MOD)
        {
            render_module *p_rend = (render_module *)p_netmod;
            displaylist *p_displist = p_rend->get_displays();
            p_displist->reset();
            display *p_disp = p_displist->next();
            while (p_disp)
            {
                string tmp = p_disp->get_hostname();
                if (tmp != hostname)
                    p_disp->send_message(tmpmsg);

                p_disp = p_displist->next();
            }
        }
        string tmp = p_netmod->get_host();
        if (tmp == hostname)
        {
            p_netmod->set_alive(0);
            CTRLGlobal::getInstance()->modUIList->delete_mod(p_netmod->get_name(), p_netmod->get_nr(), tmp);
        }
        p_netmod = CTRLGlobal::getInstance()->netList->next();
    }

    delete tmpmsg;

    CTRLGlobal::getInstance()->netList->reset();
    p_netmod = CTRLGlobal::getInstance()->netList->next();
    while (p_netmod)
    {
        string tmp = p_netmod->get_host();
        if (tmp == hostname)
        {
            if (p_netmod->get_mirror_status())
            {
                // mirror ?
                net_module *p_mirror;
                p_netmod->reset_mirror_list();
                while ((p_mirror = p_netmod->mirror_list_next()) != NULL)
                {
                    p_mirror->set_mirror_status(NOT_MIRR);
                    p_mirror->set_mirror_node(NULL);
                }
            }
            CTRLGlobal::getInstance()->netList->re_move(p_netmod->get_name(), p_netmod->get_nr(), tmp, -1);
            CTRLGlobal::getInstance()->netList->reset();
        }
        else
        {
            // remove partner renderer displays
            int mod_type = p_netmod->is_renderer();
            if (mod_type == REND_MOD)
                ((render_module *)p_netmod)->remove_display(tmp_host);
        }
        p_netmod = CTRLGlobal::getInstance()->netList->next();
    }

    //
    // remove mapeditor
    //
    userinterface *tmpui = CTRLGlobal::getInstance()->userinterfaceList->get(hostname);
    if (tmpui)
    {
        tmpui->quit();
        CTRLGlobal::getInstance()->userinterfaceList->remove(tmpui);
        delete tmpui;
    }

    //  remove Modules from modulelist
    CTRLGlobal::getInstance()->moduleList->rmv_module_list(hostname);

    //
    // update mapeditors
    //

    string mod_info;
    Message *p_msg = tmp_data->list_msg;
    if (p_msg)
    {
        // patch the LIST message
        const char *mod_list = p_msg->data.data();
        mod_info = "RMV_LIST\n";
        mod_info.append(hostname + "\n");
        mod_info.append(user_id + "\n");
        mod_info.append(mod_list + 5);
        mod_info.append("\n");
    }

    else
        cerr << endl << "ERROR: rmv_host() list_msg  ==  NULL !!!" << endl;

    //  DC-info received from remote host
    string DC_info;

#ifdef CONNECT
    DC_info = tmp_host->get_DC_list();
    DC_infp = NULL;
#endif

    // Send Message to Userinterfaces
    CTRLGlobal::getInstance()->userinterfaceList->update_all(mod_info, DC_info);

    //
    // remove crb
    //
    tmp_data->quit();
    CTRLGlobal::getInstance()->dataManagerList->remove(tmp_data);

    //
    // notify the other CRBs
    //

    tmpmsg = new Message(COVISE_MESSAGE_CRB_QUIT, string(hostname));
    CTRLGlobal::getInstance()->dataManagerList->reset();
    while ((tmp_data = CTRLGlobal::getInstance()->dataManagerList->next()) != NULL)
    {
        tmp_data->send_msg(tmpmsg);
    }
    delete tmpmsg;
    delete tmp_data;

    //
    //  rmv host from the hostlist
    //
    this->remove(tmp_host);
    delete tmp_host;

    return (1);
}

void rhost_list::mark_host()
{

    net_module *tmp_mod;
    CTRLGlobal::getInstance()->netList->reset();
    while ((tmp_mod = CTRLGlobal::getInstance()->netList->next()) != NULL)
    {

        string mod_host = tmp_mod->get_host();
        module *tmp_link = tmp_mod->get_type();
        string mod_user = tmp_link->get_user();

        rhost *tmp_rhost = this->get(mod_host, mod_user);
        tmp_rhost->mark_save();
    }
}

void rhost_list::mark_all()
{
    rhost *tmp_rhost;

    this->reset();
    while ((tmp_rhost = this->next()) != NULL)
    {
        tmp_rhost->mark_save();
    }
}

void rhost_list::reset_mark()
{
    rhost *tmp_rhost;

    this->reset();
    while ((tmp_rhost = this->next()) != NULL)
    {
        tmp_rhost->reset_mark();
    }
}

void rhost_list::print()
{
    rhost *p_host;

    reset();
    p_host = next();
    while (p_host)
    {
        p_host->print();
        p_host = next();
    }
}

//**********************************************************************
//
// 			USERINTERFACE
//
//**********************************************************************


bool userinterface::sendMessage(const Message *msg) {
    if (ui)
        return ui->send(msg);

#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "send UI\n%i %i \n %s \n", msg->sender, msg->type, msg->data);
//	fflush(msg_prot);
#endif
    return false;
}

bool userinterface::sendMessage(const UdpMessage *msg) {
    return false;
}

userinterface::userinterface()
{
    ui = NULL;
    status = "INIT";
}

void userinterface::set_host(const string &str)
{
    hostname = str;
}

void userinterface::set_userid(const string &str)
{
    userid = str;
}

void userinterface::set_passwd(const string &str)
{
    passwd = str;
}

int UIMapEditor::start(bool restart) // if restart is true a restart was done
{
    string instanz("001");

    // get Datamanager for host
    DM_data *tmp_data = CTRLGlobal::getInstance()->dataManagerList->get(hostname);
    if (tmp_data == NULL)
        return 0;

    AppModule *dmod = tmp_data->get_DM();
    if (dmod == NULL)
        return 0;

    ui = CTRLGlobal::getInstance()->controller->start_applicationmodule(USERINTERFACE, "mapeditor", dmod, instanz.c_str(), Start::Normal);
    if (ui == NULL)
        return 0;
    if (ui->connect(dmod) == 0)
        return 0;

    // send status-Message
    string tmp = status;
    if (restart)
        tmp.append("_RESTART");

    if (CTRLHandler::instance()->m_miniGUI)
        tmp.append("\nMINI_GUI");

    Message *msg = new Message(COVISE_MESSAGE_UI, tmp);
    ui->send(msg);
    delete msg;

    CTRLGlobal::getInstance()->userinterfaceList->update_ui(this);

    // wait for OK from Mapeditor

    msg = new Message;
    recv_msg(msg);

    if (msg->type == COVISE_MESSAGE_MSG_OK)
    {
        if (msg->data.data())
        {
            msg->type = COVISE_MESSAGE_UI;
            dmod->send(msg); //message for CRB that an embedded renderer is possible
        }

        delete msg;
        return 1;
    }

    delete msg;
    return 0;
}

int UISoap::start(bool)
{
    string instance("ws0001");

    // get Datamanager for host
    DM_data *tmp_data = CTRLGlobal::getInstance()->dataManagerList->get(hostname);
    if (tmp_data == NULL)
        return 0;

    AppModule *dmod = tmp_data->get_DM();
    if (dmod == 0)
        return 0;

    //Determine from config whether to use WSInterface
    bool ws_enabled = covise::coConfig::getInstance()->getBool("enable", "System.WSInterface", true);
    if (ws_enabled)
    {
        ui = CTRLGlobal::getInstance()->controller->start_applicationmodule(USERINTERFACE, "wsinterface", dmod, instance.c_str(), Start::Normal);
    }
    else
    {
        ui = 0;
    }

    if (ui == 0)
        return 0;

    if (ui->connect(dmod) == 0)
        return 0;

    // send status-Message
    string tmp = status;
    tmp.append("\n");

    Message *msg = new Message(COVISE_MESSAGE_UI, tmp);
    ui->send(msg);
    delete msg;

    CTRLGlobal::getInstance()->userinterfaceList->update_ui(this);

    // wait for OK from Mapeditor

    msg = new Message;
    recv_msg(msg);

    if (msg->type == COVISE_MESSAGE_MSG_OK)
    {
        delete msg;
        return 1;
    }

    delete msg;
    return 0;
}

int userinterface::restart()
{

    // start user interface
    start(true);

    if (ui == NULL)
        return 0;

    // send current net to UIF
    // send current controller information to ui
    Message *tmp_msg = new Message(COVISE_MESSAGE_UI, "START_READING\n");
    ui->send(tmp_msg);
    delete tmp_msg;

    // loop over all modules

    net_module *mod;
    CTRLGlobal::getInstance()->netList->reset();
    while ((mod = CTRLGlobal::getInstance()->netList->next()) != NULL)
    {
        cerr << mod->get_name() << endl;
        ostringstream mybuf;
        mybuf << "INIT\n" << mod->get_name() << "\n" << mod->get_nr() << "\n";
        mybuf << mod->get_host() << "\n" << mod->get_x_pos() << "\n" << mod->get_y_pos() << "\n";
        tmp_msg = new Message(COVISE_MESSAGE_UI, mybuf.str());
        ui->send(tmp_msg);
        delete tmp_msg;

        ostringstream os;
        os << "DESC\n";
        module *mymod = mod->get_type();
        if (mymod)
            os << mymod->create_descr();

        tmp_msg = new Message(COVISE_MESSAGE_UI, os.str());
        ui->send(tmp_msg);
        delete tmp_msg;

        ostringstream oss;
        oss << "MODULE_TITLE\n" << mod->get_name() << "\n" << mod->get_nr() << "\n";
        oss << mod->get_host() << "\n" << mod->get_title() << "\n";
        tmp_msg = new Message(COVISE_MESSAGE_UI, oss.str());
        ui->send(tmp_msg);
        delete tmp_msg;

        // send current parameter
        // only input parameter

        vector<string> name_list;
        vector<string> type_list;
        vector<string> val_list;
        vector<string> panel_list;
        int n_pc = mod->get_inpars_values(&name_list, &type_list, &val_list, &panel_list);

        // loop over all input parameters
        for (int i = 0; i < n_pc; i++)
        {
            string value = val_list[i];

            if (type_list[i] == "Browser")
            {
                CTRLHandler::instance()->handleBrowserPath(mod->get_name(), mod->get_nr(), mod->get_host(),
                                                           mod->get_host(), name_list[i], value);
            }

            ostringstream stream;
            stream << "PARAM_RESTART\n" << mod->get_name() << "\n" << mod->get_nr() << "\n" << mod->get_host() << "\n"
                   << name_list[i] << "\n" << type_list[i] << "\n" << value;
            Message *msg2 = new Message(COVISE_MESSAGE_UI, stream.str());
            ui->send(msg2);
            delete msg2;

            // send ADD_PANEL
            ostringstream mybuf2;
            mybuf2 << "ADD_PANEL\n" << mod->get_name() << "\n" << mod->get_nr() << "\n" << mod->get_host() << "\n";
            mybuf2 << name_list[i] << "\n" << panel_list[i];
            msg2 = new Message(COVISE_MESSAGE_UI, mybuf2.str());
            CTRLGlobal::getInstance()->userinterfaceList->send_all(msg2);
            delete msg2;
        }
    }

    // send connection informations
    object *tmp_obj;

    CTRLGlobal::getInstance()->objectList->reset();
    while ((tmp_obj = CTRLGlobal::getInstance()->objectList->next()) != NULL)
    {
        int i = 0;
        string buffer = tmp_obj->get_simple_connection(&i);
        if (!buffer.empty() || i != 0)
        {
            ostringstream mybuf2;
            mybuf2 << "OBJCONN2\n" << i << "\n" << buffer;
            Message *tmp_msg = new Message(COVISE_MESSAGE_UI, mybuf2.str());
            ui->send(tmp_msg);
            delete tmp_msg;
        }
    }

    //
    // send end message to UIF
    //
    tmp_msg = new Message(COVISE_MESSAGE_UI, "END_READING\ntrue");
    ui->send(tmp_msg);
    delete tmp_msg;

    return (1);
}

// test by RM
int userinterface::xstart(const string &pyFile)
{
    string instanz("001");

    // get Datamanager for host
    DM_data *tmp_data = CTRLGlobal::getInstance()->dataManagerList->get(hostname);
    if (tmp_data == NULL)
        return 0;

    AppModule *dmod = tmp_data->get_DM();
    if (dmod == NULL)
        return 0;

    string cmdStr;
#ifdef _WIN32
    cmdStr.append("..\\..\\Python\\scriptInterface.bat ");
#else
    cmdStr.append("scriptInterface ");
#endif

    if (!pyFile.empty())
        cmdStr.append(pyFile);

    ui = CTRLGlobal::getInstance()->controller->start_applicationmodule(USERINTERFACE, cmdStr.c_str(), dmod, instanz.c_str(), Start::Normal);
    if (ui == NULL)
        return 0;
    if (ui->connect(dmod) == 0)
        return 0;

    // send status-Message
    string tmp = status;
    Message *msg = new Message(COVISE_MESSAGE_UI, tmp);
    ui->send(msg);
    delete msg;

    CTRLGlobal::getInstance()->userinterfaceList->update_ui(this);

    // wait for OK from Mapeditor

    msg = new Message;
    recv_msg(msg);

    if (msg->type == COVISE_MESSAGE_MSG_OK)
    {
        rendererIsPossible = false;
        if (msg->data.data() && !strcmp(msg->data.data(), "RENDERER_INSIDE_OK"))
        {
            rendererIsPossible = true;
        }
        delete msg;
        return 1;
    }

    delete msg;
    return 0;
}

void userinterface::quit()
{
    Message *ui_msg = new Message(COVISE_MESSAGE_QUIT, "");
    ui->send(ui_msg);

#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "send UI\n%i %i \n %s \n", msg->sender, msg->type, msg->data);
//	fflush(msg_prot);
#endif

#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "receive UI\n%i %i \n %s \n", msg->sender, msg->type, msg->data);
//	fflush(msg_prot);
#endif
    delete ui_msg;
}

void userinterface::set_status(const string &str)
{
    status = str;
}

void userinterface::change_status(const string &str)
{
    status = str;
    Message *msg = new Message(COVISE_MESSAGE_UI, str);
    ui->send(msg);

#ifdef DEBUG
    fprintf(msg_prot, "---------------------------------------------------\n");
    fprintf(msg_prot, "send Status to %i \n", this->get_mod_id());
    fflush(msg_prot);
#endif

    delete msg;
}

void userinterface::change_master(const string &user, const string &host)
{
    string text = "MASTERREQ\n" + user + "\n" + host + "\n\n";
    Message *msg = new Message(COVISE_MESSAGE_UI, text);
    ui->send(msg);

#ifdef DEBUG
    fprintf(msg_prot, "---------------------------------------------------\n");
    fprintf(msg_prot, "MASTERREQ \n");
    fflush(msg_prot);
#endif

    delete msg;
}

//**********************************************************************
//
// 			UI_LIST
//
//*********************************************************************/

ui_list::ui_list()
    : Liste<userinterface>()
{
    m_slaveUpdate = false;
}

ui_list::~ui_list()
{
    ;
}

userinterface *ui_list::get(const string &hostname, const string &user)
{

    userinterface *tmp;

    this->reset();
    do
    {
        tmp = this->next();
        if (tmp == NULL)
            break;
    } while (tmp->get_host() != hostname || tmp->get_userid() != user);

    return tmp;
}

userinterface *ui_list::get(const string &hostname)
{
    userinterface *tmp;

    this->reset();
    do
    {
        tmp = this->next();
        if (tmp == NULL)
            break;
    } while (tmp->get_host() != hostname);

    return tmp;
}

userinterface *ui_list::get(int sender_no)
{
    userinterface *tmp = NULL;

    this->reset();
    while ((tmp = this->next()) != NULL)
    {
        if (sender_no == tmp->get_mod_id())
            break;
    }
    return tmp;
}

void ui_list::set_iconify(int ic)
{
    iconify = ic;
}

void ui_list::set_maximize(int maxi)
{
    maximize = maxi;
}

userinterface *ui_list::get_master()
{
    userinterface *tmp;

    this->reset();
    while ((tmp = this->next()) != NULL)
    {
        if (tmp->get_status() == "MASTER")
            break;
    }
    return tmp;
}

int ui_list::start_local_Mapeditor(const string &moduleinfo)
{
    (void)moduleinfo;

    DM_data *tmp_dm = CTRLGlobal::getInstance()->dataManagerList->get_local();

    string local_user = tmp_dm->get_user();
    string local_name = tmp_dm->get_hostname();

    userinterface *local = new UIMapEditor;

    local->set_host(local_name);
    local->set_userid(local_user);
    local->set_status("MASTER");
    local->set_passwd("LOCAL");
    int ret = local->start(false);
    if (ret == 0)
    {
        delete local;
        return 0;
    }

    this->locals.push_back(local);
    add(local);

    if (iconify)
    {
        Message *msg = new Message(COVISE_MESSAGE_UI, "ICONIFY");
        local->send(msg);
        delete msg;
    }

    if (maximize)
    {
#ifndef _AIRBUS
        Message *msg = new Message(COVISE_MESSAGE_UI, "MAXIMIZE");
#else
        Message *msg = new Message(COVISE_MESSAGE_UI, "PREPARE_CSCW");
#endif
        local->send(msg);
        delete msg;
    }
    return 1;
}

int ui_list::start_local_WebService(const string &moduleinfo)
{
    (void)moduleinfo;

    DM_data *tmp_dm = CTRLGlobal::getInstance()->dataManagerList->get_local();

    string local_name = tmp_dm->get_hostname();
    string local_user = tmp_dm->get_user();

    userinterface *local = new UISoap;

    local->set_host(local_name);
    local->set_userid(local_user);
    local->set_status("MASTER");
    local->set_passwd("LOCAL");
    int ret = local->start(false);
    if (ret == 0)
    {
        delete local;
        return 0;
    }

    this->locals.push_back(local);
    add(local);

    return 1;
}

// test by RM
int ui_list::start_local_xuif(const string &moduleinfo, const string &pyFile)
{
    (void)moduleinfo;

    DM_data *tmp_dm = CTRLGlobal::getInstance()->dataManagerList->get_local();
    string local_name = tmp_dm->get_hostname();
    string local_user = tmp_dm->get_user();

    userinterface *local = new UIMapEditor;

    local->set_host(local_name);
    local->set_userid(local_user);
    local->set_status("MASTER");
    local->set_passwd("LOCAL");
    int ret = local->xstart(pyFile);
    if (ret == 0)
    {
        delete local;
        return 0;
    }

    this->locals.push_back(local);
    add(local);

    return 1;
}

// test by RM end

bool ui_list::add_config(const string &file, const string &mapfile)
{
    char *host, *userid, *stat;
    int err;
    FILE *fp = NULL;
    Message *msg;
    string ui_host, ui_user;
    bool test = true;
    userinterface *tmp_ui;

    if (!file.empty())
    {
        if ((fp = fopen(file.c_str(), "r")) == NULL)
        {
            print_comment(__LINE__, __FILE__, " File open error! Can't read uif-description \n");
            cerr << "*                                                             *" << endl;
            cerr << "*            Could not read Session-file " << file << endl;
            cerr << "*                                                             *" << endl;
            print_exit(__LINE__, __FILE__, 1);
        }
    }

    for (;;)
    {
        if (!file.empty())
        {
            // read hostname, userid and status from file
            host = new char[80];
            err = fscanf(fp, "%s", host);
            if (err == EOF)
            {
                break;
            }
            userid = new char[80];
            err = fscanf(fp, "%s", userid);
            stat = new char[80];
            err = fscanf(fp, "%s", stat);
            status_next = stat;
            // check, if this entry exists
            test = false;
            this->reset();
            while ((tmp_ui = this->next()) != NULL)
            {
                ui_host = tmp_ui->get_host();
                ui_user = tmp_ui->get_userid();
                if (ui_host == host && ui_user == userid)
                {
                    test = true;
                    break;
                }
            }
            if ((test == false) && ((CTRLHandler::instance()->Config->getexectype(host) == COVISE_SSH) || (CTRLHandler::instance()->Config->getexectype(host) == COVISE_REMOTE_DAEMON) || (CTRLHandler::instance()->Config->getexectype(host) == COVISE_ACCESSGRID) || (CTRLHandler::instance()->Config->getexectype(host) == COVISE_RSH) || (CTRLHandler::instance()->Config->getexectype(host) == COVISE_NQS) || (CTRLHandler::instance()->Config->getexectype(host) == COVISE_MANUAL) || (CTRLHandler::instance()->Config->getexectype(host) == COVISE_SCRIPT)))
            {
                if (!config_action(mapfile, host, userid, "none"))
                {
                    test = false;
                    break;
                }
                test = true;
            }

            if (test == false)
            {
                string text = "ADDHOST\n";
                text.append(host);
                text.append("\n");
                text.append(userid);
                text.append("\nPassword\n\n");

                msg = new Message(COVISE_MESSAGE_UI, text);
                for (std::list<userinterface *>::iterator i = locals.begin();
                     i != locals.end();
                     ++i)
                {
                    (*i)->send(msg);
                }
                delete msg;
                break;
            }
        }

        if (file.empty())
        {
            break;
        }
    }

    if (fp)
        fclose(fp);

    return test;
}

int ui_list::config_action(const string &mapfile, const string &host, const string &userid, const string &passwd)
{
    return add_partner(mapfile, host, userid, passwd, "");
}

bool ui_list::slave_update()
{

    if (!m_slaveUpdate)
        return false;

    net_module *p_netmod;
    CTRLGlobal::getInstance()->netList->reset();
    while ((p_netmod = CTRLGlobal::getInstance()->netList->next()) != NULL)
    {
        int mod_type = p_netmod->is_renderer();
        if (mod_type == REND_MOD)
        {
            userinterface *master;
            master = get_master();
            string hostname = master->get_host();
            string userid = master->get_userid();
            string name = p_netmod->get_name();
            string inst = p_netmod->get_nr();
            displaylist *disps = ((render_module *)p_netmod)->get_displays();
            display *disp = disps->get(hostname, userid);
            if (disp)
            {
                ostringstream os;
                os << "UPDATE\n" << name << "\n" << inst << "\n" << hostname << "\n";
                Message *msg = new Message(COVISE_MESSAGE_RENDER, os.str());
                disp->send_message(msg);
                delete msg;
            }
        }
    }
    m_slaveUpdate = false;
    return true;
}

int ui_list::add_partner(const string &filename, const string &host, const string &userid, const string &passwd, const string &script_name)
{

    // add here the interfacelist. Advantage: the information is transmitted
    // in one message
    // action for new userinterface

    if (this->get(host))
    { // a partner already exists on that host
        string text = "Controller\n \n \n A partner already exists on host " + host + " !!!";
        sendWarning2m(text);
        return 0;
    }

    rhost *p_rhost = CTRLGlobal::getInstance()->hostList->get(host, userid);
    if (p_rhost == NULL)
    {
        // no crb exist on that host for the specified user
        // start crb
        coHostType htype(CO_PARTNER);
        if (CTRLGlobal::getInstance()->hostList->add_host(host, userid, passwd, script_name, htype) == 0)
        { // error while trying to add host
            return 0;
        }
    }

    // FIXME Should I start a WebService interface also?
    userinterface *tmp = new UIMapEditor;

    tmp->set_host(host);
    tmp->set_userid(userid);
    tmp->set_status("SLAVE");
    tmp->set_passwd(passwd);

    // start Mapeditor
    if ((tmp->start(false)) == 0)
    { // error while trying to start mapeditor
        string text = "Controller\n \n \n Mapeditor could not be started on host " + host + "!!!";
        sendWarning2m(text);
        delete tmp;
        return 0;
    }

    this->add(tmp);

    if (iconify)
    {
        Message *msg = new Message(COVISE_MESSAGE_UI, "ICONIFY");
        tmp->send(msg);
        delete msg;
    }

    if (maximize)
    {
        Message *msg = new Message(COVISE_MESSAGE_UI, "MAXIMIZE");
        tmp->send(msg);
        delete msg;
    }

    // send current net to UIF
    DM_data *dm_local = CTRLGlobal::getInstance()->dataManagerList->get_local();
    string local_name = dm_local->get_hostname();
    string local_user = dm_local->get_user();

    // send filename to UIF
    Message *tmp_msg = new Message(COVISE_MESSAGE_UI, "START_READING\n" + filename);
    tmp->send(tmp_msg);
    delete tmp_msg;

    // loop over all modules
    net_module *mod;
    CTRLGlobal::getInstance()->netList->reset();
    while ((mod = CTRLGlobal::getInstance()->netList->next()) != NULL)
    {
        ostringstream mybuf;
        mybuf << "INIT\n" << mod->get_name() << "\n" << mod->get_nr() << "\n";
        mybuf << mod->get_host() << "\n" << mod->get_x_pos() << "\n" << mod->get_y_pos() << "\n";
        tmp_msg = new Message(COVISE_MESSAGE_UI, mybuf.str());
        tmp->send(tmp_msg);
        delete tmp_msg;

        ostringstream os;
        os << "DESC\n";
        module *mymod = mod->get_type();
        if (mymod)
            os << mymod->create_descr();

        tmp_msg = new Message(COVISE_MESSAGE_UI, os.str());
        tmp->send(tmp_msg);
        delete tmp_msg;

        ostringstream oss;
        oss << "MODULE_TITLE\n" << mod->get_name() << "\n" << mod->get_nr() << "\n";
        oss << mod->get_host() << "\n" << mod->get_title() << "\n";
        tmp_msg = new Message(COVISE_MESSAGE_UI, oss.str());
        tmp->send(tmp_msg);
        delete tmp_msg;

        // send current parameter
        // only input parameter

        vector<string> name_list;
        vector<string> type_list;
        vector<string> val_list;
        vector<string> panel_list;
        int n_pc = mod->get_inpars_values(&name_list, &type_list, &val_list, &panel_list);

        // loop over all input parameters
        for (int i = 0; i < n_pc; i++)
        {
            string value = val_list[i];

            if (type_list[i] == "Browser")
            {
                CTRLHandler::instance()->handleBrowserPath(mod->get_name(), mod->get_nr(), mod->get_host(),
                                                           mod->get_host(), name_list[i], value);
            }

            ostringstream stream;
            stream << "PARAM_ADD\n" << mod->get_name() << "\n" << mod->get_nr() << "\n" << mod->get_host() << "\n"
                   << name_list[i] << "\n" << type_list[i] << "\n" << value;

            Message *msg2 = new Message(COVISE_MESSAGE_UI, stream.str());
            tmp->send(msg2);
            delete msg2;

            // send ADD_PANEL
            ostringstream mybuf;
            mybuf << "ADD_PANEL\n" << mod->get_name() << "\n" << mod->get_nr() << "\n" << mod->get_host() << "\n";
            mybuf << name_list[i] << "\n" << panel_list[i];
            msg2 = new Message(COVISE_MESSAGE_UI, mybuf.str());
            CTRLGlobal::getInstance()->userinterfaceList->send_all(msg2);
            delete msg2;
        }
    }

    // send connection informations
    CTRLGlobal::getInstance()->objectList->reset();
    object *tmp_obj;
    while ((tmp_obj = CTRLGlobal::getInstance()->objectList->next()) != NULL)
    {
        int i = 0;
        string buffer = tmp_obj->get_simple_connection(&i);
        if (!buffer.empty() || i != 0)
        {
            ostringstream text;
            text << "OBJCONN2\n" << i << "\n" << buffer;
            Message *tmp_msg = new Message(COVISE_MESSAGE_UI, text.str());
            tmp->send(tmp_msg);
            delete tmp_msg;
        }
    }

    // send end message to UIF
    tmp_msg = new Message(COVISE_MESSAGE_UI, "END_READING\ntrue");
    tmp->send(tmp_msg);
    delete tmp_msg;

    // raise partner renderers
    bool tmp_su;
    net_module *p_netmod;
    CTRLGlobal::getInstance()->netList->reset();
    while ((p_netmod = CTRLGlobal::getInstance()->netList->next()) != NULL)
    {
        int mod_type = p_netmod->is_renderer();
        if (mod_type == REND_MOD && p_netmod->get_mirror_status() != CPY_MIRR)
        {
            tmp_su = ((render_module *)p_netmod)->add_display(tmp);
            if (tmp_su)
                m_slaveUpdate = true;
        }
    }

    return (1);
}

int ui_list::rmv_partner(const string &host, const string &user_id)
{

    userinterface *p_ui = get_master();
    string master_hostname = p_ui->get_host();
    if (host == master_hostname)
    {
        sendWarning2m("Controller\n \n \n REMOVING MASTER PARTNER IS NOT ALLOWED !!!");
        return 0;
    }

    userinterface *ui = this->get(host, user_id);

    if (ui == NULL)
    {
        // no partner exists on that host
        string text = "Controller\n \n \n The partner " + user_id + " doesn't exist on host " + host + " !!!";
        sendWarning2m(text);
        return 0;
    }

    CTRLGlobal::getInstance()->netList->reset();
    net_module *p_netmod;
    while ((p_netmod = CTRLGlobal::getInstance()->netList->next()) != NULL)
    {
        int mod_type = p_netmod->is_renderer();
        if (mod_type == REND_MOD)
        {
            rhost *tmp_host = CTRLGlobal::getInstance()->hostList->get(host);
            ;
            ((render_module *)p_netmod)->remove_display(tmp_host);
        }
    }

    ui->quit();
    this->remove(ui);
    delete ui;

    return 1;
}

void ui_list::send_master(Message *msg)
{
    userinterface *tmp = this->get_master();
    if (tmp)
        tmp->send(msg);
}

void ui_list::send_slave(Message *msg)
{
    userinterface *tmp;

    this->reset();
    while ((tmp = this->next()) != NULL)
    {
        // compare the sender-number and the module-number
        AppModule *mod = tmp->get_mod();
        int id = mod->get_id();
        if (id != msg->sender)
            tmp->send(msg);
    }
}

void ui_list::sendError(const string &txt)
{
    if (txt.empty())
        return;

    Message *msg = new Message(COVISE_MESSAGE_COVISE_ERROR, txt);
    send_all(msg);
    delete msg;
}

void ui_list::sendWarning(const string &txt)
{
    if (txt.empty())
        return;

    Message *msg = new Message(COVISE_MESSAGE_WARNING, txt);
    send_all(msg);
    delete msg;
}

void ui_list::sendError2m(const string &txt)
{
    if (txt.empty())
        return;

    Message *msg = new Message(COVISE_MESSAGE_COVISE_ERROR, txt);
    send_master(msg);
    delete msg;
}

void ui_list::sendWarning2m(const string &txt)
{
    if (txt.empty())
        return;

    Message *msg = new Message(COVISE_MESSAGE_WARNING, txt);
    send_master(msg);
    delete msg;
}

bool ui_list::testid(int msg_id)
{

    bool test = false;
    userinterface *tmp;
    this->reset();
    while ((tmp = this->next()) != NULL)
    {
        AppModule *mod = tmp->get_mod();
        int id = mod->get_id();
        if (id == msg_id)
        {
            test = true;
            break;
        }
    }

    return test;
}

void ui_list::send_all(Message *msg)
{
    userinterface *tmp;
    this->reset();
    while ((tmp = this->next()) != NULL)
    {
        tmp->send(msg);
#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "send_all\n%i %i \n %s \n", msg->sender, msg->type, msg->data);
//	fflush(msg_prot);
#endif
    }
}

void ui_list::quit_and_del()
{
    userinterface *tmp;

    this->reset();
    while ((tmp = this->next()) != NULL)
    {
        tmp->quit();
#ifdef DEBUG
//	fprintf(msg_prot, "---------------------------------------------------\n");
//	fprintf(msg_prot, "send_all\n%i %i \n %s \n", msg->sender, msg->type, msg->data);
//	fflush(msg_prot);
#endif
        this->remove(tmp);
        //this->reset();
    }
}

void ui_list::change_master(int sender_no, const string &user, const string &host)
{

    // save the sender_no
    MR_sender = sender_no;

    // get master
    userinterface *tmp = this->get_master();
    // send master-request
    tmp->change_master(user, host);
}

void ui_list::force_master(const string &host)
{
    // get master
    userinterface *master = get_master();
    userinterface *newMaster = get(host);
    if (newMaster != master && newMaster != NULL)
    {
        master->change_status("SLAVE");
        newMaster->change_status("MASTER");
    }
}

void ui_list::send_new_status(const string &status)
{
    userinterface *master = this->get_master();
    // set new status at ther master-uif
    master->set_status(status);

    // change status at the asking uif
    userinterface *other = this->get(MR_sender);

    if (status == "MASTER")
        other->change_status("SLAVE");

    else
        other->change_status("MASTER");
}

void ui_list::update_ui(userinterface *ui)
{

    // send all already existing lists to new uif
    if (ui)
    {
        Message ui_msg;
        ui_msg.type = COVISE_MESSAGE_UI;
        CTRLGlobal::getInstance()->dataManagerList->reset();
        DM_data *p_data = CTRLGlobal::getInstance()->dataManagerList->next();
        while (p_data)
        {
            if (p_data->list_msg)
            {
                ui_msg.data = p_data->list_msg->data;
                ui_msg.data.setLength((int)strlen(ui_msg.data.data()) + 1);
                ui->send(&ui_msg);
            }
            else
                cerr << endl << "Controller ERROR : NULL list_msg !!!\n";
            p_data = CTRLGlobal::getInstance()->dataManagerList->next();
        }

#ifdef CONNECT
        string DC_info;
        CTRLGlobal::getInstance()->hostList->reset();
        rhost *p_host = CTRLGlobal::getInstance()->hostList->next();
        while (p_host)
        {
            DC_info = p_host->get_DC_list();
            if (!DC_info.empty())
            {
                ui_msg.data = DataHandle{ (char*)DC_info, strlen(DC_info) + 1, false };
                ui->send(&ui_msg);
            }
            p_host = CTRLGlobal::getInstance()->hostList->next();
        }
#endif
    }
}

int ui_list::update_all(const string &mod_info, const string & /*DC_info*/)
{
    if (!mod_info.empty())
    {
        Message *controllermsg = new Message(COVISE_MESSAGE_UI, mod_info);
        send_all(controllermsg);
        delete controllermsg;
    }

#ifdef CONNECT
    if (!DC_info.empty())
    {
        controllermsg = new Message(COVISE_MESSAGE_UI, DC_info);
        send_all(controllermsg);
        delete controllermsg;
    }
#endif

    return 1;
}

//************************************************************************
//
// 				UIF
//
//***********************************************************************

void uif::start(AppModule *dmod, const string &execname, const string &category, const string &key, const string &name, const string &instanz, const string &host)
{

    // instanz ist beliebig, da die UIF-Teile separat verwaltet werden
    // name ist der Name des executeables
    applmod = CTRLGlobal::getInstance()->controller->start_applicationmodule(APPLICATIONMODULE, execname.c_str(), category.c_str(), dmod, instanz.c_str(), Start::Normal);
    applmod->connect(dmod);

    // im normalen Module: receive Module description
    Message *msg = new Message;
    applmod->recv_msg(msg);

    switch (msg->type)
    {
    case COVISE_MESSAGE_EMPTY:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    case COVISE_MESSAGE_SOCKET_CLOSED:
    {
        CTRLHandler::instance()->handleClosedMsg(msg);
    }
    break;
    default:
        break;
    }

    // Auswertung ueberfluessig. evtl. im UIF-Teil das Versenden weglassen
    // im RenderModule: send status-Message MASTER/SLAVE
    ostringstream os;
    os << key << "\nUIFINFO\n" << name << "\n" << instanz << "\n" << host << "\nSTATUS\n" << status << "\n";
    string data = os.str();

    char *tmp = new char[data.length() + 1];
    strcpy(tmp, data.c_str());
    msg->data = DataHandle{ tmp, strlen(tmp) + 1 };
    msg->type = COVISE_MESSAGE_GENERIC;
    applmod->send(msg);

    delete msg;
}

void uif::delete_uif()
{
    Message *msg = new Message(COVISE_MESSAGE_QUIT, "");
    applmod->send(msg);
    delete msg;
}

void uif::set_host(const string &tmp)
{
    host = tmp;
}

void uif::set_userid(const string &tmp)
{
    userid = tmp;
}

void uif::set_passwd(const string &tmp)
{
    passwd = tmp;
}

void uif::set_status(const string &tmp)
{
    status = tmp;
}

string uif::get_host()
{
    return host;
}

string uif::get_userid()
{
    return userid;
}

string uif::get_status()
{
    return status;
}

int uif::get_procid()
{
    return applmod->get_id();
}

void uif::send_msg(Message *msg)
{
    applmod->send(msg);
}

//************************************************************************
//
// 				UIF
//
//************************************************************************

uiflist::uiflist()
    : Liste<uif>()
{
    count = 0;
}

void uiflist::create_uifs(const string &execname, const string &category, const string &key, const string &name, const string &instanz, const string &host)
{

    userinterface *tmp_ui;
    CTRLGlobal::getInstance()->userinterfaceList->reset();
    while ((tmp_ui = CTRLGlobal::getInstance()->userinterfaceList->next()) != NULL)
    {
        // fuer jeden sessionhost wird ein eigener Eintrag in die Liste erzeugt
        uif *new_uif = new uif;

        new_uif->set_host(tmp_ui->get_host());
        new_uif->set_userid(tmp_ui->get_userid());
        new_uif->set_passwd(tmp_ui->get_passwd());
        new_uif->set_status(tmp_ui->get_status());

        this->add(new_uif);
        count++;

        // select Datamanager
        DM_data *tmp_data = CTRLGlobal::getInstance()->dataManagerList->get(tmp_ui->get_host());
        AppModule *dmod = tmp_data->get_DM();

        // und ein UIF-Teil dort gestartet
        new_uif->start(dmod, execname, category, key, name, instanz, host);
    };
}

///***********************************************************************
//
// 				MODUI
//
///***********************************************************************

void modui::set_name(const string &tmp)
{
    name = tmp;
}

void modui::set_instanz(const string &tmp)
{
    instanz = tmp;
}

void modui::set_host(const string &tmp)
{
    host = tmp;
}

void modui::set_category(const string &tmp)
{
    category = tmp;
}

void modui::set_key(const string &tmp)
{
    key = tmp;
}

void modui::set_execname(const string &tmp)
{
    execname = tmp;
}

void modui::set_netmod(net_module *tmp)
{
    netmod = tmp;
}

void modui::set_nodeid(int id)
{
    nodeid = id;
}

string modui::get_name()
{
    return name;
}

string modui::get_instanz()
{
    return instanz;
}

string modui::get_host()
{
    return host;
}

int modui::get_nodeid()
{
    return nodeid;
}

string modui::get_category()
{
    return category;
}

string modui::get_key()
{
    return key;
}

string modui::get_execname()
{
    return execname;
}

net_module *modui::get_netmod()
{
    return netmod;
}

//!
//!    anlegen der uif-teile und start der execs
//!
void modui::create_uifs()
{
    uif_list = new uiflist();
    uif_list->create_uifs(execname, category, key, name, instanz, host);
}

void modui::delete_uif()
{
    uif *tmpuif;

    uif_list->reset();
    while ((tmpuif = uif_list->next()) != NULL)
    {
        tmpuif->delete_uif();
        uif_list->remove(tmpuif);
        uif_list->reset();
    };
}

void modui::send_msg(Message *msg)
{
    uif *tmpuif;
    uif_list->reset();
    while ((tmpuif = uif_list->next()) != NULL)
    {
        int tmpid = tmpuif->get_procid();
        if (tmpid != msg->sender)
            tmpuif->send_msg(msg);
    };
}

void modui::sendapp(Message *msg)
{
    netmod->send_msg(msg);
}

void modui::set_new_status()
{

    uif *tmp_uif;
    uif_list->reset();
    while ((tmp_uif = uif_list->next()) != NULL)
    {
        // host und userid des akt. uif bestimmen
        string tmp_host = tmp_uif->get_host();
        string tmp_userid = tmp_uif->get_userid();

        // ui fuer tmp_host aus ui_list holen
        userinterface *tmp_ui = CTRLGlobal::getInstance()->userinterfaceList->get(tmp_host, tmp_userid);

        // display-status fuer ui holen
        string new_status = tmp_ui->get_status();

        // neuen Status stetzen
        tmp_uif->set_status(new_status);

        // neuen Status verschicken
        ostringstream os;
        os << key << "\nUIFINFO\n" << name << "\n" << instanz << "\n" << host << "\nSTATUS\n" << new_status << "\n";
        Message *msg = new Message(COVISE_MESSAGE_GENERIC, os.str());
        tmp_uif->send_msg(msg);
        delete msg;
    };
}

///***********************************************************************
//
// 				MODUI_LIST
//
///***********************************************************************

modui_list::modui_list()
    : Liste<modui>()
{
}

modui *modui_list::get(const string &key)
{
    modui *tmp = NULL;
    this->reset();
    do
    {
        modui *tmp = this->next();
        if (tmp == NULL)
            break;
    } while (tmp->get_key() != key);

    return tmp;
}

modui *modui_list::get(int id)
{
    modui *tmp = NULL;
    this->reset();
    do
    {
        tmp = this->next();
        if (tmp == NULL)
            break;
    } while (tmp->get_nodeid() != id);

    return tmp;
}

modui *modui_list::get(const string &name, const string &nr, const string &host)
{
    modui *tmp = NULL;
    this->reset();
    while (1)
    {
        modui *tmp = this->next();
        if (tmp == NULL)
            break;

        if (tmp->get_name() == name && tmp->get_instanz() == nr && tmp->get_host() == host)
            break;
    }
    return tmp;
}

/* 
 * anlegen eines Listeneintrages
 * als Ausloeser kommt vom APP-Module eine Message, in der der
 *   - Name des Executables fuer den UIF-Teil steht
 *   - das keyword, mit dem das Module in der Liste steht und das in Messages verwendet wird
 *   - der Name, host, nr des APP-Modules
 *   - die ui_list, aus die Infos bzgl. host auf denen ein UI gestartet wird, 
 *     ausgelesen wird
 */
void modui_list::create_mod(const string &name, const string &instanz, const string &category, const string &host, const string &key, const string &executable)
{

    modui *tmp = new modui;
    this->add(tmp);

    tmp->set_name(name);
    tmp->set_instanz(instanz);
    tmp->set_category(category);
    tmp->set_host(host);
    tmp->set_key(key);
    tmp->set_execname(executable);

    // get module-link
    net_module *tmpnetmod = CTRLGlobal::getInstance()->netList->get(name, instanz, host);
    tmp->set_netmod(tmpnetmod);

    // anlegen der UIF-Teile und start der Module
    tmp->create_uifs();
}

//!
//!  Loeschen eines Listeneintrages
//!
void modui_list::delete_mod(const string &name, const string &nr, const string &host)
{
    modui *tmp;

    // uif_list durchlaufen und UIF-Teile stoppen
    if ((tmp = this->get(name, nr, host)) != NULL)
    {
        tmp->delete_uif();
        this->remove(tmp);
    }
}

//!
//!  Loeschen eines Listeneintrages
//!
void modui_list::delete_mod(int nodeid)
{
    modui *tmp;

    // uif_list durchlaufen und UIF-Teile stoppen
    if ((tmp = this->get(nodeid)) != NULL)
    {
        tmp->delete_uif();
        this->remove(tmp);
    }
}

void modui_list::set_new_status()
{
    modui *tmp_modui;

    // module durchgehen
    this->reset();
    while ((tmp_modui = this->next()))
    {
        tmp_modui->set_new_status();
    };
}
