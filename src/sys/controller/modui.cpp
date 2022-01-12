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
#include "global.h"
#include "handler.h"
#include "config.h"
#include "modui.h"
#include "subProcess.h"
#include "crb.h"

#include <messages/CRB_EXEC.h>
#include <config/CoviseConfig.h>
#include <config/coConfig.h>
#include <covise/covise_process.h>
#include <net/covise_host.h>
#include <util/covise_version.h>
#include <vrb/RemoteClient.h>

namespace covise
{
namespace controller
{
//************************************************************************
//
// 				UIF
//
//***********************************************************************

void uif::start(const controller::CRBModule &crb, const string &execname, const string &category, const string &key, const string &name, const string &instanz, const string &host)
{

    // instanz ist beliebig, da die UIF-Teile separat verwaltet werden
    // name ist der Name des executeables
    try
    {
        auto &h = crb.host.hostManager.findHost(host);
        appInfo.reset(new ModuleInfo{execname, category});
        applmod.reset(new NetModule{h, *appInfo, std::stoi(instanz)});
        applmod->start(instanz.c_str(), applmod->info().category.c_str());
        applmod->connectToCrb();

        // im normalen Module: receive Module description
        Message msg;
        applmod->recv_msg(&msg);

        switch (msg.type)
        {
        case COVISE_MESSAGE_EMPTY:
        case COVISE_MESSAGE_CLOSE_SOCKET:
        case COVISE_MESSAGE_SOCKET_CLOSED:
        {
            CTRLGlobal::getInstance()->controller->getConnectionList()->remove(msg.conn);
        }
        break;
        default:
            break;
        }

        // Auswertung ueberfluessig. evtl. im UIF-Teil das Versenden weglassen
        // im RenderModule: send status-Message MASTER/SLAVE
        ostringstream os;
        os << key << "\nUIFINFO\n"
            << name << "\n"
            << instanz << "\n"
            << host << "\nSTATUS\n"
            << status << "\n";
        string data = os.str();

        char *tmp = new char[data.length() + 1];
        strcpy(tmp, data.c_str());
        msg.data = DataHandle{tmp, strlen(tmp) + 1};
        msg.type = COVISE_MESSAGE_GENERIC;
        applmod->send(&msg);

        /* code */
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        std::cerr << "uif::start failed to find host " << host << std::endl;
    }
}

void uif::delete_uif()
{
    Message msg{COVISE_MESSAGE_QUIT, ""};
    applmod->send(&msg);
}

void uif::set_hostAddress(const string &tmp)
{
    hostAddress = tmp;
}

void uif::set_userid(const string &tmp)
{
    userid = tmp;
}

void uif::set_passwd(const string &tmp)
{
    passwd = tmp;
}

void uif::set_status(Userinterface::Status s)
{
    status = s;
}

const string &uif::get_hostAddress() const
{
    return hostAddress;
}

const string &uif::get_userid() const
{
    return userid;
}

Userinterface::Status uif::get_status() const
{
    return status;
}

const NetModule *uif::get_app() const
{
    return &*applmod;
}

int uif::get_procid() const
{
    return applmod->processId;
}

void uif::send_msg(const Message *msg) const
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

void uiflist::create_uifs(const NetModule& app, const string &execname, const string &key)
{
    for (const auto &hostIt : app.host.hostManager)
    {
        auto &host = hostIt.second;
        // fuer jeden sessionhost wird ein eigener Eintrag in die Liste erzeugt
        uif *new_uif = new uif;

        new_uif->set_hostAddress(host->userInfo().ipAdress);
        new_uif->set_userid(host->userInfo().userName);

        auto ui =  host->getProcess(sender_type::USERINTERFACE).as<Userinterface>();
        new_uif->set_status(ui->status());
        auto crb = host->getProcess(sender_type::CRB).as<CRBModule>();
        this->add(new_uif);
        count++;

        // und ein UIF-Teil dort gestartet
        new_uif->start(*crb, execname, app.info().category, key, app.info().name, std::to_string(app.instance()), host->userInfo().hostName);
    };
}

///***********************************************************************
//
// 				MODUI
//
///***********************************************************************

void modui::set_key(const string &tmp)
{
    key = tmp;
}

void modui::set_execname(const string &tmp)
{
    execname = tmp;
}

void modui::set_application(const NetModule *tmp)
{
    app = tmp;
}

void modui::set_nodeid(int id)
{
    nodeid = id;
}


int modui::get_nodeid() const
{
    return nodeid;
}

const string &modui::get_key() const
{
    return key;
}

const string &modui::get_execname() const
{
    return execname;
}

const NetModule *modui::get_application() const
{
    return app;
}

//!
//!    anlegen der uif-teile und start der execs
//!
void modui::create_uifs()
{
    uif_list = new uiflist();
    uif_list->create_uifs(*app, execname, key);
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

void modui::send_msg(const Message *msg)
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

void modui::sendapp(const Message *msg)
{
    app->send(msg);
}

void modui::set_new_status()
{

    uif *tmp_uif;
    uif_list->reset();
    while ((tmp_uif = uif_list->next()) != NULL)
    {
        auto app = tmp_uif->get_app();
        tmp_uif->set_status(app->host.getProcess(sender_type::USERINTERFACE).as<Userinterface>()->status());

        // send new status
        ostringstream os;
        os << key << "\nUIFINFO\n"
            << app->info().name << "\n"
            << app->instance() << "\n"
            << app->host.userInfo().hostName << "\nSTATUS\n"
            << Userinterface::getStatusName(tmp_uif->get_status()) << "\n";
        Message msg{COVISE_MESSAGE_GENERIC, os.str()};
        tmp_uif->send_msg(&msg);
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

        if (tmp->get_application()->info().name == name &&
         tmp->get_application()->instance() == std::stoi(nr) &&
          tmp->get_application()->host.userInfo().ipAdress == host)
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
void modui_list::create_mod(const NetModule& app, const string &key, const string &executable)
{

    modui *tmp = new modui;
    this->add(tmp);

    tmp->set_key(key);
    tmp->set_execname(executable);

    tmp->set_application(&app);

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

} // namespace controller
} // namespace covise
