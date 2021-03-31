/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "crb.h"
#include "host.h"
#include "userinterface.h"
#include "handler.h"

#include <net/message_types.h>
#include <util/covise_version.h>

using namespace covise;
using namespace covise::controller;

const std::array<const char *, static_cast<int>(Userinterface::Status::LASTDUMMY)> Userinterface::statusNames{"MASTER", "SLAVE", "Mirror", "Init"};

CRBModule::CRBModule(const RemoteHost &host, bool proxy)
    : SubProcess(moduleType, host, sender_type::CRB, proxy ? "crbProxy" : "crb")
{
}

CRBModule::~CRBModule()
{
    if (CTRLHandler::instance()->Config.getshmMode(getHost()) != ShmMode::NoShm)
    {
        Message msg{COVISE_MESSAGE_QUIT, ""};
        send(&msg);
    }
}

bool CRBModule::init()
{
    tryReceiveMessage(initMessage);
    if (!initMessage.data.data())
        return false;
    checkCoviseVersion(initMessage, getHost());
    prepareInitMessageForUIs();
    tryReceiveMessage(interfaceMessage);
    queryDataPath();

    for (const auto crb : host.hostManager.getAllModules<CRBModule>())
    {
        if (crb != this)
        {
            connectToCrb(*crb);
        }
    }
    return true;
}

bool CRBModule::checkCoviseVersion(const Message &versionMessage, const std::string &hostname)
{
    if (!versionMessage.data.data())
        return false;

    // check if we have information about partner version
    string main_version = CoviseVersion::shortVersion();
    string version_info = strchr(versionMessage.data.data(), '@');
    version_info.erase(0, 1);
    if (!version_info.empty())
    {
        string partner_version = version_info;
        if (main_version != partner_version)
        {
            string text = "Controller WARNING : main covise version = " + main_version + " and the partner version = ";
            text = text + partner_version + " from host " + hostname + " are different !!!";
            sendMaster(Message{COVISE_MESSAGE_WARNING, text});
            return false;
        }
    }
    else
    {
        string text = "Controller WARNING : main covise version = " + main_version;
        text = text + " and the partner version = \"unknown\" from host " + hostname + " are different !!!";
        sendMaster(Message{COVISE_MESSAGE_WARNING, text});
        return false;
    }
    return true;
}

void CRBModule::sendMaster(const Message &msg)
{
    for (const Userinterface *ui : host.hostManager.getAllModules<Userinterface>())
    {
        if (ui->status() == Userinterface::Master)
        {
            ui->send(&msg);
        }
    }
}

bool CRBModule::tryReceiveMessage(Message &msg)
{
    std::unique_ptr<Message> m(&msg);
    recv_msg(&msg);
    switch (msg.type)
    {
    case COVISE_MESSAGE_EMPTY:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    case COVISE_MESSAGE_SOCKET_CLOSED:
        CTRLHandler::instance()->handleClosedMsg(m);
        m.release();
        return false;
    default:
        break;
    }
    m.release();
    return true;
}

void CRBModule::prepareInitMessageForUIs()
{
    // patch Message to include hostname & user !!
    string module_info;
    module_info.append(initMessage.data.data());
    module_info.insert(5, getHost() + "\n" + host.userInfo().userName + "\n");
    DataHandle txt{module_info.length() + 1};
    strcpy(txt.accessData(), module_info.c_str());
    initMessage.data = txt;
}

void CRBModule::queryDataPath()
{
    Message msg{COVISE_MESSAGE_QUERY_DATA_PATH, ""};
    send(&msg);
    tryReceiveMessage(msg);
    if (msg.type == COVISE_MESSAGE_SEND_DATA_PATH)
        covisePath = msg.data.data();
}

bool CRBModule::connectToCrb(const SubProcess &crb)
{
    if (&crb == this)
    {
        std::cerr << "can not connect crb to itself" << std::endl;
        return false;
    }
    if (host.hostManager.proxyConn()) //crb proxy required
    {
        return connectCrbsViaProxy(crb);
    }
    return connectModuleToCrb(crb, ConnectionType::CrbToCrb);
}