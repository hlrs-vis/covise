/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "crb.h"
#include "host.h"
#include "userinterface.h"
#include "handler.h"
#include "proxyConnection.h"

#include <comsg/PROXY.h>
#include <net/message_types.h>
#include <util/covise_version.h>

using namespace covise;
using namespace covise::controller;

const std::array<const char *, static_cast<int>(Userinterface::Status::LASTDUMMY)> Userinterface::statusNames{"MASTER", "SLAVE", "Mirror", "Init"};

CRBModule::CRBModule(const RemoteHost &host)
    : SubProcess(moduleType, host, sender_type::CRB, "crb")
{
}

CRBModule::~CRBModule()
{
    if (CTRLHandler::instance()->Config.getshmMode(host.ID()) != ShmMode::NoShm)
    {
        Message msg{COVISE_MESSAGE_QUIT, ""};
        send(&msg);
    }
}

bool CRBModule::init()
{
    Message msg;
    tryReceiveMessage(msg);
    if (!msg.data.data())
        return false;

    NEW_UI uiMsg{msg};
    auto &moduleMsg = uiMsg.unpackOrCast<NEW_UI_AvailableModules>();
    
    checkCoviseVersion(moduleMsg.coviseVersion, getHost());
    initMessage = NEW_UI_PartnerInfo{host.ID(), getHost(), host.userInfo().userName, moduleMsg.coviseVersion, moduleMsg.modules, moduleMsg.categories}.createMessage();

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

bool CRBModule::checkCoviseVersion(const std::string &partnerVersion, const std::string &hostname)
{
    // check if we have information about partner version
    string main_version = CoviseVersion::shortVersion();
    if (!partnerVersion.empty())
    {
        if (main_version != partnerVersion)
        {
            string text = "Controller WARNING : main covise version = " + main_version + " and the partner version = ";
            text = text + partnerVersion + " from host " + hostname + " are different !!!";
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
    recv_msg(&msg);
    switch (msg.type)
    {
    case COVISE_MESSAGE_EMPTY:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    case COVISE_MESSAGE_SOCKET_CLOSED:
    {
        std::unique_ptr<Message> m(&msg);
        CTRLHandler::instance()->handleClosedMsg(m);
        m.release();
    }
    return false;
    default:
        break;
    }
    return true;
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

bool CRBModule::connectCrbsViaProxy(const SubProcess &toCrb)
{
    PROXY_CreateCrbProxy crbProxyRequest{toCrb.processId, processId, 30};
    sendCoviseMessage(crbProxyRequest, *host.hostManager.proxyConn());
    const std::array<const SubProcess *, 2> crbs{&toCrb, this};
    constexpr std::array<int, 2> msgTypes{COVISE_MESSAGE_PREPARE_CONTACT_DM, COVISE_MESSAGE_DM_CONTACT_DM};
    for (size_t i = 0; i < 2; i++)
    {
        Message proxyMsg;
        //receive opened port
        host.hostManager.proxyConn()->recv_msg(&proxyMsg); //produces proxy not found warning
        PROXY p{proxyMsg};
        auto &crbProxyCreated = p.unpackOrCast<PROXY_ProxyCreated>();
        //send host and port to crb
        TokenBuffer tb1;
        tb1 << crbProxyCreated.port << host.hostManager.getVrbClient().getCredentials().ipAddress;
        Message msg{msgTypes[i], tb1.getData()};
        crbs[i]->send(&msg);

        //wait for VRB to confirm the connection
        host.hostManager.proxyConn()->recv_msg(&proxyMsg);
        PROXY p2{proxyMsg};
        if (!p2.unpackOrCast<PROXY_ProxyConnected>().success)
        {
            std::cerr << "failed to connect crb on " << host.userInfo().ipAdress << " to crb on " << toCrb.host.userInfo().ipAdress << " via vrb proxy" << std::endl;
            return false;
        }
    }
    return true;
}