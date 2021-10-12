/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "exception.h"
#include "global.h"
#include "handler.h"
#include "host.h"
#include "module.h"
#include "proxyConnection.h"
#include "util.h"

#include <messages/CRB_EXEC.h>
#include <messages/PROXY.h>
#include <covise/covise.h>
#include <covise/covise_msg.h>
#include <covise/covise_process.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>

const int SIZEOF_IEEE_INT = 4;

using namespace covise;
using namespace covise::controller;

uint32_t SubProcess::processCount = 1001;

SubProcess::SubProcess(Type t, const RemoteHost &h, sender_type type, const std::string &executableName)
    : host(h), type(type), processId(processCount++), m_type(t), m_executableName(executableName)
{
}

SubProcess::SubProcess(SubProcess &&other)
    : host(other.host), type(other.type), processId(other.processId), m_type(other.m_type), m_executableName(std::move(other.m_executableName)), m_conn(other.m_conn)
{
    other.m_conn = nullptr;
}

SubProcess::~SubProcess()
{
    Message msg{COVISE_MESSAGE_QUIT};
    send(&msg);

    CTRLGlobal::getInstance()->controller->getConnectionList()->remove(m_conn);
}

const std::string &SubProcess::getHost() const
{
    return host.userInfo().ipAdress;
}

const Connection *SubProcess::conn() const
{
    return &*m_conn;
}

void SubProcess::recv_msg(Message *msg) const
{
    if (m_conn)
    {
        m_conn->recv_msg(msg);
    }
};

bool SubProcess::sendMessage(const Message *msg) const
{
    if (m_conn)
        return m_conn->sendMessage(msg);
    return false;
}

bool SubProcess::sendMessage(const UdpMessage *msg) const
{
    return false;
}

bool SubProcess::connectToCrb()
{
    return connectToCrb(host.getProcess(sender_type::CRB));
}

bool SubProcess::connectToCrb(const SubProcess &crb)
{
    return connectModuleToCrb(crb, ConnectionType::ModuleToCrb);
}

bool SubProcess::connectModuleToCrb(const SubProcess &toCrb, ConnectionType type)
{
    constexpr std::array<covise::covise_msg_type, 2> prepareMessages{COVISE_MESSAGE_PREPARE_CONTACT, COVISE_MESSAGE_PREPARE_CONTACT_DM};
    constexpr std::array<covise::covise_msg_type, 2> contactMessages{COVISE_MESSAGE_APP_CONTACT_DM, COVISE_MESSAGE_DM_CONTACT_DM};

    // Tell CRB to open a socket for the module
    Message msg{prepareMessages[type], DataHandle{}};
    if (!toCrb.send(&msg))
        return false;
    //std::cerr << "sent " << covise_msg_types_array[prepareMessages[type]] << " to crb on " << crb.host.userInfo().ipAdress << " on port " << crb.conn()->get_port() << std::endl;
    // Wait for CRB to deliver the opened port number
    do
    {
        std::unique_ptr<Message> portmsg{new Message{}};
        toCrb.recv_msg(&*portmsg);
        if (portmsg->type == COVISE_MESSAGE_PORT)
        {
            TokenBuffer tb(portmsg.get()), rtb;
            int port = 0;
            tb >> port;
            rtb << port;
            if (type == ConnectionType::CrbToCrb)
                rtb << toCrb.host.userInfo().ipAdress;

            // send to module
            msg = Message{contactMessages[type], rtb.getData()};
            send(&msg);
            //std::cerr << "sent " << covise_msg_types_array[contactMessages[type]] << " to module on " << host.userInfo().ipAdress << " on port " << conn()->get_port() << std::endl;

            return true;
        }
        else
        {
            CTRLHandler::instance()->handleMsg(portmsg); // handle all other messages
        }

    } while (true);
}

void waitForProxyMsg(std::unique_ptr<Message> &msg, const Connection &conn)
{
    while (true)
    {
        conn.recv_msg(msg.get());
        if (msg->type != COVISE_MESSAGE_PROXY)
            CTRLHandler::instance()->handleMsg(msg); // handle all other messages
        else
        {
            return;
        }
    }
}

bool SubProcess::setupConn(std::function<bool(int port, const std::string &ip)> sendConnMessage)
{
    const int timeout = m_execFlag == ExecFlag::Normal? 0 : 0; 
    if (&host == &host.hostManager.getLocalHost() || !host.proxyHost())
    {
        auto conn = setupServerConnection(processId, type, timeout, [&sendConnMessage, this](const ServerConnection &c)
                                          {
                                              m_conn = &c;
                                              return sendConnMessage(c.get_port(), host.hostManager.getLocalHost().userInfo().ipAdress);
                                          });
        if (conn)
        {
            m_conn = CTRLGlobal::getInstance()->controller->getConnectionList()->add(std::move(conn));
            CTRLGlobal::getInstance()->controller->getConnectionList()->addRemoveNotice(m_conn, [this]()
                                                                                        { m_conn = nullptr; });
            return true;
        }
        m_conn = nullptr;
        return false;
    }
    else //create proxy conn via vrb
    {
        auto proxyConn = host.hostManager.proxyConn();
        PROXY_CreateSubProcessProxie p{processId, type, timeout};
        sendCoviseMessage(p, *proxyConn);
        std::unique_ptr<Message> msg{new Message{}};
        waitForProxyMsg(msg, *proxyConn);
        PROXY proxy{*msg};
        auto conn = proxyConn->addProxy(proxy.unpackOrCast<PROXY_ProxyCreated>().port, processId, type);
        proxyConn->addRemoveNotice(conn, [this](){ m_conn = nullptr; });
        m_conn = conn;
        if (sendConnMessage(m_conn->get_port(), host.hostManager.getVrbClient().getCredentials().ipAddress))
        {
            waitForProxyMsg(msg, *proxyConn);
            PROXY pr{*msg};
            if (!pr.unpackOrCast<PROXY_ProxyConnected>().success)
            {
                cerr << "* timelimit in accept for module " << m_executableName << " exceeded!!" << endl;
                return false;
            }
            return true;
        }
        else
        {
            waitForProxyMsg(msg, *proxyConn); //receive PROXY_ProxyConnected so that it is out of msq
            return false;
        }
    }
}

bool SubProcess::start(const char *instance, const char *category)
{
    return setupConn([this, instance, category](int port, const std::string &ip)
                     {
                         auto &controllerHost = host.hostManager.getLocalHost();
                         CRB_EXEC crbExec{m_execFlag, m_executableName.c_str(), port, ip.c_str(), static_cast<int>(processId), instance,
                                          host.userInfo().ipAdress.c_str(), host.userInfo().hostName.c_str(),
                                          category, host.hostManager.getLocalHost().ID(), vrb::VrbCredentials{}, std::vector<std::string>{}};

                         host.launchProcess(crbExec);
                         return true;
                     });
}
