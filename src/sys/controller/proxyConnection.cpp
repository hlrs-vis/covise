/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "proxyConnection.h"
#include "global.h"

using namespace covise;
using namespace covise::controller;

ProxyConnection::ProxyConnection(const ControllerProxyConn &serverConn, int portOnServer, int processId, sender_type type)
{
    sock = serverConn.getSocket();
    port = portOnServer;
    sender_id = processId;
    send_type = type;
    peer_id_ = processId;
    peer_type_ = type;
    CTRLGlobal::getInstance()->controller->getConnectionList()->addRemoveNotice(&serverConn, [this]() {
        sock = nullptr;
    });
}

ProxyConnection::~ProxyConnection()
{
    sock = nullptr; //dont't delete in ~Connection()
}

int ControllerProxyConn::recv_msg(Message *msg, char *ip) const
{
    int retval = Connection::recv_msg(msg, ip);
    auto proxy = std::find_if(m_proxies.begin(), m_proxies.end(), [msg](const std::unique_ptr<ProxyConnection> &c) {
        return c->get_sender_id() == msg->sender;
    });
    if (proxy != m_proxies.end())
    {
        msg->conn = &**proxy;
    }
    else if (msg->sender != Connection::sender_id) 
    {
        std::cerr << "ControllerProxyConn did not find proxy for process " << msg->sender << std::endl;
        return 0;
    }
    return retval;
}

ProxyConnection *ControllerProxyConn::addProxy(int portOnServer, int processId, sender_type type) const
{
    std::cerr << "adding proxy for " << processId << ", type " << type << " on port " << portOnServer << std::endl;
    auto c = m_proxies.emplace(m_proxies.end(), new ProxyConnection{*this, portOnServer, processId, type});
    return &**c;
}

void ControllerProxyConn::removeProxy(ProxyConnection *proxy) const
{
    m_proxies.erase(std::remove_if(m_proxies.begin(), m_proxies.end(), [proxy](const std::unique_ptr<ProxyConnection> &c) {
                        return &*c == proxy;
                    }),
                    m_proxies.end());
}
