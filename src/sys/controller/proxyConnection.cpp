/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "proxyConnection.h"
#include "global.h"

using namespace covise;
using namespace covise::controller;

ProxyConnection::ProxyConnection(const ControllerProxyConn &serverConn, int portOnServer, int processId, sender_type type)
    : m_ctrlConn(serverConn)
{
    sock = serverConn.getSocket();
    port = portOnServer;
    sender_id = processId;
    send_type = type;
    peer_id_ = processId;
    peer_type_ = type;
    CTRLGlobal::getInstance()->controller->getConnectionList()->addRemoveNotice(&serverConn, [this]()
                                                                                { sock = nullptr; });
}

ProxyConnection::~ProxyConnection()
{
    sock = nullptr; //dont't delete in ~Connection()
}

int ProxyConnection::recv_msg(Message *msg, char *ip) const
{
    return m_ctrlConn.recv_msg(sender_id, send_type, msg, ip);
}

int ControllerProxyConn::recv_msg(Message *msg, char *ip) const
{
    if (!m_cachedMsgs.empty())
    {
        msg->copyAndReuseData(m_cachedMsgs.back());
        m_cachedMsgs.pop_back();
        return 1;
    }
    return recv_uncached_msg(msg, ip);
}

int ControllerProxyConn::recv_msg(int processID, int senderType, Message *msg, char *ip) const
{
    assert(msg);
    auto cached = std::find_if(m_cachedMsgs.begin(), m_cachedMsgs.end(), [processID, senderType](const Message &m)
                               { return m.send_type == senderType && m.sender == processID; });
    if (cached != m_cachedMsgs.end())
    {
        msg->copyAndReuseData(*cached);
        m_cachedMsgs.erase(cached);
        return 1;
    }

    while (true)
    {
        auto retval = recv_uncached_msg(msg, ip);
        if (msg->sender == processID && msg->send_type == senderType)
        {
            return retval;
        }
        else
        {
            auto c = m_cachedMsgs.emplace(m_cachedMsgs.begin(), Message{});
            c->copyAndReuseData(*msg);
        }
    }
}

int ControllerProxyConn::recv_uncached_msg(Message *msg, char *ip) const
{
    int retval = Connection::recv_msg(msg, ip);
    assert(msg->send_type != sender_type::UNDEFINED);
    auto proxy = std::find_if(m_proxies.begin(), m_proxies.end(), [msg](const std::unique_ptr<ProxyConnection> &c)
                              { return c->get_sender_id() == msg->sender; });
    if (proxy != m_proxies.end())
    {
        if (msg->type == COVISE_MESSAGE_SOCKET_CLOSED)
            removeProxy(proxy->get());
        msg->conn = &**proxy;
    }
    else if (msg->send_type != VRB && msg->sender != Connection::sender_id)
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
    auto p = std::find_if(m_proxies.begin(), m_proxies.end(), [proxy](const std::unique_ptr<ProxyConnection> &c)
                          { return &*c == proxy; });
    if (p != m_proxies.end())
    {
        for (const auto &cb : m_onRemoveCallbacks[p->get()])
            cb();
        m_proxies.erase(p);
    }
}

std::unique_ptr<Message> ControllerProxyConn::getCachedMsg() const
{
    if (m_cachedMsgs.empty())
        return nullptr;

    std::unique_ptr<Message> msg{new Message{}};
    msg->copyAndReuseData(m_cachedMsgs.back());
    m_cachedMsgs.pop_back();
    return msg;
}

void ControllerProxyConn::addRemoveNotice(const ProxyConnection *conn, const std::function<void(void)> callback) const
{
    m_onRemoveCallbacks[conn].push_back(callback);
}
