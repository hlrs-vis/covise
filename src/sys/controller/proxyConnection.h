/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CONTROLLER_PROXY_CONNECTION_H
#define CONTROLLER_PROXY_CONNECTION_H

#include <net/covise_connect.h>
#include <net/message_types.h>

#include <deque>
namespace covise
{
namespace controller
{

struct ControllerProxyConn;

struct ProxyConnection : Connection
{

    ProxyConnection(const ControllerProxyConn &serverConn, int portOnServer, int processId, sender_type type);
    virtual ~ProxyConnection();
    virtual int recv_msg(Message *msg, char *ip = nullptr) const override;

    private:
        const ControllerProxyConn &m_ctrlConn;
};

struct ControllerProxyConn : ClientConnection
{
    using ClientConnection::ClientConnection;
    virtual int recv_msg(Message *msg, char *ip = nullptr) const override;
    int recv_msg(int processID, int sender, Message *msg, char *ip = nullptr) const;
    ProxyConnection *addProxy(int portOnServer, int processId, sender_type type) const;
    void removeProxy(ProxyConnection *proxy) const;
    std::unique_ptr<Message> getCachedMsg() const;

private:
    mutable std::vector<std::unique_ptr<ProxyConnection>> m_proxies;
    mutable std::deque<Message> m_cachedMsgs;
    int recv_uncached_msg(Message *msg, char *ip = nullptr) const;
};
} // namespace controller

} // namespace covise

#endif // !CONTROLLER_PROXY_CONNECTION_H