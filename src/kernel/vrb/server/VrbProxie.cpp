/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VrbProxie.h"
#include <messages/PROXY.h>
#include <net/covise_socket.h>
#include <net/message_types.h>
#include <net/message_types.h>
#include <net/covise_host.h>
#include <cstring>
#include <algorithm>
#include <array>
using namespace vrb;
using namespace covise;

constexpr int SIZEOF_IEEE_INT = 4;
constexpr int controllerProcessID = 1000;
bool ProxyConn::sendMessage(const covise::Message *msg) const
{
  return Connection::sendMessage(msg->sender, (int)msg->send_type, msg);
}

void sendConnectionToController(int processId, int port, const MessageSenderInterface &controller)
{
  PROXY_ProxyCreated p{port};
  auto msg = p.createMessage();
  msg.sender = processId;
  msg.send_type = VRB;
  controller.send(&msg);
}

CrbProxyConn::CrbProxyConn(size_t fromProcId, size_t toProcId, const covise::MessageSenderInterface &controller, int timeout, std::function<void(const CrbProxyConn &)> disconnectedCb)
    : m_toProcId(toProcId)
    , m_fromProcId(fromProcId)
    , m_thread([this, fromProcId, toProcId, &controller, timeout, disconnectedCb]()
               {
                 std::atomic_bool m_connected{false};
                 covise::ConnectionList connList;
                 struct Connection
                 {
                   int id;
                   const covise::Connection *conn;
                   std::atomic_int *sockId;
                 };
                 std::array<Connection, 2> conns;
                 conns[0].id = fromProcId;
                 conns[1].id = toProcId;

                 conns[0].sockId = &m_fromSocketId;
                 conns[1].sockId = &m_toSocketId;
                 for (auto &conn : conns)
                 {
                   auto c = setupServerConnection(conn.id, CRB, timeout, [&conn, &controller](const covise::ServerConnection &c)
                                                  {
                                                    sendConnectionToController(conn.id, c.get_port(), controller);
                                                    return true;
                                                  });
                   PROXY_ProxyConnected pc{static_cast<bool>(c)};
                   auto msg = pc.createMessage();
                   msg.sender = conn.id;
                   msg.send_type = VRB;
                   controller.send(&msg);
                   if (!c)
                   {
                     disconnectedCb(*this);
                     return;
                   }
                   *conn.sockId = c->getSocket()->get_id();
                   conn.conn = connList.add(std::move(c));
                 }
                 m_connected = true;
                 while (conns[0].conn->is_connected() && conns[1].conn->is_connected())
                 {
                   auto conn = connList.check_for_input(5.0f);
                   if (!conn)
                   {
                     continue;
                   }

                   Message msg;
                   conn->recv_msg(&msg);
                   if (msg.type == COVISE_MESSAGE_SOCKET_CLOSED || msg.type == COVISE_MESSAGE_CLOSE_SOCKET || msg.type == COVISE_MESSAGE_QUIT)
                   {
                     break;
                   }

                   if (!msg.sender)
                   {
                     msg.sender = conns[conn == conns[0].conn].conn->get_sender_id();
                   }
                   if (msg.send_type != sender_type::UNDEFINED)
                   {
                     msg.send_type = conns[conn == conns[0].conn].conn->get_sendertype();
                   }
                   conns[conn == conns[0].conn].conn->sendMessage(&msg);
                   assert(conn == conns[0].conn || conn == conns[1].conn);
                 }
                 disconnectedCb(*this);
               })
{
}

CrbProxyConn::~CrbProxyConn()
{
  if (m_thread.joinable())
  {
    if (m_fromSocketId != 0)
      shutdownSocket(m_fromSocketId);
    if (m_toSocketId != 0)
      shutdownSocket(m_toSocketId);
    m_thread.join();
  }
}

bool isQuitMessage(const Message &msg)
{
  return msg.type == COVISE_MESSAGE_SOCKET_CLOSED ||
         msg.type == COVISE_MESSAGE_CLOSE_SOCKET ||
         msg.type == COVISE_MESSAGE_QUIT;
}

CoviseProxy::CoviseProxy(const covise::MessageSenderInterface &vrbClient)
{
  m_thread = std::thread{[this, &vrbClient]()
                         {
                           int port = 0;
                           m_controllerCon = openConn(controllerProcessID, sender_type::CONTROLLER, 0, vrbClient);
                           if (!m_controllerCon)
                           {
                             std::cerr << "CoviseProxy failed to create new ServerConnection" << std::endl;
                             return;
                           }
                           while (!m_quit)
                           {
                             deleteDisconnectedCrbProxies();
                             if (auto conn = m_conns.check_for_input(1.0f))
                             {
                               Message msg;
                               conn->recv_msg(&msg);
                               handleMessage(msg);
                             }
                           }
                         }};
}

void CoviseProxy::handleMessage(Message &msg)
{
  if (msg.conn == m_controllerCon)
  {
    if (msg.send_type == CONTROLLER && isQuitMessage(msg))
    {
      m_quit = true;
    }

    if (msg.type == COVISE_MESSAGE_PROXY)
    {
      PROXY p{msg};
      switch (p.type)
      {
      case PROXY_TYPE::CreateSubProcessProxie:
      {
        auto &createSubProcessProxie = p.unpackOrCast<PROXY_CreateSubProcessProxie>();
        addProxy(createSubProcessProxie.procID, createSubProcessProxie.senderType, createSubProcessProxie.timeout);
        return;
      }
      case PROXY_TYPE::CreateCrbProxy:
      {
        auto &crb = p.unpackOrCast<PROXY_CreateCrbProxy>();
        m_crbProxies.emplace_back(new CrbProxyConn{crb.fromProcID, crb.toProcID, *m_controllerCon, crb.timeout, [this](const CrbProxyConn &crbproxy)
                                                   {
                                                     std::lock_guard<std::mutex> g{m_crbProxyMutex};
                                                     m_disconnectedCrbProxyies.push_back(&crbproxy);
                                                   }});
        return;
      }
      case PROXY_TYPE::Abort:
      {
        auto &abort = p.unpackOrCast<PROXY_Abort>();
        for (auto proc : abort.processIds)
          abortClientConnection(proc);
      }
      break;
      default:
        break;
      }
    }
    else
    {
      const auto proxy = m_proxies.find(msg.sender);
      if (proxy != m_proxies.end())
      {
        while (proxy->second->check_for_input()) //handle module messages first in case of unhandled quit msg
        {
          Message m;
          proxy->second->recv_msg(&m);
          handleMessage(m);
          if (isQuitMessage(m))
          {
            std::cerr << "found quit msg before controller msg" << std::endl;
            return;
          }
        }

        proxy->second->sendMessage(&msg);
        if (msg.type == COVISE_MESSAGE_QUIT)
        {
          m_conns.remove(proxy->second);
          m_proxies.erase(proxy);
        }

        //std::cerr << "passing msg " << covise_msg_types_array[msg.type] << " from controller to process " << proxy->first << std::endl;
      }
      return;
    }
    if (msg.sender == m_controllerCon->get_sender_id())
    {
      //std::cerr << "bradcasting msg " << covise_msg_types_array[msg.type] << " from controller to all processes" << std::endl;
      for (const auto &proxy : m_proxies)
        proxy.second->sendMessage(&msg);
    }
  }
  else
  {
    //std::cerr << "passing msg " << covise_msg_types_array[msg.type] << " from process " << msg.conn->get_sender_id() << " to controller" << std::endl;
    msg.sender = msg.conn->get_sender_id();
    msg.send_type = msg.conn->get_sendertype();
    m_controllerCon->sendMessage(&msg);
    if (isQuitMessage(msg))
    {
      auto proxy = m_proxies.find(msg.sender);
      m_conns.remove(proxy->second);
      m_proxies.erase(proxy);
    }
  }
}

CoviseProxy::~CoviseProxy()
{
  m_quit = true;
  if (m_controllerCon && m_thread.joinable())
  {
    for (const auto &proxy : m_proxies)
    {
      if (proxy.second->getSocket())
        shutdownSocket(proxy.second->getSocket()->get_id());
    }
    m_thread.join();
  }
}

int CoviseProxy::controllerPort() const
{
  if (m_controllerCon)
  {
    return m_controllerCon->get_port();
  }
  return 0;
}

void CoviseProxy::abortClientConnection(int processId)
{
    if(!m_proxies.erase(processId))
    {
      m_crbProxies.erase(std::remove_if(m_crbProxies.begin(), m_crbProxies.end(), [processId](const std::unique_ptr<CrbProxyConn> &proxy)
                   { return proxy->m_fromProcId == processId || proxy->m_toProcId == processId; }));
    }
    //auto p = m_proxies.find(processId);
    //if (p != m_proxies.end())
    //{
    //  covise::shutdownSocket(p->second->getSocket()->get_id());
    //}
}

template <typename T>
void sendProxyMessage(const T &msg, int processID, const covise::MessageSenderInterface &sender)
{
  Message m = msg.createMessage();
  m.sender = processID;
  m.send_type = sender_type::VRB;
  sender.send(&m);
}

const Connection *CoviseProxy::openConn(int processID, sender_type type, int timeout, const covise::MessageSenderInterface &requestor)
{
  int port = 0;
  auto conn = createListeningConn<ProxyConn>(&port, processID, type);
  PROXY_ProxyCreated retMsg{port};
  sendProxyMessage(retMsg, processID, requestor);
  bool connected = false;
  double time = 0;
  float checkinterval = 0.2f;
  if (m_controllerCon) // create a proxy
  {
    while (!m_quit && !connected && (timeout == 0 || time < (float)timeout)) //eventually handle a launch msg from controller to crb or msgs between crbs
    {
      if (auto c = m_conns.check_for_input(checkinterval / 2))
      {
        Message launchMsg;
        c->recv_msg(&launchMsg);
        handleMessage(launchMsg);
        if (launchMsg.send_type == sender_type::CONTROLLER && launchMsg.type == COVISE_MESSAGE_QUIT)
        {
          break; //the launch of the crb was cancelled
        }
      }
      connected = conn->acceptOne(checkinterval / 2) >= 0;
      time += checkinterval;
    }
  }
  else //wait for initial connection
  {
    connected = conn->acceptOne(timeout) >= 0;
  }
  if (!connected)
  {
    std::cerr << "CoviseProxy connection to " << processID << " failed. timeout was " << timeout << std::endl;
  }

  if (processID != controllerProcessID) //the controller only needs to be informed if other processes connect
  {
    PROXY_ProxyConnected conMsg{connected};
    sendProxyMessage(conMsg, processID, requestor);
  }
  return m_conns.add(std::move(conn));
}

void CoviseProxy::addProxy(int processID, sender_type type, int timeout)
{
  auto p = openConn(processID, type, timeout, *m_controllerCon);
  m_proxies.insert({processID, p});
}

void CoviseProxy::deleteDisconnectedCrbProxies()
{
  std::lock_guard<std::mutex> g{m_crbProxyMutex};
  for (const auto toErase : m_disconnectedCrbProxyies)
  {
    auto it = std::find_if(m_crbProxies.begin(), m_crbProxies.end(), [toErase](const std::unique_ptr<CrbProxyConn> &c)
                           { return c.get() == toErase; });
    assert(it != m_crbProxies.end());
    m_crbProxies.erase(it);
  }
  m_disconnectedCrbProxyies.clear();
}
