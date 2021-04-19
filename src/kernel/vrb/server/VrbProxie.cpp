/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VrbProxie.h"
#include <comsg/PROXY.h>
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

void sendConnectionToCrbProxy(CrbProxyConn::Direction dir, int processId, int port, const MessageSenderInterface &crbProxy)
{
  constexpr std::array<int, 2> types{COVISE_MESSAGE_PREPARE_CONTACT_DM, COVISE_MESSAGE_DM_CONTACT_DM};
  TokenBuffer tb;
  tb << port << Host::getHostaddress();
  Message msg{types[dir], tb.getData()};
  msg.sender = processId;
  msg.send_type = CRB;
  crbProxy.send(&msg);
}

void sendConnectionToController(int processId, int port, const MessageSenderInterface &controller)
{
  PROXY_ProxyCreated p{port};
  auto msg = p.createMessage();
  msg.sender = processId;
  msg.send_type = CRB;
  controller.send(&msg);
}

bool CrbProxyConn::init(Direction dir, int processId, int timeout, const covise::MessageSenderInterface &controller, const covise::MessageSenderInterface &controllerOrProxy, covise::ConnectionList &connList)
{
  auto conn = setupServerConnection(processId, CRB, timeout, [&controller, &controllerOrProxy, dir, processId](const covise::ServerConnection &c) {
    if (&controller != &controllerOrProxy) //the crb is running as proxy, directly inform him about this connection
    {
      sendConnectionToCrbProxy(dir, processId, c.get_port(), controllerOrProxy);
    }
    else //pass msg through controller to local crb
    {
      sendConnectionToController(processId, c.get_port(), controller);
    }
    return true;
  });
  PROXY_ProxyConnected connected{conn != nullptr};
  auto msg = connected.createMessage();
  msg.send_type = CRB;
  msg.sender = processId;
  controller.send(&msg);
  if (!conn)
    return false;
  m_conns[dir] = connList.add(std::move(conn));
  m_connList = &connList;

  return true;
}

bool CrbProxyConn::tryPassMessage(const Message &msg) const
{
  auto c = std::find(m_conns.begin(), m_conns.end(), msg.conn);
  if (c != m_conns.end())
  {
    int pos = c - m_conns.begin();
    if (!m_conns[!pos]->sendMessage(&msg))
    {
      std::cerr << "failed to send message " << covise_msg_types_array[msg.type] << " to " << m_conns[!pos]->get_sender_id() << std::endl;
    }
    //else
    //  std::cerr << "passing msg " << covise_msg_types_array[msg.type] << " from process " << m_conns[pos]->get_sender_id() << " to " << m_conns[!pos]->get_sender_id() << std::endl;
    return true;
  }
  return false;
}

CrbProxyConn::~CrbProxyConn()
{
  if (m_connList)
  {
    m_connList->remove(m_conns[0]);
    m_connList->remove(m_conns[1]);
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

  m_thread = std::thread{[this, &vrbClient]() {
    int port = 0;
    m_controllerCon = openConn(controllerProcessID, sender_type::CONTROLLER, 0, vrbClient);
    if (!m_controllerCon)
    {
      std::cerr << "CoviseProxy failed to create new ServerConnection" << std::endl;
      return;
    }
    while (!m_quit)
    {
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
        std::array<size_t, 2> ids{crb.toProcID, crb.fromProcID};
        std::unique_ptr<CrbProxyConn> c{new CrbProxyConn};
        bool success = false;
        for (size_t i = 0; i < 2; i++)
        {
          auto proxy = m_proxies.find(ids[i]);
          if (proxy != m_proxies.end())
          {
            success += c->init(static_cast<CrbProxyConn::Direction>(i), ids[i], crb.timeout, *m_controllerCon, *proxy->second, m_conns);
          }
          else
          {
            success += c->init(static_cast<CrbProxyConn::Direction>(i), ids[i], crb.timeout, *m_controllerCon, *m_controllerCon, m_conns);
          }
        }
        if (success)
          m_crbProxies.push_back(std::move(c));
        else
          std::cerr << "VRB failed to create proxy connection between crbs " << crb.toProcID << " and " << crb.fromProcID << std::endl;
        return;
      }
      default:
        break;
      }
    }
    else
    {
      const auto proxy = m_proxies.find(msg.sender);
      if (proxy != m_proxies.end())
      {
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
    for (auto crbProxy = m_crbProxies.begin(); crbProxy < m_crbProxies.end(); crbProxy++)
      if (crbProxy->get()->tryPassMessage(msg))
      {
        if (isQuitMessage(msg))
        {
          m_crbProxies.erase(crbProxy);
        }
        return;
      }

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
      shutdownAndCloseSocket(proxy.second->getSocket()->get_id());
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

template <typename T>
void sendProxyMessage(const T &msg, int processID, const covise::MessageSenderInterface &sender)
{
  Message m = msg.createMessage();
  m.sender = processID;
  m.send_type = sender_type::CONTROLLER;
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
        if (c == m_controllerCon && launchMsg.type == COVISE_MESSAGE_QUIT)
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

  PROXY_ProxyConnected conMsg{connected};

  if (processID != controllerProcessID) //the controller only needs to be informed if other processes connect
  {
    sendProxyMessage(conMsg, processID, requestor);
  }
  return m_conns.add(std::move(conn));
}

void CoviseProxy::addProxy(int processID, sender_type type, int timeout)
{
  auto p = openConn(processID, type, timeout, *m_controllerCon);
  m_proxies.insert({processID, p});
}
