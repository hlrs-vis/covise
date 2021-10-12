/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRB_SERVER_PEOXIE_H
#define VRB_SERVER_PEOXIE_H

#include <net/covise_connect.h>
#include <util/coExport.h>
#include <net/message_types.h>

#include <thread>
#include <atomic>
#include <array>
#include <mutex>
namespace vrb
{

//CoviseProxy uses one connection to the controller and one connection to each subProcess that is not running on the controller's host.
//msgs are forwarded via their sender id and type
//the crbs are connected to each other via CrbProxyConn
struct ProxyConn : covise::ServerConnection //basically a Serverconnection that uses the message's original sender info when sending
{
  using ServerConnection::ServerConnection;
  bool sendMessage(const covise::Message *msg) const override;
};

struct CrbProxyConn //connects two crbs by opening a ServerCOnnection to each of them
{
  CrbProxyConn(size_t fromProcId, size_t toProcId, const covise::MessageSenderInterface &controller, int timeout, std::function<void(const CrbProxyConn&)> disconnectedCb);
  ~CrbProxyConn();

  const int m_fromProcId, m_toProcId;
private:
  std::thread m_thread;
  std::atomic_int m_fromSocketId{0}, m_toSocketId{0};
};
class VRBSERVEREXPORT CoviseProxy
{
public:
  CoviseProxy(const covise::MessageSenderInterface &vrbClient);
  CoviseProxy(const CoviseProxy &) = delete;
  CoviseProxy(CoviseProxy &&) = delete;
  CoviseProxy &operator=(const CoviseProxy &) = delete;
  CoviseProxy &operator=(CoviseProxy &&) = delete;
  ~CoviseProxy();

  int controllerPort() const;
  void abortClientConnection(int daemonId);

private:
  covise::ConnectionList m_conns;
  const covise::Connection *m_controllerCon = nullptr;
  std::map<int, const covise::Connection *> m_proxies;
  std::vector<std::unique_ptr<CrbProxyConn>> m_crbProxies;
  std::thread m_thread;
  std::atomic_bool m_quit{false};
  std::mutex m_crbProxyMutex;
  std::vector<const CrbProxyConn *> m_disconnectedCrbProxyies;
  const covise::Connection *openConn(int processID, covise::sender_type type, int timeout, const covise::MessageSenderInterface &requestor);
  void handleMessage(covise::Message &msg);

  void addProxy(int proccessID, covise::sender_type type, int timeout); //returns supProcessPort
  void deleteDisconnectedCrbProxies();
  void passMessages() const;
};
}


#endif // !VRB_SERVER_PEOXIE_H