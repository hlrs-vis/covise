/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRB_SERVER_PEOXIE_H
#define VRB_SERVER_PEOXIE_H

#include <net/covise_connect.h>
#include <util/coExport.h>

#include <thread>
#include <atomic>
namespace vrb
{


struct ProxyConn : covise::ServerConnection
{
  using ServerConnection::ServerConnection;
  bool sendMessage(const covise::Message *msg) const override;
};

struct CrbProxyConn
{
  enum Direction
  {
    To,
    From
  };
  bool init(Direction dir, int procID, int timeout, const covise::MessageSenderInterface &controller,const covise::MessageSenderInterface &controllerOrProxy, covise::ConnectionList &connList);
  bool tryPassMessage(const covise::Message &msg) const;

private:
  std::array<const covise::Connection *, 2> m_conns;   //toConn, fromConn
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

private:
  covise::ConnectionList m_conns;
  const covise::Connection *m_controllerCon = nullptr;
  std::map<int, const covise::Connection *> m_proxies;
  std::vector<std::unique_ptr<CrbProxyConn>> m_crbProxies;
  std::thread m_thread;
  std::atomic_bool m_quit{false};
  const covise::Connection *openConn(int processID, int timeout, const covise::MessageSenderInterface &requestor);
  void handleMessage(covise::Message &msg);

  void addProxy(int proccessID, int timeout); //returns supProcessPort
  void passMessages() const;
};
}


#endif // !VRB_SERVER_PEOXIE_H