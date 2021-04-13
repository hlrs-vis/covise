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
  enum Direction
  {
    To,
    From
  };
  bool init(Direction dir, int procID, int timeout, const covise::MessageSenderInterface &controller,const covise::MessageSenderInterface &controllerOrProxy, covise::ConnectionList &connList);
  bool tryPassMessage(const covise::Message &msg) const; //if msg fits one of the connections msg is forwarded to the other. Returns true if it did so.
  ~CrbProxyConn();

private:
  covise::ConnectionList *m_connList = nullptr;
  std::array<const covise::Connection *, 2> m_conns; //toConn, fromConn
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
  const covise::Connection *openConn(int processID, covise::sender_type type, int timeout, const covise::MessageSenderInterface &requestor);
  void handleMessage(covise::Message &msg);

  void addProxy(int proccessID, covise::sender_type type, int timeout); //returns supProcessPort
  void passMessages() const;
};
}


#endif // !VRB_SERVER_PEOXIE_H