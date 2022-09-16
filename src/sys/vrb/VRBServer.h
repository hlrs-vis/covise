/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRB_SERVER_H
#define VRB_SERVER_H


#include <QObject>
#include <qsocketnotifier.h>

#include <string>
#include <map>
#include <set>
#include <memory>

#include <vrb/SessionID.h>
#include <vrb/server/VrbMessageHandler.h>
#include <net/covise_connect.h>
#include <net/udpMessage.h>


class QTreeWidgetItem;
class QSocketNotifier;

namespace vrb
{
class VrbServerRegistry;
class VRBClientList;
}
extern vrb::VRBClientList *vrbClients;

//
//
class VRBServer : public QObject, public vrb::ServerInterface
{
Q_OBJECT
private slots:

    void processMessages(float waitTime = 0.0001f);
	void processUdpMessages();
public:
    VRBServer(bool gui);
    void loop();

	bool startUdpServer();
    bool openServer(bool printport);
    void closeServer();
    void removeConnection(const covise::Connection *conn) override;
    int getPort();
    int getUdpPort();

private:
    bool m_gui;
    const covise::ServerConnection *sConn = nullptr;
	const covise::UDPConnection* udpConn = nullptr;
    std::unique_ptr<QSocketNotifier> serverSN;

    covise::ConnectionList connections;
    std::unique_ptr<vrb::VrbMessageHandler> handler;
    int m_tcpPort, m_udpPort; // port Number (default: 31800) covise.config: VRB.TCPPort
  
    covise::Message msg;
	covise::UdpMessage udpMsg;
    bool requestToQuit = false;

    void VRBServer::addClient();


};
#endif
