/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRB_SERVER_H
#define VRB_SERVER_H

#include <string>
#include <QObject>
#include <QString>
#include <map>
#include <set>
#include <memory>
#include <vrb/SessionID.h>
#include <vrb/server/VrbMessageHandler.h>
#include <net/covise_connect.h>


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

    void processMessages();
	void processUdpMessages();
public:
    VRBServer(bool gui);
    ~VRBServer();
    void loop();

	bool startUdpServer();
    int openServer();
    void closeServer();
    void removeConnection(const covise::Connection *conn) override;


private:
    bool m_gui;
    QPixmap *pix_master = NULL;
    QPixmap *pix_slave = NULL;
    const covise::ServerConnection *sConn = nullptr;
	const covise::UDPConnection* udpConn = nullptr;
    QSocketNotifier *serverSN = nullptr;

    vrb::VrbMessageHandler *handler;

    covise::ConnectionList connections;
    int m_tcpPort, m_udpPort; // port Number (default: 31800) covise.config: VRB.TCPPort
  
    covise::Message *msg = nullptr;
	covise::UdpMessage* udpMsg = nullptr;
	char* ip = new char[16];
    bool requestToQuit = false;

};
#endif


