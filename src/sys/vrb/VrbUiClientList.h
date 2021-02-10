/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRBUICLIENTLIST_H
#define VRBUICLIENTLIST_H


#include <vrb/server/VrbClientList.h>

class QSocketNotifier;
class QTreeWidgetItem;
class QPixmap;
class QLabel;
class VRBCurve;
namespace covise
{
class Connenction;
class UDPConnection;
}

class VrbUiClient : public vrb::VRBSClient
{
public: 
    VrbUiClient(const covise::Connection *c, const covise::UDPConnection* udpc, QSocketNotifier *sn, covise::TokenBuffer &tb);
    ~VrbUiClient();

    void setMaster(int clientID) override;

    void setSession(const vrb::SessionID & id) override;

    QSocketNotifier *getSN();

    QTreeWidgetItem *myItem;
    VRBCurve *myCurves[4];
    QLabel *myLabels[8];
    static QPixmap *pix_master;
    static QPixmap *pix_slave;

private:
    QSocketNotifier *socketNotifier;
};

class VrbUiClientList : public vrb::VRBClientList
{
public:
    using VRBClientList::get;
    VrbUiClient* get(QSocketNotifier *sn);
};
extern VrbUiClientList uiClients;
#endif // !VRBUICLIENTLIST_H
