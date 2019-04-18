/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_VR_PARTNER_H
#define CO_VR_PARTNER_H

/*! \file
 \brief  a partner in a collaborative session

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2001
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   July 2001
 */

#include <util/coTypes.h>
#include <util/DLinkList.h>
#include "ui/Owner.h"
#include <set>
#include <vrbclient/SessionID.h>

namespace covise
{
class TokenBuffer;
}
namespace vrb {
class SessionID;
}
namespace opencover
{
class VRAvatar;
namespace ui
{
class ButtonGroup;
class CollaborativePartner;
}

class COVEREXPORT coVRPartner: public ui::Owner
{

private:
    std::string hostname;
    std::string address;
    int m_id = -1;
    vrb::SessionID m_sessionID;
    std::string name;
    std::string email;
    std::string url;
    bool m_isMaster = false;
    ui::CollaborativePartner *m_ui = nullptr;
    VRAvatar *m_avatar = nullptr;

public:
    coVRPartner *setID(int id);
    const vrb::SessionID &getSessionID() const;
    void setSession(const vrb::SessionID &g);
    void setMaster(bool m);
    void setInfo(covise::TokenBuffer &tb);
    void updateUi();
    bool isMaster() const;
    void becomeMaster();
    int getID() const;
    void setFile(const char *fileName);
    void print() const;
    VRAvatar *getAvatar();
    void setAvatar(VRAvatar *avatar);
    coVRPartner();
    coVRPartner(int id);

    virtual ~coVRPartner();
    void sendHello();
};

class COVEREXPORT coVRPartnerList: public ui::Owner //, public covise::DLinkList<coVRPartner *>
{
    static coVRPartnerList *s_instance;
    coVRPartnerList();
    ui::ButtonGroup *m_group = nullptr;
    std::map<int, coVRPartner *> partners;
public:
    ~coVRPartnerList();
    coVRPartner *get(int ID);
    coVRPartner *getFirstPartner();
    void addPartner(coVRPartner *p);
    void deletePartner(int id);
    coVRPartner *changePartnerID(int oldID, int newID);
    void deleteOthers();
    int numberOfPartners() const;
    void setMaster(int id);
    coVRPartner *getMaster();
    void setSessionID(int partner, const vrb::SessionID & id);
    void sendAvatarMessage();
    void receiveAvatarMessage(covise::TokenBuffer &tb);
    void showAvatars();
    void hideAvatars();
    bool avatarsVisible();
    void print();
    ui::ButtonGroup *group();
    static coVRPartnerList *instance();

private:
    bool m_avatarsVisible;
};
}
#endif
