/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_VR_PARTNER_H
#define CO_VR_PARTNER_H

#include "ui/Owner.h"

#include <util/DLinkList.h>
#include <util/coTypes.h>
#include <vrb/RemoteClient.h>

#include <vector>
namespace covise
{
class TokenBuffer;
}
namespace vrb {
class SessionID;
}
namespace vive
{
class PartnerAvatar;
namespace ui
{
class ButtonGroup;
class CollaborativePartner;
}

class VVCORE_EXPORT vvPartner: public ui::Owner, public vrb::RemoteClient
{

private:
    ui::CollaborativePartner *m_ui = nullptr;
    PartnerAvatar *m_avatar = nullptr;

public:
    vvPartner();
    vvPartner(RemoteClient &&me);

    void changeID(int id);
    void setMaster(int clientID) override;
    void updateUi();

    void becomeMaster();

    void setFile(const char *fileName);
    PartnerAvatar *getAvatar();
    void setAvatar(PartnerAvatar *avatar);

    virtual ~vvPartner();
};

class VVCORE_EXPORT vvPartnerList: public ui::Owner 
{

public:
    typedef std::vector<std::unique_ptr<vvPartner>> ValueType;
    vvPartnerList(vvPartnerList& other) = delete;
    vvPartnerList& operator=(vvPartnerList& other) = delete;
    vvPartnerList(vvPartnerList&& other) = default;
    vvPartnerList& operator=(vvPartnerList && other) = delete;
    ~vvPartnerList();
    vvPartner *get(int ID);
    vvPartner *me();
    void addPartner(vrb::RemoteClient &&p);
    void removePartner(int id);
    void removeOthers();
    int numberOfPartners() const;
    void setMaster(int id);
    void setSessionID(int partner, const vrb::SessionID & id);
    void sendAvatarMessage();
    void receiveAvatarMessage(covise::TokenBuffer &tb);
    void showAvatars();
    void hideAvatars();
    bool avatarsVisible();
    void print();
    void updateUi();
    ui::ButtonGroup *group();
    static vvPartnerList *instance();
    ValueType::const_iterator begin() const;
    ValueType::const_iterator end() const;

private:
    static vvPartnerList *s_instance;
    ui::ButtonGroup *m_group = nullptr;
    ValueType partners;
    bool m_avatarsVisible;
    vvPartnerList();
    ValueType::iterator find(int id);
};
}
#endif
