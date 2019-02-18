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


namespace covise
{
class TokenBuffer;
}

namespace opencover
{
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
    int m_privateSessionID = 0;
    int m_publicSessionID = 0;
    std::set<int> sessions;
    std::string name;
    std::string email;
    std::string url;
    int m_group = -1;
    bool m_isMaster = false;
    ui::CollaborativePartner *m_ui = nullptr;

public:
    void setID(int id);
    void setSessionID(int id);
    void setGroup(int g);
    void setMaster(bool m);
    void setInfo(covise::TokenBuffer &tb);
    void updateUi();
    bool isMaster() const;
    void becomeMaster();
    int getID() const;
    int getPrivateSessionID() const;
    int getPublicSessionID() const;
    void setSessions(std::set<int> sessions);
    std::set<int> getSessions();
    void setFile(const char *fileName);
    void print() const;
    coVRPartner();
    coVRPartner(int id);

    virtual ~coVRPartner();
    void sendHello();
};

class COVEREXPORT coVRPartnerList: public ui::Owner, public covise::DLinkList<coVRPartner *>
{
    static coVRPartnerList *s_instance;
    coVRPartnerList();
    ui::ButtonGroup *m_group = nullptr;
public:
    ~coVRPartnerList();
    coVRPartner *get(int ID);
    void print();
    ui::ButtonGroup *group();
    static coVRPartnerList *instance();
};
}
#endif
