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

#include <OpenVRUI/coButtonMenuItem.h>

#include <util/coTypes.h>
#include <util/DLinkList.h>
namespace vrui
{
class coCheckboxGroup;
class coMenuItem;
}

namespace covise
{
class TokenBuffer;
}

namespace opencover
{
class coPartnerMenuItem;

class COVEREXPORT coVRPartner : public vrui::coMenuListener
{

private:
    char *hostname;
    char *address;
    int ID;
    char *name;
    char *email;
    char *url;
    int Group;
    bool Master;
    coPartnerMenuItem *menuEntry;
    vrui::coButtonMenuItem *fileMenuEntry;
    static vrui::coCheckboxGroup *partnerGroup;

public:
    void setID(int id);
    void setGroup(int g);
    void setMaster(int m);
    void setInfo(covise::TokenBuffer &tb);
    bool isMaster();
    void becomeMaster();
    int getID();
    void setFile(const char *fileName);
    void menuEvent(vrui::coMenuItem *);
    void print();
    coVRPartner();
    coVRPartner(int id, covise::TokenBuffer &tb);

    virtual ~coVRPartner();
    void sendHello();
};

class COVEREXPORT coVRPartnerList : public covise::DLinkList<coVRPartner *>
{
    static coVRPartnerList *s_instance;
    coVRPartnerList();
public:
    ~coVRPartnerList();
    coVRPartner *get(int ID);
    void print();
    static coVRPartnerList *instance();
};
}
#endif
