/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRPartner.h"
#include <vrbclient/VRBClient.h>
#include <OpenVRUI/coLabel.h>
#ifndef _WIN32
#include <strings.h>
#endif
#include "coVRCommunication.h"
#include "coVRPluginSupport.h"
#include "coVRCollaboration.h"
#include "coVRTui.h"
#include "coHud.h"
#include "coVRFileManager.h"
#include "MatrixSerializer.h"
#include <net/message.h>
#include <net/message_types.h>
#include <config/CoviseConfig.h>
#include "VRAvatar.h"
#include "OpenCOVER.h"
#include "ui/CollaborativePartner.h"
#include "ui/ButtonGroup.h"
#include "ui/Group.h"
#include <vrbclient/VrbClientRegistry.h>
#include <vrbclient/VRBMessage.h>
#include <osg/MatrixTransform>
#include "VRSceneGraph.h"

using namespace opencover;
using covise::coCoviseConfig;
using covise::TokenBuffer;
using covise::Message;

coVRPartnerList *coVRPartnerList::s_instance = NULL;

coVRPartner::coVRPartner()
    :ui::Owner("VRPartner-Me", cover->ui)
    ,m_id(  -1)
    ,m_isMaster(false)
    ,m_sessionID()
    ,hostname(coVRCommunication::getHostname())
    ,address(coVRCommunication::getHostaddress())
    ,name(coCoviseConfig::getEntry("value", "COVER.Collaborative.UserName", coVRCommunication::getUsername()))
    ,email (coCoviseConfig::getEntry("value", "COVER.Collaborative.Email", "covise-users@listserv.uni-stuttgart.de"))
    ,url (coCoviseConfig::getEntry("value", "COVER.Collaborative.URL", "www.hlrs.de/covise"))
{
}

coVRPartner::coVRPartner(int id)
: ui::Owner("VRPartner_"+std::to_string(id), cover->ui)
, m_id(id)
,m_sessionID()
,m_isMaster(false)
{
}

coVRPartner::~coVRPartner()
{
    delete m_avatar;
}

coVRPartner * coVRPartner::setID(int id)
{
    std::cerr << "*** coVRPartner: own ID is " << id << " ***" << std::endl;
    int oldID = m_id;
    m_id = id;
    return coVRPartnerList::instance()->changePartnerID(oldID, id);
}

const vrb::SessionID &opencover::coVRPartner::getSessionID() const
{
    return m_sessionID;
}

void coVRPartner::setFile(const char *fileName)
{
#if 0
    if (fileName && fileMenuEntry)
    {
        std::cerr << "coVRPartner::setFile: Filename: " << fileName << std::endl;
        fileMenuEntry->setLabel(fileName ? fileName : "");
    }
#endif
}

#if 0
void coVRPartner::menuEvent(coMenuItem *m)
{
    if (m == fileMenuEntry)
    {
        //TODO load file
        coButtonMenuItem *item = (coButtonMenuItem *)m;

        //Display of hud doesn't work yet
        //Most likely due to missing framebuffer refresh
        /*OpenCOVER::instance()->hud->show();
		OpenCOVER::instance()->hud->setText1("Loading File...");
		OpenCOVER::instance()->hud->setText2(item->getLabel()->getString());
      OpenCOVER::instance()->hud->update();*/
        coVRFileManager::instance()->replaceFile(item->getLabel()->getString(), coVRTui::instance()->getExtFB());
        /*OpenCOVER::instance()->hud->hide();*/
    }
}
#endif

void coVRPartner::setSession(const vrb::SessionID &g)
{
    m_sessionID = g;
}

void coVRPartner::setMaster(bool m)
{
    if (m_ui)
        m_ui->setState(m);
    m_isMaster = m;
}

void coVRPartner::becomeMaster()
{
    m_isMaster = true;
    TokenBuffer rtb;
    rtb << true;
    Message m(rtb);
    m.type = covise::COVISE_MESSAGE_VRB_SET_MASTER;
    cover->sendVrbMessage(&m);
}

bool coVRPartner::isMaster() const
{
    return m_isMaster;
}

void coVRPartner::setInfo(TokenBuffer &tb)
{
    char *tmp, *tmp2;
    tb >> address;
    tb >> name; // name
    tb >> tmp; // userInfo
    tb >> m_sessionID;
    int master = -1;
    tb >> master;
    m_isMaster = master ? true : false;

    char *c = tmp;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c != '\0')
        c++;
    tmp2 = c;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c == '\0')
        return;
    *c = '\0';
    hostname = tmp2;
    c++;

    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c != '\0')
        c++;
    tmp2 = c;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c == '\0')
        return;
    *c = '\0';
    name = tmp2;
    c++;

    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c != '\0')
        c++;
    tmp2 = c;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c == '\0')
        return;
    *c = '\0';
    email = tmp2;
    c++;

    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c != '\0')
        c++;
    tmp2 = c;
    while ((*c != '\"') && (*c != '\0'))
        c++;
    if (*c == '\0')
        return;
    *c = '\0';
    url = tmp2;
    c++;
}

void coVRPartner::updateUi()
{
    std::string menuText = std::to_string(m_id) + " " + name + "@" + hostname;
#if 0
    fileMenuEntry = new coButtonMenuItem("NoFile");
    fileMenuEntry->setMenuListener(this);
#endif

    if (!m_ui)
    {
        m_ui = new ui::CollaborativePartner("VRPartner"+std::to_string(m_id), this, coVRPartnerList::instance()->group());
        if (auto g = coVRCollaboration::instance()->partnerGroup())
        {
            g->add(m_ui);
        }
        m_ui->setCallback([this](bool state){
            // change it back
            m_ui->setState(!state, false);
        });
    }
    m_ui->setText(menuText);
    m_ui->setState(m_isMaster);
}

int coVRPartner::getID() const
{
    return m_id;
}


void coVRPartner::sendHello()
{
    std::stringstream str;
    str << "\"" << hostname << "\",\"" << name << "\",\"" << email << "\",\"" << url << "\"";

    TokenBuffer tb;
    tb << str.str();
    Message msg(tb);
    msg.type = covise::COVISE_MESSAGE_VRB_SET_USERINFO;
    cover->sendVrbMessage(&msg);
}

void coVRPartner::print() const
{
    cerr << "ID:       " << m_id << endl;
    cerr << "HostName: " << hostname << endl;
    cerr << "Address:  " << address << endl;
    cerr << "Name:     " << name << endl;
    cerr << "Email:    " << email << endl;
    cerr << "URL:      " << url << endl;
    cerr << "Group:    " << m_sessionID.toText() << endl;
    cerr << "Master:   " << m_isMaster << endl;
}

VRAvatar * opencover::coVRPartner::getAvatar()
{
    return m_avatar;
}

void opencover::coVRPartner::setAvatar(VRAvatar * avatar)
{
    m_avatar = avatar;
}

//////////////////////////////coVRPartnerList//////////////////////////////

coVRPartner *coVRPartnerList::get(int id)
{
    auto it = partners.find(id);
    if (it != partners.end())
    {
        return it->second;
    }
    return nullptr;
}

coVRPartner * opencover::coVRPartnerList::getFirstPartner()
{
    if (partners.empty())
        return nullptr;

    return partners.begin()->second;
}

void opencover::coVRPartnerList::addPartner(coVRPartner * p)
{
    if (p)
    {
        partners[p->getID()] = p;
    }
}

void opencover::coVRPartnerList::deletePartner(int id)
{
    setSessionID(id, vrb::SessionID(0, ""));
    delete partners[id];
    partners.erase(id);
}

coVRPartner *opencover::coVRPartnerList::changePartnerID(int oldID, int newID)
{
    auto p = partners.find(newID);
    if (oldID == newID)
    {
        return p->second;
    }
    auto it = partners.find(oldID);
    if (p != partners.end())
    {
        std::cerr << "there is already a partner with " << newID << "registered in coVRParnerList" << std::endl;
        return it->second;
    }
    if (it != partners.end())
    {
        std::swap(partners[newID], it->second);
        partners.erase(it);
    }
    return partners[newID];
}

void opencover::coVRPartnerList::deleteOthers()
{
    auto p = partners.begin();
    while (p != partners.end())
    {
        if (p->second->getID() != coVRCommunication::instance()->getID())
        {
            delete p->second;
            p = partners.erase(p);
        }
        else
        {
            ++p;
        }
    }
    coVRCollaboration::instance()->showCollaborative(false);
}

int opencover::coVRPartnerList::numberOfPartners() const
{
    return partners.size();
}

void opencover::coVRPartnerList::setMaster(int id)
{
    for (auto p : partners)
    {
        p.second->setMaster(false);
    }
    if (id > 0)
    {
        partners[id]->setMaster(true);
    }
}

coVRPartner * opencover::coVRPartnerList::getMaster()
{
    for (auto p : partners)
    {
        if (p.second->isMaster())
        {
            return p.second;
        }
    }
    return nullptr;
}

void opencover::coVRPartnerList::setSessionID(int partnerID, const vrb::SessionID & newSession)
{
    coVRPartner *partner = get(partnerID);

    int myID = coVRCommunication::instance()->getID();
    if(!partner)
    {
        return;
    }
    vrb::SessionID oldSession = partner->getSessionID();
    partner->setSession(newSession);
    vrb::SessionID mySession = coVRCommunication::instance()->getSessionID();
    if (partnerID == myID) //this client changed session
    {
        vrb::VrbClientRegistry::instance->resubscribe(newSession, oldSession);
        coVRCollaboration::instance()->updateSharedStates();
        bool alone = true;
        //check if other partners are in my new session
        for (auto p : partners)
        {
            if (p.first != myID && p.second->getSessionID() == newSession)
            {
                alone = false;
                if (p.second->getAvatar())
                {
                    p.second->getAvatar()->show();
                }
            }
            else
            {
                if (p.second->getAvatar())
                {
                    p.second->getAvatar()->hide();
                }
            }
        }
        if (alone)
        {
            coVRCollaboration::instance()->showCollaborative(false);
        }
        else
        {
            coVRCollaboration::instance()->showCollaborative(true);
        }

    }
    else //other client changed session
    {

        if (oldSession == mySession && !mySession.isPrivate()) //client left my session
        {
            if (partner->getAvatar())
            {
                partner->getAvatar()->hide();
            };
            bool lastInSession = true;
            for (auto p : partners)
            {
                if (p.first != myID && p.second->getSessionID() == mySession)
                {
                    lastInSession = false;
                    break;
                }
            }
           if (lastInSession)
           {
                coVRCollaboration::instance()->showCollaborative(false);
           }
        }
        if (newSession == mySession) //client joined my sessison
        {
            if (partner->getAvatar())
            {
                partner->getAvatar()->show();
            }
            coVRCollaboration::instance()->showCollaborative(true);
        }
    }

}
void opencover::coVRPartnerList::sendAvatarMessage()
{
    // all data is in object Coordinates


    VRAvatar av = VRAvatar();
    covise::TokenBuffer tb;
    tb << vrb::AVATAR;
    tb << coVRCommunication::instance()->getID();
    std::string adress(coVRCommunication::instance()->getHostaddress());
    tb << adress;
    tb << av;

    Message msg(tb);
    msg.type = covise::COVISE_MESSAGE_VRB_MESSAGE;
    cover->sendVrbMessage(&msg);

}
void opencover::coVRPartnerList::receiveAvatarMessage(covise::TokenBuffer &tb)
{
    int sender;
    std::string adress;
    tb >> sender; 
    tb >> adress;
    coVRPartner *p = get(sender);
    if (p)
    {
        VRAvatar *av = p->getAvatar();
        if (!av)
        {
            av = new VRAvatar(sender, adress);
            if (m_avatarsVisible && p->getID() != coVRCommunication::instance()->getID() && p->getSessionID() == coVRCommunication::instance()->getSessionID())
            {
                av->show();
            }
            else
            {
                av->hide();
            }
            p->setAvatar(av);
        }
        tb >> *av;
    }


}
void opencover::coVRPartnerList::showAvatars()
{
    int myID = coVRCommunication::instance()->getID();
    vrb::SessionID mySession = coVRCommunication::instance()->getSessionID();
    for (auto partner : partners)
    {
        if (partner.second->getAvatar())
        {
            if (partner.first != myID && partner.second->getSessionID() == mySession)
            {
                partner.second->getAvatar()->show();
            }
            else
            {
                partner.second->getAvatar()->hide();
            }
        }
    }
    m_avatarsVisible = true;
}
void opencover::coVRPartnerList::hideAvatars()
{
    for (auto partner : partners)
    {
        if (partner.second->getAvatar())
        {
            partner.second->getAvatar()->hide();
        }
    }
    m_avatarsVisible = false;
}
bool opencover::coVRPartnerList::avatarsVisible()
{
    return m_avatarsVisible;
}
void coVRPartnerList::print()
{
    cerr << "Num Partners: " << partners.size() << endl;
    for (auto p : partners)
    {
        p.second->print();
        cerr << endl;
}
}

ui::ButtonGroup *coVRPartnerList::group()
{
    return m_group;
}

coVRPartnerList::coVRPartnerList()
: ui::Owner("PartnerList", cover->ui)
{
    m_group = new ui::ButtonGroup("PartnerGroup", this);
    assert(!s_instance);
}

coVRPartnerList::~coVRPartnerList()
{
    auto partner = partners.begin();
    while (partner != partners.end())
    {
        delete partner->second;
        partner = partners.erase(partner);
    }

    // da sollte noch mehr geloescht werden
    s_instance = NULL;
}

coVRPartnerList *coVRPartnerList::instance()
{
    if (!s_instance)
        s_instance = new coVRPartnerList;
    return s_instance;
}

