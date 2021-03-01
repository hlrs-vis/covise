/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRPartner.h"
#include <vrb/client/VRBClient.h>
#include <vrb/PrintClientList.h>
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
#include <vrb/client/VrbClientRegistry.h>
#include <vrb/client/VRBMessage.h>
#include <osg/MatrixTransform>
#include "VRSceneGraph.h"

using namespace opencover;
using covise::coCoviseConfig;
using covise::TokenBuffer;
using covise::Message;

coVRPartnerList *coVRPartnerList::s_instance = NULL;

coVRPartner::coVRPartner()
    : ui::Owner("VRPartner-Me", cover->ui)
    , vrb::RemoteClient(vrb::Program::opencover)
{
    m_avatar = new VRAvatar(this);
}

coVRPartner::coVRPartner(RemoteClient &&me)
    : ui::Owner("VRPartner_"+std::to_string(me.ID()), cover->ui)
    , vrb::RemoteClient(std::move(me))
{
    m_avatar = new VRAvatar(this);
    updateUi();
}

coVRPartner::~coVRPartner()
{
    delete m_avatar;
}

void coVRPartner::changeID(int id)
{
    std::cerr << "*** coVRPartner: own ID is " << id << " ***" << std::endl;
    int oldID = m_id;
    m_id = id;
}

void coVRPartner::setMaster(int clientID)
{
    vrb::RemoteClient::setMaster(clientID);
    if (m_ui)
        m_ui->setState(isMaster());
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

void coVRPartner::becomeMaster()
{
    m_session.setMaster(ID());
    TokenBuffer rtb;
    rtb << true;
    Message m(rtb);
    m.type = covise::COVISE_MESSAGE_VRB_SET_MASTER;
    cover->sendVrbMessage(&m);
}

void coVRPartner::updateUi()
{
    std::string menuText = std::to_string(m_id) + " " + m_userInfo.userName + "@" + m_userInfo.hostName;
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
    m_ui->setState(isMaster());
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
    auto p = find(id);
    if (p == partners.end())
    {
        return nullptr;
    }
    return p->get();
}


coVRPartner *opencover::coVRPartnerList::me(){
    assert(partners[0]);
    return partners[0].get();
}
void opencover::coVRPartnerList::addPartner(vrb::RemoteClient &&p)
{
    partners.push_back(ValueType::value_type{new coVRPartner{std::move(p)}});
    if (partners[partners.size() - 1]->sessionID() == me()->sessionID()) //client joined my sessison
    {
        coVRCollaboration::instance()->showCollaborative(true);
    }
}

void opencover::coVRPartnerList::removePartner(int id)
{
    partners.erase(find(id));
}

void opencover::coVRPartnerList::removeOthers()
{
    auto p = partners.begin();
    while (p != partners.end())
    {
        if ((*p)->ID() != coVRCommunication::instance()->getID())
        {
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

void opencover::coVRPartnerList::setMaster(int clientID)
{
    const auto master = find(clientID);
    if (master == partners.end())
    {
        std::cerr << "failed to set master: master " << clientID << " is not a client" << std::endl;
    }
    assert(master != partners.end());
    for (auto &p : partners)
    {
        if (p->sessionID() == (*master)->sessionID())
        {
            p->setMaster(clientID);
        }
    }
}

void opencover::coVRPartnerList::setSessionID(int partnerID, const vrb::SessionID & newSession)
{
    auto partner = get(partnerID);

    int myID = coVRCommunication::instance()->getID();
    vrb::SessionID oldSession = partner->sessionID();
    partner->setSession(newSession);
    vrb::SessionID mySession = coVRCommunication::instance()->getSessionID();
    if (partnerID == myID) //this client changed session
    {
        vrb::VrbClientRegistry::instance->resubscribe(newSession, oldSession);
        coVRCollaboration::instance()->updateSharedStates();
        bool alone = true;
        //check if other partners are in my new session
        for (auto &p : partners)
        {
            if (p->ID() != myID && p->sessionID() == newSession)
            {
                alone = false;
                if (p->getAvatar())
                {
                    p->getAvatar()->show();
                }
            }
            else
            {
                if (p->getAvatar())
                {
                    p->getAvatar()->hide();
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
            bool lastInSession = true;
            for (auto &p : partners)
            {
                if (p->ID() != myID && p->sessionID() == mySession)
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
    auto p = get(sender);
    VRAvatar *av = p->getAvatar();
    if (av->init(adress))
    {
        if (m_avatarsVisible && p->ID() != coVRCommunication::instance()->getID() && p->sessionID() == coVRCommunication::instance()->getSessionID())
        {
            av->show();
        }
        else
        {
            av->hide();
        }
    }
    tb >> *av;


}
void opencover::coVRPartnerList::showAvatars()
{
    int myID = coVRCommunication::instance()->getID();
    vrb::SessionID mySession = coVRCommunication::instance()->getSessionID();
    for (auto &partner : partners)
    {
        if (partner->getAvatar())
        {
            if (partner->ID() != myID && partner->sessionID() == mySession)
            {
                partner->getAvatar()->show();
            }
            else
            {
                partner->getAvatar()->hide();
            }
        }
    }
    m_avatarsVisible = true;
}
void opencover::coVRPartnerList::hideAvatars()
{
    for (auto &partner : partners)
    {
        if (partner->getAvatar())
        {
            partner->getAvatar()->hide();
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
    std::vector<const vrb::RemoteClient *> clients(partners.size());
    std::transform(partners.begin(), partners.end(), clients.begin(), [](const std::unique_ptr<coVRPartner>& p) { return p.get(); });
    vrb::printClientInfo(clients);
}

ui::ButtonGroup *coVRPartnerList::group()
{
    return m_group;
}

coVRPartnerList::coVRPartnerList()
: ui::Owner("PartnerList", cover->ui)
{
    m_group = new ui::ButtonGroup("PartnerGroup", this);
    partners.push_back(ValueType::value_type{new coVRPartner{}}); //me at pos 0
    assert(!s_instance);
}

coVRPartnerList::~coVRPartnerList()
{
    s_instance = NULL;
}

coVRPartnerList *coVRPartnerList::instance()
{
    if (!s_instance)
        s_instance = new coVRPartnerList;
    return s_instance;
}

coVRPartnerList::ValueType::const_iterator coVRPartnerList::begin() const{
    return partners.begin();
}

coVRPartnerList::ValueType::const_iterator coVRPartnerList::end() const{
    return partners.end();
}

coVRPartnerList::ValueType::iterator coVRPartnerList::find(int id){
    return std::find_if(partners.begin(), partners.end(), [id](const std::unique_ptr<coVRPartner> &p) { return p->ID() == id; });
}
