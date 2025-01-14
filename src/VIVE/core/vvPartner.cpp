/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvPartner.h"
#include <vrb/client/VRBClient.h>
#include <vrb/PrintClientList.h>
#include <OpenVRUI/coLabel.h>
#ifndef _WIN32
#include <strings.h>
#endif
#include "vvCommunication.h"
#include "vvPluginSupport.h"
#include "vvCollaboration.h"
#include "vvPluginList.h"
#include "vvTui.h"
#include "vvHud.h"
#include "vvFileManager.h"
#include "vvMatrixSerializer.h"
#include <net/message.h>
#include <net/message_types.h>
#include <config/CoviseConfig.h>
#include "vvAvatar.h"
#include "vvVIVE.h"
#include "ui/CollaborativePartner.h"
#include "ui/ButtonGroup.h"
#include "ui/Group.h"
#include <vrb/client/VrbClientRegistry.h>
#include <vrb/client/VRBMessage.h>
#include <vsg/nodes/MatrixTransform.h>
#include "vvSceneGraph.h"

using namespace vive;
using covise::coCoviseConfig;
using covise::TokenBuffer;
using covise::Message;

vvPartnerList *vvPartnerList::s_instance = NULL;

vvPartner::vvPartner()
    : ui::Owner("VRPartner-Me", vv->ui)
    , vrb::RemoteClient(covise::Program::opencover)
{
    m_avatar = new PartnerAvatar(this);
}

vvPartner::vvPartner(RemoteClient &&me)
    : ui::Owner("VRPartner_"+std::to_string(me.ID()), vv->ui)
    , vrb::RemoteClient(std::move(me))
{
    m_avatar = new PartnerAvatar(this);
    updateUi();
}

vvPartner::~vvPartner()
{
    delete m_avatar;
}

void vvPartner::changeID(int id)
{
    std::cerr << "*** vvPartner: own ID is " << id << " ***" << std::endl;
    int oldID = m_id;
    m_id = id;
}

void vvPartner::setMaster(int clientID)
{
    vrb::RemoteClient::setMaster(clientID);
    if (m_ui)
        m_ui->setState(isMaster());
}

void vvPartner::setFile(const char *fileName)
{
#if 0
    if (fileName && fileMenuEntry)
    {
        std::cerr << "vvPartner::setFile: Filename: " << fileName << std::endl;
        fileMenuEntry->setLabel(fileName ? fileName : "");
    }
#endif
}

#if 0
void vvPartner::menuEvent(coMenuItem *m)
{
    if (m == fileMenuEntry)
    {
        //TODO load file
        coButtonMenuItem *item = (coButtonMenuItem *)m;

        //Display of hud doesn't work yet
        //Most likely due to missing framebuffer refresh
        /*vvVIVE::instance()->hud->show();
		vvVIVE::instance()->hud->setText1("Loading File...");
		vvVIVE::instance()->hud->setText2(item->getLabel()->getString());
      vvVIVE::instance()->hud->update();*/
        vvFileManager::instance()->replaceFile(item->getLabel()->getString(), vvTui::instance()->getExtFB());
        /*vvVIVE::instance()->hud->hide();*/
    }
}
#endif

void vvPartner::becomeMaster()
{
    m_session.setMaster(ID());
    TokenBuffer rtb;
    rtb << true;
    Message m(rtb);
    m.type = covise::COVISE_MESSAGE_VRB_SET_MASTER;
    vv->sendVrbMessage(&m);
}

void vvPartner::updateUi()
{
    std::string menuText = std::to_string(m_id) + " " + m_userInfo.userName + "@" + m_userInfo.hostName;
#if 0
    fileMenuEntry = new coButtonMenuItem("NoFile");
    fileMenuEntry->setMenuListener(this);
#endif

    if (!m_ui)
    {
        m_ui = new ui::CollaborativePartner("VRPartner" + std::to_string(m_id), this);
        m_ui->setCallback(
            [this](bool state)
            {
                // change it back
                m_ui->setState(!state, false);
            });
    }
    m_ui->setText(menuText);
    m_ui->setState(isMaster());
    if (vvCollaboration::instance()->partnerGroup() && m_userInfo.userType == covise::Program::opencover &&
        sessionID() == vvPartnerList::instance()->me()->sessionID())
    {
        vvCollaboration::instance()->partnerGroup()->add(m_ui);
    }
    else
    {
        vvCollaboration::instance()->partnerGroup()->remove(m_ui);
    }
}

PartnerAvatar * vive::vvPartner::getAvatar()
{
    return m_avatar;
}

void vive::vvPartner::setAvatar(PartnerAvatar * avatar)
{
    m_avatar = avatar;
}

//////////////////////////////vvPartnerList//////////////////////////////

vvPartner *vvPartnerList::get(int id)
{
    auto p = find(id);
    if (p == partners.end())
    {
        return nullptr;
    }
    return p->get();
}


vvPartner *vive::vvPartnerList::me(){
    assert(partners[0]);
    return partners[0].get();
}
void vive::vvPartnerList::addPartner(vrb::RemoteClient &&p)
{
    partners.push_back(ValueType::value_type{new vvPartner{std::move(p)}});
    if (partners[partners.size() - 1]->sessionID() == me()->sessionID()) //client joined my sessison
    {
        vvCollaboration::instance()->showCollaborative(true);
    }
    updateUi();
}

void vive::vvPartnerList::removePartner(int id)
{
    partners.erase(find(id));
    updateUi();
}

void vive::vvPartnerList::removeOthers()
{
    auto p = partners.begin();
    while (p != partners.end())
    {
        if ((*p)->ID() != vvCommunication::instance()->getID())
        {
            p = partners.erase(p);
        }
        else
        {
            ++p;
        }
    }
    vvCollaboration::instance()->showCollaborative(false);
    updateUi();
}

int vive::vvPartnerList::numberOfPartners() const
{
    return (int)partners.size();
}

void vive::vvPartnerList::setMaster(int clientID)
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

void vive::vvPartnerList::setSessionID(int partnerID, const vrb::SessionID & newSession)
{
    auto partner = get(partnerID);

    int myID = vvCommunication::instance()->getID();
    vrb::SessionID oldSession = partner->sessionID();
    partner->setSession(newSession);
    vrb::SessionID mySession = vvCommunication::instance()->getSessionID();
    if (partnerID == myID) //this client changed session
    {
        vrb::VrbClientRegistry::instance->resubscribe(newSession, oldSession);
        vvCollaboration::instance()->updateSharedStates();
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
            vvCollaboration::instance()->showCollaborative(false);
        }
        else
        {
            vvCollaboration::instance()->showCollaborative(true);
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
                vvCollaboration::instance()->showCollaborative(false);
           }
        }
        if (newSession == mySession) //client joined my sessison
        {
            vvCollaboration::instance()->showCollaborative(true);
        }
    }
    updateUi();
}

void vive::vvPartnerList::updateUi()
{
    for (auto &p: partners)
    {
        p->updateUi();
    }
}

void vive::vvPartnerList::sendAvatarMessage()
{
    // all data is in object Coordinates


    vvAvatar av = vvAvatar();
    covise::TokenBuffer tb;
    tb << vrb::AVATAR;
    tb << vvCommunication::instance()->getID();
    std::string adress(vvCommunication::instance()->getHostaddress());
    tb << adress;
    tb << av;

    Message msg(tb);
    msg.type = covise::COVISE_MESSAGE_VRB_MESSAGE;
    vv->sendVrbMessage(&msg);

}
void vive::vvPartnerList::receiveAvatarMessage(covise::TokenBuffer &tb)
{
    int sender;
    std::string adress;
    tb >> sender; 
    tb >> adress;
    auto p = get(sender);
    auto av = p->getAvatar();
    if (av->init(adress))
    {

        if(!vvPluginList::instance()->getPlugin("AnimatedAvatar") && !vvPluginList::instance()->getPlugin("FbxAvatar"))
            av->loadPartnerIcon();
        if (m_avatarsVisible && p->ID() != vvCommunication::instance()->getID() && p->sessionID() == vvCommunication::instance()->getSessionID())
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
void vive::vvPartnerList::showAvatars()
{
    int myID = vvCommunication::instance()->getID();
    vrb::SessionID mySession = vvCommunication::instance()->getSessionID();
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
void vive::vvPartnerList::hideAvatars()
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
bool vive::vvPartnerList::avatarsVisible()
{
    return m_avatarsVisible;
}
void vvPartnerList::print()
{
    std::vector<const vrb::RemoteClient *> clients(partners.size());
    std::transform(partners.begin(), partners.end(), clients.begin(), [](const std::unique_ptr<vvPartner>& p) { return p.get(); });
    vrb::printClientInfo(clients);
}

ui::ButtonGroup *vvPartnerList::group()
{
    return m_group;
}

vvPartnerList::vvPartnerList()
: ui::Owner("PartnerList", vv->ui)
{
    m_group = new ui::ButtonGroup("PartnerGroup", this);
    partners.push_back(ValueType::value_type{new vvPartner{}}); //me at pos 0
    assert(!s_instance);
    s_instance = this;
}

vvPartnerList::~vvPartnerList()
{
    assert(s_instance);
    s_instance = NULL;
}

vvPartnerList *vvPartnerList::instance()
{
    if (!s_instance)
        s_instance = new vvPartnerList;
    return s_instance;
}

vvPartnerList::ValueType::const_iterator vvPartnerList::begin() const{
    return partners.begin();
}

vvPartnerList::ValueType::const_iterator vvPartnerList::end() const{
    return partners.end();
}

vvPartnerList::ValueType::iterator vvPartnerList::find(int id){
    return std::find_if(partners.begin(), partners.end(), [id](const std::unique_ptr<vvPartner> &p) { return p->ID() == id; });
}
