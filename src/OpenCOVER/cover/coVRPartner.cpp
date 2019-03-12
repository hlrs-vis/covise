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
#include <net/message.h>
#include <net/message_types.h>
#include <config/CoviseConfig.h>
#include "VRAvatar.h"
#include "OpenCOVER.h"
#include "ui/CollaborativePartner.h"
#include "ui/ButtonGroup.h"
#include "ui/Group.h"
#include <vrbclient/VrbClientRegistry.h>

using namespace opencover;
using covise::coCoviseConfig;
using covise::TokenBuffer;
using covise::Message;

coVRPartnerList *coVRPartnerList::s_instance = NULL;

coVRPartner::coVRPartner()
    :ui::Owner("VRPartner-Me", cover->ui)
    ,m_id(  -1)
    ,m_group(-1)
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
{
    m_group = -1;
    m_isMaster = false;
}

coVRPartner::~coVRPartner()
{
    VRAvatar *a = VRAvatarList::instance()->get(address.c_str());
    if (a)
    {
        VRAvatarList::instance()->remove(a);
    }
}

void coVRPartner::setID(int id)
{
    std::cerr << "*** coVRPartner: own ID is " << id << " ***" << std::endl;
    m_id = id;
}

void opencover::coVRPartner::setSessionID(const vrb::SessionID & id)
{
    vrb::SessionID oldSession = m_sessionID;
    m_sessionID = id;
    vrb::VrbClientRegistry::instance->resubscribe(id, oldSession);
    coVRCollaboration::instance()->updateSharedStates();
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

void coVRPartner::setGroup(int g)
{
    m_group = g;
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
    int master = 1;
    rtb << master;
    Message m(rtb);
    m.type = covise::COVISE_MESSAGE_VRB_SET_MASTER;
    if (vrbc)
        vrbc->sendMessage(&m);
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
    tb >> m_group;
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
    std::string menuText = name + "@" + hostname;
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
    if (vrbc)
        vrbc->sendMessage(&msg);
}

void coVRPartner::print() const
{
    cerr << "ID:       " << m_id << endl;
    cerr << "HostName: " << hostname << endl;
    cerr << "Address:  " << address << endl;
    cerr << "Name:     " << name << endl;
    cerr << "Email:    " << email << endl;
    cerr << "URL:      " << url << endl;
    cerr << "Group:    " << m_group << endl;
    cerr << "Master:   " << m_isMaster << endl;
}

coVRPartner *coVRPartnerList::get(int id)
{
    coVRPartner *p;
    reset();
    while ((p = current()))
    {
        if (p->getID() == id)
            return p;
        next();
    }
    return NULL;
}

void coVRPartnerList::print()
{
    cerr << "Num Partners: " << num() << endl;
    coVRPartner *p;
    reset();
    while ((p = current()))
    {
        p->print();
        cerr << endl;
        next();
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
    reset();
    while (current())
        remove();

    // da sollte noch mehr geloescht werden
    s_instance = NULL;
}

coVRPartnerList *coVRPartnerList::instance()
{
    if (!s_instance)
        s_instance = new coVRPartnerList;
    return s_instance;
}
