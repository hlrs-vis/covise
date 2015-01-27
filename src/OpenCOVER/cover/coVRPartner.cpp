/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRPartner.h"
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coRowMenu.h>
#include "coPartnerMenuItem.h"
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

using namespace opencover;
using namespace vrui;
using covise::coCoviseConfig;
using covise::TokenBuffer;
using covise::Message;

coCheckboxGroup *coVRPartner::partnerGroup = new coCheckboxGroup();
coVRPartnerList partners;

coVRPartner::coVRPartner()
{
    std::string entry;
    hostname = NULL;
    address = NULL;
    name = NULL;
    email = NULL;
    url = NULL;
    ID = -1;
    Group = -1;
    Master = 0;
    menuEntry = NULL;
    fileMenuEntry = NULL;

    hostname = new char[strlen(coVRCommunication::getHostname()) + 1];
    strcpy(hostname, coVRCommunication::getHostname());
    address = new char[strlen(coVRCommunication::getHostaddress()) + 1];
    strcpy(address, coVRCommunication::getHostaddress());
    entry = coCoviseConfig::getEntry("value", "COVER.Collaborative.UserName", "noname");
    name = new char[strlen(entry.c_str()) + 1];
    strcpy(name, entry.c_str());
    entry = coCoviseConfig::getEntry("value", "COVER.Collaborative.Email", "covise@vis-mail.hlrs.de");
    email = new char[entry.length() + 1];
    strcpy(email, entry.c_str());
    entry = coCoviseConfig::getEntry("value", "COVER.Collaborative.URL", "www.hlrs.de");
    url = new char[entry.length() + 1];
    strcpy(url, entry.c_str());
}

coVRPartner::coVRPartner(int id, TokenBuffer &tb)
{
    hostname = NULL;
    name = NULL;
    email = NULL;
    address = NULL;
    url = NULL;
    ID = id;
    Group = -1;
    Master = 0;
    menuEntry = NULL;
    fileMenuEntry = NULL;
    setInfo(tb);
}

coVRPartner::~coVRPartner()
{
    VRAvatar *a = VRAvatarList::instance()->get(address);
    if (a)
    {
        VRAvatarList::instance()->remove(a);
    }
    delete menuEntry;
    delete fileMenuEntry;
}

void coVRPartner::setID(int id)
{
    ID = id;
}

void coVRPartner::setFile(const char *fileName)
{
    if (fileName && fileMenuEntry)
    {
        std::cerr << "coVRPartner::setFile: Filename: " << fileName << std::endl;
        fileMenuEntry->setLabel(fileName ? fileName : "");
    }
}

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

void coVRPartner::setGroup(int g)
{
    Group = g;
}

void coVRPartner::setMaster(int m)
{
    if (menuEntry)
        menuEntry->setState(m != 0);
    Master = m != 0;
}

void coVRPartner::becomeMaster()
{
    Master = 1;
    TokenBuffer rtb;
    rtb << 1;
    Message m(rtb);
    m.type = covise::COVISE_MESSAGE_VRB_SET_MASTER;
    vrbc->sendMessage(&m);
}

bool coVRPartner::isMaster()
{
    return Master;
}

void coVRPartner::setInfo(TokenBuffer &tb)
{
    char *tmp, *tmp2;
    tb >> tmp;
    address = new char[strlen(tmp) + 1];
    strcpy(address, tmp);
    int m;
    tb >> tmp; // name
    tb >> tmp; // userInfo
    tb >> Group;
    tb >> m;
    Master = m != 0;
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
    hostname = new char[strlen(tmp2) + 1];
    strcpy(hostname, tmp2);
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
    name = new char[strlen(tmp2) + 1];
    strcpy(name, tmp2);
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
    email = new char[strlen(tmp2) + 1];
    strcpy(email, tmp2);
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
    url = new char[strlen(tmp2) + 1];
    strcpy(url, tmp2);
    c++;

    char *menuText = new char[strlen(name) + strlen(hostname) + 4];
    sprintf(menuText, "%s@%s", name, hostname);
    delete menuEntry;
    menuEntry = new coPartnerMenuItem(menuText, Master, partnerGroup);
    fileMenuEntry = new coButtonMenuItem("NoFile");
    fileMenuEntry->setMenuListener(this);
    coVRCollaboration::instance()->collaborativeMenu->add(menuEntry);
    coVRCollaboration::instance()->collaborativeMenu->add(fileMenuEntry);
}

int coVRPartner::getID()
{
    return ID;
}

void coVRPartner::sendHello()
{
    TokenBuffer tb;
    char *infoString = new char[strlen(hostname) + strlen(name) + strlen(email) + strlen(url) + 100];
    sprintf(infoString, "\"%s\",\"%s\",\"%s\",\"%s\"", hostname, name, email, url);
    tb << infoString;
    Message msg(tb);
    msg.type = covise::COVISE_MESSAGE_VRB_SET_USERINFO;
    vrbc->sendMessage(&msg);
    delete[] infoString;
}

void coVRPartner::print()
{
    cerr << "ID:       " << ID << endl;
    if (hostname)
        cerr << "HostName: " << hostname << endl;
    if (address)
        cerr << "Address:  " << address << endl;
    if (name)
        cerr << "Name:     " << name << endl;
    if (email)
        cerr << "Email:    " << email << endl;
    if (url)
        cerr << "URL:      " << url << endl;
    cerr << "Group:    " << Group << endl;
    cerr << "Master:   " << Master << endl;
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

coVRPartnerList *coVRPartnerList::instance()
{
    static coVRPartnerList *singleton = NULL;
    if (!singleton)
        singleton = new coVRPartnerList;
    return singleton;
}
