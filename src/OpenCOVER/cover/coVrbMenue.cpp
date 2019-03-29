/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVrbMenue.h"

#include "ui/Action.h"
#include "ui/EditField.h"
#include "ui/Menu.h"
#include "ui/SelectionList.h"

#include "OpenCOVER.h"
#include "coVRPluginSupport.h"
#include "coVRCommunication.h"
#include "coVRCollaboration.h"
#include "vrbclient/VRBClient.h"
#include <net/tokenbuffer.h>
#include <net/message_types.h>
#include <cassert>
#include <vrbclient/SharedState.h>
#include <vrbclient/SessionID.h>

using namespace covise;
namespace opencover
{
VrbMenue::VrbMenue()
    :ui::Owner("VRBMenue", cover->ui)
{
    init();
}

void VrbMenue::init()
{
    menue = new ui::Menu("VrbOptions", this);
    menue->setText("Vrb");

    ioGroup = new ui::Group(menue, "ioGroup");

    saveBtn = new ui::Action(ioGroup, "SaveSession");
    saveBtn->setText("Save session");
    saveBtn->setCallback([this]()
    {
        saveSession();
    });

    loadSL = new ui::SelectionList(ioGroup, "LoadSession");
    loadSL->setText("Load session");
    loadSL->setCallback([this](int index)
    {
        loadSession(index);
    });
    loadSL->setList(savedRegistries);

    sessionGroup = new ui::Group(menue, "sessisonGroup");

    newSessionBtn = new ui::Action(sessionGroup, "newSession");
    newSessionBtn->setText("New session");
    newSessionBtn->setCallback([this](void) {
        requestNewSession("");
    });

    newSessionEf = new ui::EditField(sessionGroup, "newSessionEf");
    newSessionEf->setText("enter session name");
    newSessionEf->setCallback([this](std::string name) {
        requestNewSession(name);
    });


    sessionsSl = new ui::SelectionList(sessionGroup, "AvailableSessions");
    sessionsSl->setText("Available sessions");
    sessionsSl->setCallback([this](int id)
    {
        selectSession(id);
    });
    sessionsSl->setList(std::vector<std::string>());

    menue->setVisible(false);
}
void VrbMenue::updateState(bool state)
{
    menue->setVisible(state);
}
//io functions : private
void VrbMenue::saveSession()
{
    assert(coVRCommunication::instance()->getPrivateSessionIDx() != vrb::SessionID());
    TokenBuffer tb;
    if (coVRCommunication::instance()->getSessionID().isPrivate())
    {
        tb << coVRCommunication::instance()->getPrivateSessionIDx();
    }
    else
    {
        tb << coVRCommunication::instance()->getSessionID();
    }
    cover->getSender()->sendMessage(tb, COVISE_MESSAGE_VRB_SAVE_SESSION);
}
void VrbMenue::loadSession(int index)
{
    if (index == 0)
    {
        unloadAll();
        return;
    }
    std::vector<std::string>::iterator it = savedRegistries.begin();
    std::advance(it, index);
    loadSession(*it);
}
void VrbMenue::loadSession(const std::string &filename)
{
    TokenBuffer tb;
    tb << coVRCommunication::instance()->getID();
    if (coVRCommunication::instance()->getSessionID().isPrivate())
    {
        tb << coVRCommunication::instance()->getPrivateSessionIDx();
    }
    else
    {
        tb << coVRCommunication::instance()->getSessionID();
    }
    tb << filename;
    cover->getSender()->sendMessage(tb, COVISE_MESSAGE_VRB_LOAD_SESSION);
}
void VrbMenue::unloadAll()
{
}
//io functions : public
void VrbMenue::updateRegistries(const std::vector<std::string> &registries)
{
    savedRegistries = registries;
    savedRegistries.insert(savedRegistries.begin(), noSavedSession);
    loadSL->setList(savedRegistries);
}
//session functions : private
void VrbMenue::requestNewSession(const std::string &name)
{
    covise::TokenBuffer tb;
    tb << vrb::SessionID(coVRCommunication::instance()->getID(), name, false);
    cover->getSender()->sendMessage(tb, covise::COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION);
}
void VrbMenue::selectSession(int id)
{
    std::vector<vrb::SessionID>::iterator it = availiableSessions.begin();
    std::advance(it, id);
    if (*it != coVRCommunication::instance()->getSessionID())
    {
        //Toggle avatar visability
        coVRCollaboration::instance()->sessionChanged(it->isPrivate());
        //inform the server about the new session
        coVRCommunication::instance()->setSessionID(*it);
    }
}

//session functions : public
void VrbMenue::updateSessions(const std::vector<vrb::SessionID>& sessions)
{
    availiableSessions.clear();
    std::vector<std::string> sessionNames;
    for (const auto &session : sessions)
    {
        if (!session.isPrivate() || session.owner() ==  coVRCommunication::instance()->getID())
        {
            availiableSessions.push_back(session);
            sessionNames.push_back(session.toText());
        }
    }
    sessionsSl->setList(sessionNames);
}
void VrbMenue::setCurrentSession(const vrb::SessionID & session)
{
    bool found = false;
    int index = -1;
    for (int i = 0; i < availiableSessions.size(); i++)
    {
        if (availiableSessions[i] == session)
        {
            found = true;
            index = i;
            break;
        }
    }
    if (!found)
    {
        return;
    }
    sessionsSl->select(index);
}

}
