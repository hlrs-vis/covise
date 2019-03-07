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
#include "vrbclient/VRBClient.h"
#include <net/tokenbuffer.h>
#include <net/message_types.h>
#include <cassert>
#include <vrbclient/SharedState.h>

using namespace covise;
namespace opencover
{
VrbMenue::VrbMenue(ui::Owner *owner) 
    :m_owner(owner)
{
    init();
}
void VrbMenue::updateState(bool state)
{
    menue->setVisible(state);
}
void VrbMenue::updateRegistries(std::vector<std::string> &registries)
{
    savedRegistries = registries;
    savedRegistries.insert(savedRegistries.begin(), noSavedSession);
    loadSL->setList(savedRegistries);
}
void VrbMenue::updateSessions(std::vector<std::string>& sessions)
{
    availiableSessions = sessions;
    availiableSessions.insert(availiableSessions.begin(), noAvailiableSession);
    SessionsSl->setList(availiableSessions);
}
void VrbMenue::init()
{
    menue = new ui::Menu("VrbOptions", m_owner);
    menue->setText("Vrb");

    saveBtn.reset(new ui::Action(menue, "SaveSession"));
    saveBtn->setText("Save session");
    saveBtn->setCallback([this]()
    {
        saveSession();
    });

    loadSL.reset(new ui::SelectionList(menue, "LoadSession"));
    loadSL->setText("Load session");
    loadSL->setCallback([this](int index)
    {
        if (index == 0)
        {
            unloadAll();
            return;
        }
        std::vector<std::string>::iterator it = savedRegistries.begin();
        std::advance(it, index);
        loadSession(*it);
    });
    loadSL->setList(savedRegistries);

    newSessionBtn.reset(new ui::Action(menue, "newSession"));
    newSessionBtn->setText("New session");
    newSessionBtn->setCallback([this](void) {
        bool isPrivate = false;
        covise::TokenBuffer tb;
        tb << coVRCommunication::instance()->getID();
        tb << coVRCommunication::instance()->getPublicSessionID();
        tb << isPrivate;
        vrbc->sendMessage(tb, covise::COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION);
    });

    SessionsSl.reset(new ui::SelectionList(menue, "AvailableSessions"));
    SessionsSl->setText("Available sessions");
    SessionsSl->setCallback([this](int id)
    {
        int sessionID = 0;
        if (id != 0)
        {
            std::vector<int>::iterator it = availiableSessions.begin()
            std::advance(it, id );
            sessionID = *it;
            if (syncMode == LooseCoupling)
            {
                VRAvatarList::instance()->show();
            }
        }
        else //dont sync when in private session
        {
            UnSyncXform();
            UnSyncScale();
            VRAvatarList::instance()->hide();
        }
        //inform the server about the new session
        coVRCommunication::instance()->setSessionID(sessionID);
    });
    m_availableSessions->setList(std::vector<std::string>{"private"});


}



void VrbMenue::saveSession()
{
    assert(coVRCommunication::instance()->getPrivateSessionID() != 0);
    TokenBuffer tb;
    if (coVRCommunication::instance()->getPublicSessionID() == 0)
    {
        tb << coVRCommunication::instance()->getPrivateSessionID();
    }
    else
    {
        tb << coVRCommunication::instance()->getPublicSessionID();
    }
    if (vrbc)
    {
        vrbc->sendMessage(tb, COVISE_MESSAGE_VRB_SAVE_SESSION);
    }
}

void VrbMenue::loadSession(const std::string &filename)
{
    TokenBuffer tb;
    tb << coVRCommunication::instance()->getID();
    if (coVRCommunication::instance()->getPublicSessionID() == 0)
    {
        tb << coVRCommunication::instance()->getPrivateSessionID();
    }
    else
    {
        tb << coVRCommunication::instance()->getPublicSessionID();
    }
    tb << filename;
    if (vrbc)
    {
        vrbc->sendMessage(tb, COVISE_MESSAGE_VRB_LOAD_SESSION);
    }
}

void VrbMenue::unloadAll()
{
}


}