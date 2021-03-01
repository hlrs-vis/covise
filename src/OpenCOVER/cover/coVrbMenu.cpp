/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVrbMenu.h"

#include "OpenCOVER.h"
#include "coVRCollaboration.h"
#include "coVRCommunication.h"
#include "coVRPartner.h"
#include "coVRPluginSupport.h"
#include "ui/Action.h"
#include "ui/EditField.h"
#include "ui/FileBrowser.h"
#include "ui/Menu.h"
#include "ui/SelectionList.h"
#include "ui/Slider.h"

#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <vrb/SessionID.h>
#include <vrb/client/LaunchRequest.h>
#include <vrb/client/SharedState.h>
#include <vrb/client/VRBClient.h>

#include <cassert>
#include <sstream>

using namespace covise;
namespace opencover
{


    VrbMenu::VrbMenu()
        : ui::Owner("VRBMenu", cover->ui)
    {
        coVRCommunication::instance()->addOnConnectCallback([this]() {
            updateState(true);
        });
        coVRCommunication::instance()->addOnDisconnectCallback([this]() {
            updateState(false);
        });
    }

    void VrbMenu::initFileMenu()
    {
        //session management
        m_sessionGroup = new ui::Group("VrbGroup", this);
        m_sessionGroup->setText("VRB");
        cover->fileMenu->add(m_sessionGroup);

        m_newSessionBtn = new ui::Action(m_sessionGroup, "newSession");
        m_newSessionBtn->setText("New session");
        m_newSessionBtn->setCallback([this](void) {
            requestNewSession("");
        });

        m_newSessionEf = new ui::EditField(m_sessionGroup, "newSessionEf");
        m_newSessionEf->setText("enter session name");
        m_newSessionEf->setCallback([this](std::string name) {
            requestNewSession(name);
        });

        m_sessionsSl = new ui::SelectionList(m_sessionGroup, "CurrentSessionSl");
        m_sessionsSl->setText("Current session");
        m_sessionsSl->setCallback([this](int id) {
            selectSession(id);
        });
        m_sessionsSl->setList(std::vector<std::string>());

        m_remoteLauncher = new ui::SelectionList(m_sessionGroup, "remotePartnerSl");
        m_remoteLauncher->setText("Launch remote COVER");
        m_remoteLauncher->setList(std::vector<std::string>());
        m_remoteLauncher->setEnabled(false);
        m_remoteLauncher->setCallback([this](int index) {
            std::vector<std::string> args;
            args.push_back("-C");
            args.push_back(vrbc->getCredentials().ipAddress + ":" + std::to_string(vrbc->getCredentials().tcpPort));
            if (!coVRCommunication::instance()->getSessionID().isPrivate()) {
                args.push_back("-g");
                args.push_back(coVRCommunication::instance()->getSessionID().name());
            }
            vrb::sendLaunchRequestToRemoteLaunchers(vrb::VRB_MESSAGE{vrb::Program::opencover, getRemoteLauncherClientID(index), args}, cover);
            });

        //save and load sessions
        m_ioGroup = new ui::Group("IoGroup", this);
        m_ioGroup->setText("");
        cover->fileMenu->add(m_ioGroup);

        m_saveSession = new ui::FileBrowser(m_ioGroup, "SaveSession", true);
        m_saveSession->setText("Save session");
        m_saveSession->setFilter("*.vrbreg;");
        m_saveSession->setCallback([this](const std::string &file) {
            saveSession(file);
        });

        m_loadSession = new ui::FileBrowser(m_ioGroup, "LoadSession");
        m_loadSession->setText("Load session");
        m_loadSession->setFilter("*.vrbreg;");
        m_loadSession->setCallback([this](const std::string &file) {
            loadSession(file);
        });

        updateState(false);
    }

    void VrbMenu::updateState(bool state)
    {
        m_newSessionBtn->setEnabled(state);
        m_newSessionEf->setEnabled(state);
        m_sessionsSl->setEnabled(state);
        m_saveSession->setEnabled(state);
        m_loadSession->setEnabled(state);
    }
    //io functions : private
    void VrbMenu::saveSession(const std::string &file)
    {
        assert(coVRCommunication::instance()->getPrivateSessionID() != vrb::SessionID());
        TokenBuffer tb;
        tb << coVRCommunication::instance()->getID();
        tb << coVRCommunication::instance()->getUsedSessionID();
        tb << file;
        cover->send(tb, COVISE_MESSAGE_VRB_SAVE_SESSION);
    }

    void VrbMenu::loadSession(const std::string &filename)
    {
        coVRCommunication::instance()->loadSessionFile(filename);
    }

    //session functions : private

    void VrbMenu::requestNewSession(const std::string &name)
    {
        covise::TokenBuffer tb;
        tb << vrb::SessionID(coVRCommunication::instance()->getID(), name, false);
        covise::Message msg(tb);
        msg.type = covise::COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION;
        cover->send(&msg);
    }
    void VrbMenu::selectSession(int id)
    {
        std::vector<vrb::SessionID>::iterator it = m_availiableSessions.begin();
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
    void VrbMenu::updateSessions(const std::vector<vrb::SessionID> &sessions)
    {
        m_availiableSessions.clear();
        std::vector<std::string> sessionNames;
        int index = 0;
        for (const auto &session : sessions)
        {
            if (!session.isPrivate() || session == coVRCommunication::instance()->getPrivateSessionID())
            {
                m_availiableSessions.push_back(session);
                std::stringstream ss;
                ss << session;
                sessionNames.push_back(ss.str());
            }
            if (session == coVRCommunication::instance()->getSessionID())
            {
                index = sessionNames.size() - 1;
            }
        }
        m_sessionsSl->setList(sessionNames);
        m_sessionsSl->select(index);
    }

    void VrbMenu::updateRemoteLauncher(){

        std::vector<std::string> remoteLauncher;
        for (const auto &partner : *coVRPartnerList::instance())
        {
            if (partner->userInfo().userType == vrb::Program::VrbRemoteLauncher)
            {
                remoteLauncher.push_back(partner->userInfo().userName);
            }
        }
        m_remoteLauncher->setList(remoteLauncher);
        m_remoteLauncher->setEnabled(remoteLauncher.size() > 0);
    }

    void VrbMenu::setCurrentSession(const vrb::SessionID &session)
    {
        bool found = false;
        int index = -1;
        for (int i = 0; i < m_availiableSessions.size(); i++)
        {
            if (m_availiableSessions[i] == session)
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
        m_sessionsSl->select(index);
    }

int getRemoteLauncherClientID(int index){
    auto partner = coVRPartnerList::instance()->begin();
    int i = 0, clientID = 0;
    while (partner != coVRPartnerList::instance()->end())
    {
        if ((*partner)->userInfo().userType == vrb::Program::VrbRemoteLauncher)
        {
            if (i == index)
            {
                return (*partner)->ID();
            }
            ++i;
        }
        ++partner;
    }
    return 0;
}


} // namespace opencover
