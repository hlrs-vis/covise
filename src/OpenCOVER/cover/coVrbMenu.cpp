/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVrbMenu.h"

#include "ui/Action.h"
#include "ui/EditField.h"
#include "ui/Menu.h"
#include "ui/SelectionList.h"
#include "ui/FileBrowser.h"

#include "ui/Slider.h"

#include "OpenCOVER.h"
#include "coVRPluginSupport.h"
#include "coVRCommunication.h"
#include "coVRCollaboration.h"
#include <vrb/client/VRBClient.h>
#include <net/tokenbuffer.h>
#include <net/message_types.h>
#include <cassert>
#include <vrb/client/SharedState.h>
#include <vrb/remoteLauncher/MessageTypes.h>
#include <vrb/SessionID.h>

//test remote fetch
#include <coTabletUI.h>
#include <coVRFileManager.h>
#include <coVRMSController.h>
#include <net/message.h>
#include <fcntl.h>
#include <boost/filesystem/operations.hpp>
#include <iostream>
#include <limits>

#include <QString>
#include <unistd.h>
using namespace covise;
namespace fs = boost::filesystem;
namespace opencover
{

    RemoteLauncher::RemoteLauncher(VrbMenu *menu)
        : m_menu(menu)
    {
    }

    void RemoteLauncher::connectSignals()
    {

        connect(&m_launcher, &vrb::launcher::VrbRemoteLauncher::updateClient, this, [this](int clientID, QString clientInfo) {
            auto partner = std::find_if(m_launchPartner.begin(), m_launchPartner.end(),
                                        [clientID](const Partner &p) { return p.first == clientID; });
            if (partner == m_launchPartner.end())
            {
                m_launchPartner.emplace_back(std::make_pair(clientID, clientInfo.toStdString()));
            }
            else
            {
                partner->second = clientInfo.toStdString();
            }
            std::sort(m_launchPartner.begin(), m_launchPartner.end(), [](const Partner &p1, const Partner &p2) {
                return p1.first > p2.first;
            });
            std::vector<std::string> s(m_launchPartner.size());
            std::transform(m_launchPartner.begin(), m_launchPartner.end(), s.begin(), [](const std::pair<int, std::string> &cl) {
                return cl.second;
            });
            m_menu->m_remotePartner->setList(s);
        });

        connect(&m_launcher, &vrb::launcher::VrbRemoteLauncher::removeClient, this, [this](int clientID) {
            auto partner = std::find_if(m_launchPartner.begin(), m_launchPartner.end(),
                                        [clientID](const Partner &p) { return p.first == clientID; });
            if (partner != m_launchPartner.end())
            {
                m_launchPartner.erase(partner);
            }
            std::sort(m_launchPartner.begin(), m_launchPartner.end(), [](const Partner &p1, const Partner &p2) {
                return p1.first > p2.first;
            });
            std::vector<std::string> s(m_launchPartner.size());
            std::transform(m_launchPartner.begin(), m_launchPartner.end(), s.begin(), [](const std::pair<int, std::string> &cl) {
                return cl.second;
            });
            m_menu->m_remotePartner->setList(s);
        });
        /*
    connect(&m_launcher, &vrb::launcher::VrbRemoteLauncher::launchSignal, this, [this](vrb::launcher::Program program, std::vector<std::string> args) {
        if (program != vrb::launcher::Program::COVER)
        {
            return;
        }

        auto session = std::find(args.begin(), args.end(), "-g");
        if (session != args.end())
        {
            ++session;
        }
        if (session != args.end())
        {
            
        }
        auto it = std::find(m_availiableSessions.begin(), m_availiableSessions.end(), )
            std::advance(it, id);
        if (*it != coVRCommunication::instance()->getSessionID())
        {
            //Toggle avatar visability
            coVRCollaboration::instance()->sessionChanged(it->isPrivate());
            //inform the server about the new session
            coVRCommunication::instance()->setSessionID(*it);
        }
    });
    */
        m_menu->m_remotePartner->setCallback([this](int index) {
            std::vector<std::string> args;
            args.push_back("-C");
            args.push_back(vrbc->getCredentials().ipAddress + ":" + std::to_string(vrbc->getCredentials().tcpPort));
            args.push_back("-g");
            args.push_back(coVRCommunication::instance()->getSessionID().name());
            m_launcher.sendLaunchRequest(vrb::launcher::Program::COVER, m_launchPartner[index].first, args);
        });
    }

    void RemoteLauncher::connectVrb()
    {
        assert(vrbc);
        m_launcher.connect(vrbc->getCredentials());
    }

    VrbMenu::VrbMenu()
        : ui::Owner("VRBMenu", cover->ui), m_remoteLauncher(this)
    {
        coVRCommunication::instance()->addOnConnectCallback([this]() {
            m_remoteLauncher.connectVrb();
            m_remotePartner->setVisible(true);
        });
        coVRCommunication::instance()->addOnDisconnectCallback([this]() {
            m_remotePartner->setVisible(false);
        });
    }

    void VrbMenu::initFileMenu()
    {
        //session management
        m_sessionGroup = new ui::Group("sessisonGroup", this);
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

        //save and load sessions
        m_ioGroup = new ui::Group("Sessions", this);
        cover->fileMenu->add(m_ioGroup);

        m_saveSession = new ui::FileBrowser(m_ioGroup, "SaveSession", true);
        m_saveSession->setText("Save session");
        m_saveSession->setFilter("*.vrbreg;");
        m_saveSession->setCallback([this](const std::string &file) {
            saveSession(file);
        });
        m_saveSession->setVisible(false);

        m_loadSession = new ui::FileBrowser(m_ioGroup, "LoadSession");
        m_loadSession->setText("Load session");
        m_loadSession->setFilter("*.vrbreg;");
        m_loadSession->setCallback([this](const std::string &file) {
            loadSession(file);
        });
        m_loadSession->setVisible(false);

        m_remotePartner = new ui::SelectionList(m_sessionGroup, "remotePartnerSl");
        m_remotePartner->setText("Launch remote COVER");

        m_remotePartner->setList(std::vector<std::string>());
        m_remotePartner->setVisible(false);

        m_remoteLauncher.connectSignals();

    }
    void VrbMenu::updateState(bool state)
    {
        m_sessionGroup->setVisible(state);

        m_ioGroup->setVisible(state);
        m_saveSession->setVisible(state);
        m_loadSession->setVisible(state);
    }
    //io functions : private
    void VrbMenu::saveSession(const std::string &file)
    {
        assert(coVRCommunication::instance()->getPrivateSessionIDx() != vrb::SessionID());
        TokenBuffer tb;
        tb << coVRCommunication::instance()->getID();
        tb << coVRCommunication::instance()->getUsedSessionID();
        tb << file;
        cover->getSender()->sendMessage(tb, COVISE_MESSAGE_VRB_SAVE_SESSION);
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
        //cover->getSender()->sendMessage(tb, covise::COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION);
        //test udp
        covise::Message msg(tb);
        msg.type = covise::COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION;
        cover->sendVrbMessage(&msg);
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
            if (session.name() != "VrbRemoteLauncher" && !session.isPrivate() || session.owner() == coVRCommunication::instance()->getID())
            {
                m_availiableSessions.push_back(session);
                sessionNames.push_back(session.toText());
            }
            if (session == coVRCommunication::instance()->getSessionID())
            {
                index = sessionNames.size() - 1;
            }
        }
        m_sessionsSl->setList(sessionNames);
        m_sessionsSl->select(index);
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

    void VrbMenu::connectRemotaLauncher()
    {
        m_remoteLauncher.connectSignals();
    }

} // namespace opencover
