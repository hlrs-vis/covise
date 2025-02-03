/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvVrbMenu.h"

#include "vvVIVE.h"
#include "vvCollaboration.h"
#include "vvCommunication.h"
#include "vvPartner.h"
#include "vvPluginSupport.h"
#include "vvMessageSender.h"
#include "ui/Action.h"
#include "ui/EditField.h"
#include "ui/FileBrowser.h"
#include "ui/Menu.h"
#include "ui/SelectionList.h"
#include "ui/Slider.h"
#include "ui/View.h"

#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <vrb/SessionID.h>
#include <vrb/client/LaunchRequest.h>
#include <vrb/client/SharedState.h>
#include <vrb/client/VRBClient.h>

#include <cassert>
#include <sstream>

using namespace covise;
namespace vive
{

VrbMenu::VrbMenu() : ui::Owner("VRBMenu", vv->ui), sender(new vvMessageSender)
{
    vvCommunication::instance()->subscribeNotification(vvCommunication::Notification::Connected, [this]()
                                                        { updateState(true); });
    vvCommunication::instance()->subscribeNotification(vvCommunication::Notification::Disconnected, [this]()
                                                            { updateState(false); });
    vvCommunication::instance()->subscribeNotification(vvCommunication::Notification::SessionChanged, [this]()
                                                                {
                                                                    for (const auto cb : m_onSessionChangedCallbacks)
                                                                        cb();
                                                                    m_onSessionChangedCallbacks.clear();
                                                                    std::cerr << "handles on session changed" << std::endl;
                                                                });
}

void VrbMenu::initFileMenu()
{
    // session management
    m_sessionGroup = new ui::Group(vv->fileMenu,"VrbGroup");
    m_sessionGroup->setText("VRB");

    m_newSessionBtn = new ui::Action(m_sessionGroup, "newSession");
    m_newSessionBtn->setText("New session");
    m_newSessionBtn->setCallback([this](void)
                                    { requestNewSession(""); });

    m_newSessionEf = new ui::EditField(m_sessionGroup, "newSessionEf");
    m_newSessionEf->setText("Set session name");
    m_newSessionEf->setCallback([this](std::string name)
                                { requestNewSession(name); });
    m_newSessionEf->setVisible(false);
    m_newSessionEf->setVisible(true, ui::View::Tablet);

    m_sessionsSl = new ui::SelectionList(m_sessionGroup, "CurrentSessionSl");
    m_sessionsSl->setText("Current session");
    m_sessionsSl->setCallback([this](int id)
                                { selectSession(id); });
    m_sessionsSl->setList(std::vector<std::string>());

    m_remoteLauncher = new ui::SelectionList(m_sessionGroup, "remotePartnerSl");
    m_remoteLauncher->setText("Launch remote COVER");
    m_remoteLauncher->setList(std::vector<std::string>());
    m_remoteLauncher->setEnabled(false);
    m_remoteLauncher->setCallback([this](int index)
                                    {
                                        if(vvCommunication::instance()->getSessionID().isPrivate())
                                        {
                                        m_onSessionChangedCallbacks.push_back(std::bind(&VrbMenu::lauchRemotePartner, this, getRemoteLauncherClientID(index)));
                                        requestNewSession("");
                                        }
                                        else
                                            lauchRemotePartner(getRemoteLauncherClientID(index)); });
    m_remoteLauncher->setVisible(!vvVIVE::instance()->useVistle());

    // save and load sessions
    m_ioGroup = new ui::Group("IoGroup", this);
    m_ioGroup->setText("");
    vv->fileMenu->add(m_ioGroup);

    m_saveSession = new ui::FileBrowser(m_ioGroup, "SaveSession", true);
    m_saveSession->setText("Save session");
    m_saveSession->setFilter("*.vrbreg;");
    m_saveSession->setCallback([this](const std::string &file)
                                { saveSession(file); });

    m_loadSession = new ui::FileBrowser(m_ioGroup, "LoadSession");
    m_loadSession->setText("Load session");
    m_loadSession->setFilter("*.vrbreg;");
    m_loadSession->setCallback([this](const std::string &file)
                                { loadSession(file); });

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
// io functions : private
void VrbMenu::saveSession(const std::string &file)
{
    assert(vvCommunication::instance()->getPrivateSessionID() != vrb::SessionID());
    TokenBuffer tb;
    tb << vvCommunication::instance()->getID();
    tb << vvCommunication::instance()->getUsedSessionID();
    tb << file;
    sender->send(tb, COVISE_MESSAGE_VRB_SAVE_SESSION);
}

void VrbMenu::loadSession(const std::string &filename)
{
    vvCommunication::instance()->loadSessionFile(filename);
}

// session functions : private

void VrbMenu::requestNewSession(const std::string &name)
{
    covise::TokenBuffer tb;
    tb << vrb::SessionID(vvCommunication::instance()->getID(), name, false);
    covise::Message msg(tb);
    msg.type = covise::COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION;
    sender->send(&msg);
}
void VrbMenu::selectSession(int id)
{
    std::vector<vrb::SessionID>::iterator it = m_availiableSessions.begin();
    std::advance(it, id);
    if (*it != vvCommunication::instance()->getSessionID())
    {
        // Toggle avatar visability
        vvCollaboration::instance()->sessionChanged(it->isPrivate());
        // inform the server about the new session
        vvCommunication::instance()->setSessionID(*it);
    }
}

void VrbMenu::lauchRemotePartner(int id)
{
    std::vector<std::string> args;
    args.push_back("-C");
    const auto vrbc = vvVIVE::instance()->vrbc();
    args.push_back(vrbc->getCredentials().ipAddress() + ":" + std::to_string(vrbc->getCredentials().tcpPort()));
    if (!vvCommunication::instance()->getSessionID().isPrivate())
    {
        args.push_back("-g");
        args.push_back(vvCommunication::instance()->getSessionID().name());
    }
    vrb::sendLaunchRequestToRemoteLaunchers(
        vrb::VRB_MESSAGE{vrbc->ID(), covise::Program::opencover, id, std::vector<std::string>(), args, 0},
        sender.get());
}

// session functions : public
void VrbMenu::updateSessions(const std::vector<vrb::SessionID> &sessions)
{
    m_availiableSessions.clear();
    std::vector<std::string> sessionNames;
    int index = 0;
    for (const auto &session : sessions)
    {
        if (!session.isPrivate() || session == vvCommunication::instance()->getPrivateSessionID())
        {
            m_availiableSessions.push_back(session);
            std::stringstream ss;
            ss << session.name();
            if (!session.isPrivate())
            {
                if (auto partner = vvPartnerList::instance()->get(session.master()))
                {
                    ss << ", host: " << partner->userInfo().userName << "@" << partner->userInfo().hostName;
                }
            }
            else
                ss << ", private";

            sessionNames.push_back(ss.str());
        }
        if (session == vvCommunication::instance()->getSessionID())
        {
            index = sessionNames.size() - 1;
        }
    }
    m_sessionsSl->setList(sessionNames);
    m_sessionsSl->select(index);
}

void VrbMenu::updateRemoteLauncher()
{

    std::vector<std::string> remoteLauncher;
    for (const auto &partner : *vvPartnerList::instance())
    {
        if (partner->userInfo().userType == covise::Program::coviseDaemon)
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

int getRemoteLauncherClientID(int index)
{
    auto partner = vvPartnerList::instance()->begin();
    int i = 0, clientID = 0;
    while (partner != vvPartnerList::instance()->end())
    {
        if ((*partner)->userInfo().userType == covise::Program::coviseDaemon)
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

} // namespace vive
