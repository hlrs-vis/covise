/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVrbMenue.h"

#include "ui/Action.h"
#include "ui/EditField.h"
#include "ui/Menu.h"
#include "ui/SelectionList.h"

#include "ui/Slider.h"

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

//test remote fetch
#include <coTabletUI.h>
#include <coVRFileManager.h>
#include <coVRMSController.h>
#include <net/message.h>
#include <fcntl.h>
#include <boost/filesystem/operations.hpp>

//udp test
#include <net/udp_message_types.h>
#include <net/udpMessage.h>
using namespace covise;
namespace fs = boost::filesystem;
namespace opencover
{
VrbMenue::VrbMenue()
    :ui::Owner("VRBMenue", cover->ui)
	, test("test")
{
    init();
	std::map<std::string, std::string> m;
	m["1"] = "nicht 1";
	m["2"] = "nicht 2";
	m["3"] = "nicht 3";
	test = m;
	test.setUpdateFunction([this]()
		{
			/*std::cerr << "test map got updated: ____________________" << std::endl;
			for (auto it : test.value())
			{
				cerr << it.first << ": " << it.second << endl;
			}
			cerr << "_____________________________________________________________" << endl;*/
		});
}

void VrbMenue::init()
{
    menue = new ui::Menu("VrbOptions", this);
    menue->setText("Connections");

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
	//test
	ui::Action *testBtn = new ui::Action(sessionGroup, "testBtn");
	testBtn->setText("testBtn");
	testBtn->setCallback([this]()
		{
			covise::TokenBuffer tb;
			tb << std::string("test udp message from OpenCOVER");
			vrb::UdpMessage m(tb);
			m.type = vrb::udp_msg_type::AVATAR_HMD_POSITION;
			cover->sendVrbUdpMessage(&m);
			//static int count = 0;
			//static int loop = 0;
			//if (count < test.value().size())
			//{
			//	++count;
			//	test.changeEntry(std::to_string(count), "value at pos " + std::to_string(count));
			//}
			//else if (count > 5)
			//{
			//	count = 1;
			//	++loop;
			//	test.changeEntry(std::to_string(count), "new loop (" + std::to_string(loop) + ") at pos " + std::to_string(count));
			//}
			//else
			//{
			//	++count;
			//	std::map<std::string, std::string> m = test.value();
			//	m[std::to_string(count)] = "new entry at " + std::to_string(count);
			//	test = m;
			//}

		});
	testBtn->setEnabled(true);
}

void VrbMenue::initFileMenue()
{
	ioGroup = new ui::Group("Sessions", this);
	cover->fileMenu->add(ioGroup);
	saveBtn = new ui::Action(ioGroup, "SaveSession");
	saveBtn->setText("Save session");
	saveBtn->setCallback([this]()
		{
			saveSession();
		});
	saveBtn->setVisible(false);

	loadSL = new ui::SelectionList(ioGroup, "LoadSession");
	loadSL->setText("Load session");
	loadSL->setCallback([this](int index)
		{
			loadSession(index);
		});
	loadSL->setList(savedRegistries);
	loadSL->setVisible(false);
}
void VrbMenue::updateState(bool state)
{
    menue->setVisible(state);
	ioGroup->setVisible(state);
	saveBtn->setVisible(state);
	loadSL->setVisible(state);

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
    //cover->getSender()->sendMessage(tb, covise::COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION);
	//test udp
	covise::Message msg(tb);
	msg.type = covise::COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION;
	cover->sendVrbMessage(&msg);
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
	int index = 0;
	for (const auto &session : sessions)
    {
        if (!session.isPrivate() || session.owner() ==  coVRCommunication::instance()->getID())
        {
            availiableSessions.push_back(session);
            sessionNames.push_back(session.toText());
        }
		if (session == coVRCommunication::instance()->getSessionID())
		{
			index = sessionNames.size() - 1;
		}
    }
    sessionsSl->setList(sessionNames);
	sessionsSl->select(index);
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
