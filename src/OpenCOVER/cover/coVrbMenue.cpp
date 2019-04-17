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
using namespace covise;
namespace fs = boost::filesystem;
namespace opencover
{
VrbMenue::VrbMenue()
    :ui::Owner("VRBMenue", cover->ui)
    ,testTest("VRVMenue_testTest", 0.0, vrb::NEVER_SHARE)
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

    ////test test test//////
    testSlider = new ui::Slider(sessionGroup, "testSlider");
    testSlider->setEnabled(true);
    testSlider->setText("blablabla");
    testSlider->setBounds(1.0, 10.0);
    testSlider->setValue(0.0);
    testSlider->setCallback([this](double i, bool b) {
        if (b)
        {
            testTest = i;
        }
    });

    testTest.setUpdateFunction([this]() {
        testSlider->setValue(testTest);
    });

    requestFile = new ui::Action(sessionGroup, "requestFile");
    requestFile->setText("request File");
    requestFile->setCallback([this]() {
        std::string file = remoteFetch("C:\\src\\anipod.obj");
        coVRFileManager::instance()->loadFile(file.c_str());
    });
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
std::string VrbMenue::remoteFetch(const char *filename)
{
    char *result = 0;
    const char *buf = NULL;
    int numBytes = 0;
    static int working = 0;

    if (working)
    {
        cerr << "WARNING!!! reentered remoteFetch!!!!" << endl;
        return std::string();
    }

    working = 1;

    if (strncmp(filename, "vrb://", 6) == 0)
    {
        //Request file from VRB
        std::cerr << "VRB file, needs to be requested through FileBrowser-ProtocolHandler!" << std::endl;
        coTUIFileBrowserButton *locFB = coVRFileManager::instance()->getMatchingFileBrowserInstance(string(filename));
        std::string sresult = locFB->getFilename(filename).c_str();
        char *result = new char[sresult.size() + 1];
        strcpy(result, sresult.c_str());
        working = 0;
        return std::string(result);
    }
    else if (strncmp(filename, "agtk3://", 8) == 0)
    {
        //REquest file from AG data store
        std::cerr << "AccessGrid file, needs to be requested through FileBrowser-ProtocolHandler!" << std::endl;
        coTUIFileBrowserButton *locFB = coVRFileManager::instance()->getMatchingFileBrowserInstance(string(filename));
        working = 0;
        return std::string(locFB->getFilename(filename).c_str());
    }

    if (vrbc || !coVRMSController::instance()->isMaster())
    {
        if (coVRMSController::instance()->isMaster())
        {
            TokenBuffer rtb;
            rtb << filename;
            rtb << vrbc->getID();
            Message m(rtb);
            m.type = COVISE_MESSAGE_VRB_REQUEST_FILE;
            cover->sendVrbMessage(&m);
        }
        int message = 1;
        Message *msg = new Message;
        do
        {
            if (coVRMSController::instance()->isMaster())
            {
                if (!vrbc->isConnected())
                {
                    message = 0;
                    coVRMSController::instance()->sendSlaves((char *)&message, sizeof(message));
                    break;
                }
                else
                {
                    vrbc->wait(msg);
                }
                coVRMSController::instance()->sendSlaves((char *)&message, sizeof(message));
            }
            if (coVRMSController::instance()->isMaster())
            {
                coVRMSController::instance()->sendSlaves(msg);
            }
            else
            {
                coVRMSController::instance()->readMaster((char *)&message, sizeof(message));
                if (message == 0)
                    break;
                // wait for message from master instead
                coVRMSController::instance()->readMaster(msg);
            }
            coVRCommunication::instance()->handleVRB(msg);
        } while (msg->type != COVISE_MESSAGE_VRB_SEND_FILE);

        if ((msg->data) && (msg->type == COVISE_MESSAGE_VRB_SEND_FILE))
        {
            TokenBuffer tb(msg);
            int myID;
            tb >> myID; // this should be my ID
            tb >> numBytes;
            buf = tb.getBinary(numBytes);
            if ((numBytes > 0) && (result = tempnam(0, "VR")))
            {
#ifndef _WIN32
                int fd = open(result, O_RDWR | O_CREAT, 0777);
#else
                int fd = open(result, O_RDWR | O_CREAT | O_BINARY, 0777);
#endif
                if (fd != -1)
                {
                    if (write(fd, buf, numBytes) != numBytes)
                    {
                        //warn("remoteFetch: temp file write error\n");
                        free(result);
                        result = NULL;
                    }
                    close(fd);
                }
                else
                {
                    free(result);
                    result = NULL;
                }
            }
        }
        delete msg;
    }
    std::string pathToTmpFile = cutFileName(std::string(result)) + "/" + getFileName(std::string(filename));
    fs::rename(result, pathToTmpFile);
    working = 0;
    return pathToTmpFile;
}
std::string VrbMenue::getFileName(std::string &fileName)
{
    std::string name;
    for (size_t i = fileName.length() - 1; i > 0; --i)
    {
        if (fileName[i] == '/' || fileName[i] == '\\')
        {
            return name;
        }
        name.insert(name.begin(), fileName[i]);
    }
    cerr << "invalid file path : " << fileName << endl;
    return "";
}
std::string VrbMenue::cutFileName(std::string &fileName)
{
    std::string name = fileName;
    for (size_t i = fileName.length() - 1; i > 0; --i)
    {
        name.pop_back();
        if (fileName[i] == '/' || fileName[i] == '\\')
        {
            return name;
        }

    }
    cerr << "invalid file path : " << fileName << endl;
    return "";
}

}
