/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 1996					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			coVRCommunication.C 			*
 *									*
 *	Description		communication  class			*
 *									*
 *	Author			U.Woessner				*
 *									*
 *	Date			07 2001			*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/


#include "MarkerTracking.h"
#include "OpenCOVER.h"
#include "VRAvatar.h"
#include "VRViewer.h"
#include "coHud.h"
#include "coTUIFileBrowser/VRBData.h"
#include "coVRCollaboration.h"
#include "coVRCommunication.h"
#include "coVRConfig.h"
#include "coVRFileManager.h"
#include "coVRMSController.h"
#include "coVRPartner.h"
#include "coVRPluginList.h"
#include "coVRPluginSupport.h"
#include "coVRSelectionManager.h"
#include "coVRTui.h"
#include "coVrbMenu.h"

#include "ui/Action.h"
#include "ui/Button.h"
#include "ui/FileBrowser.h"
#include "ui/Group.h"
#include "ui/Menu.h"
#include "ui/Owner.h"
#include "ui/SelectionList.h"

#include <OpenVRUI/coNavInteraction.h>
#include <PluginUtil/PluginMessageTypes.h>
#include <config/CoviseConfig.h>
#include <grmsg/coGRMsg.h>
#include <net/covise_host.h>
#include <net/message_types.h>
#include <net/udpMessage.h>
#include <net/udp_message_types.h>
#include <sys/stat.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <ui/SelectionList.h>
#include <util/coTabletUIMessages.h>
#include <util/common.h>
#include <util/string_util.h>
#include <vrb/SessionID.h>
#include <vrb/VrbSetUserInfoMessage.h>
#include <vrb/client/SharedStateManager.h>
#include <vrb/client/VRBClient.h>
#include <vrb/client/VRBMessage.h>
#include <vrb/client/VRBMessage.h>
#include <vrb/client/VrbClientRegistry.h>
#include <vrb/client/VrbClientRegistry.h>

#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <thread>

#ifdef _WIN32
#include <io.h>
#endif

using namespace covise;
using namespace opencover;
using namespace vrb;

coVRCommunication *coVRCommunication::s_instance = NULL;

coVRCommunication *coVRCommunication::instance()
{
    if (!s_instance)
        s_instance = new coVRCommunication;
    return s_instance;
}

coVRCommunication::coVRCommunication()
//: ui::Owner("VRCommunication", cover->ui)
{
    assert(!s_instance);

    srand((unsigned)time(NULL)); // Initialize the random timer
    ignoreRemoteTransform = coCoviseConfig::isOn("COVER.IgnoreRemoteTransform", false);

    onConnectCallbacks.push_back([this]() {
    	registry->setID(me()->ID(), me()->sessionID());
        registry->registerSender(this);
    });

    onDisconnectCallbacks.push_back([this]() {
    	registry->setID(-1, me()->sessionID());
        registry->registerSender(nullptr);
    });

    registry.reset(new VrbClientRegistry(-1));
	sharedStateManager.reset(new SharedStateManager(registry.get()));
}

coVRCommunication::~coVRCommunication()
{
    onConnectCallbacks.clear();
    onDisconnectCallbacks.clear();
    onSessionChangedCallbacks.clear();

    waitMessagesCallback = nullptr;
    handleMessageCallback = nullptr;

    sharedStateManager.reset();
    registry.reset();

    assert(s_instance);
    coVRPartnerList::instance()->removePartner(me()->ID());
    delete remoteNavInteraction;
    s_instance = NULL;
}

void coVRCommunication::init()
{
	m_vrbMenu.reset(new VrbMenu());
	remoteNavInteraction = new vrui::coNavInteraction(vrui::coInteraction::NoButton, "remoteNavInteraction");
}

void handleEvents(const std::vector<std::function<void(void)>> &events)
{
    for (auto function : events)
	{
		if (function)
		{
			function();
		}
	}
}

void coVRCommunication::connected()
{
    handleEvents(onConnectCallbacks);
}

void coVRCommunication::disconnected()
{
    handleEvents(onDisconnectCallbacks);
}

void coVRCommunication::toggleClientState(bool state){
    static bool connected{false};
    if (connected && !state)
    {
        connected = false;
        cerr << "VRB requests to quit " << endl;
        disconnected();
        coVRPartnerList::instance()->removeOthers();
        m_vrbMenu->updateState(false);
        me()->setSession(vrb::SessionID());
        m_privateSessionID = vrb::SessionID();
        OpenCOVER::instance()->vrbc()->shutdown();
        OpenCOVER::instance()->restartVrbc();
        coVRCollaboration::instance()->updateSharedStates();
    }
    connected = state;
}

coVRPartner *coVRCommunication::me(){
    return coVRPartnerList::instance()->me();
}

const coVRPartner *coVRCommunication::me() const{
    return coVRPartnerList::instance()->me();
}


void coVRCommunication::update(clientRegClass *theChangedClass)
{
    //this might be obsolete
    if (theChangedClass && theChangedClass->name() == "VRMLFile")
    {
        for(auto regvar : *theChangedClass)
        {
            int remoteID = -2;
            if (sscanf(regvar.first.c_str(), "%d", &remoteID) != 1)
            {
                cerr << "coVRCommunication::update: sscanf failed" << endl;
                break;
            }
            auto p = coVRPartnerList::instance()->get(remoteID);
            if(p)
            {
                p->setFile(regvar.second->value().data());
                cerr << theChangedClass->name() << endl;
                cerr << regvar.second->value().data() << endl;
            }
        }
    }
}

int coVRCommunication::getID()
{
    int myID = me()->ID();
    if (myID < 0)
    {
        myID = randomID;
    }
    return myID;
}

const vrb::SessionID & opencover::coVRCommunication::getPrivateSessionID() const
{
    return m_privateSessionID;
}

const vrb::SessionID &opencover::coVRCommunication::getSessionID() const
{
    return me()->sessionID();
}

const vrb::SessionID &opencover::coVRCommunication::getUsedSessionID() const
{
    if (getSessionID().isPrivate())
    {
        return getPrivateSessionID();
    }
    else
    {
        return getSessionID();
    }
}

void opencover::coVRCommunication::setSessionID(const vrb::SessionID &id)
{
    if (id.isPrivate())
    {
        m_privateSessionID = id;
    }
    coVRPartnerList::instance()->setSessionID(me()->ID(), id);
    TokenBuffer tb;
    tb << id;
    tb << me()->ID();
    send(tb, COVISE_MESSAGE_VRBC_SET_SESSION);
    handleEvents(onSessionChangedCallbacks);
}

const char *coVRCommunication::getHostaddress()
{
    return Host::getHostaddress().c_str();
}

std::string coVRCommunication::getUsername()
{
    return Host::getHostname();
}

const char *coVRCommunication::getHostname()
{
    return Host::getHostname().c_str();
}

bool coVRCommunication::collaborative() // returns true, if in collaborative mode
{
    if (coVRPartnerList::instance()->numberOfPartners() > 1)
        return true;
    if (OpenCOVER::instance()->visPlugin())
        return true;
    return false;
}

bool coVRCommunication::isMaster() // returns true, if we are master
{
    if (coVRPartnerList::instance()->numberOfPartners() > 1)
    {
        return me()->isMaster();
    }
    return true;
}

void coVRCommunication::processARVideoFrame(const char *key, const char *tmp)
{
    if (!(strcmp(key, "AR_VIDEO_FRAME")) && MarkerTracking::instance()->remoteAR)
    {
        MarkerTracking::instance()->remoteAR->receiveImage(tmp);
    }
}

void coVRCommunication::processVRBMessage(covise::TokenBuffer &tb)
{
	if (!tb.getData().data())
	{
		cerr << "invalid vrb render message" << endl;
		return;
	}
	int t;
    tb >> t;
    vrb::vrbMessageType type = (vrbMessageType)t;
    switch (type)
    {
    case vrb::AVATAR:
    {
        coVRPartnerList::instance()->receiveAvatarMessage(tb);
    }
    break;
    case vrb::SYNC_MODE:
    {
        bool showAvatar;
        tb >> showAvatar;
        coVRCollaboration::instance()->showAvatars(showAvatar);
    }
        break;
    case vrb::MASTER:
    {
        coVRPartnerList::instance()->setMaster(me()->ID());
        coVRCollaboration::instance()->updateSharedStates();
        coVRCollaboration::instance()->updateUi();
    }
        break;
    case vrb::SLAVE:
    {
        int id;
        tb >> id;
        coVRPartnerList::instance()->setMaster(id); //nobody is master here?
        coVRCollaboration::instance()->updateSharedStates();
        coVRCollaboration::instance()->updateUi();
    }
        break;
    case vrb::MOVE_HAND:
    {
        cerr << "braucht das doch jemand" << endl;
        /*   mat(0,3) = 0;
         mat(1,3) = 0;
         mat(2,3) = 0;
         mat(3,3) = 1;
         int button = 0;
         sscanf(tmp, "%f %f %f %f %f %f %f %f %f %f %f %f %d %f %f", \
            &mat(0,0), &mat(0,1), &mat(0,2), \
            &mat(1,0), &mat(1,1), &mat(1,2), \
            &mat(2,0), &mat(2,1), &mat(2,2), \
            &mat(3,0), &mat(3,1), &mat(3,2) );
            &button, &(VRSceneGraph::instance()->AnalogX), &(VRSceneGraph::instance()->AnalogY));

      cover->getButton()->setButtonStatus(button);
      VRSceneGraph::instance()->updateHandMat(mat);*/
    }
        break;
    case vrb::MOVE_HEAD:
    {
        osg::Matrixd mat;
        covise::deserialize(tb, mat);
        VRViewer::instance()->updateViewerMat(mat);
    }
        break;
    case vrb::AR_VIDEO_FRAME:
    {
        const char * tmp;
        tb >> tmp;
        MarkerTracking::instance()->remoteAR->receiveImage(tmp);
    }
        break;
    case vrb::SYNC_KEYBOARD:
    {
        int type, state, code;
        tb >> type;
        tb >> state;
        tb >> code;
        fprintf(stderr, "Slave receiving SYNC_KEYBOARD msg=[%d %d %d]\n", type, state, code);
        /*  if(((sh->writePos+1)%RINGBUFLEN)==sh->readPos)
        {
           fprintf(stderr,"Keyboard Buffer Overflow!! discarding Events\n");
           sh->readPos++;
        }
        sh->keyType[sh->writePos]=type;
        sh->keyState[sh->writePos]=state;
        sh->keyKeycode[sh->writePos]=code;
        sh->writePos = ((sh->writePos+1)%RINGBUFLEN);*/
    }
    break;
    case vrb::ADD_SELECTION:
    {
        coVRSelectionManager::instance()->receiveAdd(tb);
    }
        break;
    case vrb::CLEAR_SELECTION:
    {
        coVRSelectionManager::instance()->receiveClear();
    }
        break;
    default:
    {
        cerr << type << ": unknown render message" << endl;
    }
        break;
    }

}

void coVRCommunication::becomeMaster()
{
    coVRPluginList::instance()->becomeCollaborativeMaster();
    coVRPartnerList::instance()->setMaster(me()->ID());
    me()->becomeMaster();
    coVRCollaboration::instance()->updateUi();
}

void coVRCommunication::handleVRB(const Message &msg)
{
	//fprintf(stderr,"slave: %d msgProcessed: %s\n",coVRMSController::instance()->isSlave(),covise_msg_types_array[msg->type]);

    OpenCOVER::instance()->startVrbc();
    TokenBuffer tb(&msg);
    switch (msg.type)
    {
    case COVISE_MESSAGE_VRB_SET_USERINFO:
    {
        UserInfoMessage uim(&msg);
        if (uim.hasMyInfo)      
        {
            me()->setID(uim.myClientID);
            OpenCOVER::instance()->vrbc()->setID(uim.myClientID);
            connected();
            m_privateSessionID = uim.myPrivateSession;
            coVRPartnerList::instance()->setSessionID(me()->ID(), uim.mySession);
            m_vrbMenu->setCurrentSession(uim.myClientID);
            toggleClientState(true);
            me()->updateUi();
        }
        for(auto&& cl : uim.otherClients)
        {
            coVRPartnerList::instance()->addPartner(std::move(cl));
        }
        coVRPartnerList::instance()->print();
        m_vrbMenu->updateRemoteLauncher();
        coVRCollaboration::instance()->updateSharedStates();
    }
    break;
    case COVISE_MESSAGE_VRB_SET_MASTER:
    {
        int id;
        bool masterState;
        tb >> id;
        tb >> masterState;
        if (masterState)
        {
            coVRPartnerList::instance()->setMaster(id);
        }
        coVRPartnerList::instance()->print();
        coVRCollaboration::instance()->updateSharedStates();
        coVRCollaboration::instance()->updateUi();
    }
    break;
    case COVISE_MESSAGE_VRB_QUIT:
    {
        int id;
        tb >> id;
        if (id != me()->ID())
        {
             coVRPartnerList::instance()->removePartner(id);
			 vrui::coInteractionManager::the()->resetLock(id);
        }
        if (coVRPartnerList::instance()->numberOfPartners() <= 1)
            coVRCollaboration::instance()->showCollaborative(false);
        m_vrbMenu->updateRemoteLauncher();

    }
    break;
    case COVISE_MESSAGE_VRB_GUI:
    {
        int subtype;
        tb >> subtype;
        switch (subtype)
        {
        case LOAD_FILE:
        {
            const char *fileName;
            tb >> fileName;
            coVRFileManager::instance()->loadFile(fileName);
        }
        break;
        case NEW_FILE:
        {
            coVRFileManager::instance()->replaceFile(NULL);
        }
        break;
        case DO_QUIT:
        {
            cerr << "Thank you for using COVER! " << endl;
            exit(0);
        }
        break;
        default:
        {
            cerr << "unknown VRB_GUI Subtype " << subtype << endl;
        }
        break;
        }
    }
    break;
    case COVISE_MESSAGE_RENDER_MODULE:
    {
        coVRPluginList::instance()->forwardMessage(msg.data);
        break;
    }
    case COVISE_MESSAGE_RENDER:
    {
        if(auto grmsg = grmsg::create(msg.data.data()))
        {
            cover->guiToRenderMsg(*grmsg);
        }
        else
        {
            std::cerr << "coVRCommunication received COVISE_MESSAGE_RENDER, this might be deprecated." << std::endl;
        }
        break;
    }
    case COVISE_MESSAGE_SOCKET_CLOSED:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    {
        toggleClientState(false);
    }
        break;
    case COVISE_MESSAGE_VRB_REQUEST_FILE:
    {
        coVRFileManager::instance()->sendFile(tb);
    }
    break;
    case COVISE_MESSAGE_VRB_FB_SET:
    {

        int subtype;
        int id;
        //Received a filebrowser set command
        tb >> subtype;
        tb >> id;

        VRBData *locData = this->mfbData.find(id)->second;

        if (subtype == TABLET_SET_DIRLIST)
        {
            locData->setDirectoryList(msg);
        }
        else if (subtype == TABLET_SET_FILELIST)
        {
            locData->setFileList(msg);
        }
        else if (subtype == TABLET_SET_CURDIR)
        {
            locData->setCurDir(msg);
        }
        else if (subtype == TABLET_SET_CLIENTS)
        {
            locData->setClientList(msg);
        }
        else if (subtype == TABLET_SET_DRIVES)
        {
            locData->setDrives(msg);
        }
        else if (subtype == TABLET_SET_FILE)
        {
            locData->setFile(msg);
        }
        else if (subtype == TABLET_SET_GLOBALLOAD)
        {
            // Enable loading here

            //Retrieve Data object
            const char *curl = NULL;

            tb >> curl;

            OpenCOVER::instance()->hud->show();
            OpenCOVER::instance()->hud->setText1("Replacing File...");
            OpenCOVER::instance()->hud->setText2(curl);
            //Do what you want to do with the filename

            coVRFileManager::instance()->replaceFile(curl, coVRTui::instance()->getExtFB());

            OpenCOVER::instance()->hud->hide();
        }
        else
        {
            cerr << "Unknown type!" << endl;
        }
    }
    break;
    case COVISE_MESSAGE_VRB_FB_REMREQ:
    {

        if (coVRMSController::instance()->isSlave())
            return;
        int subtype;
        //Received a filebrowser set command
        int id;
        tb >> subtype;
        tb >> id;

        VRBData *locData = this->mfbData.find(id)->second;

        if (subtype == TABLET_SET_DIRLIST)
        {
            //Call local file system operation for directory listing
            locData->setRemoteDirList(msg);
        }
        else if (subtype == TABLET_SET_FILELIST)
        {
            //Call local file system operation for file listing
            locData->setRemoteFileList(msg);
        }
        else if (subtype == TABLET_SET_DRIVES)
        {
            //Call local file system operation for file listing
            locData->setRemoteDrives(msg);
        }
        else if (subtype == TABLET_FB_FILE_SEL)
        {
            locData->setRemoteFile(msg);
        }
        else
        {
            cerr << "Unknown type!" << endl;
        }
    }
    case COVISE_MESSAGE_VRBC_SEND_SESSIONS:
    {
        int size;
        vrb::SessionID id;
        tb >> size;
        std::vector<vrb::SessionID> sessions;
        for (size_t i = 0; i < size; ++i)
        {
            tb >> id;
			if (id == getSessionID())
			{
                coVRPartnerList::instance()->setMaster(id.master());
            }
            sessions.push_back(id);
        }
        m_vrbMenu->updateSessions(sessions);
    }
    break;
    case COVISE_MESSAGE_VRBC_SET_SESSION:
    {
        int id;
        vrb::SessionID sessionID;
        tb >> id >> sessionID;
        if (id == me()->ID())
        {
            setSessionID(sessionID);
            m_vrbMenu->setCurrentSession(sessionID);
        }
        else
        {
            coVRPartnerList::instance()->setSessionID(id, sessionID);
            coVRPartnerList::instance()->print();
        }
        coVRCollaboration::instance()->updateSharedStates();
        coVRCollaboration::instance()->updateUi();
    }
    break;
    case COVISE_MESSAGE_VRBC_CHANGE_SESSION:
    {
        std::cerr << "received COVISE_MESSAGE_VRBC_CHANGE_SESSION from covise" << std::endl;
        while (!OpenCOVER::instance()->isVRBconnected())
        {
            std::cerr << "OpenCOVER waiting for VRB connection" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        while (me()->ID() == 0)
        {
            Message m;
            if (coVRMSController::instance()->isMaster())
            {
                OpenCOVER::instance()->vrbc()->wait(&m);
                coVRMSController::instance()->sendSlaves(&m);
            }
            else
            {
                coVRMSController::instance()->readMaster(&m);
            }

            assert(m.type != COVISE_MESSAGE_VRBC_CHANGE_SESSION);
            handleVRB(m);
        }

        vrb::SessionID sessionID;
        tb >> sessionID;
        coVRCollaboration::instance()->sessionChanged(sessionID.isPrivate());
        setSessionID(sessionID);
        coVRCollaboration::instance()->updateUi();
    }
    break;
    case COVISE_MESSAGE_VRB_SAVE_SESSION:
    {
        saveSessionFile(tb);
    }
    break;
    case COVISE_MESSAGE_VRB_LOAD_SESSION:
    {
        vrb::SessionID sessionID;
        tb >> sessionID;
        registry->resubscribe(sessionID);
        coVRCollaboration::instance()->updateSharedStates(true);
        coVRCollaboration::instance()->updateUi();
    }
    break;
    case COVISE_MESSAGE_VRB_MESSAGE:
    {

		if (msg.data.length() == 0)
		{
			fprintf(stderr, "empty message\n");
			return;
		}
        processVRBMessage(tb);
    }
    break;
    case COVISE_MESSAGE_VRB_REGISTRY_ENTRY_CHANGED:
    case COVISE_MESSAGE_VRB_REGISTRY_ENTRY_DELETED:
    {
        if (registry)
            registry->update(tb, msg.type);
    }
    break;
    default:

        break;
    }
}

void coVRCommunication::handleUdp(covise::UdpMessage* msg)
{
	TokenBuffer tb(msg);
	switch (msg->type)
	{
	case covise::EMPTY:
		break;
	case covise::AVATAR_HMD_POSITION:
	{
		std::string s;
		tb >> s;
		cerr << "received udp msg from client " << msg->sender << ": " << s << ""<< endl;
	}
		break;
	case covise::AVATAR_CONTROLLER_POSITION:
		break;
	default:
		coVRPluginList::instance()->UDPmessage(msg);
		break;
	}

}

void coVRCommunication::saveSessionFile(covise::TokenBuffer &tb)
{
        string fileName;
        TokenBuffer data;
        tb >> fileName;
        tb >> data;
        
        auto size = data.getData().length();
        cerr << "saving session " << fileName << " with data size " << size << endl;
        std::fstream out(fileName, std::ios_base::out | std::ios_base::binary);
        if (!out.is_open())
        {
            cerr << "error opening file " << fileName << endl;
        }
        out.write((char *)&size, sizeof(size));
        out.write(data.getData().data(), size);
}

void coVRCommunication::loadSessionFile(const std::string &fileName)
{
    int start = 0;
    const char* file_prefix_added_by_tablet_ui_but_not_from_cover = "file://";
    auto count = fileName.compare(0, sizeof(file_prefix_added_by_tablet_ui_but_not_from_cover) -1, file_prefix_added_by_tablet_ui_but_not_from_cover);
    if (!count)
    {
        start = sizeof(file_prefix_added_by_tablet_ui_but_not_from_cover) - 1;
    }
    TokenBuffer tb;
    tb << coVRCommunication::instance()->getID();
    tb << coVRCommunication::instance()->getUsedSessionID();    
    std::fstream in(fileName.substr(start), std::ios_base::in | std::ios_base::binary);
    int l = 0;
    in.read((char*)&l, sizeof(l));
    DataHandle dh{(size_t)l};
    in.read(dh.accessData(), l);
    tb << dh;
    send(tb, COVISE_MESSAGE_VRB_LOAD_SESSION);
}

int coVRCommunication::getNumberOfPartners()
{
    return coVRPartnerList::instance()->numberOfPartners();
}

Message *coVRCommunication::waitForMessage(int messageType)
{

    //todo: code for slaves
    Message *m = coVRPluginList::instance()->waitForVisMessage(messageType);
    if (!m)
    {
        m = new Message;
	int ret = 0;
        if (coVRMSController::instance()->isMaster())
        {
            ret = OpenCOVER::instance()->vrbc()->wait(m, messageType);
            coVRMSController::instance()->sendSlaves(&ret, sizeof(ret));
            if (ret != -1)
       		coVRMSController::instance()->sendSlaves(m);
        }
        else
        {
            coVRMSController::instance()->readMaster(&ret, sizeof(ret));
            if (ret != -1)
            	coVRMSController::instance()->readMaster(m);
        }
    }

    return m;
}



void opencover::coVRCommunication::addOnConnectCallback(std::function<void(void)> function)
{
	onConnectCallbacks.push_back(function);
}

void opencover::coVRCommunication::addOnDisconnectCallback(std::function<void(void)> function)
{
	onDisconnectCallbacks.push_back(function);
}

void opencover::coVRCommunication::addOnSessionChangedCallback(std::function<void(void)> function)
{
    onSessionChangedCallbacks.push_back(function);
}

void opencover::coVRCommunication::setWaitMessagesCallback(std::function<std::vector<Message*> (void)> cb)
{
	waitMessagesCallback = cb;
}

void opencover::coVRCommunication::setHandleMessageCallback(std::function<void(Message*)> cb)
{
	handleMessageCallback = cb;
}

std::vector<Message*> opencover::coVRCommunication::waitCoviseMessages()
{
	if (waitMessagesCallback)
	{
		return waitMessagesCallback();
	}
	return std::vector<Message*>();
}

void opencover::coVRCommunication::handleCoviseMessage(Message* m)
{
	if (handleMessageCallback)
	{
		handleMessageCallback(m);
	}
}

void opencover::coVRCommunication::initVrbFileMenu()
{
	m_vrbMenu->initFileMenu();
}

void coVRCommunication::setFBData(IData *data)
{
    VRBData *locData = dynamic_cast<VRBData *>(data);
    if (locData != NULL)
    {
        this->mfbData[locData->getId()] = locData;
    }
}
