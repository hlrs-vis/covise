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

#include <util/common.h>
#include <util/string_util.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <map>
#include "coVRMSController.h"
#include "coVRPluginSupport.h"
#include "coVRFileManager.h"
#include <vrbclient/VRBClient.h>
#include <vrbclient/VRBMessage.h>
#include "coVRCommunication.h"
#include "coVRPartner.h"
#include <util/coTabletUIMessages.h>
#include "coHud.h"
#include "coVRTui.h"
#include <net/covise_host.h>
#include <net/message_types.h>
#include <net/udpMessage.h>
#include <net/udp_message_types.h>
#include "coVRCollaboration.h"
#include "VRAvatar.h"
#include "coVRSelectionManager.h"
#include "VRViewer.h"
#include "coTUIFileBrowser/VRBData.h"
#include "ARToolKit.h"
#include "OpenCOVER.h"
#include "coVRAnimationManager.h"
#include "coVRConfig.h"
#include "coVRPluginList.h"

#include <PluginUtil/PluginMessageTypes.h>
#include <vrbclient/VrbClientRegistry.h>
#include <config/CoviseConfig.h>
#include <OpenVRUI/coNavInteraction.h>
#include "ui/Owner.h"
#include "ui/Action.h"
#include "ui/Button.h"
#include "ui/Group.h"
#include "ui/FileBrowser.h"
#include "ui/Menu.h"
#include "ui/SelectionList.h"

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif
#include <sys/stat.h>
#include <vrbclient/VrbClientRegistry.h>
#include <vrbclient/SharedStateManager.h>
#include <vrbclient/SessionID.h>
#include <ui/SelectionList.h>
#include "coVrbMenue.h"
#include <vrbclient/VRBMessage.h>

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



    registry.reset(new VrbClientRegistry(-1));
	sharedStateManager.reset(new SharedStateManager(registry.get()));
}
void coVRCommunication::init()
{
	me = new coVRPartner();

	coVRPartnerList::instance()->addPartner(me);
	registry->setID(me->getID(), me->getSessionID());
	registry->registerSender(cover->getSender());
	m_vrbMenue.reset(new VrbMenue());
	remoteNavInteraction = new vrui::coNavInteraction(vrui::coInteraction::NoButton, "remoteNavInteraction");
}
void opencover::coVRCommunication::connected()
{
	for (auto function : onConnectCallbacks)
	{
		if (function)
		{
			function();
		}
	}
}

void opencover::coVRCommunication::disconnected()
{
	for (auto function : onDisconnectCallbacks)
	{
		if (function)
		{
			function();
		}
	}
}

coVRCommunication::~coVRCommunication()
{
    coVRPartnerList::instance()->deletePartner(me->getID());
	delete remoteNavInteraction;
    s_instance = NULL;
}

void coVRCommunication::update(clientRegClass *theChangedClass)
{
    if (theChangedClass)
    {
        if (theChangedClass->getName() == "VRMLFile")
        {
            for (std::map<const std::string, std::shared_ptr<regVar>>::iterator it = theChangedClass->getAllVariables().begin();
                it != theChangedClass->getAllVariables().end(); ++it)
            {
                coVRPartner *p = NULL;
                int remoteID = -2;
                if (sscanf(it->first.c_str(), "%d", &remoteID) != 1)
                {
                    cerr << "coVRCommunication::update: sscanf failed" << endl;
                }
                if ((p = coVRPartnerList::instance()->get(remoteID)))
                {
                    p->setFile(it->second->getValue().data());
                    cerr << theChangedClass->getName() << endl;
                    cerr << it->second->getValue().data() << endl;
                }
            }
        }
    }
}

int coVRCommunication::getID()
{
    int myID = me->getID();
    if (myID < 0)
    {
        myID = randomID;
    }
    return myID;
}

vrb::SessionID & opencover::coVRCommunication::getPrivateSessionIDx()
{
    return m_privateSessionID;
}

const vrb::SessionID &opencover::coVRCommunication::getSessionID() const
{
    return me->getSessionID();
}

void opencover::coVRCommunication::setSessionID(const vrb::SessionID &id)
{
    if (id.isPrivate())
    {
        m_privateSessionID = id;
    }
    coVRPartnerList::instance()->setMaster(-1);
    coVRPartnerList::instance()->setSessionID(me->getID(), id);
    TokenBuffer tb;
    tb << id;
    tb << me->getID();
    sendMessage(tb, COVISE_MESSAGE_VRBC_SET_SESSION);
}

const char *coVRCommunication::getHostaddress()
{
    static char *hostaddr = NULL;
    if (!hostaddr)
    {
        Host host;
#define MAX_LENGTH_HOSTADDR 500
        hostaddr = new char[MAX_LENGTH_HOSTADDR];
        if (host.getAddress())
            strncpy(hostaddr, host.getAddress(), MAX_LENGTH_HOSTADDR);
        else
            strcpy(hostaddr, "unknown address");
        hostaddr[MAX_LENGTH_HOSTADDR - 1] = '\0';
    }

    return hostaddr;
}

std::string coVRCommunication::getUsername()
{
    std::string name("noname");
    if (auto val = getenv("USER"))
    {
        name = val;
    }
    else if (auto val = getenv("LOGNAME"))
    {
        name = val;
    }

    return name;
}

const char *coVRCommunication::getHostname()
{
    static char *hostname = NULL;
    if (!hostname)
    {
        Host host;
#define MAX_LENGTH_HOSTNAME 500
        hostname = new char[MAX_LENGTH_HOSTNAME];
        if (host.getAddress())
            strncpy(hostname, host.getName(), MAX_LENGTH_HOSTNAME);
        else
            strcpy(hostname, "unknown");
        hostname[MAX_LENGTH_HOSTNAME - 1] = '\0';
    }

    return hostname;
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
        return me->isMaster();
    }
    return true;
}
void coVRCommunication::processRenderMessage(const char *key, const char *tmp)
{
    if (!(strcmp(key, "AR_VIDEO_FRAME")) && ARToolKit::instance()->remoteAR)
    {
        ARToolKit::instance()->remoteAR->receiveImage(tmp);
    }
    else if (!(strcmp(key, "SYNC_KEYBOARD")))
    {
        fprintf(stderr, "Slave receiving SYNC_KEYBOARD msg=[%s]\n", tmp);
        int type, state, code;
        if (sscanf(tmp, "%d %d %d", &type, &state, &code) != 3)
        {
            cerr << "coVRCommunication::processRenderMessage: sscanf failed" << endl;
        }
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
    else 
    {
    cerr << " Render Message with key " << key << " not known, consider adressing new processVRBMessage function:" << endl;
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
    case vrb::TIMESTEP:
    {
        int ts;
        tb >> ts;
        coVRAnimationManager::instance()->setRemoteAnimationFrame(ts);
    }
        break;
    case vrb::TIMESTEP_ANIMATE:
    {
        bool anumRunning;
        tb >> anumRunning;
        coVRAnimationManager::instance()->setRemoteAnimate(anumRunning);
    }
        break;
    case vrb::TIMESTEP_SYNCRONIZE:
    {
        int ts;
        tb >> ts;
        coVRAnimationManager::instance()->setRemoteSynchronize(ts == 1);
    }
        break;
    case vrb::SYNC_MODE:
    {
        bool showAvatar;
        tb >> showAvatar;
        if (showAvatar)
        {
            coVRPartnerList::instance()->showAvatars();
        }
        else
        {
            coVRPartnerList::instance()->hideAvatars();
        }
    }
        break;
    case vrb::MASTER:
    {
        coVRPartnerList::instance()->setMaster(me->getID());
        coVRCollaboration::instance()->updateSharedStates();
    }
        break;
    case vrb::SLAVE:
    {
        coVRPartnerList::instance()->setMaster(-1); //nobody is master here?
        coVRCollaboration::instance()->updateSharedStates();
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
        vrb::deserialize(tb, mat);
        VRViewer::instance()->updateViewerMat(mat);
    }
        break;
    case vrb::AR_VIDEO_FRAME:
    {
        char * tmp;
        tb >> tmp;
        ARToolKit::instance()->remoteAR->receiveImage(tmp);
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
    coVRPartnerList::instance()->setMaster(me->getID());
    me->becomeMaster();
}

void coVRCommunication::handleVRB(Message *msg)
{
	//fprintf(stderr,"slave: %d msgProcessed: %s\n",coVRMSController::instance()->isSlave(),covise_msg_types_array[msg->type]);

    if (vrbc == NULL && !cover->connectedToCovise()) 
	{
        vrbc = new VRBClient("COVER", coVRConfig::instance()->collaborativeOptionsFile.c_str(), coVRMSController::instance()->isSlave());
    }
    TokenBuffer tb(msg);
    switch (msg->type)
    {
    case COVISE_MESSAGE_VRB_SET_USERINFO:
    {
        int num;
        tb >> num;
        for (int i = 0; i < num; i++)
        {
            int id;
            tb >> id;
            coVRPartner *p = coVRPartnerList::instance()->get(id);
            if (!p)
            {
                p = new coVRPartner(id);
                coVRPartnerList::instance()->addPartner(p);
            }
            p->setInfo(tb);
            p->updateUi();
        }
        if (!coVRPartnerList::instance()->getMaster() && coVRPartnerList::instance()->numberOfPartners() > 1) // no master, check if we have the lowest ID, then become Master
        {
            if (me == coVRPartnerList::instance()->getFirstPartner())
            {
                TokenBuffer rtb;
                rtb << true;
                sendMessage(rtb, COVISE_MESSAGE_VRB_SET_MASTER);
            }
        }
        coVRPartnerList::instance()->print();

        //request a new private session if we dont have one
        if (m_privateSessionID.owner() == 0)
        {
            bool isPrivate = true;
            TokenBuffer rns;
            rns << m_privateSessionID;
            sendMessage(rns, COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION);
        }

    }
    break;
    case COVISE_MESSAGE_VRB_SET_GROUP:
    {
        int id;
        tb >> id;
        vrb::SessionID session;
        tb >> session;
        coVRPartnerList::instance()->setSessionID(id, session);
        coVRPartnerList::instance()->print();
    }
    break;
    case COVISE_MESSAGE_VRB_GET_ID:
    {
        int id;
        vrb::SessionID session;
        tb >> id;
        tb >> session;
        me = me->setID(id);
		connected();
        if (session.isPrivate())
        {
            m_privateSessionID = session;
            registry->setID(id, session);
            coVRCollaboration::instance()->updateSharedStates();
        }
        else
        {
            coVRPartnerList::instance()->setSessionID(id, session);
        }
        if (vrbc)
        {
            vrbc->setID(id);
        }
		me->sendHello();

        if (m_privateSessionID.owner() == 0)
        {
            TokenBuffer rns;
            rns << vrb::SessionID(me->getID());
            sendMessage(rns, COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION);
        }
        m_vrbMenue->updateState(true);
        //prepare for saving and loading sessions on the vrb
        TokenBuffer ftb;
        ftb << id;
        sendMessage(ftb, COVISE_MESSAGE_VRB_REQUEST_SAVED_SESSION);
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
    }
    break;
    case COVISE_MESSAGE_VRB_QUIT:
    {
        int id;
        tb >> id;
        if (id != me->getID())
        {
             coVRPartnerList::instance()->deletePartner(id);
			 vrui::coInteractionManager::the()->resetLocks(id);
        }
        if (coVRPartnerList::instance()->numberOfPartners() <= 1)
            coVRCollaboration::instance()->showCollaborative(false);
    }
    break;
    case COVISE_MESSAGE_VRB_CONNECT_TO_COVISE:
    {
        char *ip;
        tb >> ip;
        int argc;
        tb >> argc;
        char **argv = new char *[argc];
        for (int i = 0; i < argc; i++)
        {
            tb >> argv[i];
        }
       //VRCoviseConnectio::covconn = new VRCoviseConnection(argc, argv);
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
            char *fileName;
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
        coVRPluginList::instance()->forwardMessage(msg->data);
    }
    break;
    case COVISE_MESSAGE_RENDER:
    {
        if (msg->data.data()[0] != 0)
        {
            std::string data(msg->data.data());
            std::vector<std::string> tokens = split(data, '\n');
            processRenderMessage(tokens[0].c_str(), tokens[1].c_str());
        }
        else
        {
            processRenderMessage(&msg->data.data()[1], &msg->data.data()[strlen(&msg->data.data()[1]) + 2]);
        }
    }
    break;
    case COVISE_MESSAGE_SOCKET_CLOSED:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    case COVISE_MESSAGE_VRB_CLOSE_VRB_CONNECTION:
    {
        cerr << "VRB requests to quit " << msg->type<< endl;
		disconnected();
        coVRPartnerList::instance()->deleteOthers();
        m_vrbMenue->updateState(false);
        me->setSession(vrb::SessionID());
        m_privateSessionID = vrb::SessionID();
        delete vrbc;
        vrbc = new VRBClient("COVER", coVRConfig::instance()->collaborativeOptionsFile.c_str(), coVRMSController::instance()->isSlave());
        coVRCollaboration::instance()->updateSharedStates();

    }
        break;
    case COVISE_MESSAGE_VRB_REQUEST_FILE:
    {
        coVRFileManager::instance()->sendFile(tb);
    }
    break;
    case COVISE_MESSAGE_VRB_CURRENT_FILE:
    {
        char *filename;
        int remoteID;
        tb >> remoteID;
        tb >> filename;
        if (!filename)
            break;
        coVRPartner *p = coVRPartnerList::instance()->get(remoteID);
        if (p)
        {
            //p->setInfo(tb);
            p->setFile(filename);
        }
        /*if(currentFile)
         {
            if(strcmp(currentFile,filename)==0)
            {
               break;
            }
         }
         delete[] currentFile;
         currentFile = new char[strlen(filename)+1];
         strcpy(currentFile,filename);
         cerr << "Loading remote file "<< filename << endl;
         cover->loadFile(filename);*/
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
            locData->setDirectoryList(*msg);
        }
        else if (subtype == TABLET_SET_FILELIST)
        {
            locData->setFileList(*msg);
        }
        else if (subtype == TABLET_SET_CURDIR)
        {
            locData->setCurDir(*msg);
        }
        else if (subtype == TABLET_SET_CLIENTS)
        {
            locData->setClientList(*msg);
        }
        else if (subtype == TABLET_SET_DRIVES)
        {
            locData->setDrives(*msg);
        }
        else if (subtype == TABLET_SET_FILE)
        {
            locData->setFile(*msg);
        }
        else if (subtype == TABLET_SET_GLOBALLOAD)
        {
            // Enable loading here

            //Retrieve Data object
            char *curl = NULL;

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
            locData->setRemoteDirList(*msg);
        }
        else if (subtype == TABLET_SET_FILELIST)
        {
            //Call local file system operation for file listing
            locData->setRemoteFileList(*msg);
        }
        else if (subtype == TABLET_SET_DRIVES)
        {
            //Call local file system operation for file listing
            locData->setRemoteDrives(*msg);
        }
        else if (subtype == TABLET_FB_FILE_SEL)
        {
            locData->setRemoteFile(*msg);
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
				getSessionID().setOwner(id.owner());
			}
            sessions.push_back(id);
        }
        m_vrbMenue->updateSessions(sessions);
    }
    break;
    case COVISE_MESSAGE_VRBC_SET_SESSION:
    {
        
        vrb::SessionID sessionID;
        tb >> sessionID;
        setSessionID(sessionID);
        m_vrbMenue->setCurrentSession(sessionID);

    }
    break;
    case COVISE_MESSAGE_VRB_REQUEST_SAVED_SESSION:
    {
        int size;
        tb >> size;
        std::vector<std::string> registries;
        for (size_t i = 0; i < size; i++)
        {
            std::string file;
            tb >> file;
            registries.push_back(file);
        }
        m_vrbMenue->updateRegistries(registries);
    }
    break;
    case COVISE_MESSAGE_VRB_LOAD_SESSION:
    {
        vrb::SessionID sessionID;
        tb >> sessionID;
        registry->resubscribe(sessionID);
        coVRCollaboration::instance()->updateSharedStates(true);
    }
    break;
    case COVISE_MESSAGE_VRB_MESSAGE:
    {

		if (msg->data.length() == 0)
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
            registry->update(tb, msg->type);
    }
    break;
    default:

        break;
    }
}

void coVRCommunication::handleUdp(vrb::UdpMessage* msg)
{
	TokenBuffer tb(msg);
	switch (msg->type)
	{
	case vrb::EMPTY:
		break;
	case vrb::AVATAR_HMD_POSITION:
	{
		std::string s;
		tb >> s;
		cerr << "received udp msg from client " << msg->sender << ": " << s << ""<< endl;
	}
		break;
	case vrb::AVATAR_CONTROLLER_POSITION:
		break;
	default:
		coVRPluginList::instance()->UDPmessage(msg->type, msg->data);
		break;
	}

}
void coVRCommunication::setCurrentFile(const char *filename)
{
	assert(true);
	//if (!filename)
    //    return;
    //me->setFile(filename);
    //TokenBuffer tb;
    //tb << filename;
    //registry->setVar(0, "VRMLFile", std::to_string(me->getID()), std::move(tb));

    //if (currentFile)
    //{
    //    if (strcmp(currentFile, filename) == 0)
    //    {
    //        return;
    //    }
    //}
    //delete[] currentFile;
    //currentFile = new char[strlen(filename) + 1];
    //strcpy(currentFile, filename);
    //TokenBuffer rtb3;
    //rtb3 << me->getID();
    //rtb3 << (char *)filename;
    //sendMessage(rtb3, COVISE_MESSAGE_VRB_CURRENT_FILE);

    //if (coVRPluginList::instance()->getPlugin("ACInterface"))
    //{
    //    TokenBuffer tb;
    //    tb << filename;
    //    cover->sendMessage(NULL, "ACInterface", PluginMessageTypes::HLRS_ACInterfaceModelLoadedPath, tb.get_length(), tb.get_data());
    //    tb.delete_data();
    //}
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
            ret = vrbc->wait(m, messageType);
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

bool opencover::coVRCommunication::sendMessage(Message * msg)
{
    return cover->sendVrbMessage(msg);
}

bool opencover::coVRCommunication::sendMessage(TokenBuffer & tb, covise_msg_type type)
{
    Message msg(tb);
    msg.type = type;
    return sendMessage(&msg);
}

void opencover::coVRCommunication::addOnConnectCallback(std::function<void(void)> function)
{
	onConnectCallbacks.push_back(function);
}

void opencover::coVRCommunication::addOnDisconnectCallback(std::function<void(void)> function)
{
	onDisconnectCallbacks.push_back(function);
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

void opencover::coVRCommunication::initVrbFileMenue()
{
	m_vrbMenue->initFileMenue();
}

void coVRCommunication::setFBData(IData *data)
{
    VRBData *locData = dynamic_cast<VRBData *>(data);
    if (locData != NULL)
    {
        this->mfbData[locData->getId()] = locData;
    }
}


