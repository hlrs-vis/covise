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
#include "coVRCollaboration.h"
#include "VRAvatar.h"
#include "coVRSelectionManager.h"
#include "VRViewer.h"
#include "coTUIFileBrowser/VRBData.h"
#include "coVRCollaboration.h"
#include "ARToolKit.h"
#include "OpenCOVER.h"
#include "coVRAnimationManager.h"
#include "coVRConfig.h"
#include "coVRPluginList.h"

#include <PluginUtil/PluginMessageTypes.h>
#include <vrbclient/VrbClientRegistry.h>
#include <config/CoviseConfig.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif
#include <sys/stat.h>
#include <vrbclient/VrbClientRegistry.h>
#include <vrbclient/SharedStateManager.h>
#include <ui/SelectionList.h>

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
{
    assert(!s_instance);

    srand((unsigned)time(NULL)); // Initialize the random timer
    ignoreRemoteTransform = coCoviseConfig::isOn("COVER.IgnoreRemoteTransform", false);

    for (int i = 0; i < NUM_LOCKS; i++)
    {
        RILockArray[i] = -1;
    }
    currentFile = NULL;
    me = new coVRPartner();
    //randomID = (int)rand(); //nessesary??

    //me->setID(randomID);
    coVRPartnerList::instance()->append(me);
    registry = new VrbClientRegistry(me->getID(), vrbc);
    new SharedStateManager(registry);
}

coVRCommunication::~coVRCommunication()
{
    delete[] currentFile;
    delete registry;

    if (coVRPartnerList::instance()->find(me))
        coVRPartnerList::instance()->remove();

    s_instance = NULL;
}

void coVRCommunication::update(clientRegClass *theChangedClass)
{
    if (theChangedClass)
    {
        if (theChangedClass->getName() == "VRMLFile")
        {
            for (std::map<const std::string, std::shared_ptr<clientRegVar>>::iterator it = theChangedClass->getAllVariables().begin();
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
                    char * value;
                    it->second->getValue() >> value;
                    p->setFile(value);
                    cerr << theChangedClass->getName() << endl;
                    cerr << value << endl;
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

std::set<int> opencover::coVRCommunication::getSessions()
{
  return me->getSessions();
}

int opencover::coVRCommunication::getPublicSessionID()
{
    return me->getPublicSessionID();
}

int opencover::coVRCommunication::getPrivateSessionID()
{
    return me->getPrivateSessionID();
}

void opencover::coVRCommunication::setSessionID(int id)
{
    me->setSessionID(id);
    TokenBuffer tb;
    tb << id;
    tb << me->getID();
    if (vrbc)
    {
        vrbc->sendMessage(tb, COVISE_MESSAGE_VRBC_SET_SESSION);
    }
}

void coVRCommunication::RILock(int lockID)
{
    int myID = getID();
    //   cerr << "tryLOCK ID: " << lockID << " myID:" << myID << " RILockArray:"<< RILockArray[lockID] <<endl;
    if (RILockArray[lockID] < myID)
    {
        char num[500];
        sprintf(num, "%d;%d", lockID, myID);
        cover->sendBinMessage("LOCK", num, strlen(num) + 1);
        RILockArray[lockID] = myID;
        cerr << "LOCK ID: " << lockID << " myID:" << myID << endl;
    }
}

void coVRCommunication::RIUnLock(int lockID)
{
    int myID = getID();
    if (RILockArray[lockID] == myID)
    {
        char num[500];
        sprintf(num, "%d;%d", lockID, myID);
        cover->sendBinMessage("UNLOCK", num, strlen(num) + 1);
        RILockArray[lockID] = -1;
        cerr << "UNLOCK ID: " << lockID << " myID:" << myID << endl;
    }
}

void coVRCommunication::RIRemoteLock(int lockID, int remoteID)
{
    if (RILockArray[lockID] < remoteID)
    {
        RILockArray[lockID] = remoteID;
        cerr << "REMOTE_LOCK ID: " << lockID << " remoteID:" << remoteID << endl;
    }
}

void coVRCommunication::RIRemoteUnLock(int lockID, int remoteID)
{
    if (RILockArray[lockID] == remoteID)
    {
        RILockArray[lockID] = -1;
        cerr << "REMOTE_UNLOCK ID: " << lockID << " remoteID:" << remoteID << endl;
    }
}

bool coVRCommunication::isRILocked(int lockID)
{
    int myID = me->getID();
    if (myID < 0)
    {
        myID = randomID;
    }
    if ((RILockArray[lockID] >= 0) && (RILockArray[lockID] != myID))
        return true;
    return false;
}

bool coVRCommunication::isRILockedByMe(int lockID)
{
    int myID = me->getID();
    if (myID < 0)
    {
        myID = randomID;
    }
    //cerr << "tryIsLOCKedByMe ID: " << lockID << " myID:" << myID << " RILockArray:"<< RILockArray[lockID] <<endl;
    if ((RILockArray[lockID] >= 0) && (RILockArray[lockID] == myID))
        return true;
    return false;
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
    if (coVRPartnerList::instance()->num() > 1)
        return true;
    if (OpenCOVER::instance()->visPlugin())
        return true;
    return false;
}

bool coVRCommunication::isMaster() // returns true, if we are master
{
    if (coVRPartnerList::instance()->num() > 1)
    {
        return me->isMaster();
    }
    return true;
}

void coVRCommunication::processRenderMessage(const char *key, const char *tmp)
{
    if (strcmp(key, "MASTER") == 0)
    {
        coVRPartnerList::instance()->reset();
        while (coVRPartner *p2 = coVRPartnerList::instance()->current())
        {
            p2->setMaster(false);
            coVRPartnerList::instance()->next();
        }
        me->setMaster(true);
        coVRCollaboration::instance()->updateSharedStates();
    }
    else if (strcmp(key, "SLAVE") == 0)
    {
        coVRPartnerList::instance()->reset();
        while (coVRPartner *p2 = coVRPartnerList::instance()->current())
        {
            p2->setMaster(false);
            coVRPartnerList::instance()->next();
        }
        coVRCollaboration::instance()->updateSharedStates();
    }
    else if (!strcmp(key, "TRANSFORM_ALL"))
    {

        if (isRILockedByMe(TRANSFORM) || ignoreRemoteTransform)
            return;
        osg::Matrixd mat;
        mat(0, 3) = 0;
        mat(1, 3) = 0;
        mat(2, 3) = 0;
        mat(3, 3) = 1;
        if (sscanf(tmp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &mat(0, 0), &mat(0, 1), &mat(0, 2),
                   &mat(1, 0), &mat(1, 1), &mat(1, 2),
                   &mat(2, 0), &mat(2, 1), &mat(2, 2),
                   &mat(3, 0), &mat(3, 1), &mat(3, 2)) != 12)
        {
            cerr << "coVRCommunication::processRenderMessage: sscanf1 failed" << endl;
        }

        coVRCollaboration::instance()->remoteTransform(mat);
    }
    else if (!strcmp(key, "SCALE_ALL"))
    {
        float d;
        if (sscanf(tmp, "%f", &d) != 1)
        {
            cerr << "coVRCommunication::processRenderMessage: sscanf2 failed" << endl;
        }
        if (isRILockedByMe(SCALE) || ignoreRemoteTransform)
            return;

        coVRCollaboration::instance()->remoteScale(d);
    }
    else if (!(strcmp(key, "TIMESTEP")))
    {
        int ts;
        if (sscanf(tmp, "%d", &ts) != 1)
        {
            cerr << "coVRCommunication::processRenderMessage: sscanf ts failed" << endl;
        }
        coVRAnimationManager::instance()->setRemoteAnimationFrame(ts);
    }
    else if (!(strcmp(key, "TIMESTEP_ANIMATE")))
    {
        int ts;
        if (sscanf(tmp, "%d", &ts) != 1)
        {
            cerr << "coVRCommunication::processRenderMessage: sscanf ts failed" << endl;
        }
        coVRAnimationManager::instance()->setRemoteAnimate(ts == 1);
    }
    else if (!(strcmp(key, "TIMESTEP_SYNCRONIZE")))
    {
        int ts;
        if (sscanf(tmp, "%d", &ts) != 1)
        {
            cerr << "coVRCommunication::processRenderMessage: sscanf ts failed" << endl;
        }
        coVRAnimationManager::instance()->setRemoteSynchronize(ts == 1);
    }
    else if (!(strcmp(key, "SYNC_MODE")))
    {
        coVRCollaboration::instance()->setSyncMode(tmp);
        coVRCollaboration::instance()->updateSharedStates();
    }
    else if (!(strcmp(key, "AvatarX")))
    {
        VRAvatarList::instance()->receiveMessage(tmp);
    }
    else if (!(strcmp(key, "MOVE_HAND")))
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
    else if (!(strcmp(key, "MOVE_HEAD")))
    {
        osg::Matrixd mat;
        if (sscanf(tmp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &mat(0, 0), &mat(0, 1), &mat(0, 2), &mat(0, 3),
                   &mat(1, 0), &mat(1, 1), &mat(1, 2), &mat(1, 3),
                   &mat(2, 0), &mat(2, 1), &mat(2, 2), &mat(2, 3),
                   &mat(3, 0), &mat(3, 1), &mat(3, 2), &mat(3, 3)) != 16)
        {
            cerr << "coVRCommunication::processRenderMessage: sscanf3 failed" << endl;
        }
        VRViewer::instance()->updateViewerMat(mat);
    }
    else if (!(strcmp(key, "LOCK")))
    {
        int lockID = 0, myID = -1;
        if (sscanf(tmp, "%d;%d", &lockID, &myID) != 2)
        {
            cerr << "coVRCommunication::processRenderMessage: sscanf4 failed" << endl;
        }
        RIRemoteLock(lockID, myID);
    }
    else if (!(strcmp(key, "UNLOCK")))
    {
        int lockID = 0, myID = -1;
        if (sscanf(tmp, "%d;%d", &lockID, &myID) != 2)
        {
            cerr << "coVRCommunication::processRenderMessage: sscanf5 failed" << endl;
        }
        RIRemoteUnLock(lockID, myID);
    }
    else if (!(strcmp(key, "AR_VIDEO_FRAME")) && ARToolKit::instance()->remoteAR)
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
    else if (!(strcmp(key, "ADD_SELECTION")))
    {
        coVRSelectionManager::instance()->receiveAdd(tmp);
    }
    else if (!(strcmp(key, "CLEAR_SELECTION")))
    {
        coVRSelectionManager::instance()->receiveClear();
    }
    
}

void coVRCommunication::becomeMaster()
{
    coVRPluginList::instance()->becomeCollaborativeMaster();
    me->becomeMaster();
}

void coVRCommunication::handleVRB(Message *msg)
{
    //fprintf(stderr,"slave: %d msgProcessed: %s\n",coVRMSController::instance()->isSlave(),covise_msg_types_array[msg->type]);
    if (registry->getVrbc() != vrbc)
    {
        registry->setVrbc(vrbc);
    }
    if (vrbc == NULL)
    {
        vrbc = new VRBClient("COVER", coVRConfig::instance()->collaborativeOptionsFile.c_str(), coVRMSController::instance()->isSlave());
        registry->setVrbc(vrbc);
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
                coVRPartnerList::instance()->append(p);
            }
            if (p->getID() != me->getID())
                p->setInfo(tb);
            p->updateUi();
        }
        coVRPartnerList::instance()->reset();
        bool haveMaster = false;
        int minID = 10000000;
        while (coVRPartner *p = coVRPartnerList::instance()->current())
        {
            if (p->isMaster())
            {
                haveMaster = true;
                break;
            }
            if (p->getID() < minID)
                minID = p->getID();
            coVRPartnerList::instance()->next();
        }
        if (!haveMaster && coVRPartnerList::instance()->num() > 1) // no master, check if we have the lowest ID, then become Master
        {
            if (me->getID() == minID)
            {
                TokenBuffer rtb;
                rtb << 1;
                Message m(rtb);
                m.type = COVISE_MESSAGE_VRB_SET_MASTER;
                if (vrbc)
                    vrbc->sendMessage(&m);
            }
        }
        coVRPartnerList::instance()->print();
        if (coVRPartnerList::instance()->num() > 1)
            coVRCollaboration::instance()->showCollaborative(true);
        //request a new private session if we dont have one
        if (me->getPrivateSessionID() == 0)
        {
            bool isPrivate = true;
            TokenBuffer rns;
            rns << me->getID();
            rns << me->getPrivateSessionID();
            rns << isPrivate;
            Message rns_m(rns);
            rns_m.type = COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION;
            if (vrbc)
            {
                vrbc->sendMessage(&rns_m);
            }
        }
    }
    break;
    case COVISE_MESSAGE_VRB_SET_GROUP:
    {
        int id;
        tb >> id;
        coVRPartner *p = coVRPartnerList::instance()->get(id);
        if (p)
        {
            int group;
            tb >> group;
            p->setGroup(group);
        }
        coVRPartnerList::instance()->print();
    }
    break;
    case COVISE_MESSAGE_VRB_GET_ID:
    {
        int id, session;
        tb >> id;
        tb >> session;
        me->setID(id);
        me->setSessionID(session);
        registry->setID(id, session);
        if (vrbc)
        {
            vrbc->setID(id);

            me->sendHello();
        }
        registry->subscribeClass(me->getPrivateSessionID(), "VRMLFile", this);
        registry->subscribeClass(me->getPrivateSessionID(), "COVERPlugins", this);
        if (currentFile)
        {
            TokenBuffer tb;
            tb << currentFile;
            registry->setVar(0, "VRMLFile", std::to_string(me->getID()), std::move(tb));
        }
        if (me->getPrivateSessionID() == 0)
        {
            bool isPrivate = true;
            TokenBuffer rns;
            rns << me->getID();
            rns << me->getPrivateSessionID();
            rns << isPrivate;
            Message rns_m(rns);
            rns_m.type = COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION;
            if (vrbc)
            {
                vrbc->sendMessage(&rns_m);
            }
        }
    }
    break;
    case COVISE_MESSAGE_VRB_SET_MASTER:
    {
        int id;
        tb >> id;
        coVRPartner *p = coVRPartnerList::instance()->get(id);
        if (p)
        {
            int masterState;
            tb >> masterState;
            p->setMaster(masterState);
        }

        coVRPartnerList::instance()->reset();
        while (coVRPartner *p2 = coVRPartnerList::instance()->current())
        {
            if (p2 != p)
                p2->setMaster(0);
            coVRPartnerList::instance()->next();
        }
        coVRPartnerList::instance()->print();
        coVRCollaboration::instance()->updateSharedStates();
    }
    break;
    case COVISE_MESSAGE_VRB_QUIT:
    {
        int id;
        tb >> id;
        coVRPartner *p = coVRPartnerList::instance()->get(id);
        if (p)
        {
            if (p != me)
                coVRPartnerList::instance()->remove();
        }
        if (coVRPartnerList::instance()->num() <= 1)
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
        // VRCoviseConnection::covconn = new VRCoviseConnection(argc, argv);
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
        coVRPluginList::instance()->forwardMessage(msg->length, msg->data);
    }
    break;
    case COVISE_MESSAGE_RENDER:
    {
        if (msg->data[0] != 0)
        {
            std::string data(msg->data);
            std::vector<std::string> tokens = split(data, '\n');
            processRenderMessage(tokens[0].c_str(), tokens[1].c_str());
        }
        else
        {
            processRenderMessage(&msg->data[1], &msg->data[strlen(&msg->data[1]) + 2]);
        }
    }
    break;

    case COVISE_MESSAGE_VRB_CLOSE_VRB_CONNECTION:
    {
        cerr << "VRB requests to quit" << endl;
        coVRPartnerList::instance()->reset();
        while (coVRPartnerList::instance()->num() > 1)
        {
            if (coVRPartnerList::instance()->current() == me)
                coVRPartnerList::instance()->next();
            coVRPartnerList::instance()->remove();
        }
        coVRCollaboration::instance()->showCollaborative(false);
        delete vrbc;
        vrbc = new VRBClient("COVER", coVRConfig::instance()->collaborativeOptionsFile.c_str(), coVRMSController::instance()->isSlave());
        registry->setVrbc(vrbc);
    }
        break;
    case COVISE_MESSAGE_VRB_REQUEST_FILE:
    {
        struct stat statbuf;
        char *filename;
        tb >> filename;
        int requestorsID;
        tb >> requestorsID;
        if (stat(filename, &statbuf) >= 0)
        {
            TokenBuffer rtb;
            rtb << requestorsID;
#ifdef _WIN32
            int fdesc = open(filename, O_RDONLY | O_BINARY);
#else
            int fdesc = open(filename, O_RDONLY);
#endif
            if (fdesc > 0)
            {
                rtb << (int)statbuf.st_size;
                char *buf = (char *)rtb.allocBinary(statbuf.st_size);
                int n = read(fdesc, buf, statbuf.st_size);
                if (n == -1)
                {
                    cerr << "coVRCommunication::handleVRB: read failed: " << strerror(errno) << endl;
                }
                close(fdesc);
            }
            else
            {
                cerr << " file access error, could not open " << filename << endl;
                rtb << 0;
            }

            Message m(rtb);
            m.type = COVISE_MESSAGE_VRB_SEND_FILE;
            if (vrbc)
                vrbc->sendMessage(&m);
        }
        else
        {
            TokenBuffer rtb;
            rtb << requestorsID;
            rtb << 0;
            Message m(rtb);
            m.type = COVISE_MESSAGE_VRB_SEND_FILE;
            if (vrbc)
                vrbc->sendMessage(&m);
        }
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
            p->setInfo(tb);
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

    case COVISE_MESSAGE_SOCKET_CLOSED:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    {
        cerr << "VRB left" << endl;
        coVRPartnerList::instance()->reset();
        while (coVRPartnerList::instance()->num() > 1)
        {
            if (coVRPartnerList::instance()->current() == me)
                coVRPartnerList::instance()->next();
            coVRPartnerList::instance()->remove();
        }
        coVRCollaboration::instance()->showCollaborative(false);
        delete vrbc;
        vrbc = new VRBClient("COVER", coVRConfig::instance()->collaborativeOptionsFile.c_str(), coVRMSController::instance()->isSlave());
        registry->setVrbc(vrbc);
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
        int id;
        tb >> size;
        std::set<int> sessions;
        for (size_t i = 0; i < size; ++i)
        {
            tb >> id;
            sessions.insert(id);
        }
        me->setSessions(sessions);
        coVRCollaboration::instance()->updateSessionSelectionList(sessions);
    }
    break;
    case COVISE_MESSAGE_VRBC_SET_SESSION:
    {
        
        int sessionID;
        tb >> sessionID;
        me->setSessionID(sessionID);
        
        TokenBuffer buf;
        buf << sessionID;
        buf << getID();
        if (vrbc)
        {
            vrbc->sendMessage(buf, COVISE_MESSAGE_VRBC_SET_SESSION);
        }
    }
    break;
    default:
        if (registry)
            registry->update(tb, msg->type);
        break;
    }
}

void coVRCommunication::setCurrentFile(const char *filename)
{
    if (!filename)
        return;
    me->setFile(filename);
    TokenBuffer tb;
    tb << filename;
    registry->setVar(0, "VRMLFile", std::to_string(me->getID()), std::move(tb));

    if (currentFile)
    {
        if (strcmp(currentFile, filename) == 0)
        {
            return;
        }
    }
    delete[] currentFile;
    currentFile = new char[strlen(filename) + 1];
    strcpy(currentFile, filename);
    TokenBuffer rtb3;
    rtb3 << me->getID();
    rtb3 << (char *)filename;
    Message m(rtb3);
    m.type = COVISE_MESSAGE_VRB_CURRENT_FILE;
    if (vrbc)
        vrbc->sendMessage(&m);

    if (coVRPluginList::instance()->getPlugin("ACInterface"))
    {
        TokenBuffer tb;
        tb << filename;
        cover->sendMessage(NULL, "ACInterface", PluginMessageTypes::HLRS_ACInterfaceModelLoadedPath, tb.get_length(), tb.get_data());
        tb.delete_data();
    }
}

int coVRCommunication::getNumberOfPartners()
{
    return coVRPartnerList::instance()->num();
}

Message *coVRCommunication::waitForMessage(int messageType)
{

    //todo: code for slaves
    Message *m = coVRPluginList::instance()->waitForVisMessage(messageType);
    if (!m)
    {
        m = new Message;
        vrbc->wait(m, messageType);
    }

    return m;
}

void coVRCommunication::setFBData(IData *data)
{
    VRBData *locData = dynamic_cast<VRBData *>(data);
    if (locData != NULL)
    {
        this->mfbData[locData->getId()] = locData;
    }
}
