/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *                                                                      *
 *                                                                      *
 *                            (C) 1996-200                              *
 *              Computer Centre University of Stuttgart                 *
 *                         Allmandring 30                               *
 *                       D-70550 Stuttgart                              *
 *                            Germany                                   *
 *									*
 *									*
 *	File            coVRCollaboration.C (Performer 2.0)		        *
 *									*
 *	Description     scene graph class                               *
 *                                                                      *
 *	Author          D. Rainer				        *
 *                 F. Foehl                                             *
 *                 U. Woessner                                          *
 *                                                                      *
 *	Date            20.08.97                                        *
 *                 10.07.98 Performer C++ Interface                     *
 *                 20.11.00 Pinboard config through covise.config       *
 ************************************************************************/

#include <util/common.h>
#include <net/message.h>
#include <net/message_types.h>

#include <config/CoviseConfig.h>
#include <vrbclient/VrbClientRegistry.h>
#include "coVRNavigationManager.h"
#include "VRSceneGraph.h"
#include "coVRCollaboration.h"
#include "coVRPluginSupport.h"
#include "coVRMSController.h"
#include "coVRCommunication.h"
#include "VRAvatar.h"
#include <osg/MatrixTransform>
#include <vrbclient/SharedStateManager.h>
#include <vrbclient/VRBClient.h>
#include "OpenCOVER.h"

#include "ui/Menu.h"
#include "ui/Action.h"
#include "ui/Button.h"
#include "ui/SelectionList.h"
#include "ui/Slider.h"

#define SYNC_MODE_GROUP 2
#define NAVIGATION_GROUP 0

using namespace opencover;
using namespace vrb;
coVRCollaboration *coVRCollaboration::s_instance = NULL;

coVRCollaboration *coVRCollaboration::instance()
{
    if (!s_instance)
        s_instance = new coVRCollaboration;
    return s_instance;
}

coVRCollaboration::coVRCollaboration()
    : ui::Owner("coVRCollaboration", cover->ui)
    , syncXform(0)
    , syncScale(0)
    , syncInterval(0.3)
    , showAvatar(1)
    , syncMode("coVRCollaboration_syncMode", LooseCoupling, ALWAYS_SHARE)
    , avatarPosition("coVRCollaboration_avatarPosition", osg::Matrix())
    , scaleFactor("coVRCollaboration_scaleFactor", 1)
{
    assert(!s_instance);

    init();
}

void coVRCollaboration::init()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nnew coVRCollaboration\n");
    VRAvatarList::instance()->hide();
    syncMode.setUpdateFunction([this]()
    {
        assert(syncMode < 3);
        if (m_collaborationMode)
            m_collaborationMode->select(syncMode);
        syncModeChanged(syncMode);
    });
    avatarPosition.setUpdateFunction([this]()
    {
        osg::Matrix m = avatarPosition;
        remoteTransform(m);
    });
    scaleFactor.setUpdateFunction([this]()
    {
        remoteScale(scaleFactor);
    });
    readConfigFile();

    // create collaborative menu
    initCollMenu();
}

coVRCollaboration::~coVRCollaboration()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete coVRCollaboration\n");

    s_instance = NULL;
}

int coVRCollaboration::readConfigFile()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRCollaboration::readConfigFile\n");

    syncInterval = covise::coCoviseConfig::getFloat("interval", "COVER.Collaborative.Sync", 0.05);
    setSyncInterval();
    setSyncMode("LOOSE");
    std::string sMode = covise::coCoviseConfig::getEntry("mode", "COVER.Collaborative.Sync");
    if (!sMode.empty())
        setSyncMode(sMode.c_str());

    return 0;
}

void coVRCollaboration::initCollMenu()
{
    m_collaborativeMenu = new ui::Menu("CollaborativeOptions", this);
    m_collaborativeMenu->setText("Collaborate");

    m_collaborationMode = new ui::SelectionList(m_collaborativeMenu, "CollaborationMode");
    m_collaborationMode->setText("Collaboration mode");
    // keep in sync with enum SyncMode:
    assert(0 == LooseCoupling);
    assert(1 == MasterSlaveCoupling);
    assert(2 == TightCoupling);
    m_collaborationMode->setList(std::vector<std::string>({"Loose", "Master/Slave", "Tight"}));
    m_collaborationMode->setCallback([this](int mode) {
        syncMode = mode;
        syncModeChanged(mode);
        if (mode == MasterSlaveCoupling)
        {
            coVRCommunication::instance()->becomeMaster();
            m_master->setEnabled(false);
            m_returnToMaster->setEnabled(false);
        }
    });
    m_collaborationMode->select(syncMode);

    //session menue
    m_availableSessions = new ui::SelectionList(m_collaborativeMenu, "AvailableSessions");
    m_availableSessions->setText("Available sessions");
    m_availableSessions->setCallback([this](int id) 
    {
        int sessionID = 0;
        if (id != 0)
        {
            std::set<int>::iterator it = m_sessions.begin();
            std::advance(it, id - 1);
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
    m_newSession = new ui::Action(m_collaborativeMenu, "NewSession");
    m_newSession->setText("New session");
    m_newSession->setCallback([this](void) {
        if (syncMode == LooseCoupling)
        {
            VRAvatarList::instance()->show();
        }
        bool isPrivate = false;
        covise::TokenBuffer tb;
        tb << coVRCommunication::instance()->getID();
        tb << coVRCommunication::instance()->getPublicSessionID();
        tb << isPrivate;
        vrbc->sendMessage(tb, covise::COVISE_MESSAGE_VRB_REQUEST_NEW_SESSION);
    });

    m_showAvatar = new ui::Button(m_collaborativeMenu, "ShowAvatar");
    m_showAvatar->setText("Show avatar");
    m_showAvatar->setState(true);
    m_showAvatar->setCallback([this](bool state){
        showAvatar = state;
        if (showAvatar)
        {
            VRAvatarList::instance()->show();
            cover->sendBinMessage("SYNC_MODE", "SHOW_AVATAR", 12);
        }
        else
        {
            VRAvatarList::instance()->hide();
            cover->sendBinMessage("SYNC_MODE", "HIDE_AVATAR", 12);
        }
    });

    m_syncInterval = new ui::Slider(m_collaborativeMenu, "SyncInterval");
    m_syncInterval->setPresentation(ui::Slider::AsDial);
    m_syncInterval->setText("Sync interval");
    m_syncInterval->setBounds(0.01, 10.0);
    m_syncInterval->setValue(syncInterval);
    m_syncInterval->setScale(ui::Slider::Logarithmic);
    m_syncInterval->setCallback([this](double value, bool moving){
        if (!moving)
        {
            syncInterval = value;
            getSyncInterval();
        }

    });

    m_master = new ui::Button(m_collaborativeMenu, "Master");
    m_master->setState(isMaster());
    m_master->setEnabled(!isMaster() && m_visible);
    m_master->setCallback([this](bool state){
        if (state)
        {
            coVRCommunication::instance()->becomeMaster();
            updateSharedStates();
        }

        m_master->setEnabled(!state && m_visible);
        m_returnToMaster->setEnabled(!state && m_visible);
    });

    m_returnToMaster = new ui::Action(m_collaborativeMenu, "ReturnToMaster");
    m_returnToMaster->setText("Return to master");
    m_returnToMaster->setEnabled(false);
    m_returnToMaster->setCallback([this](void) {
        updateSharedStates(true);
    });

    m_partnerGroup = new ui::Group(m_collaborativeMenu, "Partners");

    showCollaborative(false);
}

bool coVRCollaboration::updateCollaborativeMenu()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "coVRCollaboration::updateCollaborativeMenu\n");

    bool changed = false;

    if (oldMasterStatus != isMaster())
    {
        changed = true;
        oldMasterStatus = isMaster();
        m_master->setState(isMaster());
        m_master->setEnabled(!isMaster() && m_visible);
        m_returnToMaster->setEnabled(!isMaster() && m_visible);
    }
    if (oldSyncInterval != syncInterval)
    {
        changed = true;
        oldSyncInterval = syncInterval;
        m_syncInterval->setValue(syncInterval);
    }
    if (oldAvatarVisibility != VRAvatarList::instance()->isVisible())
    {
        changed = true;
        oldAvatarVisibility = VRAvatarList::instance()->isVisible();
        m_showAvatar->setState(VRAvatarList::instance()->isVisible());
    }

    return changed;
}

bool coVRCollaboration::update()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "coVRCollaboration::update\n");

    bool changed = false;

    double thisTime = cover->frameTime();

    bool lo = coVRCommunication::instance()->isRILocked(coVRCommunication::TRANSFORM);
    if (lo && !wasLo)
        fprintf(stderr, "TRANSFORM locked\n");
    else if (!lo && wasLo)
        fprintf(stderr, "TRANSFORM not locked\n");
	wasLo = lo;

    if (updateCollaborativeMenu())
        changed = true;

    if (syncScale)
    {
        static float last_dcs_scale_factor = 0.0;
        const float scaleFactor = VRSceneGraph::instance()->scaleFactor();
        if (scaleFactor != last_dcs_scale_factor)
        {
            last_dcs_scale_factor = scaleFactor;
            VRSceneGraph::instance()->setScaleFactor(scaleFactor, false);
        }

        if ((coVRCommunication::instance()->collaborative())
            && (syncMode != LooseCoupling)
            && ((syncMode != MasterSlaveCoupling) || (isMaster())))
        {
            if (!(coVRCommunication::instance()->isRILockedByMe(coVRCommunication::TRANSFORM))
                && !(coVRCommunication::instance()->isRILocked(coVRCommunication::TRANSFORM))
                && (syncXform == 1))
            {
                coVRCommunication::instance()->RILock(coVRCommunication::TRANSFORM);
            }
            if (!(coVRCommunication::instance()->isRILockedByMe(coVRCommunication::SCALE))
                && !(coVRCommunication::instance()->isRILocked(coVRCommunication::SCALE))
                && (syncScale == 1))
            {
                coVRCommunication::instance()->RILock(coVRCommunication::SCALE);
            }
            this->scaleFactor = scaleFactor;
            syncScale = 0;
        }
    }
    else
    {
        //this only unlocks if it is locked by me
        if ((coVRCommunication::instance()->isRILockedByMe(coVRCommunication::TRANSFORM)) && (syncXform == 0) && (syncScale == 0))
        {
            coVRCommunication::instance()->RIUnLock(coVRCommunication::TRANSFORM);
        }
        //this only unlocks if it is locked by me
        if ((coVRCommunication::instance()->isRILockedByMe(coVRCommunication::SCALE)) && (syncScale == 0))
        {
            coVRCommunication::instance()->RIUnLock(coVRCommunication::SCALE);
        }
    }

    if (syncXform)
    {
        //      cerr << "syncXform locked:" << coVRCommunication::instance()->isRILocked(coVRCommunication::TRANSFORM) << endl;

        if ((coVRCommunication::instance()->collaborative())
            && (!coVRCommunication::instance()->isRILocked(coVRCommunication::TRANSFORM))
            && (syncMode != LooseCoupling)
            && ((syncMode != MasterSlaveCoupling) || (isMaster())))
        {
            if (!(coVRCommunication::instance()->isRILockedByMe(coVRCommunication::TRANSFORM))
                && !(coVRCommunication::instance()->isRILocked(coVRCommunication::TRANSFORM))
                && (syncXform == 1))
            {
                coVRCommunication::instance()->RILock(coVRCommunication::TRANSFORM);
            }
            avatarPosition = VRSceneGraph::instance()->getTransform()->getMatrix();
        }
    }
    else
    {
        //this only unlocks if it is locked by me
        if ((coVRCommunication::instance()->isRILockedByMe(coVRCommunication::TRANSFORM)) && (syncXform == 0) && (syncScale == 0))
        {
            coVRCommunication::instance()->RIUnLock(coVRCommunication::TRANSFORM);
        }
    }

    static double lastAvatarUpdateTime = 0.0;
    if ((coVRCommunication::instance()->collaborative())
        && (thisTime > lastAvatarUpdateTime + syncInterval)
        && (VRAvatarList::instance()->isVisible())) /*&& (syncMode == LooseCoupling)*/
    {
        // in LOOSE Coupling, we transfer AVATAR data
        // changed, now we always transfer avatar data when avatars are visible
        // visibility is synchronized....!

        VRAvatarList::instance()->sendMessage();

        lastAvatarUpdateTime = thisTime;
    }

    return changed;
}

void coVRCollaboration::SyncXform() //! mark VRSceneGraph::m_objectsTransform as dirty
{
    syncXform = true;
}

void coVRCollaboration::UnSyncXform()
{
    syncXform = false;
}

void coVRCollaboration::SyncScale() //! mark VRSceneGraph::m_scaleTransform as dirty
{
    syncScale = true;
}

void coVRCollaboration::UnSyncScale()
{
    syncScale = false;
}

void coVRCollaboration::setSyncMode(const char *mode)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRCollaboration::setSyncMode\n");

    else if (strcmp(mode, "SHOW_AVATAR") == 0)
    {
        VRAvatarList::instance()->show();
    }
    else if (strcmp(mode, "HIDE_AVATAR") == 0)
    {
        VRAvatarList::instance()->hide();
    }
}

void coVRCollaboration::remoteTransform(osg::Matrix &mat)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRCollaboration::remoteTransform\n");
    if (syncMode != LooseCoupling)
    {
        VRSceneGraph::instance()->getTransform()->setMatrix(mat);
        coVRNavigationManager::instance()->adjustFloorHeight();
    }
}

void coVRCollaboration::remoteScale(float d)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRCollaboration::remoteScale\n");

    if (syncMode != LooseCoupling)
    {
        VRSceneGraph::instance()->setScaleFactor(d, false);
    }
}

void coVRCollaboration::showCollaborative(bool visible)
{
    m_visible = visible;

    if (m_collaborativeMenu)
        m_collaborativeMenu->setVisible(visible);
    if (m_collaborationMode)
        m_collaborationMode->setEnabled(visible);
    if (m_syncInterval)
        m_syncInterval->setEnabled(visible);
    if (m_master)
        m_master->setEnabled(visible && !isMaster());
    if (m_showAvatar)
        m_showAvatar->setEnabled(visible);
}

float coVRCollaboration::getSyncInterval()
{
    if (vrbc)
    {
        if (syncInterval < vrbc->getSendDelay() * 2.0)
            return vrbc->getSendDelay() * 2.0;
    }

    return syncInterval;
}

void opencover::coVRCollaboration::updateSessionSelectionList(std::set<int> ses)
{
    m_sessions = ses;
    std::vector<std::string> sessionNames;
    sessionNames.push_back("Private");
    for (auto it = m_sessions.begin(); it != m_sessions.end(); ++it)
    {
        sessionNames.push_back(std::to_string(*it));
    }
    m_availableSessions->setList(sessionNames);
}

coVRCollaboration::SyncMode coVRCollaboration::getSyncMode() const
{
    SyncMode s = LooseCoupling;
    switch (syncMode)
    {
    case 0 :
        s = LooseCoupling;
        break;
    case 1:
        s = MasterSlaveCoupling;
        break;
    case 2:
        s = TightCoupling;
        break;
    default:
        break;
    }
    return s;
}

bool coVRCollaboration::isMaster()
{
    return coVRCommunication::instance()->isMaster();
}

void coVRCollaboration::setCurrentSession(int id)
{
   
    auto it = m_sessions.begin();
    std::vector<std::string> sessionNames;
    sessionNames.push_back("Private");
    int index = 0;
    for (int i = 0; i < m_sessions.size(); ++i)
    {
        sessionNames.push_back(std::to_string(*it));
        if (*it == id)
        {
            index = i+1;
        }
        ++it;
    }
    m_availableSessions->setList(sessionNames);
    m_availableSessions->select(index);

}

void coVRCollaboration::updateSharedStates(bool force) {
    
    int privateSessionID = coVRCommunication::instance()->getPrivateSessionID();
    int publicSessionID = coVRCommunication::instance()->getPublicSessionID();
    int useCouplingModeSessionID;
    int sessionToSubscribe = publicSessionID;;

    switch (syncMode)
    {
    case opencover::coVRCollaboration::LooseCoupling:
        useCouplingModeSessionID = publicSessionID;
        break;
    case opencover::coVRCollaboration::MasterSlaveCoupling:
        if (isMaster())
        {
            useCouplingModeSessionID = publicSessionID;
            sessionToSubscribe = privateSessionID;
        }
        else
        {
            useCouplingModeSessionID = privateSessionID;
        }
        break;
    case opencover::coVRCollaboration::TightCoupling:
        useCouplingModeSessionID = publicSessionID;
        break;
    default:
        break;
    }
    if (publicSessionID == 0) //send to private if not in public session
    {
        publicSessionID = privateSessionID;
    }
    SharedStateManager::instance()->update(privateSessionID, publicSessionID, useCouplingModeSessionID, sessionToSubscribe, force);
}

ui::Menu *coVRCollaboration::menu() const
{
    return m_collaborativeMenu;
}

ui::Group *coVRCollaboration::partnerGroup() const
{
    return m_partnerGroup;
}

void coVRCollaboration::syncModeChanged(int mode) {
    switch (mode) {
    case LooseCoupling:
        VRAvatarList::instance()->show();
        m_returnToMaster->setEnabled(false);
        break;
    case MasterSlaveCoupling:
        VRAvatarList::instance()->hide();
        m_returnToMaster->setEnabled(true);
        SyncXform();
        SyncScale();
        break;
    case TightCoupling:
        VRAvatarList::instance()->hide();
        SyncXform();
        SyncScale();
        m_returnToMaster->setEnabled(false);
        break;

    }
    updateSharedStates();
}
void coVRCollaboration::setSyncInterval() 
{
    scaleFactor.setSyncInterval(syncInterval);
    avatarPosition.setSyncInterval(syncInterval);

}
namespace vrb {
template <>
void serialize<osg::Matrix>(covise::TokenBuffer &tb, const osg::Matrix &value) {
    tb << value(0, 0);  tb << value(0, 1); tb << value(0, 2);
    tb << value(1, 0);  tb << value(1, 1); tb << value(1, 2);
    tb << value(2, 0);  tb << value(2, 1); tb << value(2, 2);
    tb << value(3, 0);  tb << value(3, 1); tb << value(3, 2);

}
template<>
void deserialize<osg::Matrix>(covise::TokenBuffer &tb, osg::Matrix &value) {
    tb >> value(0, 0); tb >> value(0, 1); tb >> value(0, 2);
    tb >> value(1, 0); tb >> value(1, 1); tb >> value(1, 2);
    tb >> value(2, 0); tb >> value(2, 1); tb >> value(2, 2);
    tb >> value(3, 0); tb >> value(3, 1); tb >> value(3, 2);

}
}


