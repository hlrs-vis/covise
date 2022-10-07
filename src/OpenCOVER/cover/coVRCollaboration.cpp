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
#include <vrb/client/VrbClientRegistry.h>
#include "coVRNavigationManager.h"
#include "VRSceneGraph.h"
#include "coVRCollaboration.h"
#include "coVRPluginSupport.h"
#include "coVRMSController.h"
#include "coVRCommunication.h"
#include "coVRPartner.h"
#include "VRAvatar.h"
#include <osg/MatrixTransform>
#include <vrb/client/SharedStateManager.h>
#include <vrb/client/VRBClient.h>
#include <vrb/client/VRBMessage.h>
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
    coVRPartnerList::instance()->hideAvatars();
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
    m_collaborationMode->setCallback(
        [this](int mode)
        {
            syncMode = mode;
            syncModeChanged(syncMode);
        });
    m_collaborationMode->select(syncMode);

    m_master = new ui::Button(m_collaborativeMenu, "Master");
    m_master->setState(isMaster());
    m_master->setCallback(
        [this](bool state)
        {
            if (state)
            {
                coVRCommunication::instance()->becomeMaster();
                avatarPosition = VRSceneGraph::instance()->getTransform()->getMatrix();
                scaleFactor = VRSceneGraph::instance()->scaleFactor();
                vrb::SharedStateManager::instance()->becomeMaster();
                updateSharedStates();
            }
            updateUi();
        });

    m_returnToMaster = new ui::Action(m_collaborativeMenu, "ReturnToMaster");
    m_returnToMaster->setText("Teleport to master");
    m_returnToMaster->setCallback([this](void) { updateSharedStates(true); });

    m_showAvatar = new ui::Button(m_collaborativeMenu, "ShowAvatar");
    m_showAvatar->setText("Show avatars");
    m_showAvatar->setState(true);
    m_showAvatar->setCallback([this](bool state){
        showAvatar = state;
        if (state)
            coVRPartnerList::instance()->showAvatars();
        else
            coVRPartnerList::instance()->hideAvatars();
        covise::TokenBuffer tb;
        tb << vrb::SYNC_MODE;
        tb << showAvatar;
        covise::Message msg(tb);
        msg.type = covise::COVISE_MESSAGE_VRB_MESSAGE;
        cover->sendVrbMessage(&msg);
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

    m_partnerGroup = new ui::Group(m_collaborativeMenu, "Partners");

    showCollaborative(false);
    syncModeChanged(syncMode);
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
    }
    if (oldSyncInterval != syncInterval)
    {
        changed = true;
        oldSyncInterval = syncInterval;
        m_syncInterval->setValue(syncInterval);
    }
    if (oldAvatarVisibility != coVRPartnerList::instance()->avatarsVisible())
    {
        changed = true;
        oldAvatarVisibility = coVRPartnerList::instance()->avatarsVisible();
        m_showAvatar->setState(coVRPartnerList::instance()->avatarsVisible());
    }

    updateUi();

    return changed;
}

bool coVRCollaboration::update()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "coVRCollaboration::update\n");

    bool changed = false;
    if (updateCollaborativeMenu())
        changed = true;
	//
	double thisTime = cover->frameTime();

    //sync avatar position 
	static double lastAvatarUpdateTime = 0.0;
    if (m_visible
        && (thisTime > lastAvatarUpdateTime + syncInterval)
        && (syncMode == LooseCoupling))
    {
        coVRPartnerList::instance()->sendAvatarMessage();
        lastAvatarUpdateTime = thisTime;
    }
	if (vrui::coInteractionManager::the()->isNaviagationBlockedByme()|| syncXform) //i am navigating
	{
		//store my viewpoint in shared state to be able to reload it
        avatarPosition = VRSceneGraph::instance()->getTransform()->getMatrix();
        scaleFactor = VRSceneGraph::instance()->scaleFactor();
	syncXform = false;
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

void opencover::coVRCollaboration::sessionChanged(bool isPrivate)
{
    if (!isPrivate && syncMode == LooseCoupling)
    {
        coVRPartnerList::instance()->showAvatars();
    }
    else //dont sync when in private session
    {
        UnSyncXform();
        coVRPartnerList::instance()->hideAvatars();
    }
}

void coVRCollaboration::setSyncMode(const char *mode)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRCollaboration::setSyncMode: %s\n", mode);

    if (std::string(mode) == "LOOSE")
    {
        syncMode = LooseCoupling;
    }

    if (strcmp(mode, "SHOW_AVATAR") == 0)
    {
        coVRPartnerList::instance()->showAvatars();
    }
    else if (strcmp(mode, "HIDE_AVATAR") == 0)
    {
        coVRPartnerList::instance()->hideAvatars();
    }
}

void coVRCollaboration::remoteTransform(osg::Matrix &mat)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRCollaboration::remoteTransform\n");
    if (syncMode != LooseCoupling) // || !coVRCommunication::instance()->getSessionID().isPrivate())
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

    if (visible && syncMode == LooseCoupling)
    {
        if (showAvatar)
            coVRPartnerList::instance()->showAvatars();
    }
    else
    {
        coVRPartnerList::instance()->hideAvatars();
    }
    updateUi();
}

float coVRCollaboration::getSyncInterval()
{
    if (auto vrbc = OpenCOVER::instance()->vrbc())
    {
        if (syncInterval < vrbc->getSendDelay() * 2.0)
            return vrbc->getSendDelay() * 2.0;
    }

    return syncInterval;
}

coVRCollaboration::SyncMode coVRCollaboration::getCouplingMode() const
{
	return (coVRCollaboration::SyncMode)syncMode.value();
}

bool coVRCollaboration::isMaster()
{
    return coVRCommunication::instance()->isMaster();
}

void coVRCollaboration::updateSharedStates(bool force) {
    
    vrb::SessionID privateSessionID = coVRCommunication::instance()->getPrivateSessionID();
    vrb::SessionID publicSessionID = coVRCommunication::instance()->getSessionID();
    bool muted = false;
    if (publicSessionID.isPrivate())
    {
        publicSessionID = privateSessionID;
    }
    if (syncMode == MasterSlaveCoupling && !isMaster())
    {
        muted = true;
    }
    SharedStateManager::instance()->update(privateSessionID, publicSessionID, muted, force);
}

ui::Menu *coVRCollaboration::menu() const
{
    return m_collaborativeMenu;
}

ui::Group *coVRCollaboration::partnerGroup() const
{
    return m_partnerGroup;
}

void coVRCollaboration::syncModeChanged(int mode)
{
    switch (mode)
    {
    case LooseCoupling:
        if (showAvatar)
            coVRPartnerList::instance()->showAvatars();
        break;
    case MasterSlaveCoupling:
        coVRPartnerList::instance()->hideAvatars();
        SyncXform();
        break;
    case TightCoupling:
        coVRPartnerList::instance()->hideAvatars();
        SyncXform();
        break;
    }
    updateSharedStates();
    updateUi();
}

void coVRCollaboration::setSyncInterval() 
{
    scaleFactor.setSyncInterval(syncInterval);
    avatarPosition.setSyncInterval(syncInterval);
}

void coVRCollaboration::showAvatars(bool visible)
{
    showAvatar = visible;
    updateUi();
}

void coVRCollaboration::updateUi()
{
    switch (syncMode)
    {
    case LooseCoupling:
        m_showAvatar->setEnabled(m_visible);
        m_master->setEnabled(false);
        m_returnToMaster->setEnabled(false);
        break;
    case MasterSlaveCoupling:
        m_showAvatar->setEnabled(m_visible);
        m_master->setEnabled(m_visible && !isMaster());
        m_returnToMaster->setEnabled(!isMaster());
        break;
    case TightCoupling:
        m_showAvatar->setEnabled(false);
        m_master->setEnabled(false);
        m_returnToMaster->setEnabled(false);
        break;
    }

    if (isMaster())
    {
        m_master->setText("Master");
    }
    else
    {
        m_master->setText("Become master");
    }
    m_showAvatar->setState(showAvatar);

    if (m_collaborativeMenu)
        m_collaborativeMenu->setVisible(m_visible);
    if (m_collaborationMode)
        m_collaborationMode->setEnabled(m_visible);
    if (m_syncInterval)
        m_syncInterval->setEnabled(m_visible);
}
