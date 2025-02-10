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
 *	File            vvCollaboration.C (Performer 2.0)		        *
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
#include "vvNavigationManager.h"
#include "vvSceneGraph.h"
#include "vvCollaboration.h"
#include "vvPluginSupport.h"
#include "vvMSController.h"
#include "vvCommunication.h"
#include "vvPartner.h"
#include "vvAvatar.h"
#include <vsg/nodes/MatrixTransform.h>
#include <vrb/client/SharedStateManager.h>
#include <vrb/client/VRBClient.h>
#include <vrb/client/VRBMessage.h>
#include "vvVIVE.h"

#include "ui/Menu.h"
#include "ui/Action.h"
#include "ui/Button.h"
#include "ui/SelectionList.h"
#include "ui/Slider.h"

#define SYNC_MODE_GROUP 2
#define NAVIGATION_GROUP 0

using namespace vive;
using namespace vrb;
vvCollaboration *vvCollaboration::s_instance = NULL;

vvCollaboration *vvCollaboration::instance()
{
    if (!s_instance)
        s_instance = new vvCollaboration;
    return s_instance;
}

vvCollaboration::vvCollaboration()
    : ui::Owner("vvCollaboration", vv->ui)
    , syncXform(0)
    , syncInterval(0.3f)
    , showAvatar(1)
    , syncMode("vvCollaboration_syncMode", LooseCoupling, ALWAYS_SHARE)
    , avatarPosition("vvCollaboration_avatarPosition", vsg::dmat4())
    , scaleFactor("vvCollaboration_scaleFactor", 1)
{
    assert(!s_instance);

    init();
}

void vvCollaboration::init()
{
    if (vv->debugLevel(2))
        fprintf(stderr, "\nnew vvCollaboration\n");
    vvPartnerList::instance()->hideAvatars();
    syncMode.setUpdateFunction([this]()
                               {
        assert(syncMode < 3);
        if (m_collaborationMode)
            m_collaborationMode->select(syncMode);
        syncModeChanged(syncMode);
        updated = true;
    });
    avatarPosition.setUpdateFunction([this]()
    {
        updated = true;
        vsg::dmat4 m = avatarPosition;
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

vvCollaboration::~vvCollaboration()
{
    if (vv->debugLevel(2))
        fprintf(stderr, "\ndelete vvCollaboration\n");

    s_instance = NULL;
}

int vvCollaboration::readConfigFile()
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvCollaboration::readConfigFile\n");

    syncInterval = covise::coCoviseConfig::getFloat("interval", "VIVE.Collaborative.Sync", 0.05f);
    setSyncInterval();
    setSyncMode("LOOSE");
    std::string sMode = covise::coCoviseConfig::getEntry("mode", "VIVE.Collaborative.Sync");
    if (!sMode.empty())
        setSyncMode(sMode.c_str());

    return 0;
}

void vvCollaboration::initCollMenu()
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
            if(syncMode == MasterSlaveCoupling)
            {
                vvCommunication::instance()->becomeMaster();
                vrb::SharedStateManager::instance()->becomeMaster();
                updateSharedStates();
            }
            if(syncMode == TightCoupling || syncMode == MasterSlaveCoupling)
            {
                looseCouplingDeactivated = true;
                SyncXform();
            }
            avatarPosition = vvSceneGraph::instance()->getTransform()->matrix;
        });
    m_collaborationMode->select(syncMode);

    m_master = new ui::Button(m_collaborativeMenu, "Master");
    m_master->setState(isMaster());
    m_master->setCallback(
        [this](bool state)
        {
            if (state)
            {
                vvCommunication::instance()->becomeMaster();
                avatarPosition = vvSceneGraph::instance()->getTransform()->matrix;
                scaleFactor = vvSceneGraph::instance()->scaleFactor();
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
            vvPartnerList::instance()->showAvatars();
        else
            vvPartnerList::instance()->hideAvatars();
        covise::TokenBuffer tb;
        tb << vrb::SYNC_MODE;
        tb << showAvatar;
        covise::Message msg(tb);
        msg.type = covise::COVISE_MESSAGE_VRB_MESSAGE;
        vv->sendVrbMessage(&msg);
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
            syncInterval = (float)value;
            getSyncInterval();
        }

    });

    m_partnerGroup = new ui::Group(m_collaborativeMenu, "Partners");

    showCollaborative(false);
    syncModeChanged(syncMode);
}

bool vvCollaboration::updateCollaborativeMenu()
{
    if (vv->debugLevel(5))
        fprintf(stderr, "vvCollaboration::updateCollaborativeMenu\n");

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
    if (oldAvatarVisibility != vvPartnerList::instance()->avatarsVisible())
    {
        changed = true;
        oldAvatarVisibility = vvPartnerList::instance()->avatarsVisible();
        m_showAvatar->setState(vvPartnerList::instance()->avatarsVisible());
    }

    if (changed)
        updateUi();

    return changed;
}

bool vvCollaboration::update()
{
    if (vv->debugLevel(5))
        fprintf(stderr, "vvCollaboration::update\n");

    bool changed = false;
    if (updateCollaborativeMenu())
        changed = true;
	if(updated)
    {
        changed = true;
        updated = false;
    }
    double thisTime = vv->frameTime();

    //sync avatar position 
	static double lastAvatarUpdateTime = 0.0;
    if (m_visible
        && (thisTime > lastAvatarUpdateTime + syncInterval)
        && (syncMode == LooseCoupling))
    {
        vvPartnerList::instance()->sendAvatarMessage();
        lastAvatarUpdateTime = thisTime;
    }


	if (vrui::coInteractionManager::the()->isNaviagationBlockedByme()|| syncXform) //i am navigating
	{
		//store my viewpoint in shared state to be able to reload it
        avatarPosition = vvSceneGraph::instance()->getTransform()->matrix;
        scaleFactor = vvSceneGraph::instance()->scaleFactor();
        syncXform = false;
	}
    if(looseCouplingDeactivated)
    {
        avatarPosition.push();
        scaleFactor.push();
        looseCouplingDeactivated = false;
    }
    return changed;
}

void vvCollaboration::SyncXform() //! mark vvSceneGraph::m_objectsTransform as dirty
{
    syncXform = true;
}

void vvCollaboration::UnSyncXform()
{
    syncXform = false;
}

void vive::vvCollaboration::sessionChanged(bool isPrivate)
{
    if (!isPrivate && syncMode == LooseCoupling)
    {
        vvPartnerList::instance()->showAvatars();
    }
    else //dont sync when in private session
    {
        UnSyncXform();
        vvPartnerList::instance()->hideAvatars();
    }
}

void vvCollaboration::setSyncMode(const char *mode)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvCollaboration::setSyncMode: %s\n", mode);

    if (std::string(mode) == "LOOSE")
    {
        syncMode = LooseCoupling;
    }

    if (strcmp(mode, "SHOW_AVATAR") == 0)
    {
        vvPartnerList::instance()->showAvatars();
    }
    else if (strcmp(mode, "HIDE_AVATAR") == 0)
    {
        vvPartnerList::instance()->hideAvatars();
    }
}

void vvCollaboration::remoteTransform(vsg::dmat4 &mat)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvCollaboration::remoteTransform\n");
    if (syncMode != LooseCoupling) // || !vvCommunication::instance()->getSessionID().isPrivate())
    {
        vvSceneGraph::instance()->getTransform()->matrix = (mat);
        vvNavigationManager::instance()->adjustFloorHeight();
    }
}

void vvCollaboration::remoteScale(float d)
{
    if (vv->debugLevel(3))
        fprintf(stderr, "vvCollaboration::remoteScale\n");

    if (syncMode != LooseCoupling)
    {
        vvSceneGraph::instance()->setScaleFactor(d, false);
    }
}

void vvCollaboration::showCollaborative(bool visible)
{
    m_visible = visible;

    if (visible && syncMode == LooseCoupling)
    {
        if (showAvatar)
            vvPartnerList::instance()->showAvatars();
    }
    else
    {
        vvPartnerList::instance()->hideAvatars();
    }
    updateUi();
}

float vvCollaboration::getSyncInterval()
{
    if (auto vrbc = vvVIVE::instance()->vrbc())
    {
        if (syncInterval < vrbc->getSendDelay() * 2.0)
            return vrbc->getSendDelay() * 2.0f;
    }

    return syncInterval;
}

vvCollaboration::SyncMode vvCollaboration::getCouplingMode() const
{
	return (vvCollaboration::SyncMode)syncMode.value();
}

bool vvCollaboration::isMaster()
{
    return vvCommunication::instance()->isMaster();
}

void vvCollaboration::updateSharedStates(bool force) {
    
    vrb::SessionID privateSessionID = vvCommunication::instance()->getPrivateSessionID();
    vrb::SessionID publicSessionID = vvCommunication::instance()->getSessionID();
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

ui::Menu *vvCollaboration::menu() const
{
    return m_collaborativeMenu;
}

ui::Group *vvCollaboration::partnerGroup() const
{
    return m_partnerGroup;
}

void vvCollaboration::syncModeChanged(int mode)
{
    switch (mode)
    {
    case LooseCoupling:
        if (showAvatar)
            vvPartnerList::instance()->showAvatars();
        break;
    case MasterSlaveCoupling:
    case TightCoupling:
        vvPartnerList::instance()->hideAvatars();
        break;
    }
    updateSharedStates();
    updateUi();
}

void vvCollaboration::setSyncInterval() 
{
    scaleFactor.setSyncInterval(syncInterval);
    avatarPosition.setSyncInterval(syncInterval);
}

void vvCollaboration::showAvatars(bool visible)
{
    showAvatar = visible;
    updateUi();
}

void vvCollaboration::updateUi()
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
