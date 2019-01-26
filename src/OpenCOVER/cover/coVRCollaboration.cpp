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
#include <config/CoviseConfig.h>

#include "coVRNavigationManager.h"
#include "VRSceneGraph.h"
#include "coVRCollaboration.h"
#include "coVRPluginSupport.h"
#include "coVRMSController.h"
#include "coVRCommunication.h"
#include "VRAvatar.h"
#include <osg/MatrixTransform>

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
{
    assert(!s_instance);

    init();
}

void coVRCollaboration::init()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nnew coVRCollaboration\n");

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
    m_collaborationMode->setCallback([this](int mode){
        switch(mode) {
        case LooseCoupling:
            setSyncMode("LOOSE");
            VRAvatarList::instance()->show();
            cover->sendBinMessage("SYNC_MODE", "LOOSE", 6);
            break;
        case MasterSlaveCoupling:
            setSyncMode("MS");
            VRAvatarList::instance()->hide();
            cover->sendBinMessage("SYNC_MODE", "MS", 3);
            SyncXform();
            SyncScale();
            break;
        case TightCoupling:
            setSyncMode("TIGHT");
            VRAvatarList::instance()->hide();
            cover->sendBinMessage("SYNC_MODE", "TIGHT", 6);
            SyncXform();
            SyncScale();
            break;
        }
    });
    m_collaborationMode->select(syncMode);

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
            syncInterval = value;
    });

    m_master = new ui::Button(m_collaborativeMenu, "Master");
    m_master->setState(isMaster());
    m_master->setEnabled(!isMaster() && m_visible);
    m_master->setCallback([this](bool state){
        if (state)
            coVRCommunication::instance()->becomeMaster();
        m_master->setEnabled(!state && m_visible);
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

bool
coVRCollaboration::update()
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
        static double lastUpdateTime = 0.0;
        const float scaleFactor = VRSceneGraph::instance()->scaleFactor();
        if (scaleFactor != last_dcs_scale_factor)
        {
            last_dcs_scale_factor = scaleFactor;
            VRSceneGraph::instance()->setScaleFactor(scaleFactor, false);
        }

        if ((coVRCommunication::instance()->collaborative())
            && (thisTime > lastUpdateTime + syncInterval)
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

            char msg[500];
            sprintf(msg, "%f", scaleFactor);
            cover->sendBinMessage("SCALE_ALL", msg, strlen(msg) + 1);

            syncScale = 0;
            lastUpdateTime = thisTime;
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

        static double xlastUpdateTime = 0.0;
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

            if ((thisTime > xlastUpdateTime + syncInterval))
            {
                //cerr << "TRANSFORM_ALL:" << endl;
                osg::Matrix dcs_mat = VRSceneGraph::instance()->getTransform()->getMatrix();
                // send new xform dcs mat to other COVERs
                char msg[500];
                sprintf(msg, "%f %f %f %f %f %f %f %f %f %f %f %f",
                        dcs_mat(0, 0), dcs_mat(0, 1), dcs_mat(0, 2),
                        dcs_mat(1, 0), dcs_mat(1, 1), dcs_mat(1, 2),
                        dcs_mat(2, 0), dcs_mat(2, 1), dcs_mat(2, 2),
                        dcs_mat(3, 0), dcs_mat(3, 1), dcs_mat(3, 2));
                cover->sendBinMessage("TRANSFORM_ALL", msg, strlen(msg) + 1);

                xlastUpdateTime = thisTime;
                syncXform = false;
            }
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

    if (strcmp(mode, "LOOSE") == 0)
    {
        syncMode = LooseCoupling;
        if (m_collaborationMode)
            m_collaborationMode->select(syncMode);
        VRAvatarList::instance()->show();
    }
    else if (strcmp(mode, "MS") == 0)
    {
        syncMode = MasterSlaveCoupling;
        if (m_collaborationMode)
            m_collaborationMode->select(syncMode);
        VRAvatarList::instance()->hide();
    }
    else if (strcmp(mode, "TIGHT") == 0)
    {
        syncMode = TightCoupling;
        if (m_collaborationMode)
            m_collaborationMode->select(syncMode);
        VRAvatarList::instance()->hide();
    }
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

coVRCollaboration::SyncMode coVRCollaboration::getSyncMode() const
{
    return syncMode;
}

bool coVRCollaboration::isMaster()
{
    return coVRCommunication::instance()->isMaster();
}

ui::Menu *coVRCollaboration::menu() const
{
    return m_collaborativeMenu;
}

ui::Group *coVRCollaboration::partnerGroup() const
{
    return m_partnerGroup;
}
