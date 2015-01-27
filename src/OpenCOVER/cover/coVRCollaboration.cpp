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
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <osg/MatrixTransform>

#include <vrbclient/VRBClient.h>
#include "OpenCOVER.h"

#define SYNC_MODE_GROUP 2
#define NAVIGATION_GROUP 0

using namespace vrui;
using namespace opencover;

coVRCollaboration *coVRCollaboration::instance()
{
    static coVRCollaboration *singleton = NULL;
    if (!singleton)
        singleton = new coVRCollaboration;
    return singleton;
}

coVRCollaboration::coVRCollaboration()
    : syncXform(0)
    , syncScale(0)
    , syncInterval(0.3)
    , collButton(NULL)
    , showAvatar(1)
    , Loose(NULL)
    , Tight(NULL)
    , MasterSlave(NULL)
{
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
    // Create collaborative menu:
    coCheckboxGroup *cbg = new coCheckboxGroup();
    collaborativeMenu = new coRowMenu("CollaborativeOptions");
    Loose = new coCheckboxMenuItem("Loose", true, cbg);
    Loose->setMenuListener(this);
    Tight = new coCheckboxMenuItem("Tight", false, cbg);
    Tight->setMenuListener(this);
    MasterSlave = new coCheckboxMenuItem("Master/Slave", false, cbg);
    MasterSlave->setMenuListener(this);
    ShowAvatar = new coCheckboxMenuItem("Show avatar", true);
    ShowAvatar->setMenuListener(this);
    SyncInterval = new coPotiMenuItem("Sync Interval", 0.0, 5, syncInterval);
    SyncInterval->setMenuListener(this);
    Master = new coCheckboxMenuItem("Master", false);
    Master->setMenuListener(this);
    collaborativeMenu->add(Loose);
    collaborativeMenu->add(Tight);
    collaborativeMenu->add(MasterSlave);
    collaborativeMenu->add(ShowAvatar);
    collaborativeMenu->add(SyncInterval);
    collaborativeMenu->add(Master);
}

void coVRCollaboration::updateCollaborativeMenu()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "coVRCollaboration::updateCollaborativeMenu\n");

    static bool oldMasterStatus = true;
    static float oldSyncInterval = -1;
    static bool oldAvatarVisibility = true;
    if (oldMasterStatus != isMaster())
    {
        oldMasterStatus = isMaster();
        Master->setState(isMaster());
    }
    if (oldSyncInterval != syncInterval)
    {
        oldSyncInterval = syncInterval;
        SyncInterval->setValue(syncInterval);
    }
    if (oldAvatarVisibility != VRAvatarList::instance()->isVisible())
    {
        oldAvatarVisibility = VRAvatarList::instance()->isVisible();
        ShowAvatar->setState(VRAvatarList::instance()->isVisible());
    }
}

void coVRCollaboration::menuEvent(coMenuItem *menuItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRCollaboration::menuEvent\n");

    if (menuItem == Loose)
    {
        setSyncMode("LOOSE");
        VRAvatarList::instance()->show();
        cover->sendBinMessage("SYNC_MODE", "LOOSE", 6);
    }
    else if (menuItem == Tight)
    {
        setSyncMode("TIGHT");
        VRAvatarList::instance()->hide();
        cover->sendBinMessage("SYNC_MODE", "TIGHT", 6);
        SyncXform();
        SyncScale();
    }
    else if (menuItem == MasterSlave)
    {
        setSyncMode("MS");
        VRAvatarList::instance()->hide();
        cover->sendBinMessage("SYNC_MODE", "MS", 3);
        SyncXform();
        SyncScale();
    }
    else if (menuItem == ShowAvatar)
    {
        showAvatar = ShowAvatar->getState();
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
    }
    else if (menuItem == Master)
    {
        if (Master->getState())
        {
            coVRCommunication::instance()->becomeMaster();
        }
    }
    else if (menuItem == SyncInterval)
    {
        syncInterval = SyncInterval->getValue();
    }
}

void
coVRCollaboration::update()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "coVRCollaboration::update\n");

    double thisTime = cover->frameTime();

    bool lo = coVRCommunication::instance()->isRILocked(coVRCommunication::TRANSFORM);
    static bool olo = false;
    if (lo && !olo)
        fprintf(stderr, "TRANSFORM locked\n");
    else if (!lo && olo)
        fprintf(stderr, "TRANSFORM not locked\n");
    olo = lo;

    updateCollaborativeMenu();

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
}

void coVRCollaboration::setSyncMode(const char *mode)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "coVRCollaboration::setSyncMode\n");

    if (strcmp(mode, "LOOSE") == 0)
    {
        syncMode = LooseCoupling;
        if (Loose)
            Loose->setState(true);
        if (MasterSlave)
            MasterSlave->setState(false);
        if (Tight)
            Tight->setState(false);
        VRAvatarList::instance()->show();
    }
    else if (strcmp(mode, "MS") == 0)
    {
        syncMode = MasterSlaveCoupling;
        if (Loose)
            Loose->setState(false);
        if (MasterSlave)
            MasterSlave->setState(true);
        if (Tight)
            Tight->setState(false);
        VRAvatarList::instance()->hide();
    }
    else if (strcmp(mode, "TIGHT") == 0)
    {
        syncMode = TightCoupling;
        if (Loose)
            Loose->setState(false);
        if (MasterSlave)
            MasterSlave->setState(false);
        if (Tight)
            Tight->setState(true);
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
    if (visible && collButton == NULL)
    {
        collButton = new coSubMenuItem("Collaborative");
        collButton->setMenu(collaborativeMenu);
        cover->getMenu()->add(collButton);
    }
    else if (!visible && collButton != NULL)
    {
        collButton->closeSubmenu();
        delete collButton;
        collButton = NULL;
    }
}

void
coVRCollaboration::tightCouplingCallback(void *sceneGraph, buttonSpecCell *spec)
{
    if (spec->state)
    {
        ((coVRCollaboration *)sceneGraph)->setSyncMode("TIGHT");
        VRAvatarList::instance()->hide();
        cover->sendBinMessage("SYNC_MODE", "TIGHT", 6);
    }
}

void
coVRCollaboration::looseCouplingCallback(void *sceneGraph, buttonSpecCell *spec)
{
    if (spec->state)
    {
        ((coVRCollaboration *)sceneGraph)->setSyncMode("LOOSE");
        VRAvatarList::instance()->show();
        cover->sendBinMessage("SYNC_MODE", "LOOSE", 6);
    }
}

void
coVRCollaboration::msCouplingCallback(void *sceneGraph, buttonSpecCell *spec)
{
    if (spec->state)
    {
        ((coVRCollaboration *)sceneGraph)->setSyncMode("MS");
        VRAvatarList::instance()->hide();
        cover->sendBinMessage("SYNC_MODE", "MS", 3);
    }
}

void
coVRCollaboration::showAvatarCallback(void *sceneGraph, buttonSpecCell *spec)
{
    if (spec->state)
    {
        ((coVRCollaboration *)sceneGraph)->showAvatar = true;
        VRAvatarList::instance()->show();
    }
    else
    {
        ((coVRCollaboration *)sceneGraph)->showAvatar = false;
        VRAvatarList::instance()->hide();
    }
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
