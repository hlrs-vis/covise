/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VR_COLLABORATION_H
#define VR_COLLABORATION_H

/*! \file
 \brief  handle collaboration menu

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2001
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#include <util/coExport.h>
#include <util/common.h>
#include <set>
#include <osg/Matrix>
#include <vrbclient/SharedState.h>
#include "ui/Owner.h"


namespace osg
{
class Group;
}
namespace vrb {
template <>
void serialize<osg::Matrix>(covise::TokenBuffer &tb, const osg::Matrix &value);

template <>
void deserialize<osg::Matrix>(covise::TokenBuffer &tb, osg::Matrix &value);
}
namespace opencover
{

namespace ui
{
class Group;
class Menu;
class Button;
class Slider;
class SelectionList;
class Action;
}


class COVEREXPORT coVRCollaboration: public ui::Owner
{
    static coVRCollaboration *s_instance;
    coVRCollaboration();

public:
    enum SyncMode
    {
        LooseCoupling,
        MasterSlaveCoupling,
        TightCoupling
    };

private:
    int readConfigFile();
    void initCollMenu();
    void setSyncInterval();
    std::set<int> m_sessions;
    bool syncXform = false;
    bool syncScale = false;
	bool wasLo = false;
    float syncInterval;
    bool oldMasterStatus = true;
    float oldSyncInterval = -1;
    bool oldAvatarVisibility = true;

public:
    virtual ~coVRCollaboration();
    void config();
    void showCollaborative(bool visible);
    static coVRCollaboration *instance();
    int showAvatar;
    vrb::SharedState<int> syncMode; ///0: LooseCoupling, 1: MasterSlaveCoupling, 2 TightCoupling
    vrb::SharedState<osg::Matrix> avatarPosition;
    vrb::SharedState<float> scaleFactor;
    float getSyncInterval();
    void updateSessionSelectionList(std::set<int> ses);
    // returns collaboration mode
    SyncMode getSyncMode() const;

    void setSyncMode(const char *mode); // set one of "LOOSE", "MS", "TIGHT"

    void updateSharedStates(bool force = false);
    // returns true if this COVER ist master in a collaborative session
    bool isMaster();
    void setCurrentSession(int id);
    // Collaborative menu:
    bool m_visible = false;
    ui::Menu *m_collaborativeMenu = nullptr;
    ui::Group *m_partnerGroup = nullptr;
    ui::Button *m_showAvatar = nullptr;
    ui::Button *m_master = nullptr;
    ui::Action *m_returnToMaster = nullptr;
    ui::Action *m_newSession = nullptr;
    ui::Slider *m_syncInterval = nullptr;
    ui::SelectionList *m_collaborationMode = nullptr;
    ui::SelectionList *m_availableSessions = nullptr;
    ui::Menu *menu() const;
    ui::Group *partnerGroup() const;

    bool updateCollaborativeMenu();
    void syncModeChanged(int mode);
    void init();

    bool update();

    void SyncXform();
    void UnSyncXform();
    void SyncScale();
    void UnSyncScale();

    void remoteTransform(osg::Matrix &mat);
    void remoteScale(float d);
};
}
#endif
