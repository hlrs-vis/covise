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

#include <OpenVRUI/coRowMenu.h>
#include <osg/Matrix>

namespace vrui
{
class coSubMenuItem;
class coPotiMenuItem;
class coCheckboxMenuItem;
}

namespace osg
{
class Group;
}

namespace opencover
{
class COVEREXPORT coVRCollaboration : public vrui::coMenuListener
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
    void addMenuItem(osg::Group *itemGroup);

    int readConfigFile();
    void initCollMenu();

    bool syncXform;
    bool syncScale;
	bool wasLo = false;
    float syncInterval;

public:
    virtual ~coVRCollaboration();
    void config();
    vrui::coSubMenuItem *collButton;
    void showCollaborative(bool visible);
    static coVRCollaboration *instance();
    int showAvatar;
    SyncMode syncMode;
    float getSyncInterval();

    // returns collaboration mode
    SyncMode getSyncMode() const;

    void setSyncMode(const char *mode); // set one of "LOOSE", "MS", "TIGHT"

    // returns true if this COVER ist master in a collaborative session
    bool isMaster();

    // Collaborative menu:
    vrui::coRowMenu *collaborativeMenu;
    vrui::coCheckboxMenuItem *Loose;
    vrui::coCheckboxMenuItem *Tight;
    vrui::coCheckboxMenuItem *MasterSlave;
    vrui::coCheckboxMenuItem *ShowAvatar;
    vrui::coCheckboxMenuItem *Master;
    vrui::coPotiMenuItem *SyncInterval;

    // process key events
    void menuEvent(vrui::coMenuItem *);
    void updateCollaborativeMenu();

    void init();

    void update();

    void SyncXform() //! mark VRSceneGraph::m_objectsTransform as dirty
    {
        syncXform = true;
    }
    void UnSyncXform()
    {
        syncXform = false;
    }
    void SyncScale() //! mark VRSceneGraph::m_scaleTransform as dirty
    {
        syncScale = true;
    }
    void UnSyncScale()
    {
        syncScale = false;
    }

    void remoteTransform(osg::Matrix &mat);
    void remoteScale(float d);
};
}
#endif
