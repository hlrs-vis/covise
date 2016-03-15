/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef _CrawlerPlugin_H
#define _CrawlerPlugin_H
/****************************************************************************\
**                                                            (C)2009 HLRS  **
**                                                                          **
** Description: Varant plugin                                               **
**                                                                          **
this plugin uses the "CrawlerPlugin" attribute, setted from the VarianMarker module
to show/hide several CrawlerPlugins in the cover menu (CrawlerPlugins item)
**                                                                          **
** Author: A.Gottlieb                                                       **
**                                                                          **
** History:                                                                 **
** Jul-09  v1                                                               **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "PxPhysicsAPI.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <cover/coVRPlugin.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <cover/coVRSelectionManager.h>
#include <util/coExport.h>
#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
#include <cover/coVRLabel.h>


using namespace covise;
using namespace opencover;
using namespace vrui;
using namespace std;
using namespace physx;
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

class CrawlerPlugin : public coVRPlugin, public coMenuListener, public coTUIListener
{
    friend class mySensor;
public:
    static CrawlerPlugin *plugin;

    CrawlerPlugin();
    ~CrawlerPlugin();

    // this will be called in PreFrame
    void preFrame();

    void menuEvent(coMenuItem *menu_CrawlerPluginitem);
    // this will be called if a COVISE object arrives
    bool init();
    void message(int type, int len, const void *buf);
    void tabletEvent(coTUIElement *);
    
    unsigned int numActiveActor;
    PxFoundation*				gFoundation;
    PxPhysics*					gPhysics;

    PxDefaultCpuDispatcher*		gDispatcher;
    PxScene*					gScene;

    PxCooking*					gCooking;

    PxMaterial*					gMaterial;
    
    PxConvexMesh* createConvexMesh(const PxVec3* verts, const PxU32 numVerts, PxPhysics& physics, PxCooking& cooking);
private:
    coMenu *cover_menu;
    coSubMenuItem *button;
    coRowMenu *crawler_menu;

    coTUITab *CrawlerPluginTab;
    coTUIToggleButton *CrawlerPluginTUIItem;
    coTUIComboBox *CrawlerPluginTUIcombo;
    std::map<std::string, coTUIToggleButton *> tui_header_trans;

    bool firsttime;


    PxDefaultAllocator			gAllocator;
    PxDefaultErrorCallback		gErrorCallback;


    PxVisualDebuggerConnection*	gConnection;

    // Actor Globals
    PxRigidStatic*				gGroundPlane;

    void cleanupPhysics();
    void stepPhysics();
    
    bool loadWRL(const char *path, PxU32 &xDimension, PxU32 &zDimension, PxU32 &xSpacing, PxU32 &zSpacing, PxReal &heightscale, std::vector<vector<PxReal>> &HeightMatrix);
    void initPhysics();
};
#endif

//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
