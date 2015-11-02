/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: TangiblePosition Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "TrackObjects.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <appl/RenderInterface.h>
#include <cover/coVRTui.h>
#include <PluginUtil/coBaseCoviseInteractor.h>
#include <OpenVRUI/coToolboxMenu.h>

#include <vrml97/vrml/VrmlNodeCOVER.h>

using namespace covise;
using vrml::VrmlNodeCOVER;
using vrml::theCOVER;

TrackObjects::TrackObjects()
{
}

bool TrackObjects::init()
{
    pinboardEntry = new coSubMenuItem("Tangible");
    execButton = new coButtonMenuItem("RestartSimulation");
    execButton->setMenuListener(this);
    TangibleSimulationMenu = new coRowMenu("TangibleSimulationMenu");
    TangibleSimulationMenu->add(execButton);
    cover->getMenu()->add(pinboardEntry);
    pinboardEntry->setMenu(TangibleSimulationMenu);
    cerr << "toolbar" << endl;
    if (cover->getToolBar() != NULL)
    {

        cerr << "toolbar Initialized" << endl;
        ToolbarButton = new coIconButtonToolboxItem("AKToolbar/Restart");
        ToolbarButton->setMenuListener(this);
        cover->getToolBar()->insert(ToolbarButton, 0);
    }

    TrackObjectsTab = new coTUITab("TrackObjects", coVRTui::instance()->mainFolder->getID());
    trackObjects = new coTUIToggleButton("TrackObjects", TrackObjectsTab->getID());
    TrackObjectsTab->setPos(0, 0);
    trackObjects->setPos(0, 0);
    trackObjects->setEventListener(this);
    
    x=y=z=h=p=r=0.0;
    
    posX = new coTUIEditFloatField("posX", TrackObjectsTab->getID());
    posY = new coTUIEditFloatField("posY", TrackObjectsTab->getID());
    posZ = new coTUIEditFloatField("posZ", TrackObjectsTab->getID());
    rotH = new coTUIEditFloatField("rotH", TrackObjectsTab->getID());
    rotP = new coTUIEditFloatField("rotP", TrackObjectsTab->getID());
    rotR = new coTUIEditFloatField("rotR", TrackObjectsTab->getID());
    posX->setEventListener(this);
    posY->setEventListener(this);
    posZ->setEventListener(this);
    rotH->setEventListener(this);
    rotP->setEventListener(this);
    rotR->setEventListener(this);
    posX->setValue(x);
    posY->setValue(y);
    posZ->setValue(z);
    rotH->setValue(h);
    rotP->setValue(p);
    rotR->setValue(r);
    posX->setPos(0, 2);
    posY->setPos(1, 2);
    posZ->setPos(2, 2);
    rotH->setPos(0, 3);
    rotP->setPos(1, 3);
    rotR->setPos(2, 3);

    return true;
}

// this is called if the plugin is removed at runtime
TrackObjects::~TrackObjects()
{
    fprintf(stderr, "TrackObjects::~TrackObjects\n");
    delete execButton;
    delete TangibleSimulationMenu;
    delete pinboardEntry;
    for (list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->decRefCount();
    }
}

void TrackObjects::preFrame()
{    
     cover->getObjectsXform()->setMatrix(tmpMat);
     coVRCollaboration::instance()->SyncXform();
}
void TrackObjects::menuEvent(coMenuItem *m)
{
    if (m == execButton || m == ToolbarButton)
    {
        updateAndExec();
    }
}

void TrackObjects::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == RestartSimulation)
    {
        updateAndExec();
    }
}

void TrackObjects::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == posX)
    {
        x = posX->getValue();
    }
    else if (tUIItem == posY)
    {
        y = posY->getValue();
    }
    else if (tUIItem == posZ)
    {
        z = posZ->getValue();
    }
    if (tUIItem == rotH)
    {
        h = rotH->getValue();
    }
    else if (tUIItem == rotP)
    {
        p = rotP->getValue();
    }
    else if (tUIItem == rotR)
    {
        r = rotR->getValue();
    }
}


COVERPLUGIN(TrackObjects)
