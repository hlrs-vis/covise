/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: TrackObjects Plugin (does nothing)                          **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                 **
**                                                                          **
** History:  								                                 **
** Nov-01  v1	    				       		                             **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "TrackObjects.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <cover/coInteractor.h>
#include <OpenVRUI/coToolboxMenu.h>
#include <cover/coVRCollaboration.h>
#include <config/CoviseConfig.h>

using namespace covise;


TObject::TObject(const std::string &n)
{
    name = n;
    const std::string conf = "COVER.Plugin.TrackObjects.Objects.Object:" + name;
    offsetCoord.xyz[0] = coCoviseConfig::getFloat("x",conf,0.0);
    offsetCoord.xyz[1] = coCoviseConfig::getFloat("y",conf,0.0);
    offsetCoord.xyz[2] = coCoviseConfig::getFloat("z",conf,0.0);
    offsetCoord.hpr[0] = coCoviseConfig::getFloat("h",conf,0.0);
    offsetCoord.hpr[1] = coCoviseConfig::getFloat("p",conf,0.0);
    offsetCoord.hpr[2] = coCoviseConfig::getFloat("r",conf,0.0);

    offsetCoord.makeMat(offset);
}
TObject::~TObject()
{
}
void TObject::setOffset(float x, float y, float z, float h, float p, float r)
{
    offsetCoord.xyz[0] = x;
    offsetCoord.xyz[1] = y;
    offsetCoord.xyz[2] = z;
    offsetCoord.hpr[0] = h;
    offsetCoord.hpr[1] = p;
    offsetCoord.hpr[2] = r;
    offsetCoord.makeMat(offset);
}
TrackObjects::TrackObjects()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool TrackObjects::init()
{
    pinboardEntry = new coSubMenuItem("TrackObjects");
    execButton = new coButtonMenuItem("RestartSimulation");

    execButton->setMenuListener(this);
    TangibleSimulationMenu = new coRowMenu("TrackObjects");
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

    std::vector<std::string> names = coCoviseConfig::getScopeNames("COVER.Plugin.TrackObjects.Objects", "Object");

    if (names.empty())
    {
        cout << "One Offset must be configured!" << endl;
        names.push_back("default");
    }
    for (size_t i = 0; i < names.size(); ++i)
    {
        TObjects.push_back(new TObject(names[i]));
    }
    currentObject = TObjects[0];

    TrackObjectsTab = new coTUITab("TrackObjects", coVRTui::instance()->mainFolder->getID());
    trackObjects = new coTUIToggleButton("TrackObjects", TrackObjectsTab->getID());

    objectChoiceLabel = new coTUILabel("Objec:",TrackObjectsTab->getID());
    objectChoice = new coTUIComboBox("Objects", TrackObjectsTab->getID());
    bodyChoiceLabel = new coTUILabel("Body:",TrackObjectsTab->getID());
    bodyChoice = new coTUIComboBox("Body", TrackObjectsTab->getID());
    getOffset = new coTUIButton("get offset", TrackObjectsTab->getID());
    snap = new coTUIButton("snap", TrackObjectsTab->getID());

    TrackObjectsTab->setPos(5, 0);
    trackObjects->setPos(0, 0);
    objectChoiceLabel->setPos(1,0);
    bodyChoiceLabel->setPos(2,0);
    objectChoice->setPos(1,1);
    bodyChoice->setPos(2,1);
    getOffset->setPos(4,1);
    snap->setPos(4,0);
    objectChoice->setEventListener(this);
    bodyChoice->setEventListener(this);
    trackObjects->setEventListener(this);
    getOffset->setEventListener(this);
    snap->setEventListener(this);
    for (size_t i = 0; i < names.size(); ++i)
    {
        objectChoice->addEntry(names[i]);
    }
    std::string bodyName = coCoviseConfig::getEntry("value","COVER.Plugin.TrackObjects.BodyName", "");
    if(bodyName.length()==0 && Input::instance()->getNumBodies()>0)
    {
        bodyName = Input::instance()->getBody(0)->name();
    }
    for (size_t i = 0; i < Input::instance()->getNumBodies(); ++i)
    {
        std::string name = Input::instance()->getBody(i)->name();
        bodyChoice->addEntry(name);
        if(name == bodyName)
        {
            bodyChoice->setSelectedEntry(i);
        }
    }


    x=y=z=h=p=r=0.0;

    trackingBody = Input::instance()->getBody(bodyName);

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
    if(trackingBody && trackObjects->getState())
    {
        osg::Matrix currentMat = trackingBody->getMat();
        if(currentMat != oldMat)
        {
            oldMat = currentMat;
            osg::Matrix tmpMat;
            tmpMat = currentObject->offset * currentMat ;
            cover->getObjectsXform()->setMatrix(tmpMat);
            coVRCollaboration::instance()->SyncXform();
        }
    }
}

void TrackObjects::menuEvent(coMenuItem *m)
{
    if (m == execButton || m == ToolbarButton)
    {
    }
}

void TrackObjects::tabletPressEvent(coTUIElement *tUIItem)
{
}

void TrackObjects::tabletEvent(coTUIElement *tUIItem)
{
    if(tUIItem == objectChoice)
    {
        currentObject = TObjects[objectChoice->getSelectedEntry()];
        updateTUI();
    }
    else if(tUIItem == bodyChoice)
    {
        trackingBody = Input::instance()->getBody(bodyChoice->getSelectedText());
    }
    else if (tUIItem == posX)
    {
        currentObject->offsetCoord.xyz[0] = posX->getValue();
	
    currentObject->offsetCoord.makeMat(currentObject->offset);
        updateTUI();
    }
    else if (tUIItem == posY)
    {
        currentObject->offsetCoord.xyz[1] = posY->getValue();
    currentObject->offsetCoord.makeMat(currentObject->offset);
        updateTUI();
    }
    else if (tUIItem == posZ)
    {
        currentObject->offsetCoord.xyz[2] = posZ->getValue();
    currentObject->offsetCoord.makeMat(currentObject->offset);
        updateTUI();
    }
    else if (tUIItem == rotH)
    {
        currentObject->offsetCoord.hpr[0] = rotH->getValue();
    currentObject->offsetCoord.makeMat(currentObject->offset);
        updateTUI();
    }
    else if (tUIItem == rotP)
    {
        currentObject->offsetCoord.hpr[1] = rotP->getValue();
    currentObject->offsetCoord.makeMat(currentObject->offset);
        updateTUI();
    }
    else if (tUIItem == rotR)
    {
        currentObject->offsetCoord.hpr[2] = rotR->getValue();
    currentObject->offsetCoord.makeMat(currentObject->offset);
        updateTUI();
    }
    else if (tUIItem == getOffset)
    {
        if(trackingBody)
        {
            osg::Matrix invTB = osg::Matrix::inverse(trackingBody->getMat());
            osg::Matrix tmpMat = cover->getObjectsXform()->getMatrix();
            currentObject->offset = tmpMat * invTB;
            currentObject->offsetCoord = currentObject->offset;
            updateTUI();
        }
    }
    else if (tUIItem == snap)
    {
        snapTo45Degrees(&currentObject->offset);
        currentObject->offsetCoord = currentObject->offset;
        updateTUI();
    }

}

void TrackObjects::updateTUI()
{

    posX->setValue(currentObject->offsetCoord.xyz[0]);
    posY->setValue(currentObject->offsetCoord.xyz[1]);
    posZ->setValue(currentObject->offsetCoord.xyz[2]);
    rotH->setValue(currentObject->offsetCoord.hpr[0]);
    rotP->setValue(currentObject->offsetCoord.hpr[1]);
    rotR->setValue(currentObject->offsetCoord.hpr[2]);
}

COVERPLUGIN(TrackObjects)
