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

#include "TangiblePositionPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <cover/coInteractor.h>
#include <OpenVRUI/coToolboxMenu.h>

#include <vrml97/vrml/VrmlNodeCOVER.h>


using namespace covise;
using vrml::VrmlNodeCOVER;
using vrml::theCOVER;

TangiblePositionPlugin::TangiblePositionPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool TangiblePositionPlugin::init()
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

    TangibleTab = new coTUITab("TangiblePosition", coVRTui::instance()->mainFolder->getID());
    RestartSimulation = new coTUIButton("RestartSimulation", TangibleTab->getID());
    TangibleTab->setPos(0, 0);
    RestartSimulation->setPos(0, 0);
    RestartSimulation->setEventListener(this);

    return true;
}

// this is called if the plugin is removed at runtime
TangiblePositionPlugin::~TangiblePositionPlugin()
{
    fprintf(stderr, "TangiblePositionPlugin::~TangiblePositionPlugin\n");
    delete execButton;
    delete TangibleSimulationMenu;
    delete pinboardEntry;
    for (list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->decRefCount();
    }
}

void
TangiblePositionPlugin::newInteractor(const RenderObject *, coInteractor *inter)
{
    if (strcmp(inter->getPluginName(), "TangiblePosition") == 0)
    {
        inter->incRefCount();
        interactors.push_back(inter);
        cerr << "removing Interactor from Module " << inter->getModuleName() << endl;
    }
}

void
TangiblePositionPlugin::removeObject(const char *objName, bool)
{
    for (list<coInteractor *>::iterator it = interactors.begin();
         it != interactors.end(); it++)
    {
        if ((*it)->getObjName() && objName)
        {
            if (strcmp((*it)->getObjName(), objName) == 0)
            {

                cerr << "removing Interactor from Module " << (*it)->getModuleName() << endl;
                (*it)->decRefCount();
                interactors.erase(it);
                break;
            }
        }
    }
}

void TangiblePositionPlugin::menuEvent(coMenuItem *m)
{
    if (m == execButton || m == ToolbarButton)
    {
        updateAndExec();
    }
}

void TangiblePositionPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == RestartSimulation)
    {
        updateAndExec();
    }
}

void TangiblePositionPlugin::tabletEvent(coTUIElement *tUIItem)
{
    (void)tUIItem;
}

void TangiblePositionPlugin::updateAndExec()
{
    for (list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        coInteractor *inter = *it;
        if (!inter)
            continue;

        bool execNeeded = false;
        for (int i=0; i<9; ++i)
        {
            if (i==0)
                continue;
            std::stringstream str;
            str << "pos_cube_" << i;

            if (theCOVER)
            {
                if ((theCOVER->transformations[i - 1][3 * 4]) < 400 && (theCOVER->transformations[i - 1][3 * 4]) > -400 && (theCOVER->transformations[i - 1][3 * 4] != 0 || theCOVER->transformations[i - 1][3 * 4 + 1] != 0))
                {
                    int numElem = 0;
                    float *ptr;
                    if (inter->getFloatVectorParam(str.str(), numElem, ptr) == -1)
                        continue;
                    execNeeded = true;
                    if (numElem == 3)
                    {
                        // x = x
                        // y = -z (da die transformation in vrml koordinaten vorliegt.
                        // z = alter z-wert
                        fprintf(stderr, "name: %s, posx:%f, posy:%f\n", str.str().c_str(), (float)theCOVER->transformations[i - 1][3 * 4], (float)-theCOVER->transformations[i - 1][3 * 4 + 2]);
                        inter->setVectorParam(str.str().c_str(), (float)theCOVER->transformations[i - 1][3 * 4], (float)-theCOVER->transformations[i - 1][3 * 4 + 2], ptr[2]);
                    }
                    else
                    {
                        cover->notify(Notify::Error, "wrong parameter type, float vector [3] expected!");
                    }
                    /* SCBooth
                       if(i-1 == 0)
                       inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[i-1][3*4], (float)-theCOVER->transformations[i-1][3*4+2], 1.5);
                       if(i-1 == 1)
                       inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[i-1][3*4], (float)-theCOVER->transformations[i-1][3*4+2], 0.38);
                       if(i-1 == 2)
                       inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[i-1][3*4], (float)-theCOVER->transformations[i-1][3*4+2], 0.38);
                       if(i-1 == 3)
                       inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[i-1][3*4], (float)-theCOVER->transformations[i-1][3*4+2], 0.38);
                       if(i-1 == 4)
                       inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[i-1][3*4], (float)-theCOVER->transformations[i-1][3*4+2], 0.92);
                       if(i-1 == 5)
                       inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[i-1][3*4], (float)-theCOVER->transformations[i-1][3*4+2], 0.92);
                       if(i-1 == 6)
                       inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[i-1][3*4], (float)-theCOVER->transformations[i-1][3*4+2], 0.92);
                       if(i-1 == 7)
                       inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[i-1][3*4], (float)-theCOVER->transformations[i-1][3*4+2], 0.67);
                       */
                }
            }
        }

        for (int i=0; i<20; ++i)
        {
            if (i==0)
                continue;

            std::stringstream str;
            str << "pos_rack_" << i;

            if (theCOVER)
            {
                if ((theCOVER->transformations[i - 1][3 * 4]) < 400 && (theCOVER->transformations[i - 1][3 * 4]) > -400 && (theCOVER->transformations[i - 1][3 * 4] != 0 || theCOVER->transformations[i - 1][3 * 4 + 1] != 0))
                {
                    int numElem = 0;
                    float *ptr;
                    if (inter->getFloatVectorParam(str.str(), numElem, ptr) == -1)
                        continue;
                    execNeeded = true;
                    if (numElem == 3)
                    {
                        // x = x
                        // y = -z (da die transformation in vrml koordinaten vorliegt.
                        // z = alter z-wert
                        fprintf(stderr, "name: %s, posx:%f, posy:%f\n", str.str().c_str(), (float)theCOVER->transformations[i - 1][3 * 4], (float)-theCOVER->transformations[i - 1][3 * 4 + 2]);
                        inter->setVectorParam(str.str().c_str(), (float)theCOVER->transformations[i - 1][3 * 4], (float)-theCOVER->transformations[i - 1][3 * 4 + 2], ptr[2]);
                    }
                    else
                    {
                        cover->notify(Notify::Error, "wrong parameter type, float vector [3] expected!");
                    }
                }
            }
        }

        if (execNeeded)
            inter->executeModule();
    }
}

COVERPLUGIN(TangiblePositionPlugin)
