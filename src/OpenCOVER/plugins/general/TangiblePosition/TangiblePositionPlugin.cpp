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
#include <PluginUtil/coBaseCoviseInteractor.h>
#include <OpenVRUI/coToolboxMenu.h>

#include <vrml97/vrml/VrmlNodeCOVER.h>

#ifdef USE_COVISE
#include <appl/RenderInterface.h>
#endif

using namespace covise;
using vrml::VrmlNodeCOVER;
using vrml::theCOVER;

TangiblePositionPlugin::TangiblePositionPlugin()
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
TangiblePositionPlugin::newInteractor(RenderObject *, coInteractor *inter)
{
    if (strcmp(inter->getPluginName(), "TangiblePosition") == 0)
    {
        inter->incRefCount();
        interactors.push_back(inter);
        cerr << "removing Interactor from Module " << inter->getModuleName() << endl;
    }
}

void TangiblePositionPlugin::addObject(RenderObject *container,
                                       RenderObject *obj, RenderObject *normObj,
                                       RenderObject *colorObj, RenderObject *texObj,
                                       osg::Group *root,
                                       int numCol, int colorBinding, int colorPacking,
                                       float *r, float *g, float *b, int *packedCol,
                                       int numNormals, int normalBinding,
                                       float *xn, float *yn, float *zn, float transparency)
{
    (void)container;
    (void)obj;
    (void)normObj;
    (void)colorObj;
    (void)texObj;
    (void)root;
    (void)numCol;
    (void)colorBinding;
    (void)colorPacking;
    (void)r;
    (void)g;
    (void)b;
    (void)packedCol;
    (void)numNormals;
    (void)normalBinding;
    (void)xn;
    (void)yn;
    (void)zn;
    (void)transparency;
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
        coBaseCoviseInteractor *inter = dynamic_cast<coBaseCoviseInteractor *>(*it);
        if (!inter)
            continue;

        int i;
        for (i = 0; i < inter->getNumParam(); i++)
        {
            if (strncmp(inter->getParaName(i), "pos_cube_", 9) == 0)
            {
                int paramnum = 0;
                if (sscanf(inter->getParaName(i) + 9, "%d", &paramnum) != 1)
                {
                    cerr << "TangiblePositionPlugin::updateAndExec: sscanf failed" << endl;
                }
                if (theCOVER)
                {
                    if (paramnum < 9 && (theCOVER->transformations[paramnum - 1][3 * 4]) < 400 && (theCOVER->transformations[paramnum - 1][3 * 4]) > -400 && (theCOVER->transformations[paramnum - 1][3 * 4] != 0 || theCOVER->transformations[paramnum - 1][3 * 4 + 1] != 0))
                    {
                        int numElem = 0;
                        float *ptr;
                        inter->getFloatVectorParam(i, numElem, ptr);
                        if (numElem == 3)
                        {
                            // x = x
                            // y = -z (da die transformation in vrml koordinaten vorliegt.
                            // z = alter z-wert
                            fprintf(stderr, "name: %s, posx:%f, posy:%f\n", inter->getParaName(i), (float)theCOVER->transformations[paramnum - 1][3 * 4], (float)-theCOVER->transformations[paramnum - 1][3 * 4 + 2]);
                            inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[paramnum - 1][3 * 4], (float)-theCOVER->transformations[paramnum - 1][3 * 4 + 2], ptr[2]);
                        }
                        else
                        {
#ifdef USE_COVISE
                            CoviseRender::sendError("wrong parameter type, float vector [3] expected!");
#else
                            printf("wrong parameter type, float vector [3] expected!");
#endif
                        }
                        /* SCBooth
						if(paramnum-1 == 0)
						inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[paramnum-1][3*4], (float)-theCOVER->transformations[paramnum-1][3*4+2], 1.5);
						if(paramnum-1 == 1)
						inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[paramnum-1][3*4], (float)-theCOVER->transformations[paramnum-1][3*4+2], 0.38);
						if(paramnum-1 == 2)
						inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[paramnum-1][3*4], (float)-theCOVER->transformations[paramnum-1][3*4+2], 0.38);
						if(paramnum-1 == 3)
						inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[paramnum-1][3*4], (float)-theCOVER->transformations[paramnum-1][3*4+2], 0.38);
						if(paramnum-1 == 4)
						inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[paramnum-1][3*4], (float)-theCOVER->transformations[paramnum-1][3*4+2], 0.92);
						if(paramnum-1 == 5)
						inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[paramnum-1][3*4], (float)-theCOVER->transformations[paramnum-1][3*4+2], 0.92);
						if(paramnum-1 == 6)
						inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[paramnum-1][3*4], (float)-theCOVER->transformations[paramnum-1][3*4+2], 0.92);
						if(paramnum-1 == 7)
						inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[paramnum-1][3*4], (float)-theCOVER->transformations[paramnum-1][3*4+2], 0.67);
						*/
                    }
                }
            }
            else if (strncmp(inter->getParaName(i), "pos_rack_", 9) == 0)
            {
                int paramnum = 0;
                if (sscanf(inter->getParaName(i) + 9, "%d", &paramnum) != 1)
                {
                    cerr << "TangiblePositionPlugin::updateAndExec: sscanf failed" << endl;
                }
                if (theCOVER)
                {
                    if (paramnum < 20 && (theCOVER->transformations[paramnum - 1][3 * 4]) < 400 && (theCOVER->transformations[paramnum - 1][3 * 4]) > -400 && (theCOVER->transformations[paramnum - 1][3 * 4] != 0 || theCOVER->transformations[paramnum - 1][3 * 4 + 1] != 0))
                    {
                        int numElem = 0;
                        float *ptr;
                        inter->getFloatVectorParam(i, numElem, ptr);
                        if (numElem == 3)
                        {
                            // x = x
                            // y = -z (da die transformation in vrml koordinaten vorliegt.
                            // z = alter z-wert
                            fprintf(stderr, "name: %s, posx:%f, posy:%f\n", inter->getParaName(i), (float)theCOVER->transformations[paramnum - 1][3 * 4], (float)-theCOVER->transformations[paramnum - 1][3 * 4 + 2]);
                            inter->setVectorParam(inter->getParaName(i), (float)theCOVER->transformations[paramnum - 1][3 * 4], (float)-theCOVER->transformations[paramnum - 1][3 * 4 + 2], ptr[2]);
                        }
                        else
                        {
#ifdef USE_COVISE
                            CoviseRender::sendError("wrong parameter type, float vector [3] expected!");
#else
                            printf("wrong parameter type, float vector [3] expected!");
#endif
                        }
                    }
                }
            }
        }
        inter->executeModule();
    }
}

COVERPLUGIN(TangiblePositionPlugin)
