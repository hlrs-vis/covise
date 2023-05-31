/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>

#include <osg/MatrixTransform>
#include <osg/Geode>

#include "CubePlugin.h"
#include <cover/RenderObject.h>
#include <cover/coInteractor.h>

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <string>

using namespace osg;
using std::string;

char *CubePlugin::currentObjectName = NULL;
CubePlugin *CubePlugin::plugin = NULL;

void CubePlugin::newInteractor(const RenderObject *cont, coInteractor *inter)
{
    if (cover->debugLevel(1))
        fprintf(stderr, "\n--- coVRNewInteractor containerName=[%s]\n", cont->getName());

    if (strcmp(inter->getPluginName(), "Cube") == 0)
    {
        // this example can only handle one interactor
        // --> if a new Cube Interactor arrives, it will be regarded
        if (CubePlugin::currentObjectName && string(cont->getName()).find(CubePlugin::currentObjectName) != 0)
        {
            CubePlugin::plugin->remove(CubePlugin::currentObjectName);
        }

        add(inter);

        //store the basename ModuleName_OUT_PortNo --> erasing index
        string contName(cont->getName());
        int stripIndex = contName.rfind("_");
        contName.erase(stripIndex, contName.length());

        currentObjectName = new char[strlen(contName.c_str()) + 1];
        strcpy(CubePlugin::currentObjectName, contName.c_str());
    }
}

void CubePlugin::removeObject(const char *contName, bool r)
{

    if (cover->debugLevel(1))
        fprintf(stderr, "\n--- coVRRemoveObject containerName=[%s]\n", contName);

    if (currentObjectName && contName)
    {
        // if object to be deleted is the interactor object then it has to be regarded
        if (string(contName).find(currentObjectName) == 0)
        {
            // replace is handeled in add
            if (!r)
            {
                remove(contName);
            }
        }
    }
}

//-----------------------------------------------------------------------------

CubePlugin::CubePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    // get the parameter names (this is hardcoded)
    plugin = this;
    centerParamName = "center";
    sizeParamName = "size";

    if (cover->debugLevel(3))
    {
        fprintf(stderr, "centerParamName=[%s]\n", centerParamName);
        fprintf(stderr, "sizeParamName=[%s]\n", sizeParamName);
    }
}

bool CubePlugin::init()
{
    if (cover->debugLevel(1))
        fprintf(stderr, "\n    new CubePlugin\n");
    inter = NULL;
    firsttime = true;

    cubeSubmenu = NULL;
    pinboardButton = NULL;
    sizePoti = NULL;
    moveCheckbox = NULL;

    // create the cube interactor
    wireCube = new CubeInteractor(coInteraction::ButtonA, "Cube", coInteraction::Medium);

    // and hide it, as long as 'move cube' is not selected
    wireCube->hide();

    return true;
}

CubePlugin::~CubePlugin()
{
    if (cover->debugLevel(1))
        fprintf(stderr, "\n    delete CubePlugin\n");

    if (pinboardButton)
        delete pinboardButton;
    deleteSubmenu();
    delete wireCube;

    inter->decRefCount();
    inter = NULL;
}

void
CubePlugin::createSubmenu()
{

    // create the submenu and attach it to the pinboard button
    cubeSubmenu = new coRowMenu("Cube");
    pinboardButton->setMenu(cubeSubmenu);

    // create the size poti
    sizePoti = new coPotiMenuItem("cube size", 1.0, 100.0, 10.0);
    sizePoti->setMenuListener(this);

    // create the checkbox menu item, start state is false
    // checkboxgroup=NULL ->togglebutton
    moveCheckbox = new coCheckboxMenuItem("move cube", false, NULL);
    moveCheckbox->setMenuListener(this);

    // add poti and checkbox to the menu
    cubeSubmenu->add(sizePoti);
    cubeSubmenu->add(moveCheckbox);
}

void
CubePlugin::deleteSubmenu()
{

    if (sizePoti)
    {
        delete sizePoti;
        sizePoti = NULL;
    }

    if (moveCheckbox)
    {
        delete moveCheckbox;
        moveCheckbox = NULL;
    }

    if (cubeSubmenu)
    {
        delete cubeSubmenu;
        cubeSubmenu = NULL;
    }
}

void
CubePlugin::add(coInteractor *in)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\n    CubePlugin::add\n");

    // if the interactor arrives firsttime
    if (firsttime)
    {
        firsttime = false;

        if (cover->debugLevel(3))
            fprintf(stderr, "firsttime\n");

        // create the button for the pinboard
        pinboardButton = new coSubMenuItem("Cube ...");

        // create submenu
        createSubmenu();

        // add the button to the pinboard
        cover->getMenu()->add(pinboardButton);

        // save the interactor for feedback
        inter = in;
        inter->incRefCount();
    }

    // get the size and center of the module
    int n;
    float *center, sizeMin, sizeMax, sizeVal;
    in->getFloatVectorParam(centerParamName, n, center);
    in->getFloatSliderParam(sizeParamName, sizeMin, sizeMax, sizeVal);

    if (cover->debugLevel(3))
    {
        fprintf(stderr, "center=[%f %f %f]\n", center[0], center[1], center[2]);
        fprintf(stderr, "size=[%f %f %f]\n", sizeMin, sizeMax, sizeVal);
    }

    // update the position and size of the cube interactor
    wireCube->setCenter(Vec3(center[0], center[1], center[2]));
    wireCube->setSize(sizeVal);

    // update the poti
    sizePoti->setValue(sizeVal);
    sizePoti->setMin(sizeMin);
    sizePoti->setMax(sizeMax);
}

void
CubePlugin::remove(const char * /*objName*/)
// remove is only called by coVRAddObject or coVRRemoveObject
// objName does not need to be regarded (already done)
{
    if (cover->debugLevel(1))
        fprintf(stderr, "\n    CubePlugin::remove\n");

    cover->getMenu()->remove(pinboardButton); // braucht man das ?
    if (pinboardButton)
        delete pinboardButton;

    deleteSubmenu();

    inter->decRefCount();

    printf("wirewireCube->registered: %d\n", wireCube->isRegistered());
    if (wireCube->isRegistered())
    {
        coInteractionManager::the()->unregisterInteraction(wireCube);
        sizePoti->setMenuListener(NULL);
        wireCube->hide();
    }
    firsttime = true;

    delete[] CubePlugin::currentObjectName;
    CubePlugin::currentObjectName = NULL;
}

void
CubePlugin::preFrame()
{
    Vec3 center;
    ///wireCube->update(); braucht man wohl nicht?

    if (wireCube->wasStopped())
    {
        center = wireCube->getCenter();
        inter->setVectorParam(centerParamName, center[0], center[1], center[2]);
        inter->executeModule();
        // beep
        fflush(stdout);
        fprintf(stdout, "\a");
        fflush(stdout);
    }
}

void CubePlugin::menuEvent(coMenuItem *menuItem)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "CubePlugin::menuEvent for %s\n", menuItem->getName());

    if (menuItem == sizePoti)
    {
        wireCube->setSize(sizePoti->getValue());
        if (moveCheckbox->getState())
        {
            inter->setSliderParam(sizeParamName, sizePoti->getMin(), sizePoti->getMax(), sizePoti->getValue());
            inter->executeModule();
        }
    }
    else //if (menuItem == moveCheckbox)
    {
        if (moveCheckbox->getState())
        {
            // enable interaction
            coInteractionManager::the()->registerInteraction(wireCube);
            // show the wireframe cube
            wireCube->show();
        }
        else
        {
            coInteractionManager::the()->unregisterInteraction(wireCube);
            wireCube->hide();
        }
    }
}

void CubePlugin::menuReleaseEvent(coMenuItem *menuItem)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "CubePlugin::menuReleaseEvent for %s\n", menuItem->getName());
    if (inter)
    {
        inter->setSliderParam(sizeParamName, sizePoti->getMin(), sizePoti->getMax(), sizePoti->getValue());
        inter->executeModule();
    }
}

COVERPLUGIN(CubePlugin)
