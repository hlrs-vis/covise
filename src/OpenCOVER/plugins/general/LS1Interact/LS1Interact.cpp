/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>

#include "LS1Interact.h"
#include <cover/RenderObject.h>
#include <cover/coInteractor.h>

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <string>

using namespace osg;
using std::string;

char *LS1Plugin::currentObjectName = NULL;
LS1Plugin *LS1Plugin::plugin = NULL;

void LS1Plugin::newInteractor(const RenderObject *cont, coInteractor *inter)
{
    if (cover->debugLevel(1))
        fprintf(stderr, "\n--- coVRNewInteractor containerName=[%s]\n", cont->getName());

    if (strcmp(inter->getPluginName(), "LS1") == 0)
    {
        if (LS1Plugin::currentObjectName && string(cont->getName()).find(LS1Plugin::currentObjectName) != 0)
        {
            LS1Plugin::plugin->remove(LS1Plugin::currentObjectName);
        }

        add(inter);

        //store the basename ModuleName_OUT_PortNo --> erasing index
        string contName(cont->getName());
        int stripIndex = contName.rfind("_");
        contName.erase(stripIndex, contName.length());

        currentObjectName = new char[strlen(contName.c_str()) + 1];
        strcpy(LS1Plugin::currentObjectName, contName.c_str());
    }
}

void LS1Plugin::removeObject(const char *contName, bool r)
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

LS1Plugin::LS1Plugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    TempParamName = "Temp";

    if (cover->debugLevel(3))
    {
        fprintf(stderr, "TempParamName=[%s]\n", TempParamName);
    }
}

bool LS1Plugin::init()
{
    if (cover->debugLevel(1))
        fprintf(stderr, "\n    new LS1Plugin\n");
    inter = NULL;
    firsttime = true;
    ls1Submenu = NULL;
    pinboardButton = NULL;
    tempPoti = NULL;

    return true;
}

bool LS1Plugin::destroy()
{
    return true;
}

LS1Plugin::~LS1Plugin()
{
    if (pinboardButton)
        delete pinboardButton;
    deleteSubmenu();

    inter->decRefCount();
    inter = NULL;
}

void LS1Plugin::createSubmenu()
{
    ls1Submenu = new coRowMenu("LS1");
    pinboardButton->setMenu(ls1Submenu);

    tempPoti = new coPotiMenuItem("LS1 temp", 0.1, 10.0, 1.0);
    tempPoti->setMenuListener(this);

    ls1Submenu->add(tempPoti);
}

void LS1Plugin::deleteSubmenu()
{

    if (tempPoti)
    {
        delete tempPoti;
        tempPoti = NULL;
    }

    if (ls1Submenu)
    {
        delete ls1Submenu;
        ls1Submenu = NULL;
    }
}

void LS1Plugin::add(coInteractor *in)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\n    LS1Plugin::add\n");
    if (firsttime)
    {
        firsttime = false;

        if (cover->debugLevel(3))
            fprintf(stderr, "firsttime\n");

        pinboardButton = new coSubMenuItem("LS1...");
        createSubmenu();
        cover->getMenu()->add(pinboardButton);
        inter = in;
        inter->incRefCount();
    }
    float MinTemp, MaxTemp, ValTemp;
    in->getFloatSliderParam(0, MinTemp, MaxTemp, ValTemp);

    if (cover->debugLevel(3))
    {
        fprintf(stderr, "Temperatures=[%f %f %f]\n", MinTemp, MaxTemp, ValTemp);
    }

    // Temp. Update
    tempPoti->setValue(ValTemp);
    tempPoti->setMin(MinTemp);
    tempPoti->setMax(MaxTemp);
}

void LS1Plugin::remove(const char *)
{
    if (cover->debugLevel(1))
        fprintf(stderr, "\n    LS1Plugin::remove\n");

    cover->getMenu()->remove(pinboardButton);
    if (pinboardButton)
        delete pinboardButton;

    deleteSubmenu();
    inter->decRefCount();
    firsttime = true;

    delete[] LS1Plugin::currentObjectName;
    LS1Plugin::currentObjectName = NULL;
}

void LS1Plugin::preFrame()
{
}

void LS1Plugin::menuEvent(coMenuItem *menuItem)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "LS1Plugin::menuEvent for %s\n", menuItem->getName());

    if (menuItem == tempPoti)
    {
        inter->setSliderParam(TempParamName, tempPoti->getMin(), tempPoti->getMax(), tempPoti->getValue());
        inter->executeModule();
    }
}

void LS1Plugin::menuReleaseEvent(coMenuItem *menuItem)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "LS1Plugin::menuReleaseEvent for %s\n", menuItem->getName());
    if (inter)
    {
        inter->setSliderParam(TempParamName, tempPoti->getMin(), tempPoti->getMax(), tempPoti->getValue());
        inter->executeModule();
    }
}
COVERPLUGIN(LS1Plugin)
