/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>

#include <osg/MatrixTransform>
#include <osg/Geode>

#include "SwitchDataPlugin.h"
#include <cover/RenderObject.h>
#include <cover/coInteractor.h>
#include <cover/coHud.h>

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>

#include <cover/coVRTui.h>

using namespace osg;
using namespace vrui;

void SwitchDataPlugin::newInteractor(const RenderObject *cont, coInteractor *inter)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- coVRNewInteractor containerName=[%s]\n", cont->getName());

    if (strcmp(inter->getPluginName(), "SwitchData") == 0)
    {
        // this example can only handle one interactor
        // --> if a new SwitchData Interactor arrives, it will be regarded
        if (currentObjectName && string(cont->getName()).find(currentObjectName) != 0)
        {
            //remove(currentObjectName, false);
        }
        else
        {
            add(inter);

            //store the basename ModuleName_OUT_PortNo --> erasing index
            string contName(cont->getName());
            int stripIndex = contName.rfind("_");
            contName.erase(stripIndex, contName.length());

            currentObjectName = new char[strlen(contName.c_str()) + 1];
            strcpy(currentObjectName, contName.c_str());
        }
    }
}

void SwitchDataPlugin::removeObject(const char *contName, bool r)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "\n--- coVRRemoveObject containerName=[%s], replace=%d\n", contName, (int)r);

    if (currentObjectName && contName)
    {
        // if object to be deleted is the interactor object then it has to be regarded
        //if (string(contName).find(currentObjectName) == 0) {
        if (!strcmp(currentObjectName, contName))
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

SwitchDataPlugin::SwitchDataPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, firsttime(true)
, inter(NULL)
, checkboxgroup(NULL)
, numChoices(0)
, choices(NULL)
, tuiTab(NULL)
, currentObjectName(NULL)
{
    // get the parameter names (this is hardcoded)
    choiceParamName = "switch";

    if (cover->debugLevel(4))
    {
        fprintf(stderr, "choiceParamName=[%s]\n", choiceParamName);
    }
}

bool SwitchDataPlugin::init()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n    new SwitchDataPlugin\n");

    firsttime = true;
    inter = NULL;

    pluginSubmenu = NULL;
    pinboardButton = NULL;

    tuiTab = new coTUITab("SwitchData", coVRTui::instance()->mainFolder->getID());
    tuiTab->setPos(0, 0);

    return true;
}

SwitchDataPlugin::~SwitchDataPlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n    delete SwitchDataPlugin\n");

    if (pinboardButton)
    {
        delete pinboardButton;
        pinboardButton = NULL;
    }

    deleteSubmenu();

    if (inter)
    {
        inter->decRefCount();
        inter = NULL;
    }
}

void
SwitchDataPlugin::createSubmenu(int numChoices, char **choiceValues, int currentChoice)
{
    if (!pinboardButton)
    {
        // create the button for the pinboard
        pinboardButton = new coSubMenuItem("SwitchData...");

        // add the button to the pinboard
        cover->getMenu()->add(pinboardButton);

        // create the submenu and attach it to the pinboard button
        if (!pluginSubmenu)
            pluginSubmenu = new coRowMenu("SwitchData");

        pinboardButton->setMenu(pluginSubmenu);
    }
    if (!checkboxgroup)
        checkboxgroup = new coCheckboxGroup();
    if (!tuiTab)
    {
        tuiTab = new coTUITab("SwitchData", coVRTui::instance()->mainFolder->getID());
        tuiTab->setPos(0, 0);
    }

    bool recreate = true;
    if (numChoices != this->numChoices)
    {
        recreate = true;
    }
    else
    {
        for (int i = 0; i < numChoices; ++i)
        {
            if (strcmp(choiceValues[i], this->choices[i]) != 0)
            {
                recreate = true;
                break;
            }
        }
    }

    this->numChoices = numChoices;
    this->choices = choiceValues;

    if (recreate)
    {
        while (!tuiButtons.empty())
        {
            delete tuiButtons.back();
            tuiButtons.pop_back();
        }

        while (!checkboxes.empty())
        {
            checkboxgroup->remove(checkboxes.back());
            delete checkboxes.back();
            checkboxes.pop_back();
        }

        // create the checkbox menu item, start state is false
        // checkboxgroup=NULL ->togglebutton
        for (int i = 0; i < numChoices; ++i)
        {
            coCheckboxMenuItem *box = new coCheckboxMenuItem(choiceValues[i], false, checkboxgroup);
            pluginSubmenu->add(box);
            box->setMenuListener(this);
            checkboxes.push_back(box);

            coTUIToggleButton *button = new coTUIToggleButton(choiceValues[i], tuiTab->getID());
            button->setPos(0, i);
            button->setEventListener(this);
            tuiButtons.push_back(button);
        }
    }
    if (currentChoice >= 0 && currentChoice < int(tuiButtons.size()))
    {
        checkboxgroup->setState(checkboxes[currentChoice], true);
        tuiButtons[currentChoice]->setState(true);
    }
}

void
SwitchDataPlugin::deleteSubmenu()
{

    while (!tuiButtons.empty())
    {
        delete tuiButtons.back();
        tuiButtons.pop_back();
    }
    delete tuiTab;
    tuiTab = NULL;

    while (!checkboxes.empty())
    {
        checkboxgroup->remove(checkboxes.back());
        delete checkboxes.back();
        checkboxes.pop_back();
    }

    numChoices = 0;
    choices = NULL;

    delete checkboxgroup;
    checkboxgroup = NULL;

    delete pluginSubmenu;
    pluginSubmenu = NULL;
}

void
SwitchDataPlugin::add(coInteractor *in)
{
    // if the interactor arrives firsttime
    if (firsttime)
    {
        firsttime = false;

        if (cover->debugLevel(4))
            fprintf(stderr, "firsttime\n");

        // save the interactor for feedback
        inter = in;
        inter->incRefCount();
    }

    if (cover->debugLevel(3))
        fprintf(stderr, "\n    SwitchDataPlugin::add\n");

    // get the size and center of the module
    int currentChoice, numChoices;
    char **choices;
    in->getChoiceParam("", numChoices, choices, currentChoice);
    // create submenu
    createSubmenu(numChoices, choices, currentChoice);

    if (cover->debugLevel(4))
    {
        fprintf(stderr, "choice=%d\n", currentChoice);
    }
}

void
SwitchDataPlugin::remove(const char * /*objName*/, bool removeMenu)
// remove is only called by coVRAddObject or coVRRemoveObject
// objName does not need to be regarded (already done)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\n    SwitchDataPlugin::remove\n");

    if (removeMenu)
    {
        if (pinboardButton)
        {
            cover->getMenu()->remove(pinboardButton); // braucht man das ?
            delete pinboardButton;
            pinboardButton = NULL;
        }

        deleteSubmenu();
    }

    inter->decRefCount();
    inter = NULL;

    firsttime = true;

    delete[] currentObjectName;
    currentObjectName = NULL;
}

void
SwitchDataPlugin::preFrame()
{
}

void
SwitchDataPlugin::showHud(const std::string &text)
{
    coHud *hud = coHud::instance();
    hud->setText1("Please wait:");
    hud->setText2((std::string("Loading ") + text + "...").c_str());
    hud->setText3("");
    hud->show();
    hud->redraw();
    hud->hideLater();
}

void SwitchDataPlugin::menuEvent(coMenuItem *menuItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "SwitchDataPlugin::menuEvent for %s\n", menuItem->getName());

    if (inter)
    {
        for (int i = 0; i < int(checkboxes.size()); ++i)
        {
            if (dynamic_cast<coCheckboxMenuItem *>(menuItem) == checkboxes[i])
            {
                inter->setChoiceParam(choiceParamName, numChoices, choices, i);
                inter->executeModule();
                showHud(choices[i]);
                break;
            }
        }
    }
}

void SwitchDataPlugin::tabletEvent(coTUIElement *tuiItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "SwitchDataPlugin::tabletEvent for %s\n", tuiItem->getName().c_str());

    if (inter)
    {
        for (int i = 0; i < int(tuiButtons.size()); ++i)
        {
            if (dynamic_cast<coTUIToggleButton *>(tuiItem) == tuiButtons[i])
            {
                inter->setChoiceParam(choiceParamName, numChoices, choices, i);
                inter->executeModule();
                showHud(choices[i]);
                break;
            }
        }
    }
}

void SwitchDataPlugin::menuReleaseEvent(coMenuItem *menuItem)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "SwitchDataPlugin::menuReleaseEvent for %s\n", menuItem->getName());
}

COVERPLUGIN(SwitchDataPlugin)
