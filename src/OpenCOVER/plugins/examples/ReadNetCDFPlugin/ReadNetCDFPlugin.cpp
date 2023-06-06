/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>

#include <osg/MatrixTransform>
#include <osg/Geode>

#include "ReadNetCDFPlugin.h"
#include <cover/RenderObject.h>
#include <cover/coInteractor.h>
#include <cover/coVRTui.h>

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <sstream>
#include <vector>

#define numParams 3 // FIXME: fixed # of Var. according to ReadNetCDF Module

using namespace osg;
using namespace opencover;
using namespace vrui;

char *CDFPlugin::currentObjectName = NULL;
CDFPlugin *CDFPlugin::plugin = NULL;

static const char *ParamFile = "NC_file";
static const char *ParamGridX = "GridOutX";
static const char *ParamGridY = "GridOutY";
static const char *ParamGridZ = "GridOutZ";
static std::vector<std::string> ParamVars;

// -----------------------------------------------------------------------------
// constructor
// -----------------------------------------------------------------------------
CDFPlugin::CDFPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    cbxGridOutX = NULL;
    cbxGridOutY = NULL;
    cbxGridOutZ = NULL;
    lblGridOutX = NULL;
    lblGridOutY = NULL;
    lblGridOutZ = NULL;

    for (int i = 0; i < numParams; ++i)
    {
        std::stringstream str;
        str << "Variable" << i;
        ParamVars.push_back(str.str());
    }
}

// -----------------------------------------------------------------------------
// destructor
// -----------------------------------------------------------------------------
CDFPlugin::~CDFPlugin()
{
    if (cover->debugLevel(1))
    {
        fprintf(stderr, "call CDFPlugin::~CDFPlugin()\n");
    }

    if (pinboardButton)
    {
        delete pinboardButton;
    }
    deleteSubmenu();

    for (vector<coTUIComboBox *>::iterator it = vecComboBox->begin();
         it != vecComboBox->end(); ++it)
    {
        delete *it;
    }
    for (vector<coTUILabel *>::iterator it = vecVarLabel->begin();
         it != vecVarLabel->end(); ++it)
    {
        delete *it;
    }

    delete paramTab;
    //delete sldTabSlider;
    //delete btnTabSwitch;
    delete fbbFileBrowser;
    delete lblFilename;
    delete lblGridOutX;
    delete lblGridOutY;
    delete lblGridOutZ;
    delete cbxGridOutX;
    delete cbxGridOutY;
    delete cbxGridOutZ;
    delete vecVarLabel;
    delete vecComboBox;

    interactor->decRefCount();
    interactor = NULL;
}

// -----------------------------------------------------------------------------
// init plugin
// -----------------------------------------------------------------------------
bool CDFPlugin::init()
{
    if (cover->debugLevel(1))
    {
        fprintf(stderr, "call CDFPlugin::init()\n");
    }

    // init of class attributes
    interactor = NULL;
    firsttime = true;
    cubeSubmenu = NULL;
    pinboardButton = NULL;
    sizePoti = NULL;
    vecVarLabel = new std::vector<coTUILabel *>;
    vecComboBox = new std::vector<coTUIComboBox *>;

    // init TabletUI
    paramTab = new coTUITab("ReadNetCDF",
                            coVRTui::instance()->mainFolder->getID());
    paramTab->setPos(0, 0);

    // sldTabSlider = new coTUIFloatSlider("sldTabSlider",paramTab->getID());
    // sldTabSlider->setEventListener(this);
    // sldTabSlider->setPos(0,1);
    // sldTabSlider->setMin(1);
    // sldTabSlider->setMax(100);
    // sldTabSlider->setValue(50);

    // btnTabSwitch = new coTUIButton("btnTabSwitch",paramTab->getID());
    // btnTabSwitch->setEventListener(this);
    // btnTabSwitch->setPos(0,2);

    lblNCFile = new coTUILabel("lblNCFile", paramTab->getID());
    lblNCFile->setPos(0, 0);
    lblNCFile->setLabel("NC File :");

    fbbFileBrowser = new coTUIFileBrowserButton("browse",
                                                paramTab->getID());
    fbbFileBrowser->setPos(1, 0);
    fbbFileBrowser->setEventListener(this);
    fbbFileBrowser->setMode(coTUIFileBrowserButton::OPEN);
    fbbFileBrowser->setFilterList("*.*");

    lblFilename = new coTUILabel("lblFilename",
                                 paramTab->getID());
    lblFilename->setPos(2, 0);
    lblFilename->setLabel("no data file selected");

    if (lblGridOutX != NULL)
    {
        delete lblGridOutX;
    }
    lblGridOutX = new coTUILabel("lblGridOutX", paramTab->getID());
    lblGridOutX->setPos(0, 1);
    lblGridOutX->setLabel("GridOutX :");

    if (cbxGridOutX != NULL)
    {
        delete cbxGridOutX;
    }
    cbxGridOutX = new coTUIComboBox("cbxGridOutX", paramTab->getID());
    cbxGridOutX->setPos(1, 1);
    cbxGridOutX->addEntry("---");
    cbxGridOutX->setSelectedEntry(0);
    cbxGridOutX->setEventListener(this);

    if (lblGridOutY != NULL)
    {
        delete lblGridOutY;
    }
    lblGridOutY = new coTUILabel("lblGridOutY", paramTab->getID());
    lblGridOutY->setPos(0, 2);
    lblGridOutY->setLabel("GridOutY :");

    if (cbxGridOutY != NULL)
    {
        delete cbxGridOutY;
    }
    cbxGridOutY = new coTUIComboBox("cbxGridOutY", paramTab->getID());
    cbxGridOutY->setPos(1, 2);
    cbxGridOutY->addEntry("---");
    cbxGridOutY->setSelectedEntry(0);
    cbxGridOutY->setEventListener(this);

    if (lblGridOutZ != NULL)
    {
        delete lblGridOutZ;
    }
    lblGridOutZ = new coTUILabel("lblGridOutZ", paramTab->getID());
    lblGridOutZ->setPos(0, 3);
    lblGridOutZ->setLabel("GridOutZ");

    if (cbxGridOutZ != NULL)
    {
        delete cbxGridOutZ;
    }
    cbxGridOutZ = new coTUIComboBox("cbxGridOutZ", paramTab->getID());
    cbxGridOutZ->setPos(1, 3);
    cbxGridOutZ->addEntry("---");
    cbxGridOutZ->setSelectedEntry(0);
    cbxGridOutZ->setEventListener(this);

    // delete old combo boxes and labels

    for (vector<coTUIComboBox *>::iterator it = vecComboBox->begin();
         it != vecComboBox->end(); ++it)
    {
        delete *it;
    }
    vecComboBox->clear();

    for (vector<coTUILabel *>::iterator it = vecVarLabel->begin();
         it != vecVarLabel->end(); ++it)
    {
        delete *it;
    }
    vecVarLabel->clear();

    // create new combo boxes and labels

    for (int i = 0; i < numParams; ++i)
    {
        ostringstream ss;
        ss << i;
        string strName = "lblVariables" + ss.str();
        string strTabName = "Variable " + ss.str() + " :";

        coTUILabel *lblNewLabel = new coTUILabel(strName.c_str(), paramTab->getID());
        lblNewLabel->setPos(0, i + 4);
        lblNewLabel->setLabel(strTabName.c_str());
        vecVarLabel->push_back(lblNewLabel);

        strName = "cbxVariables" + ss.str();

        coTUIComboBox *cbxNewCombo = new coTUIComboBox(strName.c_str(), paramTab->getID());
        cbxNewCombo->setPos(1, i + 4);
        cbxNewCombo->addEntry("---");
        cbxNewCombo->setSelectedEntry(0);
        cbxNewCombo->setEventListener(this);
        vecComboBox->push_back(cbxNewCombo);
    }
    return true;
}

// -----------------------------------------------------------------------------
// create new interactor
// -----------------------------------------------------------------------------
void CDFPlugin::newInteractor(const RenderObject *cont, coInteractor *inter)
{
    if (cover->debugLevel(1))
    {
        fprintf(stderr,
                "call CDFPlugin::newInteractor(), containerName=[%s]\n",
                cont->getName());
    }

    if (strcmp(inter->getPluginName(), "ReadNetCDFPlugin") == 0)
    {
        if (CDFPlugin::currentObjectName && string(cont->getName()).find(CDFPlugin::currentObjectName) != 0)
        {
            CDFPlugin::plugin->remove(CDFPlugin::currentObjectName);
        }

        add(inter);

        //store the basename ModuleName_OUT_PortNo --> erasing index
        string contName(cont->getName());
        int stripIndex = contName.rfind("_");
        contName.erase(stripIndex, contName.length());
        currentObjectName = new char[strlen(contName.c_str()) + 1];
        strcpy(CDFPlugin::currentObjectName, contName.c_str());
    }
}

// -----------------------------------------------------------------------------
// add
// -----------------------------------------------------------------------------
void CDFPlugin::add(coInteractor *inter)
{
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "    CDFPlugin::add\n");
    }

    // if the interactor arrives firsttime
    if (firsttime)
    {
        firsttime = false;

        if (cover->debugLevel(3))
            fprintf(stderr, "firsttime\n");

        // create the button for the pinboard
        pinboardButton = new coSubMenuItem("NetCDF...");

        // create submenu
        createSubmenu();

        // add the button to the pinboard
        cover->getMenu()->add(pinboardButton);

        // save the interactor for feedback
        interactor = inter;
        interactor->incRefCount();
    }

    // get  filename from interactor
    char *filename;
    inter->getFileBrowserParam(ParamFile, filename);

    string strg = string(filename);
    int found = strg.find(" ");
    strg.copy(filename, 0, int(found));
    filename[int(found)] = '\0';
    lblFilename->setLabel(filename);
    fbbFileBrowser->setCurDir(filename);

    // get choice parameters from interactor

    int num;
    char **labels;
    int active;
    inter->getChoiceParam(ParamGridX, num, labels, active);
    if (cbxGridOutX != NULL)
    {
        delete cbxGridOutX;
    }
    cbxGridOutX = new coTUIComboBox("cbxGridOutX", paramTab->getID());
    cbxGridOutX->setPos(1, 1);
    for (int i = 0; i < num; ++i)
    {
        cbxGridOutX->addEntry(labels[i]);
    }
    cbxGridOutX->setSelectedEntry(active);
    cbxGridOutX->setEventListener(this);

    if (cbxGridOutY != NULL)
    {
        delete cbxGridOutY;
    }
    inter->getChoiceParam(ParamGridY, num, labels, active);
    cbxGridOutY = new coTUIComboBox("cbxGridOutY", paramTab->getID());
    cbxGridOutY->setPos(1, 2);
    for (int i = 0; i < num; ++i)
    {
        cbxGridOutY->addEntry(labels[i]);
    }
    cbxGridOutY->setSelectedEntry(active);
    cbxGridOutY->setEventListener(this);

    if (cbxGridOutZ != NULL)
    {
        delete cbxGridOutZ;
    }
    inter->getChoiceParam(ParamGridZ, num, labels, active);
    cbxGridOutZ = new coTUIComboBox("cbxGridOutZ", paramTab->getID());
    cbxGridOutZ->setPos(1, 3);
    for (int i = 0; i < num; ++i)
    {
        cbxGridOutZ->addEntry(labels[i]);
    }
    cbxGridOutZ->setSelectedEntry(active);
    cbxGridOutZ->setEventListener(this);

    // delete comboboxes and labels

    for (vector<coTUIComboBox *>::iterator it = vecComboBox->begin();
         it != vecComboBox->end(); ++it)
    {
        delete *it;
    }
    vecComboBox->clear();

    for (vector<coTUILabel *>::iterator it = vecVarLabel->begin();
         it != vecVarLabel->end(); ++it)
    {
        delete *it;
    }
    vecVarLabel->clear();

    // create new comboboxes and labels with input from interactor

    for (int i = 0; i < numParams; ++i)
    {
        ostringstream ss;
        ss << i;
        string strName = "lblVariables" + ss.str();
        string strTabName = "Variable " + ss.str() + " :";

        inter->getChoiceParam(ParamVars[i], num, labels, active);

        coTUILabel *lblNewLabel = new coTUILabel(strName.c_str(), paramTab->getID());
        lblNewLabel->setPos(0, i + 4);
        lblNewLabel->setLabel(strTabName.c_str());
        vecVarLabel->push_back(lblNewLabel);

        strName = "cbxVariables" + ss.str();

        coTUIComboBox *cbxNewCombo = new coTUIComboBox(strName.c_str(), paramTab->getID());
        cbxNewCombo->setPos(1, i + 4);
        for (int i = 0; i < num; ++i)
        {
            cbxNewCombo->addEntry(labels[i]);
        }
        cbxNewCombo->setSelectedEntry(active);
        cbxNewCombo->setEventListener(this);
        vecComboBox->push_back(cbxNewCombo);
    }
}

// -----------------------------------------------------------------------------
// remove object
// -----------------------------------------------------------------------------
void CDFPlugin::removeObject(const char *contName, bool r)
{
    if (cover->debugLevel(1))
    {
        fprintf(stderr,
                "call CDFPlugin::removeObject(), containerName=[%s]\n",
                contName);
    }

    // if object to be deleted is the interactor
    // object then it has to be regarded
    if (currentObjectName && contName)
    {
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

// -----------------------------------------------------------------------------
// remove
// remove is only called by coVRAddObject or coVRRemoveObject
// objName does not need to be regarded (already done)
// -----------------------------------------------------------------------------
void CDFPlugin::remove(const char *)
{
    if (cover->debugLevel(1))
    {
        fprintf(stderr, "    CDFPlugin::remove\n");
    }

    cover->getMenu()->remove(pinboardButton);
    if (pinboardButton)
    {
        delete pinboardButton;
    }

    deleteSubmenu();

    interactor->decRefCount();
    firsttime = true;

    delete[] CDFPlugin::currentObjectName;
    CDFPlugin::currentObjectName = NULL;
}

// -----------------------------------------------------------------------------
// createSubmenu
// -----------------------------------------------------------------------------
void CDFPlugin::createSubmenu()
{
    // create the submenu and attach it to the pinboard button
    cubeSubmenu = new coRowMenu("NetCDF");
    pinboardButton->setMenu(cubeSubmenu);

    // create checkboxes
    cbxSwitch = new coCheckboxMenuItem("switch", false, NULL);
    cbxSwitch->setMenuListener(this);

    // create the size poti
    sizePoti = new coPotiMenuItem("cdf size", 1.0, 100.0, 10.0);
    sizePoti->setMenuListener(this);

    // create slider
    sldSlider = new coSliderMenuItem("Slider", 1.0, 10.0, 5.0);
    sldSlider->setMenuListener(this);

    // add poti and checkbox to the menu
    cubeSubmenu->add(cbxSwitch);
    cubeSubmenu->add(sldSlider);
    cubeSubmenu->add(sizePoti);
}

// -----------------------------------------------------------------------------
// deleteSubmenu
// -----------------------------------------------------------------------------
void CDFPlugin::deleteSubmenu()
{
    if (sizePoti)
    {
        delete sizePoti;
        sizePoti = NULL;
    }

    if (cbxSwitch)
    {
        delete cbxSwitch;
        cbxSwitch = NULL;
    }

    if (cubeSubmenu)
    {
        delete cubeSubmenu;
        cubeSubmenu = NULL;
    }
}

// -----------------------------------------------------------------------------
// menuEvent
// -----------------------------------------------------------------------------
void CDFPlugin::menuEvent(coMenuItem *menuItem)
{
    if (cover->debugLevel(3))
    {
        fprintf(stderr, "call CDFPlugin::menuEvent for %s\n",
                menuItem->getName());
    }

    if (menuItem == cbxSwitch)
    {
        if (cbxSwitch->getState())
        {
            fprintf(stderr, "     cbxSwitch on\n");
        }
        else
        {
            fprintf(stderr, "     cbxSwitch off\n");
        }
    }
}

// -----------------------------------------------------------------------------
// tabletEvent
// -----------------------------------------------------------------------------
void CDFPlugin::tabletEvent(coTUIElement *tUIItem)
{
    // if (tUIItem == sldTabSlider)
    // {
    // if (cover->debugLevel(1))
    // {
    //    fprintf(stderr,"    CDFPlugin::tabletEvent (sldSlider)\n");
    // }
    // }
    if (tUIItem == fbbFileBrowser)
    {
        if (cover->debugLevel(1))
        {
            fprintf(stderr, "    CDFPlugin::tabletEvent (fbbFileBrowser)\n");
        }

        lblFilename->setLabel(fbbFileBrowser->getSelectedPath());
        interactor->setStringParam(ParamFile,
                                   fbbFileBrowser->getSelectedPath().c_str());
        interactor->executeModule();
    }

    if (tUIItem == cbxGridOutX)
    {
        if (cover->debugLevel(1))
        {
            fprintf(stderr, "    CDFPlugin::tabletEvent (cbxGridOutX)\n");
        }
        int num;
        char **labels;
        int active;
        interactor->getChoiceParam(ParamGridX, num, labels, active);
        interactor->setChoiceParam(ParamGridX,
                                   num, labels,
                                   cbxGridOutX->getSelectedEntry());
        interactor->executeModule();
    }
    if (tUIItem == cbxGridOutY)
    {
        if (cover->debugLevel(1))
        {
            fprintf(stderr, "    CDFPlugin::tabletEvent (cbxGridOutY)\n");
        }
        int num;
        char **labels;
        int active;
        interactor->getChoiceParam(ParamGridY, num, labels, active);
        interactor->setChoiceParam(ParamGridY,
                                   num, labels,
                                   cbxGridOutY->getSelectedEntry());
        interactor->executeModule();
    }
    if (tUIItem == cbxGridOutZ)
    {
        if (cover->debugLevel(1))
        {
            fprintf(stderr, "    CDFPlugin::tabletEvent (cbxGridOutZ)\n");
        }
        int num;
        char **labels;
        int active;
        interactor->getChoiceParam(ParamGridZ, num, labels, active);
        interactor->setChoiceParam(ParamGridZ,
                                   num, labels,
                                   cbxGridOutZ->getSelectedEntry());
        interactor->executeModule();
    }
    int i = 0;
    for (vector<coTUIComboBox *>::iterator it = vecComboBox->begin();
         it != vecComboBox->end(); ++it)
    {
        if (tUIItem == *it)
        {
            fprintf(stderr, "    CDFPlugin::tabletEvent (Var-ComboBox)\n");

            int num;
            char **labels;
            int active;
            interactor->getChoiceParam(ParamVars[i], num, labels, active);
            interactor->setChoiceParam(ParamVars[i].c_str(),
                                       num, labels,
                                       (*it)->getSelectedEntry());
            interactor->executeModule();
        }
        ++i;
    }
}

// -----------------------------------------------------------------------------
// tabletPressEvent
// -----------------------------------------------------------------------------
void CDFPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    // if (tUIItem == btnTabSwitch)
    // {
    //    if (cover->debugLevel(1))
    // {
    //    fprintf(stderr,"    CDFPlugin::tabletEvent (btnSwitch)\n");
    // }
    // }
}

// -----------------------------------------------------------------------------

COVERPLUGIN(CDFPlugin)
