/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   PDB
//
// Author:        Philip Weber
//
// Creation Date: 2005-11-29
//
// **************************************************************************

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coRectButtonGeometry.h>
#include <OpenVRUI/coValuePoti.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coSlider.h>
#include <OpenVRUI/sginterface/vruiButtons.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRNavigationManager.h>
#include <string>
#ifndef WIN32
#include <sys/time.h>
#include <unistd.h>
#else
#include <sys/timeb.h>
#include <direct.h>
#endif
#include <util/unixcompat.h>
#include <osg/Node>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osgDB/ReadFile>
#include <osgUtil/Optimizer>
#include <osgUtil/Statistics>

#include <vrml97/vrml/Player.h>
#include <vrml97/vrml/VrmlNodeCOVER.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlMFString.h>
#include <vrml97/vrml/Doc.h>
#include <vrml97/vrml/coEventQueue.h>
#include <vrml97/vrml/VrmlNamespace.h>

#include <config/CoviseConfig.h>
#include <iostream>
#include <fstream>
#include <math.h>

#include "../../general/Vrml97/SystemCover.h"
#include "../../general/Vrml97/ViewerOsg.h"

#include <osgGA/GUIEventAdapter>
#include <virvo/vvtoolshed.h>

#include <cover/RenderObject.h>

// CUI:
#include <osgcaveui/FileBrowser.h>
#include <osgcaveui/CUI.h>

#include "PDBPlugin.h"
#ifdef WIN32
#pragma warning (disable : 4018)
#endif

using namespace osg;
using namespace osgDB;
using namespace std;
using namespace cui;

using vrui::vruiButtons;
using covise::coCoviseConfig;

#define PI 3.14159265

coVRPlugin *PDBPluginModule = NULL;

LowDetailTransVisitor *PDBPickBox::lowdetailVisitor = NULL;
HighDetailTransVisitor *PDBPickBox::highdetailVisitor = NULL;
FrameVisitor *PDBPickBox::frameVisitor = NULL;

// reference array for use with dials
static const char REFARRAY[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static const int NUM_DIALS = 4;
static const int MESSAGE_PREFRAME = 4;
static const string WGET("wget -np ");
static const string WGETCGI("wget -O /dev/null \"www.molmovdb.org/cgi-bin/download.cgi?ID=");
static const string WGETLOC("www.molmovdb.org/tmp/");
//static const string CYLINDER("cyl");
//static const string HELIX("hel");

static const string RIBBON("rib");
static const string STICK("stix");
static const string CARTOON("cart");
static const string SURFACE("surf");

static const string PDB_EXT(".pdb");
#ifdef WIN32
static const string PYMOL_exe("pymol.exe");
#else
static const string PYMOL_exe("pymol");
#endif
static const string PYMOL_opt = "-qcr";
//static const string PYMOL_script = "pymol-pdb2wrl-batch.py";
static const string PYMOL_script = "batch.py";
static string PYMOL;
static const string WRL_EXT(".wrl");
static const string PROTEIN("Protein: ");
#ifdef WIN32
static const string REMOVE("del ");
static const string TAR("c:\\programme\\7-zip\\7z.exe ");
#else
static const string REMOVE("rm ");
static const string TAR("tar -xzf ");
#endif
static const string TAR_EXT(".tar.gz");
static const string LOADING_MESSAGE("Loading PDB morph...");

static const string ERROR_LOADING("Error reading structure ");
//static const string PDB_ROOT("PDB_ROOT");
PDBPlugin *plugin = NULL;
COVERPLUGIN(PDBPlugin)

//-----------------------------------------
// class PDBPlugin
//-----------------------------------------

/// Constructor
PDBPlugin::PDBPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, viewer(NULL)
, vrmlScene(NULL)
, player(NULL)
{
    uidcounter = 0;
}

bool PDBPlugin::init()
{
    plugin = this;
    PDBPluginModule = this;

    PYMOL = PYMOL_exe + " " + PYMOL_opt + " " + getenv("COVISEDIR") + "/scripts/pdb/" + PYMOL_script + " -- ";

    // default button size
    float proteinButtonSize[] = { 80, 15 };

    // protein menu positions
    float px[] = { 0, 60 };
    float py[] = { 135, 120, 105, 90, 60, 45, 30, 15, 0, 75 };

    // pdb loader menu
    float x[6] = { 0, 20, 40, 60, 80, 100 };
    float potiSize = 0.3;
    float y[3] = { 75, 60, 45 };
    float z = 1;
    float labelSize = 8;
    float buttonSize[2] = { 80, 20 };

    // get host name
    char temp[2000];
    if (gethostname(temp, sizeof(temp)) == -1)
    {
        myHost.append("NoName");
    }
    else
    {
        myHost.append(temp);
    }

    screenResizeDelay = 0;
    loadDelay = -1; // initalize load delay
    loadingMorph = false;
    currentFrame = 0; // initalize current frame
    _selectedPickBox = NULL;
    _restrictedMode = false;
    _galaxyMode = false;

    // coVRPluginSupport

    //create panel
    panel = new coPanel(new coFlatPanelGeometry(coUIElement::BLACK));
    handle = new coPopupHandle("PDBLoader");

    //set currentPath
    const char *current_path = getcwd(NULL, 0);
    currentPath = current_path;

    //set default mol directory
    relativePymolPath = coCoviseConfig::getEntry("COVER.Plugin.PDB.PDBPluginPymolDir");
    if (relativePymolPath.empty())
    {
        cerr << "WARNING: PDBPlugin.PDBPluginPymolDir is not specified in the config file" << endl;
    }

    //set radius for galaxy
    _radius = coCoviseConfig::getFloat("COVER.Plugin.PDB.Radius", 3.0);

    //set radius for galaxy
    _maxstructures = coCoviseConfig::getInt("COVER.Plugin.PDB.MaxStructures", 150);

    //set file size (galaxy)
    _filesize = coCoviseConfig::getFloat("COVER.Plugin.PDB.FileSize", 150000.0);

    //set file size (galaxy)
    _ringnum = (int)coCoviseConfig::getFloat("COVER.Plugin.PDB.NumPerRing", 12.0);

    //set default fade distance
    fadedist = coCoviseConfig::getFloat("COVER.Plugin.PDB.FadeDistance", 1.0);

    //set default viewer distance
    viewdist = coCoviseConfig::getFloat("COVER.Plugin.PDB.ViewerDistance", 1.0);

    //set default viewer distance
    _defaultScale = coCoviseConfig::getFloat("COVER.Plugin.PDB.Scale", 1.0);

    //set default marker size
    _markerSize = coCoviseConfig::getFloat("COVER.Plugin.PDB.MarkerSize", 70.0);

    //set default temp directory
    relativeTempPath = coCoviseConfig::getEntry("COVER.Plugin.PDB.PDBPluginTempDir");
    if (relativeTempPath.empty())
    {
        cerr << "WARNING: COVER.Plugin.PDB.PDBPluginTempDir is not specified in the config file" << endl;
        relativeTempPath = current_path;
    }

    //set pdb url
    auto pdbURL = coCoviseConfig::getEntry("COVER.Plugin.PDB.PDBUrl");
    if (pdbURL.empty())
    {
        pdburl = "www.pdb.org/pdb/files/";
    }
    else
    {
        pdburl = pdbURL.c_str();
    }

    //set animation url
    animationurl = coCoviseConfig::getEntry("COVER.Plugin.PDB.AnimationURL");
    if (animationurl.empty())
    {
        animationurl = "www.molmovdb.org/uploads/";
    }

    topsan = new TopsanViewer();
    sview = new SequenceViewer(this);

    //create a pop up menu for structure interaction
    proteinHandle = new coPopupHandle("ProteinControl");
    proteinFrame = new coFrame("UI/Frame");
    proteinPanel = new coPanel(new coFlatPanelGeometry(coUIElement::BLACK));
    proteinLabel = new coLabel;
    proteinLabel->setString(PROTEIN);
    proteinLabel->setPos(px[0], py[0], z);
    proteinLabel->setFontSize(labelSize);
    proteinHandle->addElement(proteinFrame);
    proteinDeleteButton = new coPushButton(new coRectButtonGeometry(proteinButtonSize[0], proteinButtonSize[1], "PDBPlugin/delete"), this);
    proteinResetScaleButton = new coPushButton(new coRectButtonGeometry(proteinButtonSize[0], proteinButtonSize[1], "PDBPlugin/reset"), this);
    alignButton = new coPushButton(new coRectButtonGeometry(proteinButtonSize[0], proteinButtonSize[1], "PDBPlugin/align"), this);

    coloroverrideCheckBox = new coToggleButton(new coFlatButtonGeometry("UI/haken"), this);
    coloroverrideLabel = new coLabel;
    coloroverrideLabel->setString("Hue 0verride");
    coloroverrideLabel->setPos(px[0], py[5], z);
    coloroverrideLabel->setFontSize(labelSize);
    coloroverrideCheckBox->setSize(8);
    coloroverrideCheckBox->setPos(50, py[5], z);
    coloroverrideCheckBox->setState(false, false);

    surfaceCheckBox = new coToggleButton(new coFlatButtonGeometry("UI/haken"), this);
    surfaceLabel = new coLabel;
    surfaceLabel->setString("Surface view");
    surfaceLabel->setPos(px[0], py[1], z);
    surfaceLabel->setFontSize(labelSize);
    surfaceCheckBox->setSize(8);
    surfaceCheckBox->setPos(50, py[1], z);
    surfaceCheckBox->setState(false, false);

    cartoonCheckBox = new coToggleButton(new coFlatButtonGeometry("UI/haken"), this);
    cartoonLabel = new coLabel;
    cartoonLabel->setString("Cartoon view");
    cartoonLabel->setPos(px[0], py[2], z);
    cartoonLabel->setFontSize(labelSize);
    cartoonCheckBox->setSize(8);
    cartoonCheckBox->setPos(50, py[2], z);
    cartoonCheckBox->setState(true, false); // initalize the cartoon view as the default

    stickCheckBox = new coToggleButton(new coFlatButtonGeometry("UI/haken"), this);
    stickLabel = new coLabel;
    stickLabel->setString("Stick view");
    stickLabel->setPos(px[0], py[3], z);
    stickLabel->setFontSize(labelSize);
    stickCheckBox->setSize(8);
    stickCheckBox->setPos(50, py[3], z);
    stickCheckBox->setState(false, false);

    ribbonCheckBox = new coToggleButton(new coFlatButtonGeometry("UI/haken"), this);
    ribbonLabel = new coLabel;
    ribbonLabel->setString("Ribbon view");
    ribbonLabel->setPos(px[0], py[9], z);
    ribbonLabel->setFontSize(labelSize);
    ribbonCheckBox->setSize(8);
    ribbonCheckBox->setPos(50, py[9], z);
    ribbonCheckBox->setState(false, false);

    sequenceCheckBox = new coToggleButton(new coFlatButtonGeometry("UI/haken"), this);
    sequenceLabel = new coLabel;
    sequenceLabel->setString("Seq Mark");
    sequenceLabel->setPos(px[0], py[4], z);
    sequenceLabel->setFontSize(labelSize);
    sequenceCheckBox->setSize(8);
    sequenceCheckBox->setPos(50, py[4], z);
    sequenceCheckBox->setState(false, false);

    proteinScalePoti = new coValuePoti("Scale", this, "Volume/valuepoti-bg");
    proteinHuePoti = new coValuePoti("Hue", this, "Volume/valuepoti-bg");
    proteinResetScaleButton->setSize(30);
    proteinDeleteButton->setSize(30);
    alignButton->setSize(30);
    proteinResetScaleButton->setPos(px[0], py[7], z);
    alignButton->setPos(px[0], py[6], z);
    proteinDeleteButton->setPos(px[0], py[8], z);
    proteinScalePoti->setPos(px[1], py[7], z);
    proteinScalePoti->setMin(0);
    proteinScalePoti->setMax(10);
    proteinHuePoti->setPos(px[1], py[5], z);
    proteinHuePoti->setMin(0);
    proteinHuePoti->setMax(1);
    proteinPanel->addElement(proteinLabel);
    proteinPanel->addElement(coloroverrideLabel);
    proteinPanel->addElement(coloroverrideCheckBox);

    proteinPanel->addElement(ribbonLabel);
    proteinPanel->addElement(ribbonCheckBox);
    proteinPanel->addElement(stickLabel);
    proteinPanel->addElement(stickCheckBox);
    proteinPanel->addElement(cartoonLabel);
    proteinPanel->addElement(cartoonCheckBox);
    proteinPanel->addElement(surfaceLabel);
    proteinPanel->addElement(surfaceCheckBox);
    proteinPanel->addElement(sequenceLabel);
    proteinPanel->addElement(sequenceCheckBox);

    //proteinPanel->addElement(alignButton);
    proteinPanel->addElement(proteinResetScaleButton);
    proteinPanel->addElement(proteinDeleteButton);
    proteinPanel->addElement(proteinScalePoti);
    proteinPanel->addElement(proteinHuePoti);
    proteinPanel->setScale(5);
    proteinPanel->resize();
    proteinFrame->addElement(proteinPanel);
    proteinHandle->setVisible(false);

    //create message panel
    messageHandle = new coPopupHandle("Messages");
    messageLabel = new coLabel();
    messageLabel->setString(ERROR_LOADING);
    messageLabel->setPos(0, 0, z);
    messageLabel->setFontSize(50);

    //attach a switch node to the root
    mainNode = new osg::Switch();
    mainNode->setAllChildrenOn();

    cover->getObjectsRoot()->addChild((osg::Node *)mainNode.get());

    // create arrays for labels and dials
    labelChar = new coLabel *[NUM_DIALS];
    potiChar = new coValuePoti *[NUM_DIALS];
    stringChar = new string *[NUM_DIALS];

    // Create dial for reading a value in
    // add first set of dials
    for (int i = 0; i < NUM_DIALS; i++)
    {
        // create a dial that will be used to set a character for loading user input
        potiChar[i] = new coValuePoti("", this, "Volume/valuepoti-bg");
        potiChar[i]->setLabelVisible(false);
        potiChar[i]->setPos(x[i], y[0], z);
        potiChar[i]->setSize(potiSize);
        potiChar[i]->setMin(0);
        potiChar[i]->setMax(35);
        potiChar[i]->setValue((i == 0) ? 4 : ((i == 1) ? 17 : ((i == 2) ? 17 : 11)));
        potiChar[i]->setInteger(true);

        // initialize the strings for the labels
        stringChar[i] = new string(1, REFARRAY[((int)potiChar[i]->getValue())]);

        // create a label show output of matching indexed dial
        labelChar[i] = new coLabel();
        labelChar[i]->setString(*stringChar[i]);
        labelChar[i]->setPos(x[i], y[1], z);
        labelChar[i]->setFontSize(labelSize);

        // add items to the panel
        panel->addElement(labelChar[i]);
        panel->addElement(potiChar[i]);
    }

    //create a PDBOption menu
    selectMenu = new coSubMenuItem("PDB Viewer");
    structureMenu = new coSubMenuItem("Proteins");
    animationMenu = new coSubMenuItem("Protein Morphs");
    faderMenu = new coSubMenuItem("Protein Fader");
    selectRow = new coRowMenu("PDB Viewer");
    structureRow = new coRowMenu("Proteins");
    animationRow = new coRowMenu("Protein Morphs");
    faderRow = new coRowMenu("Protein Fader");
    loaderMenuCheckbox = new coCheckboxMenuItem("Loader", false);
    //alignCheckbox = new coCheckboxMenuItem("Align Point", false);
    distviewMenuPoti = new coPotiMenuItem("Distance from viewer", 0, 200, viewdist);
    fadeareaMenuPoti = new coPotiMenuItem("Fade area", 0, 50, fadedist);
    faderesetButton = new coButtonMenuItem("Reset Settings");
    fadeonCheckbox = new coCheckboxMenuItem("Fade On", false);
    highdetailCheckbox = new coCheckboxMenuItem("High Detail", true);
    clearSceneButton = new coButtonMenuItem("Delete All");
    scalePoti = new coPotiMenuItem("Protein size", 0, 10, 1);
    movableCheckbox = new coCheckboxMenuItem("Movable", true);
    browserButton = new coButtonMenuItem("Protein Browser");
    layoutMenuItem = new coSubMenuItem("Layout Menu");
    layoutMenu = new coRowMenu("Layout Menu");
    saveLayoutMenuItem = new coSubMenuItem("Save Layout");
    saveLayoutMenu = new coRowMenu("Save Layout");
    loadLayoutMenuItem = new coSubMenuItem("Load Layout");
    loadLayoutMenu = new coRowMenu("Load Layout");
    layoutButtonGrid = new coButtonMenuItem("Grid layout");
    layoutButtonCylinder = new coButtonMenuItem("Cylinder layout");
    radiusPoti = new coPotiMenuItem("Spacing/Radius", 1, 10000, 1000);
    proteinsPerLevelPoti = new coPotiMenuItem("Proteins Per Level", 1, 15, 5);
    resetButton = new coButtonMenuItem("Clear Alignment");
    nameVisibleCheckbox = new coCheckboxMenuItem("Show name", false);
    topsanVisibleCheckbox = new coCheckboxMenuItem("Show Topsan Info", false);
    sviewVisibleCheckbox = new coCheckboxMenuItem("Sequence Viewer", false);
    alVisibleCheckbox = new coCheckboxMenuItem("Alignment Menu", false);

    selectRow->add(structureMenu);
    selectRow->add(animationMenu);
    selectRow->add(loaderMenuCheckbox);
    //selectRow->add(faderMenu);
    //selectRow->add(alignCheckbox);
    selectRow->add(alVisibleCheckbox);
    selectRow->add(browserButton);
    //selectRow->add(highdetailCheckbox);
    selectRow->add(scalePoti);
    selectRow->add(layoutMenuItem);
    layoutMenuItem->setMenu(layoutMenu);
    layoutMenu->add(loadLayoutMenuItem);
    loadLayoutMenuItem->setMenu(loadLayoutMenu);
    layoutMenu->add(saveLayoutMenuItem);
    saveLayoutMenuItem->setMenu(saveLayoutMenu);
    layoutMenu->add(layoutButtonGrid);
    layoutMenu->add(layoutButtonCylinder);
    layoutMenu->add(radiusPoti);
    layoutMenu->add(proteinsPerLevelPoti);
    selectRow->add(nameVisibleCheckbox);
    selectRow->add(topsanVisibleCheckbox);
    selectRow->add(sviewVisibleCheckbox);
    selectRow->add(resetButton);
    selectRow->add(clearSceneButton);
    //selectRow->add(movableCheckbox);
    selectMenu->setMenu(selectRow);
    structureMenu->setMenu(structureRow);
    animationMenu->setMenu(animationRow);
    faderMenu->setMenu(faderRow);
    faderRow->add(fadeonCheckbox);
    faderRow->add(distviewMenuPoti);
    faderRow->add(fadeareaMenuPoti);
    faderRow->add(faderesetButton);
    layoutMenuItem->setMenuListener(this);
    saveLayoutMenuItem->setMenuListener(this);
    loadLayoutMenuItem->setMenuListener(this);
    loaderMenuCheckbox->setMenuListener(this);
    //alignCheckbox->setMenuListener(this);
    fadeonCheckbox->setMenuListener(this);
    distviewMenuPoti->setMenuListener(this);
    fadeareaMenuPoti->setMenuListener(this);
    //highdetailCheckbox->setMenuListener(this);
    faderesetButton->setMenuListener(this);
    clearSceneButton->setMenuListener(this);
    layoutButtonGrid->setMenuListener(this);
    layoutButtonCylinder->setMenuListener(this);
    radiusPoti->setMenuListener(this);
    proteinsPerLevelPoti->setMenuListener(this);
    browserButton->setMenuListener(this);
    nameVisibleCheckbox->setMenuListener(this);
    topsanVisibleCheckbox->setMenuListener(this);
    resetButton->setMenuListener(this);
    scalePoti->setMenuListener(this);
    //movableCheckbox->setMenuListener(this);

    for (int i = 0; i < 10; i++)
    {
        saveLayoutButtons[i] = new coButtonMenuItem((string("Slot ") + string(1, (char)'0' + i)).c_str());
        saveLayoutMenu->add(saveLayoutButtons[i]);
        saveLayoutButtons[i]->setMenuListener(this);
    }

    readSavedLayout();

    for (int i = 0; i < 10; i++)
    {
        if (layouts[i] != NULL)
        {
            loadLayoutButtons[i] = new coButtonMenuItem((string("Slot ") + string(1, (char)'0' + i)).c_str());
            loadLayoutMenu->add(loadLayoutButtons[i]);
            loadLayoutButtons[i]->setMenuListener(this);
        }
    }

    lastlayout = NONE;

    loadinglayout = false;

    alHandle = new coPopupHandle(string("Alignment"));
    alFrame = new coFrame();
    alPanel = new SizedPanel(new coFlatPanelGeometry(coUIElement::BLACK));

    alHandle->addElement(alFrame);
    alFrame->addElement(alPanel);
    alHandle->setVisible(false);

    alreset = NULL;
    alclear = NULL;

    updateAlignMenu();

    updateFadeValues(); // initalize fade values

    // create pop up window for moused over name
    nameFrame = new coFrame("UI/Frame");
    nameMessage = new coLabel;
    nameMessage->setString("Name:         ");
    nameMessage->setPos(0, 0, z);
    nameMessage->setFontSize(50);
    nameFrame->addElement(nameMessage);
    nameFrame->fitToChild();

    nameHandle = new coPopupHandle("Protein");
    nameHandle->addElement(nameFrame);
    nameHandle->setVisible(false);
    nameHandle->update();

    //add buttons to animations and protein structures from config
    readMenuConfigData("COVER.Plugin.PDB.Animations", animationVec, *animationRow);
    readMenuConfigData("COVER.Plugin.PDB.Structures", structureVec, *structureRow);

    // add button to panel to initiate a search
    loadButton = new coPushButton(new coRectButtonGeometry(buttonSize[0], buttonSize[1], "PDBPlugin/load"), this);
    loadButton->setSize(40);
    loadButton->setPos(x[4], y[0]);
    panel->addElement(loadButton);
    panel->setScale(5); // set size of panel and contents
    panel->resize();
    frame = new coFrame("UI/Frame");
    frame->addElement(panel);
    handle->addElement(frame);

    //message handler frame
    messageFrame = new coFrame("UI/Frame");
    messageFrame->addElement(messageLabel);
    messageFrame->fitToChild();
    messageHandle->addElement(messageFrame);
    // Create main menu button
    selectMenu->setMenuListener(this);
    cover->getMenu()->add(selectMenu);

    // Create interaction class for CUI support:
    assert(cover->getScene() != NULL);
    _interaction = new cui::Interaction(cover->getScene(), cover->getObjectsRoot());
    _interaction->_wandR->setCursorType(cui::InputDevice::INVISIBLE);
    _interaction->_head->setCursorType(cui::InputDevice::NONE);

    //add the align location
    alignpoint = new AlignPoint(_interaction, Vec3(0, 0, 0), Vec3(1, 1, 1), Widget::COL_RED, Widget::COL_YELLOW, Widget::COL_GREEN, 0.6, 25., sview, this);
    alignpoint->addListener(this);
    alignpoint->setVisible(false);

    struct BrowserMessage bm;

    bm.plugin = this;
    strcpy(bm.ext, "tif");
    bm.preview = false;

    cover->sendMessage(this, "FileBrowser", REGISTER_EXT, sizeof(struct BrowserMessage), &bm);

    strcpy(bm.ext, "pdb");
    bm.preview = true;

    cover->sendMessage(this, "FileBrowser", REGISTER_EXT, sizeof(struct BrowserMessage), &bm);

    return true;
}

/// Destructor
PDBPlugin::~PDBPlugin()
{
    // remove elements from node
    clearSwitch();

    cover->getObjectsRoot()->removeChild((osg::Node *)mainNode.get());

    // Delete frame components:
    for (int i = 0; i < NUM_DIALS; i++)
    {
        delete potiChar[i];
        delete labelChar[i];
        delete stringChar[i];
    }
    delete[] stringChar;
    delete[] potiChar;
    delete[] labelChar;
    delete messageHandle;
    delete handle;
    delete frame;
    delete structureMenu;
    delete selectMenu;
    delete animationMenu;

    // delete main components
    delete alignpoint;
    delete _interaction;
}

// toggle button for PDB load frame
void PDBPlugin::menuEvent(coMenuItem *menuItem)
{
    PDBPluginMessage mm;

    // listen for initPDB frame to open close
    if (menuItem == loaderMenuCheckbox)
    {
        handle->setVisible(loaderMenuCheckbox->getState());
    }
    /*else if(menuItem == movableCheckbox)
  {
    setAllMovable(movableCheckbox->getState());
  }*/
    else if (menuItem == highdetailCheckbox)
    {
        mm.high_detail_on = highdetailCheckbox->getState();
        mm.token = HIGH_DETAIL;
        cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME_OTHERS,
                           HIGH_DETAIL, sizeof(PDBPluginMessage), &mm);
    }
    else if (menuItem == fadeareaMenuPoti)
    {
        updateFadeValues();
    }
    else if (menuItem == distviewMenuPoti)
    {
        updateFadeValues();
    }
    else if (menuItem == scalePoti)
    {
        float value = 0.001;
        if (scalePoti->getValue() > 0.001)
            value = scalePoti->getValue();

        mm.scale = value;
        mm.token = SCALE_ALL;
        cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                           1, sizeof(PDBPluginMessage), &mm);
    }
    else if (menuItem == clearSceneButton)
    {
        mm.token = CLEAR_ALL;
        cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                           1, sizeof(PDBPluginMessage), &mm);
    }
    else if (menuItem == layoutButtonGrid)
    {
        //cerr << "layout" << endl;
        //layoutProteins();
        mm.token = LAYOUT_GRID;
        cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                           1, sizeof(PDBPluginMessage), &mm);
    }
    else if (menuItem == layoutButtonCylinder)
    {
        mm.token = LAYOUT_CYL;
        cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                           1, sizeof(PDBPluginMessage), &mm);
    }
    else if (menuItem == resetButton)
    {
        //cerr << "reset positions" << endl;
        resetPositions();
    }
    else if (menuItem == faderesetButton)
    {
        distviewMenuPoti->setValue(viewdist);
        fadeareaMenuPoti->setValue(fadedist);
        updateFadeValues();
    }
    //else if(menuItem == alignCheckbox)
    //{
    //alignpoint->setVisible(alignCheckbox->getState());
    //}
    else if (menuItem == browserButton)
    {
        string temp = coCoviseConfig::getEntry("COVER.Plugin.PDB.BrowserDirectory");
        setFileBrowserVisible(true);
        cerr << "showing file browser for directory " << temp << endl;
        if (temp != "")
        {
            changeFileBrowserDir(temp);
        }
    }
    else if (menuItem == radiusPoti || menuItem == proteinsPerLevelPoti)
    {
        switch (lastlayout)
        {
        case NONE:
            return;
        case GRID:
            menuEvent(layoutButtonGrid);
            break;
        case CYLINDER:
            menuEvent(layoutButtonCylinder);
            break;
        default:
            return;
        }
    }

    for (int i = 0; i < 10; i++)
    {
        if (menuItem == saveLayoutButtons[i])
        {
            makeLayout(i);
            writeSavedLayout();
            return;
        }
    }

    for (int i = 0; i < 10; i++)
    {
        if (menuItem != NULL && menuItem == loadLayoutButtons[i])
        {
            loadLayout(i);
            return;
        }
    }
    // check if a default structure was choosen

    selectedMenuButton(menuItem);
}

void PDBPickBox::setView(string &view, bool state)
{
    //cerr << "setView for " << view << endl;
    if (state) // in want turned on
    {
        if (!_isMorph)
        {
            osg::Node *current = getVersion(view);

            if (!current) // load the view from the cache
            {
                // need to load view from cache
                osgUtil::Optimizer optimizer;

                chdir((plugin->getRelativeTempPath()).c_str());

                //ReaderWriterIV::quality = 0.2;
                //cerr << "Loading file: " << (plugin->getRelativeTempPath()).c_str() + _name + view + WRL_EXT << endl;
                current = readNodeFile((plugin->getRelativeTempPath()).c_str() + _name + view + WRL_EXT);
                osg::StateSet *state = current->getOrCreateStateSet();
                state->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);

                //ReaderWriterIV::quality = 1.0;
                current->setName(view);

                //((Group*)_mainSwitch->getChild(0))->addChild(current);  // wont work for morphs
                chdir((plugin->getCurrentPath()).c_str());

                //osgUtil::StatsVisitor stats;
                //current->accept(stats);
                //cerr << "before num geodes is " << stats._numInstancedGeode << endl;
                //stats.reset();

                if (current)
                {
                    optimizer.optimize(current, osgUtil::Optimizer::MERGE_GEOMETRY
                                                | osgUtil::Optimizer::MERGE_GEODES);

                    //current->accept(stats);
                    //cerr << "after num geodes is " << stats._numInstancedGeode << endl;
                    //stats.reset();
                }

                setVersion(current);
            }

            ((Group *)_mainSwitch->getChild(0))->addChild(current); // wont work for morphs
        }
        else
        {
            Group *current = getVersionMorph(view);

            if (!current)
            {
                current = new Group();
                // need to load view from cache
                osgUtil::Optimizer optimizer;

                chdir((plugin->getRelativeTempPath()).c_str());
                for (unsigned int i = 0; i < _mainSwitch->getNumChildren(); i++)
                {
                    string basename = _name + "-ff";
                    char buf[10];
                    sprintf(buf, "%d", i);
                    Node *tempnode = readNodeFile((plugin->getRelativeTempPath()).c_str() + basename + buf + view + WRL_EXT);
                    osg::StateSet *state = tempnode->getOrCreateStateSet();
                    state->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
                    tempnode->setName(view);

                    optimizer.optimize(tempnode, osgUtil::Optimizer::MERGE_GEOMETRY
                                                 | osgUtil::Optimizer::MERGE_GEODES);

                    current->addChild(tempnode);
                }
                chdir((plugin->getCurrentPath()).c_str());
                setVersionMorph(current);
            }
            for (unsigned int i = 0; i < _mainSwitch->getNumChildren(); i++)
            {
                ((Group *)_mainSwitch->getChild(i))->addChild(current->getChild(i));
            }
        }
    }
    else // want to remove from the current view
    {
        if (!_isMorph)
        {
            Group *group = ((Group *)_mainSwitch->getChild(0)); // wont work for mrphs
            for (unsigned int i = 0; i < group->getNumChildren(); i++)
            {
                if (group->getChild(i)->getName() == view)
                {
                    group->removeChild(i);
                }
            }
        }
        else
        {
            for (unsigned int j = 0; j < _mainSwitch->getNumChildren(); j++)
            {
                Group *group = ((Group *)_mainSwitch->getChild(j)); // wont work for mrphs
                for (unsigned int i = 0; i < group->getNumChildren(); i++)
                {
                    if (group->getChild(i)->getName() == view)
                    {
                        group->removeChild(i);
                    }
                }
            }
        }
    }
}

bool PDBPickBox::isViewEnabled(const string &view)
{
    Group *group = ((Group *)_mainSwitch->getChild(0));
    for (unsigned int i = 0; i < group->getNumChildren(); i++)
    {
        if (group->getChild(i)->getName() == view)
            return true;
    }
    return false;
}

void PDBPickBox::setVersion(osg::Node *node)
{
    _versions->addChild(node);
}

osg::Node *PDBPickBox::getVersion(string &type)
{
    for (unsigned int i = 0; i < _versions->getNumChildren(); i++)
    {
        if (_versions->getChild(i)->getName() == type)
        {
            return _versions->getChild(i);
        }
    }
    return NULL;
}

void PDBPickBox::setVersionMorph(osg::Group *node)
{
    _versions->addChild(node);
}

osg::Group *PDBPickBox::getVersionMorph(string &type)
{
    for (unsigned int i = 0; i < _versions->getNumChildren(); i++)
    {
        if (((Group *)_versions->getChild(i))->getNumChildren() > 0)
        {
            if (((Group *)_versions->getChild(i))->getChild(0)->getName() == type)
            {
                return (Group *)_versions->getChild(i);
            }
        }
    }
    return NULL;
}

void PDBPlugin::updateFadeValues()
{
    if (PDBPickBox::lowdetailVisitor)
    {
        PDBPickBox::lowdetailVisitor->setArea(fadeareaMenuPoti->getValue());
        PDBPickBox::lowdetailVisitor->setDistance(distviewMenuPoti->getValue());
    }
    if (PDBPickBox::highdetailVisitor)
    {
        PDBPickBox::highdetailVisitor->setArea(fadeareaMenuPoti->getValue());
        PDBPickBox::highdetailVisitor->setDistance(distviewMenuPoti->getValue());
    }
}

osg::Node *PDBPlugin::loadStructure(string &filename)
{

    std::string fullpath = relativeTempPath + filename;
    cerr << "full path: " << fullpath << endl;

    osg::Node *node = coVRFileManager::instance()->loadFile(fullpath.c_str());
    if (!node)
    {
        node = readNodeFile(fullpath.c_str());
        if (node)
        {
            osgUtil::Optimizer optimizer;
            optimizer.optimize(node, osgUtil::Optimizer::SHARE_DUPLICATE_STATE & osgUtil::Optimizer::MERGE_GEOMETRY & osgUtil::Optimizer::MERGE_GEODES);
        }
        else
        {
            cerr << "Error: Loading model" << endl;
            node = new osg::Node();
        }
    }

    node->setName(CARTOON);

    MatrixTransform *mroot = dynamic_cast<MatrixTransform *>(node);
    if (mroot)
        mroot = dynamic_cast<MatrixTransform *>(mroot->getChild(0));
    if (mroot)
    {
        cerr << "Root is a MatrixTransform\n" << endl;
        cerr << "Has " << mroot->getNumChildren() << " children\n";
        //mroot = dynamic_cast<MatrixTransform*>(mroot->getChild(0));
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                cerr << mroot->getMatrix()(i, j) << " ";
            }
            cerr << endl;
        }
    }

    return node;
}

// abstract
void PDBPlugin::potiValueChanged(float, float newvalue, coValuePoti *poti, int)
{
    if (poti == proteinScalePoti)
    {
        if (_selectedPickBox)
        {
            if (newvalue < 0.005)
                newvalue = 0.005;

            // send message to adjust the hue and color of the structure
            PDBPluginMessage mm;
            setMMString(mm.filename, _selectedPickBox->getName());
            mm.token = SCALE_PROTEIN;
            mm.scale = newvalue;
            cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                               1, sizeof(PDBPluginMessage), &mm);
        }
    }
    else if (poti == proteinHuePoti)
    {
        if (_selectedPickBox)
        {
            // send message to adjust the hue and color of the structure
            PDBPluginMessage mm;
            setMMString(mm.filename, _selectedPickBox->getName());
            mm.token = ADJUST_HUE;
            mm.hue = newvalue;
            mm.hue_mode_on = coloroverrideCheckBox->getState();
            cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                               1, sizeof(PDBPluginMessage), &mm);
        }
    }
}

// check for button events
void PDBPlugin::buttonEvent(coButton *cobutton)
{

    PDBPluginMessage mm;
    string filename;

    //check which dials to load file from
    if (cobutton == loadButton)
    {
        // read in file name
        for (int i = 0; i < NUM_DIALS; i++)
        {
            filename.append(*stringChar[i]);
        }

        // create message content
        setMMString(mm.filename, filename);
        mm.token = LOAD_FILE;
        cerr << "Clicked load button\n";
        cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                           LOAD_FILE, sizeof(PDBPluginMessage), &mm);
        handle->update();
    }
    else if (cobutton == proteinDeleteButton) // need to change to send messages
    {
        // send message to adjust the hue and color of the structure
        if (_selectedPickBox)
        {
            PDBPluginMessage mm;
            setMMString(mm.hostname, getMyHost());
            setMMString(mm.filename, _selectedPickBox->getName());
            mm.token = REMOVE_PROTEIN;
            mm.uid = _selectedPickBox->uid;
            cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                               1, sizeof(PDBPluginMessage), &mm);
        }
    }
    else if (cobutton == proteinResetScaleButton)
    {
        // send message to adjust the hue and color of the structure
        if (_selectedPickBox)
        {
            PDBPluginMessage mm;
            setMMString(mm.filename, _selectedPickBox->getName());
            mm.token = RESET_SCALE;
            cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                               1, sizeof(PDBPluginMessage), &mm);
        }
    }
    else if (cobutton == coloroverrideCheckBox)
    {
        if (_selectedPickBox)
        {
            // send message to adjust the hue and color of the structure
            PDBPluginMessage mm;
            setMMString(mm.filename, _selectedPickBox->getName());
            mm.token = ADJUST_HUE;
            mm.hue = proteinHuePoti->getValue();
            mm.hue_mode_on = coloroverrideCheckBox->getState();
            cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                               1, sizeof(PDBPluginMessage), &mm);
        }
    }
    else if (cobutton == cartoonCheckBox)
    {
        if (_selectedPickBox)
        {
            // send message to adjust the hue and color of the structure
            PDBPluginMessage mm;
            setMMString(mm.filename, _selectedPickBox->getName());
            mm.token = VIEW;
            setMMString(mm.viewtype, CARTOON);
            mm.view_on = cartoonCheckBox->getState();
            cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                               1, sizeof(PDBPluginMessage), &mm);
        }
    }
    else if (cobutton == stickCheckBox)
    {
        if (_selectedPickBox)
        {
            // send message to adjust the hue and color of the structure
            PDBPluginMessage mm;
            setMMString(mm.filename, _selectedPickBox->getName());
            mm.token = VIEW;
            setMMString(mm.viewtype, STICK);
            mm.view_on = stickCheckBox->getState();
            cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                               1, sizeof(PDBPluginMessage), &mm);
        }
    }
    else if (cobutton == surfaceCheckBox)
    {
        if (_selectedPickBox)
        {
            // send message to adjust the hue and color of the structure
            PDBPluginMessage mm;
            setMMString(mm.filename, _selectedPickBox->getName());
            mm.token = VIEW;
            setMMString(mm.viewtype, SURFACE);
            mm.view_on = surfaceCheckBox->getState();
            cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                               1, sizeof(PDBPluginMessage), &mm);
        }
    }
    else if (cobutton == ribbonCheckBox)
    {
        if (_selectedPickBox)
        {
            // send message to adjust the hue and color of the structure
            PDBPluginMessage mm;
            setMMString(mm.filename, _selectedPickBox->getName());
            mm.token = VIEW;
            setMMString(mm.viewtype, RIBBON);
            mm.view_on = ribbonCheckBox->getState();
            cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                               1, sizeof(PDBPluginMessage), &mm);
        }
    }
    else if (cobutton == sequenceCheckBox)
    {
        if (_selectedPickBox)
        {
            // send a message to PDB sequence
            /*SequenceMessage sm;
      sm.x = _selectedPickBox->getSequenceMarker()->getPosition().x();
      sm.y = _selectedPickBox->getSequenceMarker()->getPosition().z();
      sm.z = -_selectedPickBox->getSequenceMarker()->getPosition().y();
      sm.on = sequenceCheckBox->getState();
      assert((_selectedPickBox->getName()).size() < 5);
      strcpy(sm.filename, (_selectedPickBox->getName()).c_str());
      cover->sendMessage(PDBPluginModule,
            coVRPluginSupport::TO_SAME,
            MOVE_MARK,
            sizeof(SequenceMessage),
            &sm);*/

            // send a message to update the state of whether the marker is on or off
            PDBPluginMessage mm;
            setMMString(mm.filename, _selectedPickBox->getName());
            mm.token = MARK;
            mm.mark_on = sequenceCheckBox->getState();
            mm.uid = _selectedPickBox->uid;
            cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                               1, sizeof(PDBPluginMessage), &mm);
        }
    }
    else if (cobutton == alignButton)
    {
        //need to make copy then remove object from scene (maybe leave the orignal in the warehouse)
        if (_selectedPickBox)
        {
            // send message to set the align point
            PDBPluginMessage mm;
            setMMString(mm.filename, _selectedPickBox->getName());
            mm.token = PDB_ALIGN;
            mm.view_on = surfaceCheckBox->getState();
            cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                               1, sizeof(PDBPluginMessage), &mm);
        }
    }
    else if (cobutton == alclear)
    {
        alclear->setState(false, false);
        clearAlign();
        return;
    }
    else if (cobutton == alreset)
    {
        alreset->setState(false, false);
        alignReset();
        return;
    }

    for (int i = 0; i < alButtons.size(); i++)
    {
        if (cobutton == alButtons[i].first)
        {
            alignProteins(i);
            return;
        }
    }

    for (int i = 0; i < alButtons.size(); i++)
    {
        if (cobutton == alButtons[i].second)
        {
            if (alButtons[i].first->getState())
            {
                if (alButtons[i].second->getState())
                {
                    for (int j = 0; j < alButtons.size(); j++)
                    {
                        if (j == i)
                        {
                            continue;
                        }
                        if (alButtons[j].second->getState())
                        {
                            alignpoint->setMovable(_proteins[j]->getName(), false);
                            alButtons[j].second->setState(false, false);
                        }
                    }
                    alignpoint->setMovable(_proteins[i]->getName(), true);
                }
                else
                {
                    alignpoint->setMovable(_proteins[i]->getName(), false);
                }
            }
            else
            {
                alButtons[i].second->setState(false, false);
            }
            return;
        }
    }
}

void PDBPickBox::setHue(float hue)
{
    _hue = hue;

    if (_overridecolor)
        overrideHue(true);
}

bool PDBPickBox::isOverrideColor()
{
    return _overridecolor;
}

void PDBPickBox::overrideHue(bool override)
{
    _overridecolor = override;

    // transform hue in a vec4
    float r, g, b;
    Vec4 color;
    vvToolshed::HSBtoRGB(_hue, 1, 1, &r, &g, &b);
    color[0] = r;
    color[1] = g;
    color[2] = b;
    color[3] = 1.0;

    // set the color and override the material nodes of all children
    StateSet *state = _node->getOrCreateStateSet();
    Material *material = new Material;
    material->setDiffuse(Material::FRONT_AND_BACK, color);
    if (override)
        state->setAttributeAndModes(material, StateAttribute::OVERRIDE);
    else
        state->setAttributeAndModes(material, StateAttribute::ON);
}

/// Called whenever the current timestep changes
void PDBPlugin::setTimestep(int ts)
{
    vector<PDBPickBox *>::iterator it = _aniProteins.begin();
    for (; it < _aniProteins.end(); it++)
    {
        (*it)->setTimestep(ts);
    }
}

/** Update osgcaveui interaction.
*/
void PDBPlugin::updateOSGCaveUI()
{
    static int lastStatus = 0;
    static int lastButton = 0;
    static int lastState = 0;
    int button = 0, state = 0;
    int newStatus;

    _interaction->_wandR->setI2W(cover->getPointerMat());
    _interaction->_head->setI2W(cover->getViewerMat());

    newStatus = cover->getPointerButton()->getState();
    switch (newStatus)
    {
    case 0:
        button = 0;
        state = 0;
        break;
    case vruiButtons::ACTION_BUTTON:
        button = 0;
        state = 1;
        break;
    case vruiButtons::XFORM_BUTTON:
        button = 1;
        state = 1;
        break;
    case vruiButtons::DRIVE_BUTTON:
        button = 2;
        state = 1;
        break;
    default:
        break;
    }
    if (lastStatus != newStatus)
    {
        if (lastState == 1)
        {
            _interaction->_wandR->buttonStateChanged(lastButton, 0);
        }
        if (state == 1) //&& !_removingProtein)
        {
            _interaction->_wandR->buttonStateChanged(button, 1);
        }
    }

    lastStatus = newStatus;
    lastButton = button;
    lastState = state;

    _interaction->action();

    //_removingProtein = false;
}

/// Called before each frame
void PDBPlugin::preFrame()
{
    auto mode = coVRNavigationManager::instance()->getMode();
    setAllMovable(mode == coVRNavigationManager::XForm);

    // check if the show panel should be visible
    nameHandle->setVisible(nameVisibleCheckbox->getState());

    topsan->setVisible(topsanVisibleCheckbox->getState());
    sview->setVisible(sviewVisibleCheckbox->getState());
    alHandle->setVisible(alVisibleCheckbox->getState());

    // check if restricted mode is enabled !!!!!!!!!!!!!!! ( need to fix )
    if (_restrictedMode)
    {
        //set objectsRoot x,y trans to the users x,y
        Vec3 trans = (cover->getViewerMat() * cover->getInvBaseMat()).getTrans();
        Matrix mat = cover->getBaseMat();
        trans.set(trans.x(), trans.y(), 0);
        mat.setTrans(trans);
        cover->getObjectsXform()->setMatrix(mat);
        _restrictedMode = false;
    }

    updateOSGCaveUI(); // get current interaction parameters from system and pass to CUI

    // low detail traversals  /* need to redo */
    /*vector<PDBPickBox*>::iterator it = _proteins.begin();
  for(;it < _proteins.end(); it++)
  {
    // temp (set all settings the same way until gui built)
    (*it)->setDetailLevel(highdetailCheckbox->getState());

    //PDBPickBox::frameVisitor->setFadeOn(fadeonCheckbox->getState());
    PDBPickBox::frameVisitor->setFadeOn(false);
    PDBPickBox::frameVisitor->setHighDetailOn((*it)->getDetailLevel());
    (*it)->getNode()->accept(*PDBPickBox::frameVisitor);

    // compue detail level if fade mode is not enabled
    if(!fadeonCheckbox->getState())
    	(*it)->computeDetailLevel();
    else
    	(*it)->computeFadeLevel();
  }*/

    // check if interaction panel should be displayed for a protein
    if (_selectedPickBox)
    {
        proteinLabel->setString(PROTEIN + _selectedPickBox->getName());
        proteinScalePoti->setValue(_selectedPickBox->getScale());
        proteinHuePoti->setValue(_selectedPickBox->getHue());
        coloroverrideCheckBox->setState(_selectedPickBox->isOverrideColor());

        //update visibility buttons
        surfaceCheckBox->setState(_selectedPickBox->isViewEnabled(SURFACE));
        stickCheckBox->setState(_selectedPickBox->isViewEnabled(STICK));
        cartoonCheckBox->setState(_selectedPickBox->isViewEnabled(CARTOON));
        ribbonCheckBox->setState(_selectedPickBox->isViewEnabled(RIBBON));
        if (!_selectedPickBox->isMorph())
        {
            sequenceCheckBox->setState(_selectedPickBox->getSequenceMarker()->isVisible());
        }
        else
        {
            sequenceCheckBox->setState(false);
        }

        proteinHandle->setVisible(true);
    }
    else
    {
        proteinLabel->setString(PROTEIN);
        proteinScalePoti->setValue(1);
        proteinHandle->setVisible(false);
    }

    if (handle->isVisible()) // only applicable if it is visible
    {
        for (int i = 0; i < NUM_DIALS; i++)
        {
            // update labels corressponding to dials
            *stringChar[i] = REFARRAY[((int)potiChar[i]->getValue())];
            labelChar[i]->setString(*stringChar[i]);
        }
    }

    // check if frame is different then update users
    if (_aniProteins.size() > 0 && (currentFrame != coVRAnimationManager::instance()->getAnimationFrame()))
    {
        PDBPluginMessage mm;
        mm.framenumber = currentFrame;
        mm.token = FRAME_UPDATE;
        currentFrame = coVRAnimationManager::instance()->getAnimationFrame();
    }

    // decrement loadDelay if it exists
    if (loadDelay >= 0)
        loadDelay--;

    //when load delay is zero load applicable
    if (loadDelay == 0)
    {

        //cerr << "trying to load after delay: " << loadFile << "\n";
        /*if(loadingMorph)
    {
      if(downloadPDBFile(loadFile, YALE))
        loadPDB(loadFile, YALE);

      // remove temporary animation files !!!!!!!! NEED TO TEST, MAY DELETE FILES BEFORE ALL NODES LOAD STRUCTURES !!!!!!!
      if(coVRMSController::instance()->isMaster()) {

    	// remove frame models 
    	chdir(relativeTempPath.c_str());
    	system("rm -f FF*.wrl");
    	chdir(currentPath.c_str());
      }
    }
    else
    {
      // testing if this is the area the slaves try and load no existing files before they exist!!!!!
      int status = 0;
      if(coVRMSController::instance()->isMaster()) {
  	    // check if the file exists to open it
	    chdir(relativeTempPath.c_str());

	    // TODO change if can not open the file here since master get model from server and convert // (quick fix will do on another thread later)	 
  	    ifstream file((loadFile + CARTOON + WRL_EXT).c_str());
  	    if(file.is_open())
          	status = 1;
	    else
	    {
		//send a message to download the file
		
		// download models required and convert them
      		if(downloadPDBFile(loadFile, UCSD))
       	   		loadPDB(loadFile, UCSD);
            }
  	    file.close(); // close the file
	    chdir(currentPath.c_str());
	 
        coVRMSController::instance()->sendSlaves((char *)&status,sizeof(int));
      }
      else
      {
	 // slaves waiting until the files is loaded
         coVRMSController::instance()->readMaster((char *)&status,sizeof(int));
      }
      */

        // check if I can open the file locally
        //if(status)
        //{
        // remove loading message
        hideMessage();
        if (!loadingMorph)
        {
            Node *node = loadVRMLFile(loadFile);
            if (node != NULL)
            {
                Switch *timeSwitch = new Switch;
                timeSwitch->addChild(node);

                // Create PDB PickBox:
                //cerr << "loading from pre frame" << endl;
                PDBPickBox *pdbBox = new PDBPickBox(timeSwitch, _interaction, Vec3(0, 0, 0), Vec3(1, 1, 1), Widget::COL_RED, Widget::COL_YELLOW, Widget::COL_GREEN, loadFile, _defaultScale);
                pdbBox->addListener(this);

                pdbBox->uid = uidcounter;

                // add sequence marker
                SequenceMarker *mark = new SequenceMarker(pdbBox, Marker::CONE, _interaction, 80., .4);
                mark->setColor(Vec4(184, 115, 51, 1));
                mark->addMarkerListener(this);
                pdbBox->setSequenceMarker(mark);
                //pdbBox->addChild(mark->getNode());
                pdbBox->addMarker(mark->getNode());

                mark->setVisible(false);

                sview->load(loadFile);
                topsan->load(loadFile);

                if (loadinglayout)
                {
                    pdbBox->setNodeMat(layouts[layoutnum]->protmat[loadnum].first);
                    pdbBox->setScaleMat(layouts[layoutnum]->protmat[loadnum].second);
                }

                // Add PDB box to scenegraph:
                _proteins.push_back(pdbBox);
                mainNode.get()->addChild(pdbBox->getNode());

                updateAlignMenu(ADD, pdbBox->uid);

                // remove loading message
                //hideMessage();
            }
            if (loadinglayout)
            {
                loadLayoutHelper();
            }
        }
        else
        {
            int counter = 0;
            string basename = loadFile + "-ff";
            char buf[10];
            sprintf(buf, "%d", counter);
            Switch *timeSwitch = new Switch;
            string framename = basename + buf;
            Node *node = loadVRMLFile(framename);
            while (node != NULL)
            {
                counter++;
                timeSwitch->addChild(node);
                sprintf(buf, "%d", counter);
                framename = basename + buf;
                node = loadVRMLFile(framename);
            }
            if (counter != 0)
            {
                PDBPickBox *pdbBox = new PDBPickBox(timeSwitch, _interaction, Vec3(0, 0, 0), Vec3(1, 1, 1), Widget::COL_RED, Widget::COL_YELLOW, Widget::COL_GREEN, loadFile, _defaultScale);
                pdbBox->addListener(this);

                pdbBox->setMorph(true);

                pdbBox->uid = uidcounter;

                _aniProteins.push_back(pdbBox);
                mainNode.get()->addChild(pdbBox->getNode());

                updateNumTimesteps();
            }
        }
        /*else {
	// download file from server as it doesnt exist locally
      	if(downloadPDBFile(loadFile, UCSD))
       	   loadPDB(loadFile, UCSD);
      }*/
        //}
        //screenResizeDelay = MESSAGE_PREFRAME;
    }

    setTimestep(currentFrame);

    // decrement resize delay
    //if(screenResizeDelay >= 0)
    //screenResizeDelay--;

    // resize screen to view all
    //if(!screenResizeDelay)
    //{
    //VRSceneGraph::instance()->viewAll();    // disabled so that sizes of multiple proteins are comparable
    //}

    // make sure submenu checkboxes match up
    loaderMenuCheckbox->setState(handle->isVisible());
    if (!selectRow->isVisible())
    {
        selectMenu->closeSubmenu();
    }

    if (!structureRow->isVisible())
        structureMenu->closeSubmenu();

    if (!animationRow->isVisible())
        animationMenu->closeSubmenu();

    // resize the sequence markers so they are always one size
    setSequenceMarkSizeAll(_markerSize);

    messageHandle->update();
    handle->update();
}

// make sure the marks stay a set size
void PDBPlugin::setSequenceMarkSizeAll(float size)
{
    vector<PDBPickBox *>::iterator it = _proteins.begin();
    for (; it < _proteins.end(); it++)
    {
        SequenceMarker *temp = (*it)->getSequenceMarker();
        temp->setSize(size * (1 / cover->getScale()) * (1 / (*it)->getScale()));
    }
}

void PDBPlugin::makeLayout(int i)
{
    if (layouts[i])
    {
        delete layouts[i];
    }

    layouts[i] = new struct PDBSavedLayout;

    layouts[i]->objmat = cover->getXformMat();
    layouts[i]->scale = cover->getScale();
    for (int j = 0; j < _proteins.size(); j++)
    {
        layouts[i]->protmat.push_back(pair<Matrix, Matrix>(_proteins[j]->getNodeMat(), _proteins[j]->getScaleMat()));
        layouts[i]->protname.push_back(_proteins[j]->getName());
    }
}

void PDBPlugin::loadLayout(int i)
{
    PDBPluginMessage mm;
    mm.token = CLEAR_ALL;
    cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                       1, sizeof(PDBPluginMessage), &mm);

    cover->setXformMat(layouts[i]->objmat);
    cover->setScale(layouts[i]->scale);

    if (layouts[i]->protname.size() > 0)
    {
        loadinglayout = true;
        loadnum = 0;
        layoutnum = i;

        setMMString(mm.filename, layouts[i]->protname[loadnum]);
        mm.token = LOAD_FILE;
        //filename = itEntry->fileName;
        cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                           LOAD_FILE, sizeof(PDBPluginMessage), &mm);
    }
}

void PDBPlugin::loadLayoutHelper()
{
    loadnum++;

    PDBPluginMessage mm;

    if (loadnum < layouts[layoutnum]->protname.size())
    {
        setMMString(mm.filename, layouts[layoutnum]->protname[loadnum]);
        mm.token = LOAD_FILE;
        //filename = itEntry->fileName;
        cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                           LOAD_FILE, sizeof(PDBPluginMessage), &mm);
    }
    else
    {
        loadinglayout = false;
    }
}

void PDBPlugin::writeSavedLayout()
{
    for (int i = 0; i < 10; i++)
    {
        if (layouts[i] != NULL)
        {
            loadLayoutMenu->remove(loadLayoutButtons[i]);
            delete loadLayoutButtons[i];
        }
    }

    string layoutpath = coCoviseConfig::getEntry("COVER.Plugin.PDB.LayoutDir");
    for (int i = 0; i < 10; i++)
    {
        if (!layouts[i])
        {
            continue;
        }

        std::ofstream lfile;
        lfile.open((layoutpath + "/Slot " + string(1, (char)'0' + i) + ".cfg").c_str(), ios::trunc);

        lfile << layouts[i]->scale << "\n";
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 4; k++)
            {
                lfile << layouts[i]->objmat(j, k) << " ";
            }
        }

        lfile << "\n";

        for (int j = 0; j < layouts[i]->protname.size(); j++)
        {
            lfile << layouts[i]->protname[j] << "\n";
            for (int k = 0; k < 4; k++)
            {
                for (int l = 0; l < 4; l++)
                {
                    lfile << layouts[i]->protmat[j].first(k, l) << " ";
                }
            }
            lfile << "\n";
            for (int k = 0; k < 4; k++)
            {
                for (int l = 0; l < 4; l++)
                {
                    lfile << layouts[i]->protmat[j].second(k, l) << " ";
                }
            }
            lfile << "\n";
        }

        lfile.close();
    }

    for (int i = 0; i < 10; i++)
    {
        if (layouts[i] != NULL)
        {
            loadLayoutButtons[i] = new coButtonMenuItem((string("Slot ") + string(1, (char)'0' + i)).c_str());
            loadLayoutMenu->add(loadLayoutButtons[i]);
            loadLayoutButtons[i]->setMenuListener(this);
        }
    }
}

void PDBPlugin::readSavedLayout()
{
    string layoutpath = coCoviseConfig::getEntry("COVER.Plugin.PDB.LayoutDir");
    for (int i = 0; i < 10; i++)
    {
        std::ifstream cfile;
        cfile.open((layoutpath + "/Slot " + string(1, (char)'0' + i) + ".cfg").c_str(), ios::in);

        if (!cfile.fail())
        {
            //cerr << "Reading Slot: " << i << endl;
            layouts[i] = new struct PDBSavedLayout;
            if (!cfile.eof())
            {
                cfile >> layouts[i]->scale;
            }

            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    cfile >> layouts[i]->objmat(j, k);
                }
            }

            char name[25];
            cfile >> name;
            while (!cfile.eof())
            {
                //cerr << "Name: " << name << endl;
                layouts[i]->protname.push_back(string(name));
                Matrix m1, m2;
                for (int j = 0; j < 4; j++)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        cfile >> m1(j, k);
                    }
                }
                for (int j = 0; j < 4; j++)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        cfile >> m2(j, k);
                    }
                }
                layouts[i]->protmat.push_back(pair<Matrix, Matrix>(m1, m2));
                cfile >> name;
            }
        }
        else
        {
            layouts[i] = NULL;
        }
        cfile.close();
    }
}

void PDBPlugin::updateAlignMenu(alignenum e, int uid)
{
    switch (e)
    {
    case ADD:
    {
        int i;
        for (i = 0; i < _proteins.size(); i++)
        {
            if (uid == _proteins[i]->uid)
            {
                break;
            }
        }
        coTextButtonGeometry *tbg = new coTextButtonGeometry(alButtonWidth, alButtonHeight, _proteins[i]->getName());
        tbg->setColors(0.1, 0.1, 0.1, 1.0, 0.9, 0.9, 0.9, 1.0);
        coToggleButton *t1 = new coToggleButton(tbg, this);
        coToggleButton *c1 = new coToggleButton(new coFlatButtonGeometry("UI/haken"), this);

        alPanel->addElement(t1);
        alPanel->addElement(c1);
        alButtons.push_back(pair<coToggleButton *, coToggleButton *>(c1, t1));
        break;
    }
    case ALIGN_DELETE:
    {
        int i;
        std::vector<std::pair<coToggleButton *, coToggleButton *> >::iterator it = alButtons.begin();
        for (i = 0; i < _proteins.size(); i++)
        {
            if (uid == _proteins[i]->uid)
            {
                break;
            }
            it++;
        }

        if (alButtons[i].first->getState())
        {
            alButtons[i].first->setState(false, false);
            alignpoint->setAlign(i);
        }

        alPanel->removeElement(alButtons[i].first);
        alPanel->removeElement(alButtons[i].second);
        delete alButtons[i].first;
        delete alButtons[i].second;

        alButtons.erase(it);

        break;
    }
    case DELETE_ALL:
    {
        for (int i = 0; i < alButtons.size(); i++)
        {
            alPanel->removeElement(alButtons[i].first);
            alPanel->removeElement(alButtons[i].second);
            delete alButtons[i].first;
            delete alButtons[i].second;
        }

        alButtons.clear();
        buttonEvent(alclear);
        break;
    }
    }
    updateAlignMenu();
}

void PDBPlugin::updateAlignMenu()
{

    float Bh = 40;
    float space = 5;
    float Hwidth = 200;

    alButtonWidth = 150;
    alButtonHeight = Bh;

    for (int i = 0; i < alButtons.size(); i++)
    {
        alButtons[i].first->setPos(0, ((float)alButtons.size() - 1 - i) * (space + Bh) + space, 1.0);
        alButtons[i].second->setPos(50, ((float)alButtons.size() - 1 - i) * (space + Bh), 1.0);
    }

    if (!alreset)
    {
        float width = 175;
        alreset = new coToggleButton(new coTextButtonGeometry(width, Bh, "Reset"), this);
        alclear = new coToggleButton(new coTextButtonGeometry(width, Bh, "Clear"), this);
        alPanel->addElement(alreset);
        alPanel->addElement(alclear);

        alreset->setPos((Hwidth / 2.0f) - (width / 2.0f), -(space + Bh), 1.0);
        alclear->setPos((Hwidth / 2.0f) - (width / 2.0f), -2.0 * (space + Bh));
    }

    alPanel->setSize(Hwidth, ((float)alButtons.size()) * (Bh + space) + 2.0 * Bh + 2.0 * space);
}

SequenceMarker *PDBPickBox::getSequenceMarker()
{
    return _mark;
}

/// clears the mainSwitch node
void PDBPlugin::clearSwitch()
{
    // set remove protein flag
    //_removingProtein = true;

    _restrictedMode = false;
    _galaxyMode = false;

    // reset selected PickPox
    _selectedPickBox = NULL;

    // Delete pickboxes:
    vector<PDBPickBox *>::iterator it = _proteins.begin();
    for (; it < _proteins.end(); it++)
    {
        delete (*it);
    }
    for (int i = 0; i < _aniProteins.size(); i++)
    {
        delete _aniProteins[i];
    }
    _aniProteins.clear();
    _proteins.clear();

    // removes align point children
    alignpoint->removeChildren();
    updateAlignMenu(DELETE_ALL, 0);
}

// downloads PDB file from a server
bool PDBPlugin::downloadPDBFile(string &filename, DataBankType dbtype)
{
    switch (dbtype)
    {
    case UCSD:
    {
        int status = 0;

        // change to relative directory
        chdir(relativeTempPath.c_str());

        // create buffer to get the file
        cmdInput.erase().append(WGET).append(pdburl).append(filename).append(PDB_EXT);

        if (coVRMSController::instance()->isMaster())
        {
            /* convert pdb to vrml */
            status = system(cmdInput.c_str());

            coVRMSController::instance()->sendSlaves((char *)&status, sizeof(int));
        }
        else
        {
            coVRMSController::instance()->readMaster((char *)&status, sizeof(int));
        }

        chdir(currentPath.c_str()); // change back to default directory

        if (status)
        {
            //showMessage(ERROR_STRUCTURE_MESSAGE);
            return false;
        }
        break;
    }
    case YALE:
    {
        char buf[10];
        buf[0] = 'a';

        if (coVRMSController::instance()->isMaster())
        {
            //change to working directory
            chdir(relativeTempPath.c_str());

            // execute remove cgi script to create tar
            //cmdInput.erase().append(WGETCGI).append(filename).append("\"");
            //system(cmdInput.c_str());

            // download remote tar file
            cmdInput.erase().append(WGET).append(WGETLOC).append(filename).append(TAR_EXT);
            system(cmdInput.c_str());

            // load in morph structures
            cmdInput.erase().append(TAR).append(filename).append(TAR_EXT);
            system(cmdInput.c_str());

            // rename frame set to filename-ff##.pdb
            cmdInput.erase().append("rename ff ").append(filename).append("-ff *.pdb");
            system(cmdInput.c_str());

            // remove tar file
            cmdInput.erase().append(REMOVE).append(filename).append(TAR_EXT);
            system(cmdInput.c_str());

            // change back to current directory
            chdir(currentPath.c_str());

            coVRMSController::instance()->sendSlaves(buf, 1);
        }
        else
        {
            coVRMSController::instance()->readMaster(buf, 1);
        }
        break;
    }
    }

    return true;
}

/// load menu button information
void PDBPlugin::readMenuConfigData(const char *menu, vector<PDBFileEntry> &menulist, coRowMenu &subMenu)
{
    coCoviseConfig::ScopeEntries e = coCoviseConfig::getScopeEntries(menu);
    for (const auto &entry : e)
    {
        coButtonMenuItem *temp = new coButtonMenuItem(entry.first);
        subMenu.add(temp);
        temp->setMenuListener(this);
        menulist.push_back(PDBFileEntry(entry.first.c_str(), entry.second.c_str(), (coMenuItem *)temp));
    }
}

/// search for selected option and load image/animation
void PDBPlugin::selectedMenuButton(coMenuItem *menuItem)
{
    PDBPluginMessage mm;
    //setMMString(mm.filename,
    //string filename;

    // check structures vector for pointer (if found exit)
    vector<PDBFileEntry>::iterator itEntry = structureVec.begin();
    for (; itEntry < structureVec.end(); itEntry++)
    {
        if (itEntry->fileMenuItem == menuItem)
        {
            //load current structure
            setMMString(mm.filename, itEntry->fileName);
            mm.token = LOAD_FILE;
            //filename = itEntry->fileName;
            cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                               LOAD_FILE, sizeof(PDBPluginMessage), &mm);
            return; //exit
        }
    }

    // check animation vector for pointer (if found exit)
    itEntry = animationVec.begin();
    for (; itEntry < animationVec.end(); itEntry++)
    {
        if (itEntry->fileMenuItem == menuItem)
        {

            //load current animation
            //filename = itEntry->fileName;
            mm.token = LOAD_FILE;
            setMMString(mm.filename, itEntry->fileName);
            cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                               LOAD_FILE, sizeof(PDBPluginMessage), &mm);
            return; //exit
        }
    }
}

void PDBPlugin::setMMString(char *dst, string src)
{
    assert(src.size() < 20);
    strcpy(dst, src.c_str());
}

/** loads helix and cylinder representations of a PDB structure
*/
bool PDBPlugin::loadPDBFile(string filename, Switch * /*node*/)
{
    int status = 0;
    if (coVRMSController::instance()->isMaster())
    {
        // convert pdb to wrl
        chdir(relativePymolPath.c_str());

        cmdInput.erase().append(PYMOL).append(relativeTempPath.c_str());
        std::cerr << "running 1: " << cmdInput << std::endl;
        status = system(cmdInput.c_str());

        chdir(currentPath.c_str());

        coVRMSController::instance()->sendSlaves((char *)&status, sizeof(int));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&status, sizeof(int));
    }

    // check whether the file loading was successful
    if (status)
    {
        if (!loadingMorph)
            showMessage(ERROR_LOADING); //output error

        return false; //exit loading failed
    }

    // try loading the file
    //Node* newNode = loadVRMLFile(filename);

    //if(newNode != NULL) {
    //node->addChild(newNode);

    return true;
    //}

    // node was null therefore file didnt exist to load
    //return false;
}

/// loads a single structure or morph
void PDBPlugin::loadPDB(string &filename, DataBankType dbtype)
{
    static bool animPluginExists = false;

    // auto load a vrml animation scene (tests)
    Switch *timeSwitch = new Switch;
    if (dbtype == YALE)
    {
        /*int counter = 0;
    bool loaded = true;
    while(loaded)
    {
      char buf[32];
      sprintf(buf, "FF%d", counter);
      filename.erase().append(buf);

      loaded = loadPDBFile(filename, timeSwitch);
      counter++;
    }*/

        loadPDBFile(filename, timeSwitch);

        if (!animPluginExists)
        {
            initCoverAnimation();
            animPluginExists = true;
        }
    }
    else
    {
        loadPDBFile(filename, timeSwitch);
    }
    timeSwitch->unref();
    /*
  // Create PDB PickBox:
  cerr << "Loading from load PDB" << endl;
  PDBPickBox* pdbBox = new PDBPickBox(timeSwitch, _interaction, Vec3(0,0,0), Vec3(1, 1, 1), Widget::COL_RED, Widget::COL_YELLOW
      , Widget::COL_GREEN, filename, _defaultScale);
  pdbBox->addListener(this);

  // Add PDB box to scenegraph:
  _proteins.push_back(pdbBox);
  mainNode.get()->addChild(pdbBox->getNode());
  updateNumTimesteps();

  // remove loading
  hideMessage();*/
}

// turn on animation control
void PDBPlugin::initCoverAnimation()
{
    coVRAnimationManager::instance()->setAnimationSpeed(12); // set default framerate
    coVRAnimationManager::instance()->enableAnimation(false);
    coVRAnimationManager::instance()->requestAnimationFrame(currentFrame);
}

/** Load a VRML file into an OSG node.
*/
osg::Node *PDBPlugin::loadVRMLFile(string &filename)
{
    chdir(relativeTempPath.c_str());

    // check for file existing (if doesnt exist exit passing null pointer)
    string vrmlFilename;
    vrmlFilename = filename + CARTOON + WRL_EXT;

    //PHILIP added touch for file system problem
    system(("touch -c " + vrmlFilename).c_str());

    std::ifstream file(vrmlFilename.c_str());
    if (!file.is_open()) // file doesnt exist
    {
        cerr << "could not open " << relativeTempPath + filename + CARTOON + WRL_EXT << endl;
        file.close(); // close the file
        chdir(currentPath.c_str());
        return NULL;
    }
    file.close(); // close the file

    // only adding a single model to the switch node by default
    Switch *newSwitch = new Switch;
    newSwitch->setAllChildrenOn();
    ((Group *)newSwitch)->addChild(loadStructure(vrmlFilename));

    chdir(currentPath.c_str());
    return newSwitch;
}

/// clean up old files
void PDBPlugin::cleanFiles(string &filename)
{
    // let the master clean up the files
    if (coVRMSController::instance()->isMaster())
    {
        // change to relative directory
        chdir(relativeTempPath.c_str());

        // remove temp files
        cmdInput.erase().append(REMOVE).append(filename).append(PDB_EXT);
        system(cmdInput.c_str());

        // want to remove morph frames
        if (filename.find("ff", 0) == 0)
        {
            cmdInput.erase().append(REMOVE).append(filename).append(CARTOON).append(WRL_EXT);
            system(cmdInput.c_str());
            cmdInput.erase().append(REMOVE).append(filename).append(STICK).append(WRL_EXT);
            system(cmdInput.c_str());
            cmdInput.erase().append(REMOVE).append(filename).append(SURFACE).append(WRL_EXT);
            system(cmdInput.c_str());
        }

        // change back to current directory
        chdir(currentPath.c_str());
    }
}

string PDBPlugin::getRelativeTempPath()
{
    return relativeTempPath;
}

string PDBPlugin::getCurrentPath()
{
    return currentPath;
}

/// enable user message
void PDBPlugin::showMessage(const string &message)
{
    messageLabel->setHighlighted(true);
    messageLabel->setString(message);
    messageHandle->setVisible(true);
    messageHandle->update();
}

/// disable user message
void PDBPlugin::hideMessage()
{
    messageHandle->setVisible(false);
    messageHandle->update();
}

// try reading in the file
void PDBPlugin::readSetFile(string file, vector<string> *list)
{
    chdir(relativeTempPath.c_str());
    char imageName[20];
    std::ifstream setFile(file.c_str());

    if (setFile.is_open())
    {
        while (!setFile.eof())
        {
            setFile.getline(imageName, 19);
            list->push_back(string(imageName));
        }
        setFile.close();
    }
    chdir(currentPath.c_str());
}

// used to create a ring of height structures high
void PDBPlugin::calcTrans(int index, int num_per_ring, float radius, Vec3f &trans)
{
    //check if num_per ring is greater than 0
    float baseangle = PI;
    if (num_per_ring > 0)
        baseangle = PI * ((index * 2.) / num_per_ring);
    int layer = index / num_per_ring;
    float x = radius * cos(baseangle);
    float y = radius * sin(baseangle);
    float z = ((radius * .4) * layer); //offset start the layers at the bottom of the screen
    trans.set(x, y, z);
}

void PDBPlugin::loadWarehouseFiles(vector<string> &list)
{
    //enable restricted mode
    _restrictedMode = true;
    _galaxyMode = true;

    // change into a special directory with all the proteins already converted in it
    chdir(relativeTempPath.c_str());

    cerr << "Directory " << relativeTempPath.c_str() << endl;

    int index = 0; //index in warehouse structure

    //reset world scale back to default
    cover->setScale(1.);

    vector<string>::iterator it = list.begin();
    for (; it < list.end(); it++)
    {
        // check the file size
        string sizetestfilename = (*it);
        if (sizetestfilename.empty())
            break;

        sizetestfilename.append(CARTOON).append(WRL_EXT);

        cerr << "FILE TRYING TO OPEN " << sizetestfilename << endl;
        long size = 0;
        std::ifstream currentfile(sizetestfilename.c_str());
        if (currentfile.is_open())
        {
            currentfile.seekg(0, ios::end);
            size = currentfile.tellg();
            cerr << "FILE SIZE " << size << endl;
            currentfile.close();
        }

        // if below restricted filesize
        if (size < _filesize && (index <= _maxstructures))
        {

            Switch *timeSwitch = new Switch;
            // copy loader list function into here
            string filename = (*it);

            string vrmlFilename;
            Switch *newSwitch = new Switch;
            vrmlFilename = filename + CARTOON + WRL_EXT;

            newSwitch->addChild(loadStructure(vrmlFilename));
            timeSwitch->addChild(newSwitch);

            // Create PDB PickBox: Initalzie default scale to 0.015
            PDBPickBox *pdbBox = new PDBPickBox(timeSwitch, _interaction, Vec3(0, 0, 0), Vec3(1, 1, 1), Widget::COL_RED, Widget::COL_YELLOW, Widget::COL_GREEN, filename, 0.015);
            pdbBox->addListener(this);

            // Add PDB box to scenegraph:
            _proteins.push_back(pdbBox);
            mainNode.get()->addChild(pdbBox->getNode());

            // adjust transform to create a ring of structures 6 layers high
            Vec3f trans;
            calcTrans(index, _ringnum, _radius, trans);
            pdbBox->setOrigin(trans); // SET THE DEFAULT LOCATION TRANS IN PDBPICKBOX
            pdbBox->resetOrigin(); // MOVE PROTEIN TO THAT LOCATION

            index++;
            updateNumTimesteps();
        }
    }
    //change back to the default directory
    chdir(currentPath.c_str());

    resetAllScale();

    cerr << "TOTAL NUMBER OF STRUCTURES " << index << endl;
}

void PDBPlugin::moveSequenceMarker(SequenceMessage *mm)
{
    //cerr << "got move message" << endl;
    // use the name to the active protein
    if (mm)
    {
        // iterate and pick matching pickbox
        vector<PDBPickBox *>::iterator it = _proteins.begin();
        for (; it < _proteins.end(); it++)
        {
            //cerr << "loadedname: " << (*it)->getName() << " vs filename: " << mm->filename << endl;
            if ((*it)->getName() == mm->filename && (*it)->getSequenceMarker()->isVisible())
            {
                //cerr << "found match" << endl;
                //cerr << "x: " << mm->x << " y: " << mm->y << " z: " << mm->z << endl;
                //(*it)->getSequenceMarker()->setPosition(Vec3(mm->x, mm->y, mm->z));
                (*it)->getSequenceMarker()->setPosition(Vec3(mm->x, -mm->z, mm->y));
                break;
            }
        }
    }

    // adjust the location of marker
}

void PDBPlugin::message(int toWhom, int type, int, const void *buf)
{

    // use type to determine if communicating between plugins
    if (type == MOVE_MARK)
    {
        //cerr << "found a move message" << endl;

        SequenceMessage *sm = (SequenceMessage *)buf;
        //sview->set(sm->filename);
        sview->setLoc(sm->x, sm->y, sm->z);
        moveSequenceMarker(sm);
        return;
    }

    // assume normal messsage (this is a bad idea with messages possibly sent to all plugins !!! need to fix!!!!)
    (void)type;

    // use message type to determine if the message is PDBPlugin or Sequence message
    PDBPluginMessage *mm = (PDBPluginMessage *)buf;

    // requires a string *added*
    switch (mm->token)
    {
    case LOAD_FILE:
    {
        //cerr << " load called" << endl;

        uidcounter++;

        //topsan->load(mm->filename);

        // check if file is a protein play list
        vector<string> list;
        string filename(mm->filename);

        readSetFile(filename, &list);
        int totalsize = 0;

        //find the total number of structures to load
        // (also load and convert any pdb files that are not in the local directory
        if (coVRMSController::instance()->isMaster())
        {

            vector<string>::iterator it = list.begin();
            for (; it < list.end(); it++)
            {
                // check the file size
                chdir(relativeTempPath.c_str());
                string sizetestfilename = (*it) + CARTOON + WRL_EXT;
                // create an input file stream
                std::ifstream currentfile;
                currentfile.open(sizetestfilename.c_str());
                // if the file doesnt exist download it
                if (!currentfile.is_open())
                {

                    // need to do this local else issue with enbedded control messages
                    cmdInput.erase().append(WGET).append(pdburl).append((*it)).append(PDB_EXT);

                    // download to file
                    int status = system(cmdInput.c_str());

                    if (!status)
                    {
                        // convert pdb to wrl
                        chdir(relativePymolPath.c_str());

                        cmdInput.erase().append(PYMOL).append(relativeTempPath.c_str());
                        std::cerr << "running 2: " << cmdInput << std::endl;
                        system(plugin->cmdInput.c_str());

                        chdir(currentPath.c_str());
                    }

                    // change to correct directory
                    chdir(relativeTempPath.c_str());
                    currentfile.open(sizetestfilename.c_str());
                }

                // openfile and test size
                //cerr << "trying to open: " << sizetestfilename.c_str() << endl;
                //currentfile.open(sizetestfilename.c_str());
                if (currentfile.is_open())
                {
                    currentfile.seekg(0, ios::end);
                    if (currentfile.tellg() < _filesize)
                        totalsize++; // FIND THE TOTAL NUMBER OF PROTEINS BELOW A CERTAIN SIZE //
                }
                currentfile.close(); // close the file is open
            }

            chdir(currentPath.c_str());
            coVRMSController::instance()->sendSlaves((char *)&totalsize, sizeof(int));
        }
        else
        {
            coVRMSController::instance()->readMaster((char *)&totalsize, sizeof(int));
        }

        // if structures exist in the list load them in
        if (totalsize)
        {
            // show loading message
            showMessage("loading " + filename + " warehouse");

            // call ware house loader method
            loadWarehouseFiles(list);

            //hide the message
            hideMessage();
        }
        else // cause of loading an single structure e.g. not the list
        {
            // call load single file structure
            string filename(mm->filename);

            // check the file size
            chdir(relativeTempPath.c_str());
            if (filename.size() == 4)
            {
                string testfilename = filename + CARTOON + WRL_EXT;

                // create an input file stream
                std::ifstream currentfile;

                currentfile.open(testfilename.c_str());
                // if the file doesnt exist download it
                if (!currentfile.is_open())
                {
                    if (downloadPDBFile(filename, UCSD))
                        loadPDB(filename, UCSD);
                }
                currentfile.close(); // close the file is open
            }
            else
            {
                string testfilename = filename + "-ff1" + CARTOON + WRL_EXT;

                // create an input file stream
                std::ifstream currentfile;

                currentfile.open(testfilename.c_str());
                // if the file doesnt exist download it
                if (!currentfile.is_open())
                {
                    if (downloadPDBFile(filename, YALE))
                        loadPDB(filename, YALE);
                }
                currentfile.close(); // close the file is open
            }
            chdir(currentPath.c_str());

            loadSingleStructure(filename);
        }
        break;
    }
    // requires a bool for high detail *added*
    case HIGH_DETAIL:
    {
        if (mm->high_detail_on)
        {
            highdetailCheckbox->setState(true);
        }
        else
        {
            highdetailCheckbox->setState(false);
        }
        break;
    }
    // requires a filename *added*
    /*case LOAD_ANI:
    {
      // call load PDB morph
      string filename(mm->filename);
      loadAniStructures(filename);
      break;
    }*/
    // requires a filename *added*
    case MOVE:
    {
        // call move function
        move(mm);
        break;
    }
    case STOP_MOVE:
    {
        // call stopmove function
        stopmove(mm);
        break;
    }
    case REMOVE_PROTEIN:
    {
        removeProtein(mm);
        break;
    }
    case SCALE_PROTEIN:
    {
        setScale(mm);
        break;
    }
    case SCALE_ALL:
    {
        setAllScale(mm);
        break;
    }
    case RESET_SCALE:
    {
        resetScale(mm);
        break;
    }
    // requires a frame number *added*
    case FRAME_UPDATE:
    {
        // change frame
        //int* curFrame = (int*)buf;
        //currentFrame = *curFrame;
        coVRAnimationManager::instance()->requestAnimationFrame(mm->framenumber);
        break;
    }
    // requires nothing *complete*
    case CLEAR_ALL:
    {
        // remove animation menu if it exists
        coVRAnimationManager::instance()->showAnimMenu(false);

        topsan->clear();
        sview->clear();

        //clear node of proteins
        clearSwitch();
        //updateAlignMenu(DELETE_ALL, 0);
        break;
    }
    case ADJUST_HUE:
    {
        // set the hue of the structure
        setHue(mm);
        break;
    }
    case VIEW:
    {
        // set the surface view
        setView(mm);
        break;
    }
    case MARK:
    {
        // enable/disable mark

        enableMark(mm);
        break;
    }
    case PDB_ALIGN:
    {
        //alignProtein(mm);
        break;
    }
    case LAYOUT_GRID:
    {
        layoutProteinsGrid();
        break;
    }
    case LAYOUT_CYL:
    {
        layoutProteinsCylinder();
        break;
    }
    }
}

void PDBPlugin::loadSingleStructure(string &filename)
{
    string message("Loading PDB structures ");
    message.append(filename);
    showMessage(message);
    if (filename.size() == 4)
    {
        loadingMorph = false;
    }
    else
    {
        loadingMorph = true;
    }
    loadDelay = MESSAGE_PREFRAME;
    loadFile = filename;
}
/*
void PDBPlugin::loadAniStructures(string& filename)
{
  string message("Loading PDB animation");
  loadDelay = MESSAGE_PREFRAME;
  showMessage(LOADING_MESSAGE);
  loadingMorph = true;
  loadFile = filename;    
}*/

bool PDBPlugin::fileBrowserEvent(cui::coFileBrowser *, std::string &file, std::string &ext, int, int)
{

    //cerr << "Browser Event: got " << file << endl;

    size_t last = file.find_last_of('/');
    string filename;

    if (last != string::npos && last + 1 < file.size())
    {
        filename = file.substr(last + 1, file.size() - (last + 1));
        if (filename.size() > 8)
        {
            filename = string("z") + filename.substr(filename.size() - 7, 7);
        }
        else if (filename.size() < 8)
        {
            char temp[5];
            temp[0] = 'z';
            temp[1] = (char)'0' + (uidcounter / 100) % 10;
            temp[2] = (char)'0' + (uidcounter / 10) % 10;
            temp[3] = (char)'0' + uidcounter % 10;
            temp[4] = '\0';
            filename = temp;
            filename = filename + ".pdb";
        }
    }
    else
    {
        return false;
    }

    if (ext == "pdb")
    {
        int status = 0;
        if (coVRMSController::instance()->isMaster())
        {
            system((string("cp ") + file + " " + relativeTempPath + "/" + filename).c_str());

            // convert pdb to wrl
            chdir(relativePymolPath.c_str());

            cmdInput.erase().append(PYMOL).append(relativeTempPath.c_str());
            std::cerr << "running 3: " << cmdInput << std::endl;
            status = system(cmdInput.c_str());

            chdir(currentPath.c_str());

            coVRMSController::instance()->sendSlaves((char *)&status, sizeof(int));
        }
        else
        {
            coVRMSController::instance()->readMaster((char *)&status, sizeof(int));
        }
    }

    PDBPluginMessage mm;
    mm.token = LOAD_FILE;

    //cerr << "Loading PDB ID: " << pdbID << endl;
    if (ext != "pdb")
    {
        setMMString(mm.filename, file.substr(file.length() - 8, 4));
    }
    else
    {
        setMMString(mm.filename, filename.substr(filename.length() - 8, 4));
    }
    cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME,
                       LOAD_FILE, sizeof(PDBPluginMessage), &mm);
    return true;
}

void PDBPlugin::setAllScale(PDBPluginMessage *mm)
{
    if (mm)
    {
        vector<PDBPickBox *>::iterator it = _proteins.begin();
        for (; it < _proteins.end(); it++)
        {
            (*it)->setScale(mm->scale);
        }
        it = _aniProteins.begin();
        for (; it < _aniProteins.end(); it++)
        {
            (*it)->setScale(mm->scale);
        }
    }
}

void PDBPlugin::resetAllScale()
{
    vector<PDBPickBox *>::iterator it = _proteins.begin();
    for (; it < _proteins.end(); it++)
    {
        (*it)->resetScale();
    }
    it = _aniProteins.begin();
    for (; it < _aniProteins.end(); it++)
    {
        (*it)->resetScale();
    }
}

void PDBPlugin::setAllMovable(bool movable)
{
    vector<PDBPickBox *>::iterator it = _proteins.begin();
    for (; it < _proteins.end(); it++)
    {
        (*it)->setMovable(movable);
    }
    it = _aniProteins.begin();
    for (; it < _aniProteins.end(); it++)
    {
        (*it)->setMovable(movable);
    }
}

void PDBPlugin::updateNumTimesteps()
{
    int maxSteps = 0;

    vector<PDBPickBox *>::iterator it = _aniProteins.begin();
    for (; it < _aniProteins.end(); it++)
    {
        if ((*it)->getNumTimesteps() > maxSteps)
        {
            maxSteps = (*it)->getNumTimesteps();
        }
    }

    coVRAnimationManager::instance()->setNumTimesteps(maxSteps, this);
}

/**
  * Button Event for a PDBPickBox
  **/
void PDBPlugin::pickBoxButtonEvent(cui::PickBox *interactPick, cui::InputDevice *dev, int button)
{
    // load structure menu
    if (button == 2)
    {
        if (dev->getButtonState(button) == 1) // button pressed?
        {
            // make sure pick box is PDBPickBox
            if (!dynamic_cast<PDBPickBox *>(interactPick))
                return;

            if (_selectedPickBox) // if true unselect pickbox
            {
                if (_selectedPickBox == (PDBPickBox *)interactPick) //if selecting same item close pick box
                    _selectedPickBox = NULL;
                else
                    _selectedPickBox = (PDBPickBox *)interactPick; // selecting a different item change
            }
            else // pick box does not exist
            {
                PDBPickBox *pickbox = dynamic_cast<PDBPickBox *>(interactPick);
                if (pickbox)
                    _selectedPickBox = pickbox;
                else
                    _selectedPickBox = NULL;
            }
        }
    }
    PDBPickBox *pickbox = dynamic_cast<PDBPickBox *>(interactPick);
    if (pickbox && !pickbox->isMorph())
    {
        topsan->set(pickbox->getName());
    }
}

/**
 * Pick box move event listener
 **/
void PDBPlugin::pickBoxMoveEvent(cui::PickBox *, cui::InputDevice *) {}

void PDBPlugin::resetPositions()
{
    //Vec3 trans(0.,0.,0.);
    vector<PDBPickBox *>::iterator it = _proteins.begin();
    for (; it < _proteins.end(); it++)
    {
        (*it)->resetOrigin();
    }

    //remove copys from plane of the screen
    alignpoint->removeChildren();
}

void PDBPlugin::setNamePopup(string name)
{
    nameMessage->setString(name);
    nameHandle->update();
}

void PDBPlugin::layoutProteinsCylinder()
{
    Vec3 posObj, posWld;
    int proteinsPerRing = (int)proteinsPerLevelPoti->getValue();

    float radsPerProtein = 2.0f * M_PI / (float)proteinsPerRing;

    float radius = radiusPoti->getValue();
    float dz = radius / 10.0 * cover->getScale();

    int ring, col, i;
    vector<PDBPickBox *>::iterator it = _proteins.begin();
    for (i = 0; it < _proteins.end(); it++, ++i)
    {
        ring = i / proteinsPerRing;
        col = i % proteinsPerRing;

        posWld[0] = sin(((float)(col)) * radsPerProtein) * radius;
        posWld[1] = cos(((float)(col)) * radsPerProtein) * radius;

        if (ring % 2)
        {
            posWld[2] = ((float)((ring / 2) + 1)) * -dz;
        }
        else
        {
            posWld[2] = ((float)((ring / 2))) * dz;
        }
        posObj = posWld * cover->getInvBaseMat();
        (*it)->setOrigin(posObj);
        (*it)->setPosition(posObj);
    }
    lastlayout = CYLINDER;
}

void PDBPlugin::layoutProteinsGrid()
{
    Vec3 posObj, posWld;
    float x, y, z;
    float sx, sy, sz; // start positions
    float dx, dy, dz; // distances between proteins
    int mx = (int)proteinsPerLevelPoti->getValue(); // maximum # proteins per dimension
    int i;

    sx = 0.0f;
    //sy = 400.0f;
    sz = 0.0f;
    //dx = 400.0f;
    dy = 0.0f;
    //dz = 400.0f;
    x = sx;
    y = 0.;
    z = sz;

    sy = dx = dz = radiusPoti->getValue() / 10.0 * cover->getScale();

    int row, col;
    vector<PDBPickBox *>::iterator it = _proteins.begin();
    for (i = 0; it < _proteins.end(); it++, ++i)
    {
        row = i / mx;
        col = i % mx;
        //posWld[0] = x;
        posWld[1] = sy;
        //posWld[2] = z;
        if (col % 2)
        {
            posWld[0] = ((float)((col / 2) + 1)) * -dx;
        }
        else
        {
            posWld[0] = ((float)((col / 2))) * dx;
        }
        if (row % 2)
        {
            posWld[2] = ((float)((row / 2) + 1)) * -dz;
        }
        else
        {
            posWld[2] = ((float)((row / 2))) * dz;
        }
        posObj = posWld * cover->getInvBaseMat();
        (*it)->setOrigin(posObj);
        (*it)->setPosition(posObj);
    }
    lastlayout = GRID;
}

/*
void PDBPlugin::layoutProteins()
{
  Vec3 posObj, posWld;
  float x,y,z;
  float sx,sy,sz;   // start positions
  float dx,dy,dz;   // distances between proteins
  int mx = 5;       // maximum # proteins per dimension
  int i;

  sx = -1000.0f;
  sy = 0.0f;
  sz = 1000.0f;
  dx = 400.0f;
  dy = -400.0f;
  dz = -400.0f;
  x = sx;
  y = sy;
  z = sz;

  vector<PDBPickBox*>::iterator it = _proteins.begin();
  for(i=0; it < _proteins.end(); it++, ++i)
  {
    posWld[0] = x;
    posWld[1] = y;
    posWld[2] = z;
    posObj = posWld * cover->getInvBaseMat();
    (*it)->setOrigin(posObj);
    (*it)->setPosition(posObj);
    x += dx;
    if (((i+1) % mx) == 0)
    {
      x  = sx;
      z += dz;
    }
  }
}
*/

AlignPickBox::AlignPickBox(MatrixTransform *node, cui::Interaction *interaction, const Vec3 &min, const Vec3 &max, const Vec4 &c1, const Vec4 &c2, const Vec4 &c3)
    : PickBox(interaction, min, max, c1, c2, c3)
{
    rootmt = new MatrixTransform();
    _isMoving = false;
    _isSelected = false;
    setMovable(true);
    setShowWireframe(false);
    _mainSwitch = node;
    rootmt->addChild(_node.get());
    _scale->addChild(_mainSwitch);
    _origin.set(0., 0., 0.);
    setScale(1.0); // set default scale size

    _scale->setNodeMask(_scale->getNodeMask() | (2));

    // disable intersection with the switch node
    node->setNodeMask(_scale->getNodeMask() & (~2));

    Matrix m;
    m.makeIdentity();
    ComputeBBVisitor *compBB = new ComputeBBVisitor(m);
    //_scale->accept(*compBB);
    _mainSwitch->accept(*compBB);
    BoundingBox bb = compBB->getBound();
    delete compBB;
    compBB = NULL;

    // shift structure to the center of the bounding box
    boxCenter = bb.center();

    Vec3 distance = (bb.corner(0) - bb.center());
    boxSize = Vec3(fabs(distance[0]) * 2, fabs(distance[1]) * 2, fabs(distance[2]) * 2);

    setBoxSize(Vec3(0, 0, 0));
}

AlignPickBox::~AlignPickBox()
{
}

MatrixTransform *AlignPickBox::getRoot()
{
    return rootmt;
}

void AlignPickBox::cursorEnter(InputDevice *dev)
{
    if (!_interactionA->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(_interactionA);
    }
    PickBox::cursorEnter(dev);
}

void AlignPickBox::cursorLeave(InputDevice *dev)
{
    //_selected = false;
    if (_interactionA->isRegistered())
    {
        coInteractionManager::the()->unregisterInteraction(_interactionA);
    }
    PickBox::cursorLeave(dev);
}

void AlignPickBox::cursorUpdate(InputDevice *dev)
{
    if (dev != dev->_interaction->_wandR)
        return;
    if (_isMoving)
    {
        // Compute difference matrix between last and current wand:
        Matrix invLastWand2w = Matrix::inverse(_lastWand2w);
        Matrix wDiff = invLastWand2w * dev->getI2W();

        // Volume follows wand movement:
        Matrix mark2w = CUI::computeLocal2Root(_node.get());
        Matrix w2o = Matrix::inverse(CUI::computeLocal2Root(rootmt));
        _node->setMatrix(mark2w * wDiff * w2o);
        //Matrix m;
        //m.makeScale(1.0/cover->getScale(), 1.0/cover->getScale(), 1.0/cover->getScale());
        //_node->setMatrix(_node->getMatrix() * Matrix::inverse(w2o) * wDiff * w2o);
        //Vec3 vec = _node->getMatrix().getTrans();
        //cerr << "MarkerPos: x: " << vec.x() << " y: " << vec.y() << " z: " << vec.z() << endl;
    }
    PickBox::cursorUpdate(dev);
}

void AlignPickBox::buttonEvent(InputDevice *dev, int button)
{
    if (dev != dev->_interaction->_wandR)
        return;

    switch (button)
    {
    case 0:
        if (dev->getButtonState(button) == 1) // button pressed?
        {
            _isMoving = true;
        }
        else if (dev->getButtonState(button) == 0) // button released?
        {
            _isMoving = false;
        }
        break;
    default:
        break;
    }
    PickBox::buttonEvent(dev, button); // call parent
}

void AlignPickBox::resetPos()
{
    Matrix m;
    m.makeIdentity();
    _node->setMatrix(m);
}

void AlignPickBox::setMovable(bool b)
{
    if (b)
    {
        setBoxSize(boxCenter, boxSize);
    }
    else
    {
        setBoxSize(Vec3(0, 0, 0));
    }
}

//-----------------------------------------
// class PDBPickBox
//-----------------------------------------

PDBPickBox::PDBPickBox(Switch *node, cui::Interaction *interaction, const Vec3 &min, const Vec3 &max, const Vec4 &c1, const Vec4 &c2, const Vec4 &c3,
                       string &name, float scale, Matrix *basemat)
    : PickBox(interaction, min, max, c1, c2, c3)
{
    adjustmt = new MatrixTransform();
    _isMoving = false;
    _isFading = false;
    _isHighDetail = true;
    _isSelected = false;
    _overridecolor = false;
    _isMorph = false;
    _name = name;
    _defaultscale = scale;
    _hue = 0.;
    setMovable(true);
    setShowWireframe(false);
    _mainSwitch = node;
    adjustmt->addChild(_mainSwitch);
    _scale->addChild(adjustmt);
    _currentTimestep = 0;
    _origin.set(0., 0., 0.);
    _mainSwitch->setSingleChildOn(_currentTimestep);
    setScale(1.0); // set default scale size

    //this->setShowWireframe(true);

    /*Matrix ident;
  ident.makeIdentity();
  MatrixTransform * tempmt = dynamic_cast<MatrixTransform *>(_mainSwitch);
  if(tempmt)
  {
    tempmt->setMatrix(ident);
  }*/

    // initalize -versions to hold the different views
    _versions = new osg::Group;

    // reenable the intersction test for the scale node else can not intersect with the marker
    _scale->setNodeMask(_scale->getNodeMask() | (2));

    // disable intersection with the switch node
    node->setNodeMask(_scale->getNodeMask() & (~2));

    //add current version to the versions node (cartoon)
    setVersion(((Group *)node->getChild(0))->getChild(0));
    //cerr << "version name is " << ((Group*)node->getChild(0))->getChild(0)->getName() << endl;

    if (node->getNumChildren() > 1)
    {
        Group *g = new Group();
        for (unsigned int i = 0; i < node->getNumChildren(); i++)
        {
            g->addChild(((Group *)node->getChild(i))->getChild(0));
        }
        setVersionMorph(g);
    }

    //set bounding size
    //ComputeBBVisitor* compBB = new ComputeBBVisitor(_scale->getMatrix());
    Matrix m;
    m.makeIdentity();
    ComputeBBVisitor *compBB = new ComputeBBVisitor(m);
    //_scale->accept(*compBB);
    _mainSwitch->accept(*compBB);
    BoundingBox bb = compBB->getBound();
    delete compBB;
    compBB = NULL;

    // shift structure to the center of the bounding box
    boxCenter = bb.center();
    //boxCenter = Vec3(boxCenter.x(), boxCenter.z(), -boxCenter.y());
    //cerr << boxCenter.x() << " " << boxCenter.y() << " " << boxCenter.z() << endl;
    //osg::Matrix recenterMat = _scale->getMatrix();
    //recenterMat.setTrans(-boxCenter);
    //_scale->setMatrix(recenterMat);
    Matrix amt;
    amt.makeTranslate(-boxCenter);
    adjustmt->setMatrix(amt);
    Vec3 distance = (bb.corner(0) - bb.center());
    Vec3 boxSize(fabs(distance[0]) * 2, fabs(distance[1]) * 2, fabs(distance[2]) * 2);

    setBoxSize(boxSize);

    //default protein location
    if (basemat)
        setMatrix(*basemat);
    //cerr << "init pdb" << endl;
    if (!lowdetailVisitor)
        lowdetailVisitor = new LowDetailTransVisitor;
    if (!highdetailVisitor)
        highdetailVisitor = new HighDetailTransVisitor;
    if (!frameVisitor)
        frameVisitor = new FrameVisitor;
    //cerr << "finished init pdb" << endl;
}

PDBPickBox::~PDBPickBox() {}

void PDBPickBox::setMorph(bool b)
{
    _isMorph = b;
}

bool PDBPickBox::isMorph()
{
    return _isMorph;
}

void PDBPickBox::addChild(osg::Node *node)
{
    _scale->addChild(node);
}

void PDBPickBox::addMarker(osg::Node *node)
{

    adjustmt->addChild(node);
}

void PDBPickBox::cursorEnter(InputDevice *dev)
{
    if (!_interactionA->isRegistered() && _isMovable)
        coInteractionManager::the()->registerInteraction(_interactionA);
    plugin->setNamePopup("Name:" + _name);
    string imagefilename;
    imagefilename.append(_name);
    imagefilename.append(".tif");
    //plugin->loadDataImage(imagefilename);

    PickBox::cursorEnter(dev);
}

void PDBPickBox::cursorLeave(InputDevice *dev)
{
    if (_interactionA->isRegistered())
        coInteractionManager::the()->unregisterInteraction(_interactionA);
    plugin->setNamePopup("Name:         ");
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    string imagefilename;
    //plugin->loadDataImage(imagefilename);

    PickBox::cursorLeave(dev);
}

void PDBPickBox::cursorUpdate(InputDevice *dev)
{
    if (dev != dev->_interaction->_wandR)
        return;
    if (_isMovable)
    {
        if (_isMoving)
        {
            //check lock
            if (_lockname.size() == 0)
            {
                Matrix i2w = dev->getI2W();
                move(_lastWand2w, i2w);

                // send move message
                PDBPluginMessage mm;
                plugin->setMMString(mm.filename, _name);
                plugin->setMMString(mm.hostname, plugin->getMyHost());
                mm.trans = getMatrix();
                mm.token = MOVE;
                cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME_OTHERS,
                                   1, sizeof(PDBPluginMessage), &mm);
            }
        }
    }

    PickBox::cursorUpdate(dev);
}

void PDBPickBox::buttonEvent(InputDevice *dev, int button)
{
    if (dev != dev->_interaction->_wandR)
        return;

    switch (button)
    {
    case 0:
        if (dev->getButtonState(button) == 1) // button pressed?
        {
            _isMoving = true;
        }
        else if (dev->getButtonState(button) == 0) // button released?
        {
            _isMoving = false;
            PDBPluginMessage mm;
            plugin->setMMString(mm.filename, _name);
            plugin->setMMString(mm.hostname, plugin->getMyHost());
            mm.token = STOP_MOVE;
            cover->sendMessage(PDBPluginModule, coVRPluginSupport::TO_SAME_OTHERS,
                               1, sizeof(PDBPluginMessage), &mm);
        }
        break;
    default:
        break;
    }
    PickBox::buttonEvent(dev, button); // call parent
}

bool PDBPickBox::getLock(string hostname)
{
    if ((hostname == _lockname) || (_lockname.size() == 0))
    {
        _lockname = hostname;
        return true;
    }
    else
    {
        return false;
    }
}

void PDBPickBox::unlock(string sysname)
{
    if (sysname == _lockname)
        _lockname.erase();
}

void PDBPlugin::move(PDBPluginMessage *mm)
{
    if (mm)
    {
        // iterate and find pick with matching name
        vector<PDBPickBox *>::iterator it = _proteins.begin();
        for (; it < _proteins.end(); it++)
        {
            if ((*it)->getName() == mm->filename)
            {
                if ((*it)->getLock(mm->hostname))
                {
                    (*it)->setMatrix(mm->trans);
                    break;
                }
            }
        }
        it = _aniProteins.begin();
        for (; it < _aniProteins.end(); it++)
        {
            if ((*it)->getName() == mm->filename)
            {
                if ((*it)->getLock(mm->hostname))
                {
                    (*it)->setMatrix(mm->trans);
                    break;
                }
            }
        }
    }
}

string PDBPlugin::getMyHost()
{
    return myHost;
}

void PDBPlugin::setHue(PDBPluginMessage *mm)
{
    if (mm)
    {
        // iterate and pick matching pickbox
        vector<PDBPickBox *>::iterator it = _proteins.begin();
        for (; it < _proteins.end(); it++)
        {
            if ((*it)->getName() == mm->filename)
            {
                (*it)->setHue(mm->hue);
                (*it)->overrideHue(mm->hue_mode_on);
                break;
            }
        }
        it = _aniProteins.begin();
        for (; it < _aniProteins.end(); it++)
        {
            if ((*it)->getName() == mm->filename)
            {
                (*it)->setHue(mm->hue);
                (*it)->overrideHue(mm->hue_mode_on);
                break;
            }
        }
    }
}

void PDBPlugin::setView(PDBPluginMessage *mm)
{
    if (mm)
    {
        // iterate and pick matching pickbox
        vector<PDBPickBox *>::iterator it = _proteins.begin();
        for (; it < _proteins.end(); it++)
        {
            if ((*it)->getName() == mm->filename)
            {
                string type(mm->viewtype);
                (*it)->setView(type, mm->view_on);
                break;
            }
        }
        it = _aniProteins.begin();
        for (; it < _aniProteins.end(); it++)
        {
            if ((*it)->getName() == mm->filename)
            {
                string type(mm->viewtype);
                (*it)->setView(type, mm->view_on);
                break;
            }
        }
    }
}

void PDBPlugin::removeProtein(PDBPluginMessage *mm)
{
    if (mm)
    {
        // need to remove the pickbox from the vector
        vector<PDBPickBox *>::iterator it = _proteins.begin();
        for (; it < _proteins.end(); it++)
        {
            //if((*it)->getName() == mm->filename)
            if ((*it)->uid == mm->uid)
            {
                updateAlignMenu(ALIGN_DELETE, mm->uid);
                // indiate a protein is being removed
                //_removingProtein = true;
                if ((*it)->getSequenceMarker()->isVisible())
                {
                    sview->clearDisplay();
                }
                if (_selectedPickBox)
                {
                    if (_selectedPickBox->getName() == mm->filename)
                    {
                        // set selected pickbox back to NULL
                        _selectedPickBox = NULL;

                        // hide the menu
                        proteinHandle->setVisible(false);
                    }
                }

                delete (*it);

                _proteins.erase(it);
                topsan->remove(mm->filename);
                sview->remove(mm->filename);
                return;
            }
        }

        it = _aniProteins.begin();
        for (; it < _aniProteins.end(); it++)
        {
            //if((*it)->getName() == mm->filename)
            if ((*it)->uid == mm->uid)
            {
                // indiate a protein is being removed
                //_removingProtein = true;
                if (_selectedPickBox)
                {
                    if (_selectedPickBox->getName() == mm->filename)
                    {
                        // set selected pickbox back to NULL
                        _selectedPickBox = NULL;

                        // hide the menu
                        proteinHandle->setVisible(false);
                    }
                }

                delete (*it);

                _aniProteins.erase(it);
                updateNumTimesteps();
                return;
            }
        }
    }
}

void PDBPlugin::setScale(PDBPluginMessage *mm)
{
    if (mm)
    {
        // iterate and pick matching pickbox
        vector<PDBPickBox *>::iterator it = _proteins.begin();
        for (; it < _proteins.end(); it++)
        {
            if ((*it)->getName() == mm->filename)
            {
                (*it)->setScale(mm->scale);
                break;
            }
        }
        it = _aniProteins.begin();
        for (; it < _aniProteins.end(); it++)
        {
            if ((*it)->getName() == mm->filename)
            {
                (*it)->setScale(mm->scale);
                break;
            }
        }
    }
}

void PDBPlugin::enableMark(PDBPluginMessage *mm)
{
    if (mm)
    {
        // iterate and pick matching pickbox
        vector<PDBPickBox *>::iterator it = _proteins.begin();
        for (; it < _proteins.end(); it++)
        {
            //if((*it)->getName() == mm->filename)
            if ((*it)->uid == mm->uid)
            {
                if (mm->mark_on)
                {
                    for (vector<PDBPickBox *>::iterator i = _proteins.begin(); i != _proteins.end(); i++)
                    {
                        (*i)->getSequenceMarker()->setVisible(false);
                    }
                    sview->clearDisplay();
                    sview->set(mm->filename);
                }
                else
                {
                    sview->clearDisplay();
                }

                (*it)->getSequenceMarker()->setVisible(mm->mark_on);
                break;
            }
        }
    }
}

/*void PDBPlugin::alignProtein(PDBPluginMessage* mm)
{
  if(mm)
  {
    // iterate and pick matching pickbox
    vector<PDBPickBox*>::iterator it = _proteins.begin();
    for(; it < _proteins.end(); it++)
    {
      if((*it)->getName() == mm->filename)
      {
	alignpoint->addChild((*it)->copy(), string(mm->filename));
        break;
      }
    }
  }
}*/

void PDBPlugin::alignProteins(int i)
{
    alignpoint->setAlign(i);
}

void PDBPlugin::clearAlign()
{
    alignpoint->removeChildren();
}

void PDBPlugin::alignReset()
{
    alignpoint->reset();
}

void PDBPlugin::resetScale(PDBPluginMessage *mm)
{
    if (mm)
    {
        // iterate and pick matching pickbox
        vector<PDBPickBox *>::iterator it = _proteins.begin();
        for (; it < _proteins.end(); it++)
        {
            if ((*it)->getName() == mm->filename)
            {
                (*it)->resetScale();
                break;
            }
        }
    }
}

void PDBPlugin::stopmove(PDBPluginMessage *mm)
{
    if (mm)
    {
        // iterate and pick matching pickbox
        vector<PDBPickBox *>::iterator it = _proteins.begin();
        for (; it < _proteins.end(); it++)
        {
            if ((*it)->getName() == mm->filename)
            {
                (*it)->unlock(mm->hostname);
                break;
            }
        }
    }
}

void PDBPickBox::setTimestep(int step)
{
    _currentTimestep = step;
    if (_currentTimestep >= _mainSwitch->getNumChildren())
        _currentTimestep = _mainSwitch->getNumChildren() - 1;
    else if (_currentTimestep < 0)
        _currentTimestep = 0;
    _mainSwitch->setSingleChildOn(_currentTimestep);
}

int PDBPickBox::getNumTimesteps()
{
    return _mainSwitch->getNumChildren();
}

string PDBPickBox::getName()
{
    return _name;
}

/*void PDBPickBox::setDetailLevel(bool detaillevel)
{
  _isHighDetail = detaillevel;
}

bool PDBPickBox::getDetailLevel()
{
  return _isHighDetail;
}

void PDBPickBox::setFade(bool fade)
{
  _isFading = fade;
}

bool PDBPickBox::getFade()
{
  return _isFading;
}*/

bool PDBPickBox::isMoving()
{
    return _isMoving;
}

void PDBPickBox::resetScale()
{
    setScale(_defaultscale);
}

/**
 * Need to continually execute during each preframe
 * */
/*void PDBPickBox::computeFadeLevel()
{
  //if(_isFading) 
  //{
     lowdetailVisitor->enableTransparency(true);
     lowdetailVisitor->setScale(cover->getScale());
     ((Switch*)_mainSwitch->getChild(_currentTimestep))->getChild(0)->accept(*lowdetailVisitor);
     //highdetailVisitor->enableTransparency(true);
     //highdetailVisitor->setScale(cover->getScale());
     //((Switch*)_mainSwitch->getChild(_currentTimestep))->getChild(1)->accept(*highdetailVisitor);
  //}
}*/

void PDBPickBox::setOrigin(osg::Vec3 trans)
{
    _origin = trans;
}

void PDBPickBox::resetOrigin()
{
    setPosition(_origin);
}

/**
 * Need to execute only once when selection first made
 * */
/*void PDBPickBox::computeDetailLevel()
{
 if(!_isFading)
 {
    if(_isHighDetail)
    {
       highdetailVisitor->enableTransparency(false);
       _mainSwitch->getChild(_currentTimestep)->accept(*highdetailVisitor);
    }
    else
    {
      lowdetailVisitor->enableTransparency(false);
      _mainSwitch->getChild(_currentTimestep)->accept(*lowdetailVisitor);
    }
  }
}*/

// creates a deep copy of the node structure of this pickbox
Group *PDBPickBox::copy()
{
    return new Group(*_mainSwitch, CopyOp::DEEP_COPY_ALL);
}

float PDBPickBox::getHue()
{
    return _hue;
}

// align point
AlignPoint::AlignPoint(cui::Interaction *interaction, const Vec3 &min, const Vec3 &max, const Vec4 &c1, const Vec4 &c2, const Vec4 &c3, float hue, float size, SequenceViewer *sview, PDBPlugin *pdbp)
    : PickBox(interaction, min, max, c1, c2, c3)
{
    inter = interaction;
    _isMoving = false;
    _hue = hue;
    _basehue = 0.0;
    Geode *geode = new Geode;
    TessellationHints *hints = new TessellationHints();
    hints->setDetailRatio(0.3f);

    setShowWireframe(false);

    _sview = sview;

    _pdbp = pdbp;

    Vec3 boxCenter(0.0f, 0.0f, 0.0f);
    Box *boxShape = new Box(boxCenter, size);
    _shapeDrawable = new ShapeDrawable(boxShape);
    _shapeDrawable->setTessellationHints(hints);
    _shapeDrawable->setColor(calculateColor(_hue));
    geode->addDrawable(_shapeDrawable);
    _shapeDrawable->setUseDisplayList(false);

    StateSet *stateSet = geode->getOrCreateStateSet();
    stateSet->setMode(GL_RESCALE_NORMAL, StateAttribute::ON);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    // create a branch for the copied proteins (reason to rescale them);
    _protein_size = new MatrixTransform;
    Matrix scalemat;
    scalemat.makeScale(Vec3(15., 15., 15.));
    _protein_size->setMatrix(scalemat);
    _scale->addChild(_protein_size);
    _scale->addChild(geode);

    cover->getScene()->addChild(_node.get());

    mammothpath = coCoviseConfig::getEntry("COVER.Plugin.PDB.MAMMOTHDir");
    if (mammothpath == "")
    {
        cerr << "PDB: MAMMOTHDir not set in config file." << endl;
    }
    PDBpath = coCoviseConfig::getEntry("COVER.Plugin.PDB.PDBDir");
}

Vec4 AlignPoint::calculateColor(float hue)
{
    float r, g, b;
    Vec4 color;
    vvToolshed::HSBtoRGB(hue, 1, 1, &r, &g, &b);
    color[0] = r;
    color[1] = g;
    color[2] = b;
    color[3] = 1.0;
    return color;
}

void AlignPoint::setColor(Vec4 vec)
{
    _shapeDrawable->setColor(vec);
}

void AlignPoint::setVisible(bool visible)
{
    if (visible)
    {
        _node->setNodeMask(~0);
    }
    else
    {
        _node->setNodeMask(0);
    }
}

void AlignPickBox::setColor(Vec4 v)
{
    color = v;
}

void AlignPoint::setAlign(int index)
{
    bool add = false;
    string iname = _pdbp->_proteins[index]->getName();

    if (!_pdbp->alButtons[index].first->getState())
    {
        vector<osg::Group *>::iterator it = list.begin();
        vector<std::string>::iterator it2 = namelist.begin();
        std::vector<AlignPickBox *>::iterator it3 = pbvec.begin();
        for (int i = 0; i < namelist.size(); i++)
        {
            if (iname == namelist[i])
            {
                _protein_size->removeChild(list[i]);
                list.erase(it);
                namelist.erase(it2);
                delete pbvec[i];
                pbvec.erase(it3);
            }
            it++;
            it2++;
            it3++;
        }
    }
    else
    {

        for (int i = 0; i < namelist.size(); i++)
        {
            if (iname == namelist[i])
            {
                //do stuff
                return;
            }
        }
        add = true;
        namelist.push_back(iname);
    }

    /*for(int i = 0; i < allist.size(); i++)
  {
    cerr << allist[i] << endl;
  }*/

    if (namelist.size() == 0)
    {
        this->setVisible(false);
        removeChildren();
    }
    else
    {
        this->setVisible(true);
    }

    if (add)
    {

        Group *grp = _pdbp->_proteins[index]->copy();

        MatrixTransform *child = new MatrixTransform();

        child->addChild(grp);

        AlignPickBox *apb = new AlignPickBox(child, inter, Vec3(0, 0, 0), Vec3(1, 1, 1), Widget::COL_RED, Widget::COL_YELLOW, Widget::COL_GREEN);

        pbvec.push_back(apb);

        list.push_back(apb->getRoot());
        _protein_size->addChild(apb->getRoot());
    }

    for (int i = 0; i < pbvec.size(); i++)
    {
        _basehue = (0.15 * ((float)i + 1.0));

        pbvec[i]->setColor(calculateColor(_basehue));

        // need to set the protein color to a default color
        StateSet *state = pbvec[i]->getRoot()->getOrCreateStateSet();
        Material *material = new Material;
        material->setDiffuse(Material::FRONT_AND_BACK, calculateColor(_basehue));
        material->setAlpha(Material::FRONT_AND_BACK, 0.75);
        state->setAttributeAndModes(material, StateAttribute::OVERRIDE);
        state->setMode(GL_BLEND, StateAttribute::ON);
    }

    for (int i = 0; i < _pdbp->_proteins.size(); i++)
    {
        float tx, ty, tz;
        bool tstate;
        tx = _pdbp->alButtons[i].second->getXpos();
        ty = _pdbp->alButtons[i].second->getYpos();
        tz = _pdbp->alButtons[i].second->getZpos();
        tstate = _pdbp->alButtons[i].second->getState();
        _pdbp->alPanel->removeElement(_pdbp->alButtons[i].second);
        delete _pdbp->alButtons[i].second;
        coTextButtonGeometry *tempbg = new coTextButtonGeometry(_pdbp->alButtonWidth, _pdbp->alButtonHeight, _pdbp->_proteins[i]->getName());
        if (!_pdbp->alButtons[i].first->getState())
        {
            tempbg->setColors(0.1, 0.1, 0.1, 1.0, 0.9, 0.9, 0.9, 1.0);
            tstate = false;
        }
        else
        {
            string alname = _pdbp->_proteins[i]->getName();
            int j;
            for (j = 0; j < namelist.size(); j++)
            {
                if (alname == namelist[j])
                {
                    break;
                }
            }
            tempbg->setColors(0.1, 0.1, 0.1, 1.0, pbvec[j]->color.x(), pbvec[j]->color.y(), pbvec[j]->color.z(), pbvec[j]->color.w());
        }
        _pdbp->alButtons[i].second = new coToggleButton(tempbg, _pdbp);
        _pdbp->alButtons[i].second->setPos(tx, ty, tz);
        _pdbp->alButtons[i].second->setState(tstate, false);
        _pdbp->alPanel->addElement(_pdbp->alButtons[i].second);
    }

    if (namelist.size() == 0)
    {
        return;
    }

    char *tempath = getcwd(NULL, 0);

    chdir(mammothpath.c_str());

    int status;
    if (coVRMSController::instance()->isMaster())
    {
        std::ofstream jobfile;
        jobfile.open((mammothpath + "/list").c_str(), ios::trunc);
        jobfile << "MAMMOTH\n";
        for (int i = 0; i < namelist.size(); i++)
        {
            jobfile << PDBpath + "/" + namelist[i] + ".pdb\n";
        }
        jobfile << "\n";
        jobfile.close();

        system((string("mammothmult.g77 list -rot")).c_str());

        sleep(1);

        coVRMSController::instance()->sendSlaves((char *)&status, sizeof(int));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&status, sizeof(int));
    }

    chdir(tempath);

    vector<Vec3> center;
    vector<Matrix> rotation;
    vector<Vec3> translation;
    float r11, r12, r13, r21, r22, r23, r31, r32, r33, xc, yc, zc, xt, yt, zt;

    std::ifstream rotfile;
    rotfile.open((mammothpath + "/list-FINAL.rot").c_str(), ios::in);

    if (rotfile.fail())
    {
        cerr << "Problem reading file " << mammothpath + "/list-FINAL.rot" << endl;
    }

    string line;

    if (!rotfile.eof())
    {
        getline(rotfile, line);
    }

    bool read = false;

    while (!rotfile.eof())
    {
        int pos = line.find("Str#");

        if (pos == 0)
        {
            getline(rotfile, line);
            read = true;
            continue;
        }

        if (read)
        {
            sscanf(line.c_str(), "%*s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f", &r11, &r12, &r13, &r21, &r22, &r23, &r31, &r32, &r33, &xc, &yc, &zc, &xt, &yt, &zt);
            center.push_back(Vec3(-xc, -yc, -zc));
            translation.push_back(Vec3(xt, yt, zt));
            Matrix m;
            m.makeIdentity();
            m(0, 0) = r11;
            m(1, 0) = r12;
            m(2, 0) = r13;
            m(0, 1) = r21;
            m(1, 1) = r22;
            m(2, 1) = r23;
            m(0, 2) = r31;
            m(1, 2) = r32;
            m(2, 2) = r33;

            rotation.push_back(m);
        }

        getline(rotfile, line);
    }

    Matrix centermat, transmat;

    Matrix comp;

    comp.makeIdentity();

    comp(2, 1) = 1;
    comp(1, 1) = 0;
    comp(1, 2) = -1;
    comp(2, 2) = 0;

    for (int i = 0; i < list.size(); i++)
    {
        centermat.makeTranslate(center[i]);
        transmat.makeTranslate(translation[i]);

        ((MatrixTransform *)list[i])->setMatrix(comp * _sview->getInverse(namelist[i]).second * _sview->getInverse(namelist[i]).first * centermat * rotation[i] * transmat);
    }

    Matrix m;
    m.makeIdentity();
    ComputeBBVisitor *compBB = new ComputeBBVisitor(m);
    _protein_size->accept(*compBB);
    BoundingBox bb = compBB->getBound();
    delete compBB;
    compBB = NULL;

    Vec3 boxCenter = bb.center();

    Vec3 distance = (bb.corner(0) - bb.center());
    Vec3 boxSize(fabs(distance[0]) * 2, fabs(distance[1]) * 2, fabs(distance[2]) * 2);

    setBoxSize(boxCenter, boxSize);
}

void AlignPoint::removeChildren()
{
    //remove all children
    for (int i = 0; i < list.size(); i++)
    {
        _protein_size->removeChild(list[i]);
        delete pbvec[i];
    }
    list.clear();
    namelist.clear();
    pbvec.clear();

    for (int i = 0; i < _pdbp->alButtons.size(); i++)
    {
        _pdbp->alButtons[i].first->setState(false, false);
        _pdbp->alButtons[i].second->setState(false, false);
    }

    for (int i = 0; i < _pdbp->_proteins.size(); i++)
    {
        float tx, ty, tz;
        bool tstate;
        tx = _pdbp->alButtons[i].second->getXpos();
        ty = _pdbp->alButtons[i].second->getYpos();
        tz = _pdbp->alButtons[i].second->getZpos();
        tstate = false;
        _pdbp->alPanel->removeElement(_pdbp->alButtons[i].second);
        delete _pdbp->alButtons[i].second;
        coTextButtonGeometry *tempbg = new coTextButtonGeometry(_pdbp->alButtonWidth, _pdbp->alButtonHeight, _pdbp->_proteins[i]->getName());
        tempbg->setColors(0.1, 0.1, 0.1, 1.0, 0.9, 0.9, 0.9, 1.0);
        _pdbp->alButtons[i].second = new coToggleButton(tempbg, _pdbp);
        _pdbp->alButtons[i].second->setPos(tx, ty, tz);
        _pdbp->alButtons[i].second->setState(tstate, false);
        _pdbp->alPanel->addElement(_pdbp->alButtons[i].second);
    }

    this->setVisible(false);

    //reset basehue
    _basehue = 0.0;
}

void AlignPoint::reset()
{
    for (int i = 0; i < pbvec.size(); i++)
    {
        pbvec[i]->resetPos();
    }
}

void AlignPoint::setMovable(string name, bool set)
{
    for (int i = 0; i < namelist.size(); i++)
    {
        if (name == namelist[i])
        {
            pbvec[i]->setMovable(set);
            return;
        }
    }
}

void AlignPoint::cursorEnter(InputDevice *dev)
{
    if (!_interactionA->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(_interactionA);
    }
    setColor(calculateColor(1 - _hue));
    PickBox::cursorEnter(dev);
}

void AlignPoint::cursorLeave(InputDevice *dev)
{
    if (_interactionA->isRegistered())
    {
        coInteractionManager::the()->unregisterInteraction(_interactionA);
    }
    setColor(calculateColor(_hue));
    PickBox::cursorLeave(dev);
}

void AlignPoint::cursorUpdate(InputDevice *dev)
{
    if (dev != dev->_interaction->_wandR)
        return;

    // Compute difference matrix between last and current wand:
    if (_isMoving)
    {
        // Compute difference matrix between last and current wand:
        Matrix invLastWand2w = Matrix::inverse(_lastWand2w);
        Matrix wDiff = invLastWand2w * dev->getI2W();

        // Volume follows wand movement:
        Matrix box2w = getB2W();

        // need to make adjust due to _scale node used in calculation
        Matrix invScale = Matrix::inverse(_scale->getMatrix());
        _node->setMatrix(invScale * box2w * wDiff);
    }
    PickBox::cursorUpdate(dev);
}

void AlignPoint::buttonEvent(InputDevice *dev, int button)
{
    if (dev != dev->_interaction->_wandR)
        return;

    switch (button)
    {
    case 0:
        if (dev->getButtonState(button) == 1) // button pressed?
        {
            _isMoving = true;
        }
        else if (dev->getButtonState(button) == 0) // button released?
        {
            _isMoving = false;
        }
        break;
    default:
        break;
    }
    PickBox::buttonEvent(dev, button); // call parent
}
/*
void AlignPoint::setBoxSize()
{
  //ComputeBBVisitor* compBB = new ComputeBBVisitor(_scale->getMatrix());
  //_scale->accept(*compBB);
  //BoundingBox bb = compBB->getBound();
  //delete compBB;
  //compBB = NULL;
  
  
  Vec3 distance(100.,100.,100.);
  //Vec3 distance = (bb.corner(0) - bb.center());
  Vec3 boxSize(fabs(distance[0])*2 ,fabs(distance[1])*2, fabs(distance[2])*2);

  PickBox::setBoxSize(boxSize);
}
*/
AlignPoint::~AlignPoint()
{
    cover->getScene()->removeChild(_node.get());
}

int PDBPlugin::chdir(const char *dir)
{
    int ret = ::chdir(dir);
    if (ret == -1)
    {
        cerr << "changing directory to " << dir << "failed: " << strerror(errno) << endl;
    }
    return ret;
}

int PDBPlugin::system(const char *cmd)
{
    int ret = ::system(cmd);
    if (ret == -1)
    {
        cerr << "executing " << cmd << "failed: " << strerror(errno) << endl;
    }
    return ret;
}

bool PDBPlugin::loadDataImage(string id)
{

    // try reading in the file
    osg::Geode *geode = NULL;
    if ((geode = createImage(id)) == NULL)
    {
        imageMat->removeChild(imageGeode);
        return false;
    }

    // set geode to image geode
    imageGeode = geode;

    // add geode to scene graph
    imageMat->addChild(imageGeode);
    cerr << "Added image" << endl;
    return true;
}

Geode *PDBPlugin::createImage(string &filename)
{
    const float WIDTH = 256.0f;
    const float HEIGHT = 256.0f;
    const float ZPOS = 0.0f;

    Geometry *geom = new Geometry();
    Texture2D *icon = new Texture2D();
    Image *image = NULL;

    // check if the file exists
    chdir(relativeTempPath.c_str());
    image = osgDB::readImageFile(filename);
    chdir(currentPath.c_str());

    // check if the file exists
    if (!image)
    {
        cerr << "File does not exist" << filename << endl;
        return NULL;
    }

    icon->setImage(image);
    Vec3Array *vertices = new Vec3Array(4);
    // bottom left
    (*vertices)[0].set(-WIDTH / 2.0, -HEIGHT / 2.0, ZPOS);
    // bottom right
    (*vertices)[1].set(WIDTH / 2.0, -HEIGHT / 2.0, ZPOS);
    // top right
    (*vertices)[2].set(WIDTH / 2.0, HEIGHT / 2.0, ZPOS);
    // top left
    (*vertices)[3].set(-WIDTH / 2.0, HEIGHT / 2.0, ZPOS);
    geom->setVertexArray(vertices);

    Vec2Array *texcoords = new Vec2Array(4);
    (*texcoords)[0].set(0.0, 0.0);
    (*texcoords)[1].set(1.0, 0.0);
    (*texcoords)[2].set(1.0, 1.0);
    (*texcoords)[3].set(0.0, 1.0);
    geom->setTexCoordArray(0, texcoords);

    Vec3Array *normals = new Vec3Array(1);
    (*normals)[0].set(0.0f, 0.0f, 1.0f);
    geom->setNormalArray(normals);
    geom->setNormalBinding(Geometry::BIND_OVERALL);

    Vec4Array *colors = new Vec4Array(1);
    (*colors)[0].set(1.0, 1.0, 1.0, 1.0);
    geom->setColorArray(colors);
    geom->setColorBinding(Geometry::BIND_OVERALL);
    geom->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));

    // Texture:
    StateSet *stateset = geom->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, StateAttribute::OFF);
    stateset->setTextureAttributeAndModes(0, icon, StateAttribute::ON);

    Geode *imageGeode = new Geode();
    imageGeode->addDrawable(geom);

    return imageGeode;
}

void SequenceMarker::cursorUpdate(InputDevice *dev)
{
    if (dev != dev->_interaction->_wandR)
        return;
    if (_isMoving)
    {
        // move the marker (need to adjust so this works in collabrative mode)
        // Compute difference matrix between last and current wand:
        Matrix invLastWand2w = Matrix::inverse(_lastWand2w);
        Matrix wDiff = invLastWand2w * dev->getI2W();

        // Volume follows wand movement:
        Matrix mark2w = CUI::computeLocal2Root(_node.get());
        Matrix w2o = Matrix::inverse(CUI::computeLocal2Root(getProtein()->adjustmt));
        //_node->setMatrix(mark2w * wDiff * w2o);
        _node->setMatrix(_node->getMatrix() * Matrix::inverse(w2o) * wDiff * w2o);

        //float scale = 1.0/getProtein()->getScale();

        // send a message to PDB sequence
        SequenceMessage mm;
        mm.x = getPosition().x();
        mm.y = getPosition().z();
        mm.z = -getPosition().y();
        //cerr << "x " << mm.x << " ,y " << mm.y << " ,z " << mm.z << endl;
        mm.on = true;
        assert((getProtein()->getName()).size() < 5);
        strcpy(mm.filename, (getProtein()->getName()).c_str());
        cover->sendMessage(PDBPluginModule,
                           coVRPluginSupport::TO_SAME,
                           MOVE_MARK,
                           sizeof(SequenceMessage),
                           &mm);
    }
    Marker::cursorUpdate(dev);
}

PDBPickBox *SequenceMarker::getProtein()
{
    return _protein;
}

void SequenceMarker::buttonEvent(InputDevice *dev, int button)
{
    if (dev != dev->_interaction->_wandR)
        return;

    switch (button)
    {
    case 0:
        if (dev->getButtonState(button) == 1) // button pressed?
        {
            _isMoving = true;
        }
        else if (dev->getButtonState(button) == 0) // button released?
        {
            _isMoving = false;
        }
        break;
    default:
        break;
    }
    Marker::buttonEvent(dev, button); // call parent
}

// SequenceMarker Constructor
SequenceMarker::SequenceMarker(PDBPickBox *protein, GeometryType type, cui::Interaction *interaction, float size, float hue)
    : Marker(type, interaction)
{
    _isMoving = false;
    _visible = true;
    _protein = protein;
    _selected = false;
    setHue(hue);
    setSize(size);
    //Matrix mat;
    //mat.makeRotate(Vec3(0.,0.,1.), Vec3(0.,0.,-1.));
    //setMatrix(mat);

    /*if(protein) {
      ((Group*)_protein->getBase())->addChild(_node.get());
  }*/
}

Node *SequenceMarker::getBase()
{
    return _node.get();
}

SequenceMarker::~SequenceMarker(){};

void SequenceMarker::setVisible(bool vis)
{
    _visible = vis;
    if (_visible)
    {
        _node->setNodeMask(~0);
    }
    else
    {
        _node->setNodeMask(0);
    }
}

bool SequenceMarker::isVisible()
{
    return _visible;
}

void SequenceMarker::cursorEnter(InputDevice *dev)
{
    if (!_interactionA->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(_interactionA);
    }
    Marker::cursorEnter(dev);
}

void SequenceMarker::cursorLeave(InputDevice *dev)
{
    _selected = false;
    if (_interactionA->isRegistered())
    {
        coInteractionManager::the()->unregisterInteraction(_interactionA);
    }
    Marker::cursorLeave(dev);
}

void PDBPickBox::setSequenceMarker(SequenceMarker *mark)
{
    mark->setPosition(boxCenter);
    _mark = mark;

    // attach to scen graph after the scale component
    //_scale->addChild(_mark->get());
}

void PDBPlugin::markerEvent(cui::Marker *, int, int)
{
}
