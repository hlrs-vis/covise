/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef QT_CLEAN_NAMESPACE
#define QT_CLEAN_NAMESPACE
#endif

#include <util/common.h>

#include "coVRTui.h"
#include "VRSceneGraph.h"
#include "coVRFileManager.h"
#include "coVRNavigationManager.h"
#include "coVRCollaboration.h"
#include "coVRAnimationManager.h"
#include "VRViewer.h"
#include "coVRPluginSupport.h"
#include "coVRConfig.h"
#include "coVRPluginList.h"
#include "coVRCommunication.h"
#include "coVRMSController.h"
#include "coIntersection.h"
#include "ARToolKit.h"
#include "OpenCOVER.h"
#include <config/CoviseConfig.h>
#include <util/coFileUtil.h>
#include <util/environment.h>
#include <grmsg/coGRKeyWordMsg.h>
#include <osg/MatrixTransform>
#include <osgViewer/Renderer>
#include "OpenCOVER.h"
#include "coHud.h"
#include <osgDB/WriteFile>
#include <osg/ClipNode>
#include <input/input.h>
#include <input/inputdevice.h>
#include <OpenVRUI/osg/mathUtils.h> //for MAKE_EULER_MAT

using namespace opencover;
using namespace grmsg;
using namespace covise;

COVEREXPORT coVRTui *coVRTui::tui = NULL;
coVRTui *coVRTui::instance()
{
    if (tui == NULL)
        tui = new coVRTui();
    return tui;
}

coPluginEntry::coPluginEntry(const char *libName, coTUITab *tab, int n)
{
    name = new char[strlen(libName) + 1];
#ifdef WIN32
    strcpy(name, libName);
#else
    strcpy(name, libName + 3);
#endif

#ifdef WIN32
    name[strlen(name) - 4] = '\0';
#else
#ifdef __APPLE__
    name[strlen(name) - 3] = '\0';
#else
    name[strlen(name) - 3] = '\0';
#endif
#endif

    tuiEntry = new coTUIToggleButton(name, tab->getID());
    tuiEntry->setEventListener(this);
    tuiEntry->setPos(n % 5, n / 5);
}

coPluginEntry::~coPluginEntry()
{
    delete tuiEntry;
    delete[] name;
}

void coPluginEntry::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == tuiEntry)
    {
        if (tuiEntry->getState())
            cover->addPlugin(name);
        else
            cover->removePlugin(name);
    }
}

void coPluginEntry::tabletPressEvent(coTUIElement * /*tUIItem*/)
{
}

void coPluginEntry::tabletReleaseEvent(coTUIElement * /*tUIItem*/)
{
}

coPluginEntryList::coPluginEntryList(coTUITab *tab)
{
    myTab = tab;
    const char *coviseDir;
    const char *archsuffix;
    coviseDir = getenv("COVISEDIR");
    archsuffix = getenv("ARCHSUFFIX");
    char buf[1024];
    int n = 0;
#ifdef __APPLE__
    std::string bundlepath = getBundlePath();
    if (!bundlepath.empty())
    {
        sprintf(buf, "%s/Contents/PlugIns/", bundlepath.c_str());
        coDirectory *dir = coDirectory::open(buf);
        for (int i = 0; dir && i < dir->count(); i++)
        {
            if (dir->match(dir->name(i), "*.so")
                && !dir->match(dir->name(i), "*.?.?.so"))
            {
                append(new coPluginEntry(dir->name(i), myTab, n));
                n++;
            }
        }
    }
#endif
    if ((coviseDir != NULL) && (archsuffix != NULL))
    {
#ifdef WIN32
        sprintf(buf, "%s\\%s\\lib\\OpenCOVER\\plugins", coviseDir, archsuffix);
#else
        sprintf(buf, "%s/%s/lib/OpenCOVER/plugins", coviseDir, archsuffix);
#endif
        coDirectory *dir = coDirectory::open(buf);
        for (int i = 0; dir && i < dir->count(); i++)
        {
            if (dir->match(dir->name(i),
#ifdef WIN32
                           "*.dll"
#elif defined(__APPLE__)
                           "*.so"
#else
                           "*.so"
#endif
                           ))
#ifdef __APPLE__
                if (!dir->match(dir->name(i), "*.?.?.so"))
#endif
                {
                    if (!strstr(dir->name(i), "input_"))
                    {
                        append(new coPluginEntry(dir->name(i), myTab, n));
                        n++;
                    }
                }
        }
    }
    updateState();
}

coPluginEntryList::~coPluginEntryList()
{
}

void coPluginEntryList::updateState()
{
    reset();
    while (current())
    {
        if (cover->getPlugin(current()->name))
        {
            if (current()->tuiEntry->getState() == false)
                current()->tuiEntry->setState(true);
        }
        else
        {
            if (current()->tuiEntry->getState() == true)
                current()->tuiEntry->setState(false);
        }
        next();
    }
}

coVRTui::coVRTui()
    : collision(false)
    , navigationMode(coVRNavigationManager::NavNone)
    , driveSpeed(0.0)
    , ScaleValue(1.0)
    , numTimesteps(1)
    , animationCurrentFrame(0)
    , animationSpeed(1.0)
    , animationSliderMin(0.0)
    , animationSliderMax(30.)
    , animationEnabled(true)
    , animationOscillate(false)
{
    tui = this;
    lastUpdateTime = 0;
    binList = new BinList;
    mainFolder = new coTUITabFolder("testfolder");
    coverTab = new coTUITab("COVER", mainFolder->getID());
    animTab = new coTUITab("Animation", mainFolder->getID());
    pluginTab = new coTUITab("Plugins", mainFolder->getID());
    inputTUI = new coInputTUI();
    topContainer = new coTUIFrame("Buttons", coverTab->getID());
    bottomContainer = new coTUIFrame("Nav", coverTab->getID());
    rightContainer = new coTUIFrame("misc", bottomContainer->getID());

    Walk = new coTUIToggleButton("Walk", topContainer->getID());
    Drive = new coTUIToggleButton("Drive", topContainer->getID());
    Fly = new coTUIToggleButton("Fly", topContainer->getID());
    XForm = new coTUIToggleButton("Move world", topContainer->getID());
    DebugBins = new coTUIToggleButton("DebugBins", topContainer->getID(), false);
    DebugBins->setEventListener(this);

    Scale = new coTUIToggleButton("Scale", topContainer->getID());
    Collision = new coTUIToggleButton("Detect collisions", topContainer->getID());
    DisableIntersection = new coTUIToggleButton("Disable intersection", topContainer->getID());
    testImage = new coTUIToggleButton("TestImage", topContainer->getID());
    testImage->setState(false);
    Quit = new coTUIButton("Quit", topContainer->getID());
    Freeze = new coTUIToggleButton("Stop headtracking", topContainer->getID());
    Wireframe = new coTUIToggleButton("Wireframe", topContainer->getID());
    ViewAll = new coTUIButton("View all", topContainer->getID());
    Menu = new coTUIToggleButton("Hide menu", topContainer->getID());
    scaleLabel = new coTUILabel("Scale factor (log10)", topContainer->getID());
    ScaleSlider = new coTUIFloatSlider("ScaleFactor", topContainer->getID());
    Menu->setState(false);
#ifndef NOFB
    FileBrowser = new coTUIFileBrowserButton("Load file...", topContainer->getID());
    coVRCommunication::instance()->setFBData(FileBrowser->getVRBData());
    SaveFileFB = new coTUIFileBrowserButton("Save file...", topContainer->getID());
    SaveFileFB->setMode(coTUIFileBrowserButton::SAVE);
    coVRCommunication::instance()->setFBData(SaveFileFB->getVRBData());
    coVRFileManager::instance()->SetDefaultFB(FileBrowser);

#endif

    speedLabel = new coTUILabel("Navigation speed", bottomContainer->getID());
    NavSpeed = new coTUIFloatSlider("NavSpeed", bottomContainer->getID());
    viewerLabel = new coTUILabel("Viewer position", bottomContainer->getID());
    posX = new coTUIEditFloatField("ViewerX", bottomContainer->getID());
    posY = new coTUIEditFloatField("ViewerY", bottomContainer->getID());
    posZ = new coTUIEditFloatField("ViewerZ", bottomContainer->getID());
    fovLabel = new coTUILabel("FieldOfView in ", bottomContainer->getID());
    fovH = new coTUIEditFloatField("ViewerZ", bottomContainer->getID());
    FPSLabel = new coTUILabel("Fps:", bottomContainer->getID());
    CFPSLabel = new coTUILabel("Constant fps", bottomContainer->getID());
    CFPS = new coTUIEditFloatField("cfps", bottomContainer->getID());
    backgroundLabel = new coTUILabel("Background color", bottomContainer->getID());
    backgroundColor = new coTUIColorButton("background", bottomContainer->getID());
    LODScaleLabel = new coTUILabel("LODScale", bottomContainer->getID());
    LODScaleEdit = new coTUIEditFloatField("LODScaleE", bottomContainer->getID());

    driveLabel = new coTUILabel("Drive navigation", bottomContainer->getID());
    driveNav = new coTUINav("driveNav", bottomContainer->getID());
    panLabel = new coTUILabel("Pan navigation", bottomContainer->getID());
    panNav = new coTUINav("panNav", bottomContainer->getID());

    nearEdit = new coTUIEditFloatField("near", rightContainer->getID());
    farEdit = new coTUIEditFloatField("far", rightContainer->getID());
    nearLabel = new coTUILabel("near", rightContainer->getID());
    farLabel = new coTUILabel("far", rightContainer->getID());

    NavSpeed->setEventListener(this);
    Walk->setEventListener(this);
    Drive->setEventListener(this);
    Fly->setEventListener(this);
    XForm->setEventListener(this);
    Scale->setEventListener(this);
    Collision->setEventListener(this);
    DisableIntersection->setEventListener(this);
    testImage->setEventListener(this);
    ScaleSlider->setEventListener(this);
    Quit->setEventListener(this);
    Freeze->setEventListener(this);
    Wireframe->setEventListener(this);
    Menu->setEventListener(this);
    posX->setEventListener(this);
    posY->setEventListener(this);
    posZ->setEventListener(this);
    fovH->setEventListener(this);
    driveNav->setEventListener(this);
    panNav->setEventListener(this);
    ViewAll->setEventListener(this);
    CFPS->setEventListener(this);
    backgroundColor->setEventListener(this);
    LODScaleEdit->setEventListener(this);

    nearEdit->setEventListener(this);
    farEdit->setEventListener(this);

#ifndef NOFB
    FileBrowser->setEventListener(this);
    FileBrowser->setMode(coTUIFileBrowserButton::OPEN);
    SaveFileFB->setEventListener(this);
    SaveFileFB->setMode(coTUIFileBrowserButton::OPEN);
#endif

    ScaleSlider->setMin(-5.);
    ScaleSlider->setMax(+5.);
    ScaleSlider->setValue(0.);

    NavSpeed->setMin(0.03);
    NavSpeed->setMax(30);
    NavSpeed->setValue(1);

    topContainer->setPos(0, 0);
    bottomContainer->setPos(0, 1);
    rightContainer->setPos(2, 5);

    Walk->setPos(0, 0);
    Drive->setPos(0, 1);
    Fly->setPos(0, 2);
    XForm->setPos(0, 3);
    Scale->setPos(0, 4);
    Collision->setPos(0, 5);
    DebugBins->setPos(3, 0);
    DisableIntersection->setPos(1, 5);
    testImage->setPos(2, 5);

    speedLabel->setPos(0, 10);
    NavSpeed->setPos(1, 10);

    viewerLabel->setPos(0, 7);
    posX->setPos(0, 8);
    posY->setPos(1, 8);
    posZ->setPos(2, 8);
    fovLabel->setPos(3, 7);
    fovH->setPos(3, 8);

    scaleLabel->setPos(1, 3);
    ScaleSlider->setPos(1, 4);

    FPSLabel->setPos(0, 11);
    CFPSLabel->setPos(1, 11);
    CFPS->setPos(2, 11);
    CFPS->setValue(0.0);

    backgroundLabel->setPos(0, 12);
    backgroundColor->setPos(1, 12);
    backgroundColor->setColor(0, 0, 0, 1);
    LODScaleLabel->setPos(2, 12);
    LODScaleEdit->setPos(3, 12);

    Quit->setPos(2, 0);
#ifndef NOFB
    FileBrowser->setPos(2, 1);
    SaveFileFB->setPos(2, 2);
#endif
    Menu->setPos(2, 3);

    ViewAll->setPos(1, 0);
    Freeze->setPos(1, 1);
    Wireframe->setPos(1, 2);

    driveNav->setPos(0, 5);
    driveLabel->setPos(0, 6);
    panNav->setPos(1, 5);
    panLabel->setPos(1, 6);

    nearEdit->setPos(0, 1);
    farEdit->setPos(0, 3);
    nearLabel->setPos(0, 0);
    farLabel->setPos(0, 2);

    //TODO coConfig
    float xp, yp, zp;
    xp = coCoviseConfig::getFloat("x", "COVER.ViewerPosition", 0.0f);
    yp = coCoviseConfig::getFloat("y", "COVER.ViewerPosition", -2000.0f);
    zp = coCoviseConfig::getFloat("z", "COVER.ViewerPosition", 30.0f);
    viewPos.set(xp, yp, zp);
    posX->setValue(viewPos[0]);
    posY->setValue(viewPos[1]);
    posZ->setValue(viewPos[2]);
    availablePlugins = new coPluginEntryList(pluginTab);

    // Animation tab
    Animate = new coTUIToggleButton("Animate", animTab->getID());
    Animate->setState(animationEnabled);
    AnimSpeedLabel = new coTUILabel("Animation speed (fps)", animTab->getID());
    AnimSpeed = new coTUIFloatSlider("Speed", animTab->getID());
    AnimForward = new coTUIButton("Step forward", animTab->getID());
    AnimBack = new coTUIButton("Step backward", animTab->getID());
    //AnimRotate = new coTUIToggleButton("Rotate objects", animTab->getID());
    AnimTimestepLabel = new coTUILabel("Time step", animTab->getID());
    AnimTimestep = new coTUISlider("TimeStep", animTab->getID());
    AnimOscillate = new coTUIToggleButton("Oscillate", animTab->getID());
    AnimStartStepLabel = new coTUILabel("Start time step", animTab->getID());
    AnimStartStep = new coTUISlider("StartTimeStep", animTab->getID());
    AnimStopStepLabel = new coTUILabel("Stop time step", animTab->getID());
    AnimStopStep = new coTUISlider("StopTimeStep", animTab->getID());

    PresentationLabel = new coTUILabel("Presentation", animTab->getID());
    PresentationForward = new coTUIButton("Forward", animTab->getID());
    PresentationBack = new coTUIButton("Back", animTab->getID());
    PresentationStep = new coTUIEditIntField("PresStep", animTab->getID());
    PresentationForward->setEventListener(this);
    PresentationBack->setEventListener(this);
    PresentationStep->setEventListener(this);
    PresentationStep->setValue(0);
    PresentationStep->setMin(0);
    PresentationStep->setMax(1000);

    PresentationLabel->setPos(0, 10);
    PresentationBack->setPos(0, 11);
    PresentationStep->setPos(1, 11);
    PresentationForward->setPos(2, 11);

    // anim speed was configurable in 5.2 where animSpeed was a slider
    // for the dial where min=-max we use the greater value as animationSliderMax
    animationSpeed = animationSliderMax * 0.96; // 24 FPS for 25.0

    float min, max;
    min = coCoviseConfig::getFloat("min", "COVER.AnimationSpeed", animationSliderMin);
    max = coCoviseConfig::getFloat("max", "COVER.AnimationSpeed", animationSliderMax);
    animationSpeed = coCoviseConfig::getFloat("default", "COVER.AnimationSpeed", animationSpeed);

    if (min > max)
        std::swap(min, max);

    animationSliderMin = min;
    animationSliderMax = max;

    AnimSpeed->setMin(animationSliderMin);
    AnimSpeed->setMax(animationSliderMax);
    AnimSpeed->setValue(animationSpeed);

    AnimTimestep->setMin(0);
    AnimTimestep->setMax(numTimesteps);
    AnimTimestep->setValue(animationCurrentFrame);
    AnimStartStep->setMin(0);
    AnimStartStep->setMax(numTimesteps);
    AnimStartStep->setValue(animationCurrentFrame);
    AnimStopStep->setMin(0);
    AnimStopStep->setMax(numTimesteps);
    AnimStopStep->setValue(animationCurrentFrame);
    AnimOscillate->setState(animationOscillate);

    Animate->setEventListener(this);
    AnimSpeed->setEventListener(this);
    AnimForward->setEventListener(this);
    AnimBack->setEventListener(this);
    //AnimRotate->setEventListener(this);
    AnimTimestep->setEventListener(this);
    AnimOscillate->setEventListener(this);
    AnimStartStep->setEventListener(this);
    AnimStopStep->setEventListener(this);

    Animate->setPos(0, 0);
    //AnimRotate->setPos(1,0);
    AnimSpeedLabel->setPos(0, 1);
    AnimSpeed->setPos(0, 2);
    AnimBack->setPos(0, 3);
    AnimForward->setPos(1, 3);
    AnimTimestepLabel->setPos(0, 4);
    AnimTimestep->setPos(0, 5);
    AnimOscillate->setPos(2, 5);
    AnimStartStepLabel->setPos(0, 6);
    AnimStartStep->setPos(0, 7);
    AnimStopStepLabel->setPos(0, 8);
    AnimStopStep->setPos(0, 9);

    nearEdit->setValue(coVRConfig::instance()->nearClip());
    farEdit->setValue(coVRConfig::instance()->farClip());
    LODScaleEdit->setValue(coVRConfig::instance()->getLODScale());
}
void coVRTui::config()
{
#ifndef NOFB
    FileBrowser->setFilterList(coVRFileManager::instance()->getFilterList());
    SaveFileFB->setFilterList(coVRFileManager::instance()->getFilterList());
#endif
}

coVRTui::~coVRTui()
{
    delete PresentationLabel;
    delete PresentationForward;
    delete PresentationBack;
    delete PresentationStep;
    delete Animate;
    delete AnimSpeedLabel;
    delete AnimSpeed;
    delete AnimForward;
    delete AnimBack;
    //delete AnimRotate;
    delete AnimTimestepLabel;
    delete AnimTimestep;
    delete Walk;
    delete Drive;
    delete Fly;
    delete XForm;
    delete Scale;
    delete Collision;
    delete DebugBins;
    delete DisableIntersection;
    delete testImage;
    delete speedLabel;
    delete NavSpeed;
    delete ScaleSlider;
    delete coverTab;
    delete mainFolder;
    delete Quit;
    delete Freeze;
    delete Wireframe;
    delete Menu;
    delete viewerLabel;
    delete posX;
    delete posY;
    delete posZ;
    delete fovLabel;
    delete fovH;
    delete FPSLabel;
    delete CFPSLabel;
    delete CFPS;
    delete backgroundLabel;
    delete backgroundColor;
    delete LODScaleLabel;
    delete LODScaleEdit;
#ifndef NOFB
    delete FileBrowser;
    delete SaveFileFB;
#endif
    delete driveLabel;
    delete panLabel;
    delete nearLabel;
    delete farLabel;
    delete nearEdit;
    delete farEdit;
    delete topContainer;
    delete bottomContainer;
    delete rightContainer;
    delete binList;
    delete inputTUI;
}

coInputTUI::coInputTUI()
{
    inputTab = new coTUITab("Input", coVRTui::instance()->mainFolder->getID());
    
    personContainer = new coTUIFrame("pc",inputTab->getID());
    personContainer->setPos(0,0);
    personsLabel= new coTUILabel("person",personContainer->getID());
    personsLabel->setPos(0,0);
    personsChoice = new coTUIComboBox("personsCombo",personContainer->getID());
    personsChoice->setPos(1,0);
    
    personsChoice->setEventListener(this);
    for (int i = 0; i < Input::instance()->getNumPersons(); i++)
    {
        personsChoice->addEntry(Input::instance()->getPerson(i)->getName());
    }
    personsChoice->setSelectedEntry(Input::instance()->getActivePerson());
    
    
    bodiesContainer = new coTUIFrame("bc",inputTab->getID());
    bodiesContainer->setPos(1,0);
    bodiesLabel= new coTUILabel("Body",bodiesContainer->getID());
    bodiesLabel->setPos(0,0);
    bodiesChoice = new coTUIComboBox("bodiesCombo",bodiesContainer->getID());
    bodiesChoice->setPos(0,1);
    bodiesChoice->setEventListener(this);
    for (int i = 0; i < Input::instance()->getNumBodies(); i++)
    {
        bodiesChoice->addEntry(Input::instance()->getBody(i)->getName());
    }
    bodiesChoice->setSelectedEntry(Input::instance()->getActivePerson());
    
    bodyTrans[0] = new coTUIEditFloatField("xe", bodiesContainer->getID());
    bodyTransLabel[0] = new coTUILabel("x", bodiesContainer->getID());
    bodyTrans[1] = new coTUIEditFloatField("ye", bodiesContainer->getID());
    bodyTransLabel[1] = new coTUILabel("y", bodiesContainer->getID());
    bodyTrans[2] = new coTUIEditFloatField("ze", bodiesContainer->getID());
    bodyTransLabel[2] = new coTUILabel("z", bodiesContainer->getID());
    bodyRot[0] = new coTUIEditFloatField("he", bodiesContainer->getID());
    bodyRotLabel[0] = new coTUILabel("h", bodiesContainer->getID());
    bodyRot[1] = new coTUIEditFloatField("pe", bodiesContainer->getID());
    bodyRotLabel[1] = new coTUILabel("p", bodiesContainer->getID());
    bodyRot[2] = new coTUIEditFloatField("re", bodiesContainer->getID());
    bodyRotLabel[2] = new coTUILabel("r", bodiesContainer->getID());
    for (int i = 0; i < 3; i++)
    {
        bodyTrans[i]->setPos(1+i * 2 + 1, 1);
        bodyTransLabel[i]->setPos(1+i * 2, 1);
        bodyTrans[i]->setEventListener(this);
        bodyRot[i]->setPos(1+i * 2 + 1, 2);
        bodyRotLabel[i]->setPos(1+i * 2, 2);
        bodyRot[i]->setEventListener(this);
    }
    
    devicesLabel= new coTUILabel("Device",bodiesContainer->getID());
    devicesLabel->setPos(0,4);
    devicesChoice = new coTUIComboBox("devicesCombo",bodiesContainer->getID());
    devicesChoice->setPos(0,5);
    devicesChoice->setEventListener(this);
    for (int i = 0; i < Input::instance()->getNumDevices(); i++)
    {
        devicesChoice->addEntry(Input::instance()->getDevice(i)->getName());
    }
    devicesChoice->setSelectedEntry(Input::instance()->getActivePerson());
    
    deviceTrans[0] = new coTUIEditFloatField("xe", bodiesContainer->getID());
    deviceTransLabel[0] = new coTUILabel("x", bodiesContainer->getID());
    deviceTrans[1] = new coTUIEditFloatField("ye", bodiesContainer->getID());
    deviceTransLabel[1] = new coTUILabel("y", bodiesContainer->getID());
    deviceTrans[2] = new coTUIEditFloatField("ze", bodiesContainer->getID());
    deviceTransLabel[2] = new coTUILabel("z", bodiesContainer->getID());
    deviceRot[0] = new coTUIEditFloatField("he", bodiesContainer->getID());
    deviceRotLabel[0] = new coTUILabel("h", bodiesContainer->getID());
    deviceRot[1] = new coTUIEditFloatField("pe", bodiesContainer->getID());
    deviceRotLabel[1] = new coTUILabel("p", bodiesContainer->getID());
    deviceRot[2] = new coTUIEditFloatField("re", bodiesContainer->getID());
    deviceRotLabel[2] = new coTUILabel("r", bodiesContainer->getID());
    for (int i = 0; i < 3; i++)
    {
        deviceTrans[i]->setPos(1+i * 2 + 1, 5);
        deviceTransLabel[i]->setPos(1+i * 2, 5);
        deviceTrans[i]->setEventListener(this);
        deviceRot[i]->setPos(1+i * 2 + 1, 6);
        deviceRotLabel[i]->setPos(1+i * 2, 6);
        deviceRot[i]->setEventListener(this);
    }
    
    updateTUI();
}

void coInputTUI::updateTUI()
{
    TrackingBody * tb = Input::instance()->getBody(bodiesChoice->getSelectedEntry());
    osg::Matrix m = tb->getOffsetMat();
    osg::Vec3 v = m.getTrans();
    coCoord coord = m;
    for (int i = 0; i < 3; i++)
    {
        bodyTrans[i]->setValue(v[i]);
        bodyRot[i]->setValue(coord.hpr[i]);
    }
    InputDevice *id = Input::instance()->getDevice(devicesChoice->getSelectedEntry());
    m = id->getOffsetMat();
    v = m.getTrans();
    coord = m;
    for (int i = 0; i < 3; i++)
    {
        deviceTrans[i]->setValue(v[i]);
        deviceRot[i]->setValue(coord.hpr[i]);
    }
}
coInputTUI::~coInputTUI()
{
    for (int i = 0; i < 3; i++)
    {
        delete bodyTrans[i];
        delete bodyTransLabel[i];
        delete bodyRot[i];
        delete bodyRotLabel[i];
        delete deviceTrans[i];
        delete deviceTransLabel[i];
        delete deviceRot[i];
        delete deviceRotLabel[i];
    }
    
    delete personContainer;
    delete personsLabel;
    delete personsChoice;

    delete bodiesContainer;
    delete bodiesLabel;
    delete bodiesChoice;
    delete devicesLabel;
    delete devicesChoice;
    delete inputTab;
    
}

void coInputTUI::tabletEvent(coTUIElement *tUIItem)
{
    if(tUIItem == personsChoice)
    {
        Input::instance()->setActivePerson(personsChoice->getSelectedEntry());
    }
    else if(tUIItem == bodiesChoice)
    {
        updateTUI();
    }
    else if(tUIItem == devicesChoice)
    {
        updateTUI();
    }
    else if(tUIItem == bodyTrans[0] || tUIItem == bodyTrans[1] || tUIItem == bodyTrans[2] ||
            tUIItem == bodyRot[0] || tUIItem == bodyRot[1] || tUIItem == bodyRot[2])
    {
        TrackingBody * tb = Input::instance()->getBody(bodiesChoice->getSelectedEntry());
        osg::Matrix m;
        MAKE_EULER_MAT(m, bodyRot[0]->getValue(), bodyRot[1]->getValue(), bodyRot[2]->getValue());
        
        osg::Matrix translationMat;
        translationMat.makeTranslate(bodyTrans[0]->getValue(), bodyTrans[1]->getValue(), bodyTrans[2]->getValue());
        m.postMult(translationMat);
        tb->setOffsetMat(m);
    }
    else if(tUIItem == deviceTrans[0] || tUIItem == deviceTrans[1] || tUIItem == deviceTrans[2] ||
            tUIItem == deviceRot[0] || tUIItem == deviceRot[1] || tUIItem == deviceRot[2])
    {
        InputDevice *id = Input::instance()->getDevice(devicesChoice->getSelectedEntry());
        osg::Matrix m;
        MAKE_EULER_MAT(m, deviceRot[0]->getValue(), deviceRot[1]->getValue(), deviceRot[2]->getValue());
        
        osg::Matrix translationMat;
        translationMat.makeTranslate(deviceTrans[0]->getValue(), deviceTrans[1]->getValue(), deviceTrans[2]->getValue());
        m.postMult(translationMat);
        id->setOffsetMat(m);
    }
}
void coInputTUI::tabletPressEvent(coTUIElement *tUIItem)
{
}
void coInputTUI::tabletReleaseEvent(coTUIElement *tUIItem)
{
}


void coVRTui::updateState()
{
    if (availablePlugins)
        availablePlugins->updateState();
}

void coVRTui::updateFPS(double fps)
{
    char buf[300];
    sprintf(buf, "Fps: %6.2f", fps);
    FPSLabel->setLabel(buf);
}

void coVRTui::update()
{
    if (navigationMode != coVRNavigationManager::instance()->getMode())
    {
        navigationMode = coVRNavigationManager::instance()->getMode();
        switch (navigationMode)
        {
        case coVRNavigationManager::XForm:
            XForm->setState(true);
            driveLabel->setLabel("Rotate");
            panLabel->setLabel("Panning");
            break;
        case coVRNavigationManager::Scale:
            Scale->setState(true);
            driveLabel->setLabel("Scale (up/down)");
            panLabel->setLabel("Scale (right/left)");
            break;
        case coVRNavigationManager::Fly:
            Fly->setState(true);
            driveLabel->setLabel("Pitch & heading");
            panLabel->setLabel("Forward: pitch & roll");
            break;
        case coVRNavigationManager::Glide:
            Drive->setState(true);
            driveLabel->setLabel("Speed & heading");
            panLabel->setLabel("Panning");
            break;
        case coVRNavigationManager::Walk:
            Walk->setState(true);
            driveLabel->setLabel("Speed & heading");
            panLabel->setLabel("Panning");
            break;
        default:
            ;
        }
    }
    if (collision != coVRNavigationManager::instance()->getCollision())
    {
        collision = coVRNavigationManager::instance()->getCollision();
        Collision->setState(collision);
    }
    if (DisableIntersection->getState() != (coIntersection::instance()->numIsectAllNodes == 0))
    {
        DisableIntersection->setState(coIntersection::instance()->numIsectAllNodes == 0);
    }
    if (testImage->getState() != ARToolKit::instance()->testImage)
    {
        testImage->setState(ARToolKit::instance()->testImage);
    }
    if (Freeze->getState() != coVRConfig::instance()->frozen())
    {
        Freeze->setState(coVRConfig::instance()->frozen());
    }
    if (driveSpeed != coVRNavigationManager::instance()->getDriveSpeed())
    {
        driveSpeed = coVRNavigationManager::instance()->getDriveSpeed();
        NavSpeed->setValue(driveSpeed);
    }
    if (ScaleValue != cover->getScale())
    {
        ScaleValue = cover->getScale();
        ScaleSlider->setValue(log(ScaleValue) / log(10.));
    }

    if ((driveNav->down || panNav->down) && (!coVRCommunication::instance()->isRILocked(coVRCommunication::TRANSFORM)))
    {

        if (navigationMode == coVRNavigationManager::Walk || navigationMode == coVRNavigationManager::Glide)
        {
            doTabWalk();
        }

        else if (navigationMode == coVRNavigationManager::Fly)
        {
            doTabFly();
        }

        else if (navigationMode == coVRNavigationManager::Scale)
        {
            doTabScale();
        }

        else if (navigationMode == coVRNavigationManager::XForm)
        {
            doTabXform();
        }
    }

    if (numTimesteps != coVRAnimationManager::instance()->getNumTimesteps())
    {
        if (cover->frameRealTime() > lastUpdateTime + 0.5)
        {
            lastUpdateTime = cover->frameRealTime();
            numTimesteps = coVRAnimationManager::instance()->getNumTimesteps();
            if (numTimesteps > 0)
                AnimTimestep->setMax(numTimesteps - 1);
            else
                AnimTimestep->setMax(0);
            AnimTimestep->setMin(0);
        }
        AnimStopStep->setMax(numTimesteps - 1);
        AnimStopStep->setMin(0);
        AnimStartStep->setMax(numTimesteps - 1);
        AnimStartStep->setMin(0);
    }
    if (AnimStartStep->getValue() != coVRAnimationManager::instance()->getStartFrame())
        AnimStartStep->setValue(coVRAnimationManager::instance()->getStartFrame());
    if (AnimStopStep->getValue() != coVRAnimationManager::instance()->getStopFrame())
        AnimStopStep->setValue(coVRAnimationManager::instance()->getStopFrame());
    if (animationCurrentFrame != coVRAnimationManager::instance()->getAnimationFrame())
    {
        animationCurrentFrame = coVRAnimationManager::instance()->getAnimationFrame();
        AnimTimestep->setValue(animationCurrentFrame);
    }
    if (animationEnabled != coVRAnimationManager::instance()->animationRunning())
    {
        animationEnabled = coVRAnimationManager::instance()->animationRunning();
        Animate->setState(animationEnabled);
    }
    if (animationSpeed != coVRAnimationManager::instance()->getAnimationSpeed())
    {
        animationSpeed = coVRAnimationManager::instance()->getAnimationSpeed();
        AnimSpeed->setValue(animationSpeed);
    }
    if (animationOscillate != coVRAnimationManager::instance()->isOscillating())
    {
        animationOscillate = coVRAnimationManager::instance()->isOscillating();
        AnimOscillate->setState(animationOscillate);
    }
}

void coVRTui::tabletEvent(coTUIElement *tUIItem)
{

    if (tUIItem == driveNav)
    {
        mx = driveNav->x;
        my = driveNav->y;
    }
    else if (tUIItem == panNav)
    {
        mx = panNav->x;
        my = panNav->y;
    }
    else if (tUIItem == testImage)
    {
        ARToolKit::instance()->testImage = testImage->getState();
    }
    else if (tUIItem == DebugBins)
    {
        if (DebugBins->getState())
        {
            binList->refresh();
        }
        else
        {
            binList->removeAll();
        }
    }
    else if (tUIItem == PresentationForward)
    {
        PresentationStep->setValue(PresentationStep->getValue() + 1);
        char buf[100];
        sprintf(buf, "%d\ngoToStep %d", coGRMsg::KEYWORD, PresentationStep->getValue());
        coVRPluginList::instance()->guiToRenderMsg(buf);
    }
    else if (tUIItem == PresentationBack)
    {
        if (PresentationStep->getValue() > 0)
        {
            PresentationStep->setValue(PresentationStep->getValue() - 1);
        }
        char buf[100];
        sprintf(buf, "%d\ngoToStep %d", coGRMsg::KEYWORD, PresentationStep->getValue());
        coVRPluginList::instance()->guiToRenderMsg(buf);
    }
    else if (tUIItem == PresentationStep)
    {
        char buf[100];
        sprintf(buf, "%d\ngoToStep %d", coGRMsg::KEYWORD, PresentationStep->getValue());
        coVRPluginList::instance()->guiToRenderMsg(buf);
    }

    if (tUIItem == nearEdit || tUIItem == farEdit)
    {
        coVRConfig::instance()->setNearFar(nearEdit->getValue(), farEdit->getValue());
    }
    if (tUIItem == posX)
    {
        viewPos[0] = posX->getValue();
        VRViewer::instance()->setInitialViewerPos(viewPos);
        osg::Matrix viewMat;
        viewMat.makeIdentity();
        viewMat.setTrans(viewPos);
        VRViewer::instance()->setViewerMat(viewMat);
    }
    else if (tUIItem == posY)
    {
        viewPos[1] = posY->getValue();
        VRViewer::instance()->setInitialViewerPos(viewPos);
        osg::Matrix viewMat;
        viewMat.makeIdentity();
        viewMat.setTrans(viewPos);
        VRViewer::instance()->setViewerMat(viewMat);
    }
    else if (tUIItem == posZ)
    {
        viewPos[2] = posZ->getValue();
        VRViewer::instance()->setInitialViewerPos(viewPos);
        osg::Matrix viewMat;
        viewMat.makeIdentity();
        viewMat.setTrans(viewPos);
        VRViewer::instance()->setViewerMat(viewMat);
    }
    else if (tUIItem == fovH)
    {
        float fov = (fovH->getValue() / 180.0) * M_PI;
        coVRConfig::instance()->HMDViewingAngle = fovH->getValue();
        float dist = (coVRConfig::instance()->screens[0].hsize / 2.0) / tan(fov / 2.0);
        viewPos[0] = 0.0;
        viewPos[1] = coVRConfig::instance()->screens[0].xyz[1] - dist;
        fprintf(stderr, "dist = %f\n", viewPos[1]);
        viewPos[2] = 0.0;
        VRViewer::instance()->setInitialViewerPos(viewPos);
        osg::Matrix viewMat;
        viewMat.makeIdentity();
        viewMat.setTrans(viewPos);
        VRViewer::instance()->setViewerMat(viewMat);
    }
    else if (tUIItem == CFPS)
    {
        coVRConfig::instance()->setFrameRate(CFPS->getValue());
    }
    else if (tUIItem == backgroundColor)
    {
        VRViewer::instance()->setClearColor(osg::Vec4(backgroundColor->getRed(), backgroundColor->getGreen(), backgroundColor->getBlue(), backgroundColor->getAlpha()));
    }

    else if (tUIItem == NavSpeed)
    {
        driveSpeed = NavSpeed->getValue();
        coVRNavigationManager::instance()->setDriveSpeed(driveSpeed);
    }
    else if (tUIItem == ScaleSlider)
    {
        ScaleValue = powf(10.f, ScaleSlider->getValue());
        cover->setScale(ScaleValue);
    }
    else if (tUIItem == driveNav)
    {
        if (driveNav->down)
        {
            mx = driveNav->x;
            my = driveNav->y;
        }
    }
    else if (tUIItem == panNav)
    {
        if (panNav->down)
        {
            mx = panNav->x;
            my = panNav->y;
        }
    }
    else if (tUIItem == AnimTimestep)
    {
        animationCurrentFrame = AnimTimestep->getValue();
        coVRAnimationManager::instance()->setAnimationFrame(animationCurrentFrame);
    }
    else if (tUIItem == AnimSpeed)
    {
        animationSpeed = AnimSpeed->getValue();
        coVRAnimationManager::instance()->setAnimationSpeed(animationSpeed);
    }
    else if (tUIItem == AnimOscillate)
    {
        animationOscillate = AnimOscillate->getState();
        coVRAnimationManager::instance()->setOscillate(animationOscillate);
    }
    else if (tUIItem == AnimStartStep)
    {
        coVRAnimationManager::instance()->setStartFrame(AnimStartStep->getValue() - 1);
    }
    else if (tUIItem == AnimStopStep)
    {
        coVRAnimationManager::instance()->setStopFrame(AnimStopStep->getValue() - 1);
    }
    else if (tUIItem == FileBrowser)
    {
        //Retrieve Data object
        std::string filename = FileBrowser->getSelectedPath();

        //Do what you want to do with the filename
        OpenCOVER::instance()->hud->show();
        OpenCOVER::instance()->hud->setText1("Loading File...");
        OpenCOVER::instance()->hud->setText2(filename.c_str());
        coVRFileManager::instance()->loadFile(filename.c_str(), FileBrowser);
    }
    else if (tUIItem == SaveFileFB)
    {
        //Retrieve Data object
        std::string filename = SaveFileFB->getSelectedPath();

        OpenCOVER::instance()->hud->show();
        OpenCOVER::instance()->hud->setText1("Replacing File...");
        OpenCOVER::instance()->hud->setText2(filename.c_str());

        cerr << "File-Path: " << SaveFileFB->getSelectedPath().c_str() << endl;
        //Do what you want to do with the filename
        if (osgDB::writeNodeFile(*cover->getObjectsRoot(), SaveFileFB->getFilename(SaveFileFB->getSelectedPath()).c_str()))
        {
            if (cover->debugLevel(3))
                std::cerr << "Data written to \"" << SaveFileFB->getFilename(SaveFileFB->getSelectedPath()) << "\"." << std::endl;
        }

        //coVRFileManager::instance()->replaceFile(filename.c_str(), SaveFileFB);
    }
    else if (tUIItem == LODScaleEdit)
    {
        coVRConfig::instance()->setLODScale(LODScaleEdit->getValue());
        for (int i = 0; i < coVRConfig::instance()->numChannels(); i++)
            coVRConfig::instance()->channels[i].camera->setLODScale(LODScaleEdit->getValue());
    }
    OpenCOVER::instance()->hud->hide();
}

void coVRTui::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == driveNav)
    {
        mx = driveNav->x;
        my = driveNav->y;
    }
    else if (tUIItem == panNav)
    {
        mx = panNav->x;
        my = panNav->y;
    }

    if (tUIItem == Quit)
    {
        OpenCOVER::instance()->quitCallback(NULL, NULL);
    }
    else if (tUIItem == ViewAll)
    {
        VRSceneGraph::instance()->viewAll();
    }
    else if (tUIItem == Walk)
    {
        cover->enableNavigation("Walk");
    }
    else if (tUIItem == Fly)
    {
        cover->enableNavigation("Fly");
    }
    else if (tUIItem == Drive)
    {
        cover->enableNavigation("Drive");
    }
    else if (tUIItem == Scale)
    {
        cover->enableNavigation("Scale");
    }
    else if (tUIItem == XForm)
    {
        cover->enableNavigation("XForm");
    }
    else if (tUIItem == Collision)
    {
        cover->setBuiltInFunctionState("Collide", true);
    }
    else if (tUIItem == Freeze)
    {
        coVRConfig::instance()->setFrozen(true);
        cover->setBuiltInFunctionState("Freeze", true);
    }
    else if (tUIItem == Wireframe)
    {
        VRSceneGraph::instance()->setWireframe(true);
    }
    else if (tUIItem == Menu)
    {
        VRSceneGraph::instance()->setMenu(false);
    }
    else if ((tUIItem == panNav) || (tUIItem == driveNav))
    {
        startTabNav();
    }
    else if (tUIItem == Animate)
    {
        coVRAnimationManager::instance()->enableAnimation(true);
    }
    else if (tUIItem == AnimForward)
    {
        coVRAnimationManager::instance()->enableAnimation(false);
        coVRAnimationManager::instance()->setAnimationFrame(coVRAnimationManager::instance()->getAnimationFrame() + 1);
    }
    else if (tUIItem == AnimBack)
    {
        coVRAnimationManager::instance()->enableAnimation(false);
        coVRAnimationManager::instance()->setAnimationFrame(coVRAnimationManager::instance()->getAnimationFrame() - 1);
    }
}

void coVRTui::tabletReleaseEvent(coTUIElement *tUIItem)
{
    if (tUIItem == driveNav)
    {
        mx = driveNav->x;
        my = driveNav->y;
    }
    else if (tUIItem == panNav)
    {
        mx = panNav->x;
        my = panNav->y;
    }

    if (tUIItem == coverTab)
    {
        //  cerr << "entry:" << cBox->getSelectedText() << endl;
    }
    else if ((tUIItem == panNav) || (tUIItem == driveNav))
    {
        stopTabNav();
    }
    else if (tUIItem == Walk)
    {
        cover->disableNavigation("Walk");
    }
    else if (tUIItem == Fly)
    {
        cover->disableNavigation("Fly");
    }
    else if (tUIItem == Drive)
    {
        cover->disableNavigation("Drive");
    }
    else if (tUIItem == Scale)
    {
        cover->disableNavigation("Scale");
    }
    else if (tUIItem == XForm)
    {
        cover->disableNavigation("XForm");
    }
    else if (tUIItem == Collision)
    {
        cover->setBuiltInFunctionState("Collide", false);
    }
    else if (tUIItem == Freeze)
    {
        coVRConfig::instance()->setFrozen(false);
        cover->setBuiltInFunctionState("Freeze", false);
    }
    else if (tUIItem == Wireframe)
    {
        VRSceneGraph::instance()->setWireframe(false);
    }
    else if (tUIItem == Menu)
    {
        VRSceneGraph::instance()->setMenu(true);
    }
    else if (tUIItem == Animate)
    {
        coVRAnimationManager::instance()->enableAnimation(false);
    }
}

void coVRTui::doTabFly()
{
    osg::Vec3 velDir;

    osg::Matrix dcs_mat = cover->getObjectsXform()->getMatrix();
    float heading = 0.0;
    if (driveNav->down)
        heading = (mx - x0) / -300;
    float pitch = (my - y0) / -300;
    float roll = (mx - x0) / -300;
    if (driveNav->down)
        roll = 0.;

    osg::Matrix rot;
    //rot.makeEuler(heading,pitch,roll);
    MAKE_EULER_MAT(rot, heading, pitch, roll);
    dcs_mat.postMult(rot);
    // XXX
    //dcs_mat.getRow(1,velDir);                // velocity in direction of viewer
    velDir[0] = 0.0;
    if (panNav->down)
        velDir[1] = -1.0;
    else
        velDir[1] = 0;
    velDir[2] = 0.0;
    dcs_mat.postMult(osg::Matrix::translate(velDir[0] * currentVelocity,
                                            velDir[1] * currentVelocity, velDir[2] * currentVelocity));
    cover->getObjectsXform()->setMatrix(dcs_mat);
    //       navigating = true;
    coVRCollaboration::instance()->SyncXform();
}

void coVRTui::doTabXform()
{
    if (driveNav->down)
    {

        float newx, newy; //newz;//relz0
        float heading, pitch, roll;
        //rollVerti, rollHori;

        //    whichButton = 1;
        newx = mx - originX;
        newy = my - originY;

        //getPhi kann man noch aendern, wenn xy-Rotation nicht gewichtet werden soll
        heading = getPhi(newx - relx0, widthX);
        pitch = getPhi(newy - rely0, widthY);
        roll = getPhi(newy - rely0, widthY);

        //makeRot(heading, pitch, roll, headingBool, pitchBool, rollBool);

        makeRot(heading, pitch, roll, 1, 1, 1);
        relx0 = mx - originX;
        rely0 = my - originY;
    }

    //Translation funktioniert
    if (panNav->down)
    {
        //irgendwas einbauen, damit die folgenden Anweisungen vor der
        //naechsten if-Schleife nicht unnoetigerweise dauernd ausgefuehrt werden

        float newxTrans = mx - originX;
        float newyTrans = my - originY;
        float xTrans, yTrans;
        xTrans = (newxTrans - relx0);
        yTrans = (newyTrans - rely0);
        cerr << yTrans << endl;
        osg::Matrix actTransState = mat0;
        osg::Matrix trans;
        osg::Matrix doTrans;
        trans.makeTranslate(1.1 * xTrans, 0.0, -1.1 * yTrans);
        //die 1.1 sind geschaetzt, um die Ungenauigkeit des errechneten Faktors auszugleichen
        //es sieht aber nicht so aus, als wuerde man es auf diese
        //Weise perfekt hinbekommen
        doTrans.mult(actTransState, trans);
        cover->getObjectsXform()->setMatrix(doTrans);
        coVRCollaboration::instance()->SyncXform();
    }
}

void coVRTui::doTabScale()
{
    float newx, newScaleFactor; // newy,
    float maxFactor = 10.0;
    float widthX = coVRConfig::instance()->windows[0].sx;
    float ampl = widthX / 2;

    if (driveNav->down)
        newx = my - originY;
    else
        newx = mx - originX;

    //Skalierung mittels e-Funktion, kann nicht negativ werden
    newScaleFactor = exp(log(maxFactor) * ((newx - relx0) / ampl));
    cover->setScale(actScaleFactor * newScaleFactor);
    coVRCollaboration::instance()->SyncXform();
    coVRCollaboration::instance()->SyncScale();
}

void coVRTui::doTabWalk()
{
    osg::Vec3 velDir;
    osg::Matrix dcs_mat = cover->getObjectsXform()->getMatrix();
    if (driveNav->down)
    {
        float angle;

        angle = (float)((mx - x0) * cover->frameDuration()) / 300.0f;
        /*
      if((mx - x0) < 0)
      angle =  -((mx - x0)*(mx - x0) / 4000);
      else
      angle =  (mx - x0)*(mx - x0) / 4000;*/
        dcs_mat.postMult(osg::Matrix::rotate(angle, 0.0, 0.0, 1.0));
        velDir[0] = 0.0;
        velDir[1] = 1.0;
        velDir[2] = 0.0;
        currentVelocity = (my - y0) * driveSpeed * cover->frameDuration();
        dcs_mat.postMult(osg::Matrix::translate(velDir[0] * currentVelocity,
                                                velDir[1] * currentVelocity, velDir[2] * currentVelocity));
    }
    else if (panNav->down)
    {
        dcs_mat.postMult(osg::Matrix::translate((mx - x0) * driveSpeed * 0.1 * -1,
                                                0, (my - y0) * driveSpeed * 0.1 * 1));
    }
    cover->getObjectsXform()->setMatrix(dcs_mat);
    coVRCollaboration::instance()->SyncXform();
}

void coVRTui::stopTabNav()
{
    actScaleFactor = cover->getScale();

    x0 = mx;
    y0 = my;
    currentVelocity = 10;
    relx0 = x0 - originX; //relativ zum Ursprung des Koordinatensystems
    rely0 = y0 - originY; //dito
    cerr << "testit" << endl;
}

void coVRTui::startTabNav()
{
    widthX = 200;
    widthY = 170;
    originX = widthX / 2.0;
    originY = widthY / 2.0;

    osg::Matrix dcs_mat = cover->getObjectsXform()->getMatrix();
    actScaleFactor = cover->getScale();
    x0 = mx;
    y0 = my;
    mat0 = dcs_mat;
    mat0_Scale = dcs_mat;
    //cerr << "mouseNav" << endl;
    currentVelocity = 10;

    relx0 = x0 - originX; //relativ zum Ursprung des Koordinatensystems
    rely0 = y0 - originY; //dito
}

float coVRTui::getPhiZVerti(float y2, float y1, float x2, float widthX, float widthY)
{
    float phiZMax = 180.0;
    float y = (y2 - y1);
    float phiZ = 4 * pow(x2, 2) * phiZMax * y / (pow(widthX, 2) * widthY);
    if (x2 < 0.0)
        return (phiZ);
    else
        return (-phiZ);
}

float coVRTui::getPhiZHori(float x2, float x1, float y2, float widthY, float widthX)
{
    float phiZMax = 180.0;
    float x = x2 - x1;
    float phiZ = 4 * pow(y2, 2) * phiZMax * x / (pow(widthY, 2) * widthX);

    if (y2 > 0.0)
        return (phiZ);
    else
        return (-phiZ);
}

float coVRTui::getPhi(float relCoord1, float width1)
{
    float phi, phiMax = 180.0;
    phi = phiMax * relCoord1 / width1;
    return (phi);
}

void coVRTui::makeRot(float heading, float pitch, float roll, int headingBool, int pitchBool, int rollBool)
{
    osg::Matrix actRotState = cover->getObjectsXform()->getMatrix();
    osg::Matrix rot;
    osg::Matrix doRot;
    //rot.makeEuler(headingBool * heading, pitchBool * pitch, rollBool * roll);
    MAKE_EULER_MAT(rot, headingBool * heading, pitchBool * pitch, rollBool * roll);
    doRot.mult(actRotState, rot);
    cover->getObjectsXform()->setMatrix(doRot);
    coVRCollaboration::instance()->SyncXform();
}

coTUIFileBrowserButton *coVRTui::getExtFB()
{
    return SaveFileFB;
}

BinListEntry::BinListEntry(osgUtil::RenderBin *rb, int num)
{
    binNumber = rb->getBinNum();
    std::string name;
    std::stringstream ss;
    ss << binNumber;
    name = ss.str();
    tb = new coTUIToggleButton(name, coVRTui::instance()->getTopContainer()->getID(), true);
    tb->setPos(4, num);
    tb->setEventListener(this);
}

void BinListEntry::updateBin()
{
    osgUtil::RenderBin *bin = renderBin();
    if (bin)
    {
        if (tb->getState())
        {
            bin->setDrawCallback(NULL);
        }
        else
        {
            bin->setDrawCallback(new DontDrawBin);
        }
    }
}

osgUtil::RenderBin *BinListEntry::renderBin()
{
    osgUtil::RenderStage *rs = dynamic_cast<osgViewer::Renderer *>(coVRConfig::instance()->channels[0].camera->getRenderer())->getSceneView(0)->getRenderStage();
    if (rs == NULL)
        rs = dynamic_cast<osgViewer::Renderer *>(coVRConfig::instance()->channels[0].camera->getRenderer())->getSceneView(0)->getRenderStageLeft();
    if (rs)
    {
        BinRenderStage bs = *rs;
        osgUtil::RenderBin *rb = bs.getPreRenderStage();
        if (rb)
        {
            for (osgUtil::RenderBin::RenderBinList::iterator it = rb->getRenderBinList().begin(); it != rb->getRenderBinList().end(); it++)
            {
                if (it->second->getBinNum() == binNumber)
                    return it->second;
            }
        }
        rb = rs;
        for (osgUtil::RenderBin::RenderBinList::iterator it = rb->getRenderBinList().begin(); it != rb->getRenderBinList().end(); it++)
        {
            if (it->second->getBinNum() == binNumber)
                return it->second;
        }
        rb = bs.getPostRenderStage();
        if (rb)
        {
            for (osgUtil::RenderBin::RenderBinList::iterator it = rb->getRenderBinList().begin(); it != rb->getRenderBinList().end(); it++)
            {
                if (it->second->getBinNum() == binNumber)
                    return it->second;
            }
        }
    }
    return NULL;
}
BinListEntry::~BinListEntry()
{
    delete tb;
    renderBin()->setDrawCallback(NULL);
}
void BinListEntry::tabletEvent(coTUIElement *tUIItem)
{

    /*	if(tb->getState())
	{
		renderBin()->setDrawCallback(NULL);
	}
	else
	{
		renderBin()->setDrawCallback(new DontDrawBin);
	}*/
}

BinList::BinList()
{
}

BinList::~BinList()
{
    removeAll();
}
void BinList::refresh()
{
    for (BinList::iterator it = begin(); it != end(); it++)
    {
        delete *it;
    }
    clear();
    osgUtil::RenderStage *rs = dynamic_cast<osgViewer::Renderer *>(coVRConfig::instance()->channels[0].camera->getRenderer())->getSceneView(0)->getRenderStage();
    if (rs == NULL)
        rs = dynamic_cast<osgViewer::Renderer *>(coVRConfig::instance()->channels[0].camera->getRenderer())->getSceneView(0)->getRenderStageLeft();
    if (rs)
    {
        BinRenderStage bs = *rs;
        osgUtil::RenderBin *rb = bs.getPreRenderStage();
        if (rb)
        {
            for (osgUtil::RenderBin::RenderBinList::iterator it = rb->getRenderBinList().begin(); it != rb->getRenderBinList().end(); it++)
            {
                push_back(new BinListEntry(it->second, size()));
            }
        }
        rb = rs;
        for (osgUtil::RenderBin::RenderBinList::iterator it = rb->getRenderBinList().begin(); it != rb->getRenderBinList().end(); it++)
        {
            push_back(new BinListEntry(it->second, size()));
        }
        rb = bs.getPostRenderStage();
        if (rb)
        {
            for (osgUtil::RenderBin::RenderBinList::iterator it = rb->getRenderBinList().begin(); it != rb->getRenderBinList().end(); it++)
            {
                push_back(new BinListEntry(it->second, size()));
            }
        }
    }
}

void BinList::updateBins()
{
    for (BinList::iterator it = begin(); it != end(); it++)
    {
        (*it)->updateBin();
    }
}
void BinList::removeAll()
{
    for (BinList::iterator it = begin(); it != end(); it++)
    {
        delete *it;
    }
    clear();
}
