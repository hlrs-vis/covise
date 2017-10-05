/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *                                                                      *
 *                                                                      *
 *                            (C) 1996                                  *
 *              Computer Centre University of Stuttgart                 *
 *                         Allmandring 30                               *
 *                       D-70550 Stuttgart                              *
 *                            Germany                                   *
 *                                                                      *
 *                                                                      *
 *    File           VRPinboard.C (Performer 2.0)                       *
 *                                                                      *
 *    Description    menu system                                        *
 *                                                                      *
 *    Date           20.08.97                                           *
 *                                                                      *
 *    Author         Frank Foehl                                        *
 *                                                                      *
 ************************************************************************/

//#define HIGHLIGHT 1
//#define NOTEXT 1

#include <util/common.h>
#include <util/unixcompat.h>

#include <config/CoviseConfig.h>
#include <input/VRKeys.h>
#include "OpenCOVER.h"
#include "VRPinboard.h"
#include "coVRConfig.h"
#include "coVRNavigationManager.h"
#include "VRSceneGraph.h"
#include "coVRLighting.h"
#include "VRViewer.h"
#include "coVRMSController.h"
#include "coVRPluginSupport.h"
#include "coVRSelectionManager.h"

#include <OpenVRUI/osg/mathUtils.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coCheckboxGroup.h>

#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>

#include <OpenVRUI/osg/OSGVruiMatrix.h>

#define NAV_GROUP_TYPE 0;
#define FUNC_TYPE 3;
#define SWITCH_TYPE 4;

using namespace opencover;
using namespace vrui;
using covise::coCoviseConfig;

VRPinboard *VRPinboard::s_singleton = NULL;

coCheckboxGroup *opencover::groupPointerArray[100];

buttonSpecCell::buttonSpecCell(const buttonSpecCell &s)
{
    strcpy(name, s.name);
    actionType = s.actionType;
    callback = s.callback;
    calledClass = s.calledClass;
    state = s.state;
    oldState = state;
    dragState = s.dragState;
    subMenu = s.subMenu;
    strcpy(subMenuName, s.subMenuName);
    dashed = s.dashed;
    group = s.group;
    sliderMin = s.sliderMin;
    sliderMax = s.sliderMax;
    userData = s.userData;
    if (s.myMenu)
        setMenu(s.myMenu);
}

buttonSpecCell &buttonSpecCell::operator=(const buttonSpecCell &s)
{
    strcpy(name, s.name);
    actionType = s.actionType;
    callback = s.callback;
    calledClass = s.calledClass;
    state = s.state;
    oldState = state;
    dragState = s.dragState;
    subMenu = s.subMenu;
    strcpy(subMenuName, s.subMenuName);
    dashed = s.dashed;
    group = s.group;
    sliderMin = s.sliderMin;
    sliderMax = s.sliderMax;
    userData = s.userData;
    if (s.myMenu)
        setMenu(s.myMenu);

    return *this;
}

buttonSpecCell::buttonSpecCell()
{
    name[0] = '\0';
    actionType = BUTTON_FUNCTION;
    callback = NULL;
    calledClass = NULL;
    state = 0.0;
    oldState = 0.0;
    dragState = BUTTON_RELEASED;
    subMenu = NULL;
    subMenuName[0] = '\0';
    dashed = false;
    group = -1;
    sliderMin = 0.0;
    sliderMax = 0.0;
    userData = NULL;
    myMenu = NULL;
}

buttonSpecCell::~buttonSpecCell()
{
    delete[] myMenu;
}

void buttonSpecCell::setMenu(const char *n)
{
    delete[] myMenu;
    if (n)
    {
        myMenu = new char[strlen(n) + 1];
        strcpy(myMenu, n);
    }
    else
        myMenu = NULL;
}

/************************************************************************
 *                                                                      *
 *   class VRPinboard                                                   *
 *                                                                      *
 ************************************************************************/
VRPinboard *VRPinboard::instance()
{
    if (!s_singleton)
        s_singleton = new VRPinboard();
    return s_singleton;
}

VRPinboard::PinboardFunction::PinboardFunction(const char *fn,
                                               int ft,
                                               const char *bn,
                                               const char *mn,
                                               ButtonCallback cb,
                                               void *cc)
    : functionName(fn)
    , functionType(ft)
    , defButtonName(bn)
    , defMenuName(mn)
    , callback(cb)
    , callbackClass(cc)
    , isInPinboard(false)
    , customButtonName(NULL)
    , customMenuName(NULL)
{
    if (bn)
    {
        customButtonName = new char[strlen(bn) + 1];
        strcpy(customButtonName, bn);
    }
    if (mn)
    {
        customMenuName = new char[strlen(mn) + 1];
        strcpy(customMenuName, mn);
    }
}

void VRPinboard::addFunction(const char *fn, int ft, const char *bn, const char *mn, ButtonCallback cb, void *cc)
{
    functions.push_back(PinboardFunction(fn, ft, bn, mn, cb, cc));
}

/*______________________________________________________________________*/
VRPinboard::VRPinboard()
{
    assert(!s_singleton);

    customPinboard = false;
    mainMenu = addMenu("COVER", NULL);

    if (cover->debugLevel(2))
        fprintf(stderr, "\nnew VRPinboard\n");

    for (int i = 0; i < 100; i++)
    {
        groupPointerArray[i] = NULL;
    }
    /*   theToolbox = new coToolboxMenu("MainToolbar");*/
    /// allow deselection of group 0
    groupPointerArray[0] = new coCheckboxGroup(true);
    //lockInteraction = false;
    // initialize the custom entries with the def values

    addFunction("XForm", BTYPE_NAVGROUP, "move world", NULL, coVRNavigationManager::xformCallback, coVRNavigationManager::instance());
    addFunction("Scale", BTYPE_NAVGROUP, "scale world", NULL, coVRNavigationManager::scaleCallback, coVRNavigationManager::instance());
    addFunction("ViewAll", BTYPE_FUNC, "view all", NULL, VRSceneGraph::viewallCallback, VRSceneGraph::instance());
    addFunction("ResetView", BTYPE_FUNC, "reset view", "navigation", VRSceneGraph::resetviewCallback, VRSceneGraph::instance());
    addFunction("Fly", BTYPE_NAVGROUP, "fly", "navigation", coVRNavigationManager::flyCallback, coVRNavigationManager::instance());
    addFunction("Walk", BTYPE_NAVGROUP, "walk", "navigation", coVRNavigationManager::walkCallback, coVRNavigationManager::instance());
    addFunction("Drive", BTYPE_NAVGROUP, "drive", "navigation", coVRNavigationManager::driveCallback, coVRNavigationManager::instance());
    //addFunction("XFormRotate", BTYPE_NAVGROUP, "rotate", "navigation", coVRNavigationManager::xformRotateCallback, coVRNavigationManager::instance());
    //addFunction("XFormTranslate", BTYPE_NAVGROUP, "translate", "navigation", coVRNavigationManager::xformTranslateCallback, coVRNavigationManager::instance());
    addFunction("Selection", BTYPE_NAVGROUP, "select", "navigation", coVRSelectionManager::selectionCallback, coVRSelectionManager::instance());
    addFunction("ShowName", BTYPE_NAVGROUP, "show name", "navigation", coVRNavigationManager::showNameCallback, coVRNavigationManager::instance());
    //addFunction("Measure", BTYPE_NAVGROUP, "measure", "navigation", coVRNavigationManager::measureCallback, coVRNavigationManager::instance());
    //addFunction("traverseInteractors", BTYPE_NAVGROUP, "traverseInteractors", "navigation", coVRNavigationManager::traverseInteractorsCallback, coVRNavigationManager::instance());
    //addFunction("ScalePlus", BTYPE_FUNC, "scale plus", "navigation", VRSceneGraph::scalePlusCallback, VRSceneGraph::instance());
    //addFunction("ScaleMinus", BTYPE_FUNC, "scale minus", "navigation", VRSceneGraph::scaleMinusCallback, VRSceneGraph::instance());
    addFunction("DriveSpeed", BTYPE_SLIDER, "speed", "navigation", coVRNavigationManager::driveSpeedCallback, coVRNavigationManager::instance());
    addFunction("Collide", BTYPE_TOGGLE, "collision detection", "navigation", coVRNavigationManager::collideCallback, coVRNavigationManager::instance());
    addFunction("Snap", BTYPE_TOGGLE, "snap", "navigation", coVRNavigationManager::snapCallback, coVRNavigationManager::instance());

    addFunction("Freeze", BTYPE_TOGGLE, "stop headtracking", NULL, VRViewer::freezeCallback, VRViewer::instance());
    addFunction("HighQuality", BTYPE_TOGGLE, "high quality mode", "view options", VRSceneGraph::highQualityCallback, VRSceneGraph::instance());
    addFunction("CoordAxis", BTYPE_TOGGLE, "show axis", "view options", VRSceneGraph::coordAxisCallback, VRSceneGraph::instance());
    //addFunction("Specular", BTYPE_TOGGLE, "specular", "view options", coVRLighting::specularCallback, coVRLighting::instance());
    //addFunction("Spotlight", BTYPE_TOGGLE, "spotlight", "view options", coVRLighting::spotlightCallback, coVRLighting::instance());
    //addFunction("Headlight", BTYPE_TOGGLE, "headlight", "view options", coVRLighting::headlightCallback, coVRLighting::instance());
    //addFunction("OtherLights", BTYPE_TOGGLE, "other lights", "view options", coVRLighting::otherlightsCallback, coVRLighting::instance());
    addFunction("StereoSeparation", BTYPE_TOGGLE, "stereo separation", "view options", VRViewer::stereoSepCallback, VRViewer::instance());
    //addFunction("StereoEyeDistance", BTYPE_TOGGLE, "stereo separation", "view options", VRViewer::stereoSepCallback, VRViewer::instance());
    addFunction("OrthographicProjection", BTYPE_TOGGLE, "orthographic projection", "view options", VRViewer::orthographicCallback, VRViewer::instance());
    addFunction("Statistics", BTYPE_TOGGLE, "statistics", "view options", VRViewer::statisticsCallback, VRViewer::instance());
    addFunction("ClusterStatistics", BTYPE_TOGGLE, "cluster statistics", "view options", coVRMSController::statisticsCallback, coVRMSController::instance());
    addFunction("Store", BTYPE_FUNC, "store scenegraph", "misc", VRSceneGraph::storeCallback, VRSceneGraph::instance());
    addFunction("ReloadVRML", BTYPE_FUNC, "reload file", "misc", VRSceneGraph::reloadFileCallback, VRSceneGraph::instance());
    addFunction("Quit", BTYPE_FUNC, "quit", "misc", VRPinboard::quitCallback, OpenCOVER::instance());

    num_perm_functions = functions.size();

    makeQuitMenu();
}

/*______________________________________________________________________*/
VRPinboard::~VRPinboard()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete VRPinboard\n");
    s_singleton = NULL;
}

/*______________________________________________________________________*/
void
VRPinboard::configInteraction()
{
    int numEntries = 0;

    if (cover->debugLevel(3))
        fprintf(stderr, "VRPinboard::configInteraction\n");

    // read the PinboardConfig, extract the button and menu names

    coCoviseConfig::ScopeEntries pbEntries = coCoviseConfig::getScopeEntries("COVER.Pinboard");
    const char **entries = pbEntries.getValue();

    if (entries) // custom Pinboard entries
    {
        // check if there is only a scope for 1 computer or the general scope
        // because COVERPinboard{} and COVERPinboard:xxx{} isn't supported
        int n = 0;
        unsigned int nce = 0;
        // entries contains key value
        while (entries[n] != NULL)
        {
            int j = getIndex(entries[n]);
            if (j != -1)
                nce += 2;
            n += 2;
        }

        numEntries = nce;

        if ((nce - 2) > 2 * functions.size()) // two scopes->default config
        {
            fprintf(stderr, "ERROR: there is more than one scope of COVERPinboard\n");
            fprintf(stderr, "       making the default pinboard layout\n");
        }
        else // custom entries
        {
            customPinboard = true;
            for (int i = 0; i < numEntries; i += 2)
            {
                readCustomEntry(entries[i]);
            }
        }
    }
    else
    {
        //fprintf(stderr,"WARNING in VRPinboard: scope entries COVERPinboard not available\n");
    }

    // now make the permanent buttons and the execute button

    if (!customPinboard) // we make ALL permanent buttons
    {
        for (int j = 0; j < num_perm_functions; j++)
        {
            functions[j].isInPinboard = true;
            makeButton(functions[j].functionName, functions[j].defButtonName,
                       functions[j].defMenuName, functions[j].functionType,
                       functions[j].callback, functions[j].callbackClass);
        }
    }
    else // we have to keep the order in the covise.config
    // and make only the buttons configured in covise.config
    {
        for (int i = 0; i < numEntries; i += 2)
        {
            int j = getIndex(entries[i]);
            if (j != -1) //avoid misspelling
            {
                if (isPermanentEntry(entries[i]))
                {

                    if (functions[j].isInPinboard)
                        makeButton(functions[j].functionName, functions[j].customButtonName,
                                   functions[j].customMenuName, functions[j].functionType,
                                   functions[j].callback, functions[j].callbackClass);
                }
            }
        }
    }
}

/*
================
getButtonByName
================
*/

VRButton *
VRPinboard::getButtonByName(const char *buttonName)
{
    for (list<VRMenu *>::iterator m = menuList.begin(); m != menuList.end(); ++m)
    {
        VRMenu *menu = *m;
        VRButton *button = menu->namedButton(buttonName);
        if (button)
        {
            return button;
        }
    }

    return NULL;
}

/*
============
getFileName
============
*/

const char *
VRPinboard::getFileName(const char *string)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRPinboard::getFileName\n");

    const char *start = strchr(string, '"');
    start++;
    const char *end = strrchr(string, '"');

    if (!start || !end)
    {
        fprintf(stderr, "string passed in: %s \n", string);
        fprintf(stderr, "error: could not find separator - wrong usage of VRPinboard::getFileName \n");
        return NULL;
    }

    char *ret = new char[end - start + 1];
    memcpy(ret, start, end - start + 1);
    ret[end - start] = '\0';

    return ret;
}

/*
===========
makeButton
===========
*/

void
VRPinboard::makeButton(const char *functionName, const char *buttonName,
                       const char *menuName, int type, ButtonCallback callback,
                       void *inst, void *userData)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRPinboard::makeButton for %s\n", functionName);

    bool state = false;
    float min, max, val;
    std::string line;
    char str[1024];
    char title[200];

    int i = getIndex(functionName);

    if (i != -1)
        functions[i].isInPinboard = true;

    switch (type)
    {
    case BTYPE_NAVGROUP:
        line = coCoviseConfig::getEntry("COVER.NavigationMode");
        if (!line.empty())
        {
            if (strcmp(line.c_str(), functionName) == 0)
            {
                //fprintf(stderr, "setting nav %s to true\n", functionName);
                state = true;
            }
        }
        else if (strcmp("Walk", functionName) == 0)
        {
            //fprintf(stderr, "setting nav %s to true\n", functionName);
            state = true;
        }

        if (menuName)
        {
            strcpy(title, menuName);
            strcat(title, "...");
            cover->addSubmenuButton(title, NULL, menuName, false, NULL, -1, this);
        }
        cover->addGroupButton(buttonName, menuName, state, callback, 0, inst, userData);
        if (state)
        {
            cover->enableNavigation(functionName);
        }
        //fprintf(stderr, "navMenu: %s/%s, state=%d\n", menuName?menuName:"(null)", buttonName, state);
        break;

    case BTYPE_SYNCGROUP:
        line = coCoviseConfig::getEntry("mode", "COVER.Collaborative.Sync");
        if (!line.empty())
        {

            if (strcmp(line.c_str(), functionName) == 0)
            {

                state = true;
            }
        }
        if (menuName)
        {
            strcpy(title, menuName);
            strcat(title, "...");
            cover->addSubmenuButton(title, NULL, menuName, false, NULL, -1, this);
        }
        cover->addGroupButton(buttonName, menuName, state, callback, 2, inst, userData);
        break;

    case BTYPE_FUNC:
        if (menuName)
        {
            strcpy(title, menuName);
            strcat(title, "...");
            cover->addSubmenuButton(title, NULL, menuName, false, NULL, -1, this);
        }
        cover->addFunctionButton(buttonName, menuName, callback, inst, userData);
        break;

    case BTYPE_TOGGLE:

        sprintf(str, "COVER.%s", functionName);
        line = coCoviseConfig::getEntry(str);
        if (line.empty())
        {
            // we: default setting 'on' must be explicitly stated here
            if (0 == strcmp(functionName, "Headlight")
                || 0 == strcmp(functionName, "OtherLights")
                || (0 == strcmp(functionName, "StereoSeparation") && ((coVRConfig::instance()->stereoState()) || (coVRConfig::instance()->monoView() != coVRConfig::MONO_MIDDLE)))
                || 0 == strcmp(functionName, "HighQuality"))
            {
                state = true;
            }
            else
            {
                state = false;
            }
        }
        else // if line exist, use isOn: test for 'on', 'true', '1' ...
            state = coCoviseConfig::isOn(str, false);

        if (menuName)
        {
            strcpy(title, menuName);
            strcat(title, "...");
            cover->addSubmenuButton(title, NULL, menuName, false, NULL, -1, this);
        }
        cover->addToggleButton(buttonName, menuName, state, callback, inst, userData);
        cover->callButtonCallback(buttonName);

        break;

    case BTYPE_SLIDER:
        sprintf(str, "COVER.%s", functionName);
        min = coCoviseConfig::getFloat("min", str, 0.03);
        max = coCoviseConfig::getFloat("max", str, 30);
        val = coCoviseConfig::getFloat("default", str, 1.0);
        cover->addSliderButton(buttonName, menuName, min, max, val, callback, inst, userData);
        cover->callButtonCallback(buttonName);
        break;

    default:
        fprintf(stderr, "ERROR: VRPinboard::makeButton - unknow type\n");
    }
}

int
VRPinboard::getIndex(const char *functionName)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRPinboard::getIndex for %s\n", functionName);

    for (size_t i = 0; i < functions.size(); i++)
    {
        if (strcasecmp(functions[i].functionName, functionName) == 0)
            return (i);
    }
    //fprintf(stderr,"ERROR in VRPinboard - function %s is not a valid COVER function\n", functionName);
    return (-1);
}

const char *
VRPinboard::getCustomName(const char *functionName)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRPinboard::getCustomName for %s\n", functionName);

    int i = getIndex(functionName);
    if (i >= 0)
    {
        return (functions[i].customButtonName);
    }
    else
    {
        return (NULL);
    }
}

int
VRPinboard::isPermanentEntry(const char *functionName)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRPinboard::isPermanentEntry for %s\n", functionName);

    int i = getIndex(functionName);
    if (i < num_perm_functions)
    {
        return true;
    }
    else
    {

        return false;
    }
}

void VRPinboard::makeQuitMenu()
{
    string qtext, yesText, noText;

    qtext = "Really quit OpenCOVER?";
    yesText = "Quit";
    noText = "Continue";

    quitMenu_ = new coRowMenu(qtext.c_str());
    quitMenu_->setVisible(false);
    quitMenu_->setAttachment(coUIElement::RIGHT);
    OSGVruiMatrix transMatrix, scaleMatrix, rotateMatrix, matrix;

    float px = coCoviseConfig::getFloat("x", "COVER.QuitMenu.Position", 0.0);
    float py = coCoviseConfig::getFloat("y", "COVER.QuitMenu.Position", 0.0);
    float pz = coCoviseConfig::getFloat("z", "COVER.QuitMenu.Position", 0.0);
    float s = coCoviseConfig::getFloat("value", "COVER.QuitMenu.Size", 1.0f);

    matrix.makeIdentity();
    transMatrix.makeTranslate(px, py, pz);
    rotateMatrix.makeEuler(0, 90, 0);
    scaleMatrix.makeScale(s, s, s);

    matrix.mult(&scaleMatrix);
    matrix.mult(&rotateMatrix);
    matrix.mult(&transMatrix);

    quitMenu_->setTransformMatrix(&matrix);
    quitMenu_->setScale(cover->getSceneSize() / 2500);
    yesButton_ = new coButtonMenuItem(yesText.c_str());
    yesButton_->setMenuListener(this);
    cancelButton_ = new coButtonMenuItem(noText.c_str());
    cancelButton_->setMenuListener(this);
    quitMenu_->add(yesButton_);
    quitMenu_->add(cancelButton_);
}
void VRPinboard::showQuitMenu()
{
    quitMenu_->setVisible(true);
}
void VRPinboard::hideQuitMenu()
{
    quitMenu_->setVisible(false);
}
void
VRPinboard::quitCallback(void * /*sceneGraph*/, buttonSpecCell * /*spec*/)
{
    VRPinboard::instance()->showQuitMenu();
}

void
VRPinboard::menuEvent(coMenuItem *item)
{
    if (item == yesButton_)
    {
        quitMenu_->setVisible(false);
        OpenCOVER::instance()->quitCallback(NULL, NULL);
    }
    else if (item == cancelButton_)
    {
        quitMenu_->setVisible(false);
    }
}

/*______________________________________________________________________*/
VRButton *
VRPinboard::addButtonToMainMenu(buttonSpecCell *spec)
/* generates pinboard's mainMenu (if not already present) */
/*    and appends button to it */
{

    if (cover->debugLevel(4))
        fprintf(stderr, "VRPinboard::addButtonToMainMenu name=%s\n", spec->name);

    if (spec->actionType == BUTTON_SUBMENU)
    {
        spec->subMenu = this->namedMenu(spec->subMenuName);
        if (spec->subMenu == NULL)
            spec->subMenu = addMenu(spec->subMenuName, mainMenu->myMenu);
    }

    return this->mainMenu->addButton(spec);
    return NULL;
}

/*______________________________________________________________________*/
void VRPinboard::removeButtonFromMainMenu(const char *name)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRPinboard::addButtonToMainMenu for %s\n", name);

    if (mainMenu)
    {
        mainMenu->removeButton(name);
    }
}

/*______________________________________________________________________*/
void VRPinboard::removeButtonFromNamedMenu(const char *name, const char *menuName)
{

    if (cover->debugLevel(4))
    {
        if (menuName)
            fprintf(stderr, "VRPinboard::removeButtonFromNamedMenu menu=%s name=%s\n", menuName, name);
        else
            fprintf(stderr, "VRPinboard::removeButtonFromNamedMenu menu=NULL name=%s\n", name);
    }
    VRMenu *menu;
    menu = namedMenu((char *)menuName);
    if (menu)
    {
        menu->removeButton(name);
    }
}

/*______________________________________________________________________*/
VRButton *
VRPinboard::addButtonToNamedMenu(buttonSpecCell *spec, const char *menuName)
/* generates named menu (if not already present) */
/*    and appends button to it */
{
    if (cover->debugLevel(4))
    {
        if (menuName)
            fprintf(stderr, "VRPinboard::addButtonToNamedMenu menu=%s name=%s\n", menuName, spec->name);
        else
            fprintf(stderr, "VRPinboard::addButtonToNamedMenu menu=NULL name=%s\n", spec->name);
    }

    VRMenu *menu = namedMenu(menuName);
    if (menu == NULL)
        menu = addMenu(menuName, mainMenu->myMenu);

    if (spec->actionType == BUTTON_SUBMENU)
    {
        spec->subMenu = this->namedMenu(spec->subMenuName);
        if (spec->subMenu == NULL)
            spec->subMenu = addMenu(spec->subMenuName, menu->getCoMenu());
    }
    return menu->addButton(spec);
}

/*______________________________________________________________________*/
VRMenu *
VRPinboard::addMenu(const char *name, coMenu *parentMenu)
{
    if (cover->debugLevel(4))
    {
        fprintf(stderr, "VRPinboard::addMenu name=%s\n", name);
    }
    osg::Matrix dcsTransMat, dcsRotMat, dcsMat, preRot, tmp;
    float xp = 0.0, yp = 0.0, zp = 0.0;
    float h = 0, p = 0, r = 0;
    float size = 1;
    VRMenu *menu = new VRMenu(name);
    menu->myMenu = new coRowMenu(name, parentMenu);
    menu->myMenu->setMenuListener(menu);
    if (parentMenu == NULL)
    {
        menu->myMenu->setVisible(true);
    }

    menuList.push_back(menu);

    xp = coCoviseConfig::getFloat("x", "COVER.Menu.Position", 0.0);
    yp = coCoviseConfig::getFloat("y", "COVER.Menu.Position", 0.0);
    zp = coCoviseConfig::getFloat("z", "COVER.Menu.Position", 0.0);
    h = coCoviseConfig::getFloat("h", "COVER.Menu.Orientation", 0.0);
    p = coCoviseConfig::getFloat("p", "COVER.Menu.Orientation", 0.0);
    r = coCoviseConfig::getFloat("r", "COVER.Menu.Orientation", 0.0);
    size = coCoviseConfig::getFloat("COVER.Menu.Size", 0.0);

    if (size <= 0)
        size = 1;
    dcsRotMat.makeIdentity();
    MAKE_EULER_MAT(dcsRotMat, h, p, r);
    preRot.makeRotate(osg::inDegrees(90.0f), 1.0f, 0.0f, 0.0f);
    dcsTransMat.makeTranslate(xp, yp, zp);
    tmp.mult(preRot, dcsRotMat);
    dcsMat.mult(tmp, dcsTransMat);
    OSGVruiMatrix menuMatrix;
    menuMatrix.setMatrix(dcsMat);
    menu->myMenu->setTransformMatrix(&menuMatrix);
    menu->myMenu->setScale(size * (cover->getSceneSize() / 2500.0));

    return menu;
}

/*______________________________________________________________________*/
void
VRPinboard::removeMenuFromList(VRMenu *menu)
{

    if (cover->debugLevel(4))
        fprintf(stderr, "VRPinboard::removeMenuFromList\n");

    for (list<VRMenu *>::iterator m = menuList.begin(); m != menuList.end(); ++m)
    {
        if ((*m) == menu)
        {
            menuList.remove((*m));
            return;
        }
    }
}

/*______________________________________________________________________*/
VRMenu *
VRPinboard::namedMenu(const char *name)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRPinboard::namedMenu %s\n", name);

    VRMenu *menu;

    for (list<VRMenu *>::iterator m = menuList.begin(); m != menuList.end(); ++m)
    {
        menu = *m;
        if (menu->isNamedMenu(name))
            return menu;
    }
    return NULL;
}

/*______________________________________________________________________*/
bool
VRPinboard::setButtonState(const char *buttonName, float state)
/* traverse all buttons in all menus for a button 'buttonName' */
/* if found, set new state and return true, otherwise return false */
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRPinboard::setButtonState buttonName=%s state=%f\n", buttonName, state);

    bool boolean = false;

    for (list<VRMenu *>::iterator m = menuList.begin(); m != menuList.end(); ++m)
    {
        VRMenu *menu = *m;

        VRButton *button = menu->namedButton(buttonName);
        if (button)
        {
            button->setState(state);
            boolean = true;
        }
    }
    return boolean;
}

int
VRPinboard::callButtonCallback(const char *buttonName)
/* traverse all buttons in all menus for a button 'buttonName' */
/* if found, set new state and return true, otherwise return false */
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRPinboard::callButtonCallback buttonName=%s\n", buttonName);

    int result = false;

    for (list<VRMenu *>::iterator m = menuList.begin(); m != menuList.end(); ++m)
    {
        VRMenu *menu = *m;
        VRButton *button = menu->namedButton(buttonName);
        if (button)
        {
            if (button->spec.callback)
            {
                button->spec.callback(button->spec.calledClass, &(button->spec));
            }
            result = true;
        }
    }
    return result;
}

void
VRPinboard::readCustomEntry(const char *functionName)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRPinboard::readCustomEntry functionName=%s\n", functionName);

    // example: xform
    //
    // ntok=2 XFORM buttonname menuname    : xform with custom name in a custom menu
    // ntok=1 XFORM buttonname             : xform with custom name in main menu
    // ntok=1 XFORM NONE                   : no xform in menu
    // ntok=0 XFORM                        : default button and menu

    std::vector<char *> tokens;
    std::string entry = std::string("COVER.Pinboard.") + functionName;

    int num = 0;
    bool exists;
    std::string ent = coCoviseConfig::getEntry(entry, &exists);

    if (!exists)
    {
        num = -1;
    }
    else
    {
        char *tok = new char[ent.size() + 1];
        strcpy(tok, ent.c_str());

        for (char *s = tok; *s; ++s)
        {
            while (*s && *s != '"')
                ++s;
            if (!*s)
                break;
            tokens.push_back(++s);
            while (*s && *s != '"')
                ++s;
            if (!*s)
                break;
            *s = '\0';
            ++s;
            if (!*s)
                break;
        }
        num = tokens.size();
    }

    switch (num)
    {
    case -1: //this means entry XFORM is missing -> no XFORM, this case should never happen
    {
        int i = getIndex(functionName);
        if (i != -1)
            functions[i].isInPinboard = false;
    }
    break;

    case 0: // this means default button
    {
        int i = getIndex(functionName);
        if (i != -1)
            functions[i].isInPinboard = true;
        break;
    }
    case 1:
        if (strcasecmp(tokens[0], "NONE") == 0)
        {
            // this means no XFORM
            int i = getIndex(functionName);
            if (i != -1)
                functions[i].isInPinboard = false;
        }
        else // this means XFORM in main menu
        {

            // get the index of function name
            int i = getIndex(functionName);
            if (i != -1)
            {
                functions[i].isInPinboard = true;
                // we replace the name in the defaultButtonNames list
                if (functions[i].customButtonName)
                {

                    delete[] functions[i].customButtonName;
                }
                functions[i].customButtonName = new char[strlen(tokens[0]) + 1];
                strcpy(functions[i].customButtonName, tokens[0]);

                if (functions[i].customMenuName)
                {

                    delete[] functions[i].customMenuName;
                }
                functions[i].customMenuName = NULL;
            }
        }
        break;
    case 2:
    {
        // this means xform in submenu

        // append button to submenu
        int i = getIndex(functionName);
        if (i != -1)
        {
            functions[i].isInPinboard = true;

            // we replace the name in the defaultButtonNames list
            if (functions[i].customButtonName)
                delete[] functions[i].customButtonName;
            functions[i].customButtonName = NULL;
            functions[i].customButtonName = new char[strlen(tokens[0]) + 1];
            strcpy(functions[i].customButtonName, tokens[0]);

            if (functions[i].customMenuName)
                delete[] functions[i].customMenuName;
            functions[i].customMenuName = new char[strlen(tokens[1]) + 1];
            strcpy(functions[i].customMenuName, tokens[1]);
        }
    }
    break;
    default:
        fprintf(stderr, "ERROR: too many tokens in %s\n", functionName);
    }
}

coRowMenuItem *
VRPinboard::getMenuItem(const char *name)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRPinboard::getMenuItem name=%s\n", name);

    for (list<VRMenu *>::iterator m = menuList.begin(); m != menuList.end(); ++m)
    {
        VRMenu *menu = *m;

        VRButton *button = menu->namedButton(name);
        if (button)
        {
            return (button->getMenuItem());
        }
    }
    return (NULL);
}

/************************************************************************
 *                                                                      *
 *    class VRMenu                                                      *
 *                                                                      *
 ************************************************************************/

/*______________________________________________________________________*/
VRMenu::VRMenu(const char *n)
{

    if (cover->debugLevel(4))
        fprintf(stderr, "new VRMenu name=%s\n", n);

    name = new char[strlen(n) + 1];
    strcpy(name, n);
    myMenu = NULL;
}

/*______________________________________________________________________*/
VRMenu::~VRMenu()
{
    if (cover->debugLevel(4))
        fprintf(stderr, "delete VRMenu name=%s\n", name);

    delete[] myMenu;
    delete[] name;
    // remove me from the menuList
    VRPinboard::instance()->removeMenuFromList(this);
}

void VRMenu::focusEvent(bool, coMenu *)
{
    if (cover->debugLevel(5))
        fprintf(stderr, "VRMenu::focusEvent\n");

    //VRPinboard::instance()->lockInteraction = focus;
}

/*______________________________________________________________________*/
VRButton *
VRMenu::addButton(buttonSpecCell *spec)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRMenu::addButton name=%s\n", spec->name);

    VRButton *button = NULL;
    if (namedButton(spec->name) == NULL)
    {
        button = new VRButton(spec, myMenu);
        buttonList.push_back(button);
    }
    return button;
}

/*_____________________________________________________________________*/
void VRMenu::removeButton(const char *n)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRMenu::removeButton name=%s\n", n);

    //int i;
    VRButton *button = NULL;
    button = namedButton(n);
    if (button != NULL)
    {
        buttonList.remove(button);
        delete button;
    }
    else
        fprintf(stderr, "VRMenu::removeButton Button [%s] not found in menu [%s]!!!\n", n, this->name);
}

/*______________________________________________________________________*/
VRButton *
VRMenu::namedButton(const char *name)
/* return button called 'name' if present */
/* otherwise return NULL */
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRMenu::namedButton name=%s\n", name);

    for (list<VRButton *>::iterator b = buttonList.begin();
         b != buttonList.end();
         ++b)
    {
        VRButton *button = *b;
        if (button->isNamedButton(name))
            return button;
    }

    return NULL;
}

/*______________________________________________________________________*/
int
VRMenu::isNamedMenu(const char *name)
/* return true if menu's name is equal to 'name' */
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRMenu::isNamedMenu name=%s this->name=%s\n", name, this->name);

    return (strcmp(name, this->name) == 0);
}

/************************************************************************
 *                                                                      *
 *      class VRButton                                                  *
 *                                                                      *
 ************************************************************************/

/*______________________________________________________________________*/
VRButton::VRButton(buttonSpecCell *buttonSpec, coMenu *myMenu)
{

    if (cover->debugLevel(4))
        fprintf(stderr, "new VRButton name=%s\n", buttonSpec->name);

    /* copy buttonSpec struct into button */
    spec = *(buttonSpec);
    myMenuItem = NULL;

    if (spec.actionType == BUTTON_SLIDER)
    {
        //coSliderMenuItem * slider;
        myMenuItem = new coSliderMenuItem(spec.name, spec.sliderMin, spec.sliderMax, spec.state);
    }
    else if (spec.actionType == BUTTON_SWITCH)
    {
        if (spec.group >= 0 && spec.group < 100)
        {
            if (groupPointerArray[spec.group] == NULL)
            {
                groupPointerArray[spec.group] = new coCheckboxGroup();
            }
            myMenuItem = new coCheckboxMenuItem(spec.name, spec.state == 1.0, groupPointerArray[spec.group]);
        }
        else
        {
            myMenuItem = new coCheckboxMenuItem(spec.name, spec.state == 1.0);
        }
    }
    else if (spec.actionType == BUTTON_SUBMENU)
    {
        coSubMenuItem *button;
        myMenuItem = button = new coSubMenuItem(spec.name);
        VRMenu *menu;
        if ((menu = VRPinboard::instance()->namedMenu(spec.subMenuName)))
        {
            button->setMenu(menu->getCoMenu());
        }
        else
        {
            cerr << "subMenu not found" << spec.subMenu << endl;
        }
    }
    else if (spec.actionType == BUTTON_FUNCTION)
    {
        //coButtonMenuItem * button;
        myMenuItem = new coButtonMenuItem(spec.name);
    }
    if (myMenuItem)
    {
        myMenuItem->setMenuListener(this);
    }
    setState(spec.state);
    if (myMenu)
        myMenu->add(myMenuItem);
}

/*______________________________________________________________________*/
VRButton::~VRButton()
{

    if (cover->debugLevel(4))
        fprintf(stderr, "delete VRButton\n");

    delete myMenuItem;
}

/*______________________________________________________________________*/
int
VRButton::isNamedButton(const char *name)
/* return true if button's name is equal to 'name' */
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRButton::isNamedButton name=%s\n", name);

    return (strcmp(name, spec.name) == 0);
}

void VRButton::menuEvent(coMenuItem *)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRButton::menuEvent\n");

    switch (spec.actionType)
    {
    case BUTTON_FUNCTION:
        if (spec.callback)
            spec.callback((void *)spec.calledClass, &spec);
        break;
    case BUTTON_SWITCH:
        if (((coCheckboxMenuItem *)myMenuItem)->getState())
            spec.state = 1.0;
        else
            spec.state = 0.0;
        if (spec.callback)
            spec.callback((void *)spec.calledClass, &spec);
        break;

    case BUTTON_SLIDER:

        spec.state = ((coSliderMenuItem *)myMenuItem)->getValue();
        if (cover->getPointerButton()->wasPressed())
        {
            spec.dragState = BUTTON_PRESSED;
        }
        else
        {
            spec.dragState = BUTTON_DRAGGED;
        }
        if (spec.state < spec.sliderMin)
            spec.state = spec.sliderMin;
        if (spec.state > spec.sliderMax)
            spec.state = spec.sliderMax;
        if (spec.callback)
            spec.callback((void *)spec.calledClass, &spec);
        break;

    case BUTTON_SUBMENU:
        if (((coSubMenuItem *)myMenuItem)->isOpen())
            spec.state = 1.0;
        else
            spec.state = 0.0;
        if (spec.callback)
            spec.callback((void *)spec.calledClass, &spec);
        break;
    }
}

void VRButton::menuReleaseEvent(coMenuItem *)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRButton::menuEvent\n");

    switch (spec.actionType)
    {
    case BUTTON_SLIDER:

        spec.state = ((coSliderMenuItem *)myMenuItem)->getValue();

        spec.dragState = BUTTON_RELEASED;

        if (spec.state < spec.sliderMin)
            spec.state = spec.sliderMin;
        if (spec.state > spec.sliderMax)
            spec.state = spec.sliderMax;
        if (spec.callback)
            spec.callback((void *)spec.calledClass, &spec);
        break;
    }
}

/*______________________________________________________________________*/
void
VRButton::setState(float state)
/* set new button state (callback is invoked as appropriate) */
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRButton::setState state=%f\n", state);

    switch (spec.actionType)
    {
    case BUTTON_FUNCTION:
        ////if (spec.callback)
        ////    spec.callback((void*)spec.calledClass, &spec);
        break;
    case BUTTON_SWITCH:
    {
        spec.state = state;
        coCheckboxMenuItem *checkbox = (coCheckboxMenuItem *)myMenuItem;
        coCheckboxGroup *group = checkbox->getGroup();
        if (group)
        {
            group->setState(checkbox, spec.state != 0.0, true);
            checkbox->setState(spec.state != 0.0, false, true);
        }
        else
        {
            checkbox->setState(spec.state != 0.0, false, true);
        }
    }
    break;

    case BUTTON_SLIDER:
    {
        spec.state = state;
        spec.oldState = state;

        if (spec.state < spec.sliderMin)
            spec.state = spec.sliderMin;
        if (spec.state > spec.sliderMax)
            spec.state = spec.sliderMax;
        coSliderMenuItem *m = (coSliderMenuItem *)myMenuItem;
        if ((spec.sliderMin != m->getMin()) || (spec.sliderMax != m->getMax()))
        {
            m->setMin(spec.sliderMin);
            m->setMax(spec.sliderMax);
        }
        m->setValue(spec.state);
    }
    break;

    case BUTTON_SUBMENU:
        spec.state = state;
        if (spec.state)
            ((coSubMenuItem *)myMenuItem)->openSubmenu();
        else
            ((coSubMenuItem *)myMenuItem)->closeSubmenu();
        break;
        break;
    }
}

/*______________________________________________________________________*/
// Changes the displayed string
void
VRButton::setText(const char *s)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "VRButton::setText text=%s\n", s);

    myMenuItem->getLabel()->setString(s);
}

coRowMenuItem *
VRButton::getMenuItem()
{
    return (myMenuItem);
}
