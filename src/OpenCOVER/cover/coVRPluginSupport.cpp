/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRPluginSupport.h"
#include "OpenCOVER.h"
#include "coVRSelectionManager.h"
#include "VRSceneGraph.h"
#include "coVRNavigationManager.h"
#include "coVRCollaboration.h"
#include "coVRPluginList.h"
#include "coVRPlugin.h"
#include "coVRMSController.h"
#include <OpenVRUI/coUpdateManager.h>
#include <OpenVRUI/coInteractionManager.h>
#include <config/CoviseConfig.h>
#include <assert.h>
#include <net/tokenbuffer.h>
#include <net/message.h>
#include <net/message_types.h>
#include <OpenVRUI/sginterface/vruiButtons.h>
#include "input/VRKeys.h"
#include "input/input.h"
#include "input/coMousePointer.h"
#include "VRViewer.h"
#include "coTabletUI.h"
#ifdef DOTIMING
#include <util/coTimer.h>
#endif

#include <osg/MatrixTransform>
#include <osg/BoundingSphere>
#include <osg/Material>
#include <osg/ClipNode>
#include <osg/PolygonMode>
#include <util/unixcompat.h>
#include <osg/ComputeBoundsVisitor>

#ifdef __DARWIN_OSX__
#include <Carbon/Carbon.h>
#endif

// undef this to get no START/END messaged
#undef VERBOSE

#ifdef VERBOSE
#define START(msg) fprintf(stderr, "START: %s\n", (msg))
#define END(msg) fprintf(stderr, "END:   %s\n", (msg))
#else
#define START(msg)
#define END(msg)
#endif

#include <vrbclient/VRBClient.h>
#include "coVRConfig.h"

#include <grmsg/coGRKeyWordMsg.h>

using namespace opencover;
using namespace vrui;
using namespace grmsg;
using namespace covise;

opencover::coVRPluginSupport *opencover::cover = NULL;

bool coVRPluginSupport::debugLevel(int level) const
{
    return coVRConfig::instance()->debugLevel(level);
}

void
coVRPluginSupport::addGroupButton(const char *buttonName, const char *parentMenuName,
                                  int state, ButtonCallback callback, int groupId, void *classPtr, void *userData)
{
    START("coVRPluginSupport::addGroupButton");
    if (debugLevel(3))
    {
        fprintf(stderr, "\ncoVRPluginSupport::addGroupButton\n");
        if (buttonName)
            fprintf(stderr, "\t buttonName=[%s]\n", buttonName);
        else
            fprintf(stderr, "\t buttonName=NULL\n");

        if (parentMenuName)
            fprintf(stderr, "\t parentMenuName=[%s]\n", parentMenuName);
        else
            fprintf(stderr, "\t parentMenuName=NULL\n");

        fprintf(stderr, "\t state=[%d]\n", state);
        fprintf(stderr, "\t groupId=[%d]\n", groupId);
    }

    buttonSpecCell button;

    strcpy(button.name, buttonName);
    button.subMenu = NULL;
    button.actionType = BUTTON_SWITCH;
    button.callback = callback;
    button.calledClass = classPtr;
    button.state = state;
    button.dashed = false;
    button.group = groupId;
    if (userData)
    {
        button.userData = (void *)new char[strlen((char *)userData) + 1];
        strcpy((char *)button.userData, (char *)userData);
    }

    if (parentMenuName)
        VRPinboard::instance()->addButtonToNamedMenu(&button, (char *)parentMenuName);
    else
        VRPinboard::instance()->addButtonToMainMenu(&button);

    // as the constructor of buttonspeccell doesn't call the callback
    // if state =1 we do it here
    if (state)
        cover->setButtonState(button.name, (int)button.state);
}

// Add Pinboard Buttons

void
coVRPluginSupport::addToggleButton(const char *buttonName, const char *parentMenuName,
                                   int state, ButtonCallback callback, void *classPtr, void *userData)
{
    START("coVRPluginSupport::addToggleButton");
    if (debugLevel(3))
    {
        fprintf(stderr, "\ncoVRPluginSupport::addToggleButton\n");
        if (buttonName)
            fprintf(stderr, "\t buttonName=[%s]\n", buttonName);
        if (parentMenuName)
            fprintf(stderr, "\t parentMenuName=[%s]\n", parentMenuName);
        fprintf(stderr, "\t state=[%d]\n", state);
    }

    buttonSpecCell button;

    strcpy(button.name, buttonName);
    button.subMenu = NULL;
    button.actionType = BUTTON_SWITCH;
    button.callback = callback;
    button.calledClass = classPtr;
    button.state = state;
    button.dashed = false;
    button.group = -1;
    if (userData)
    {
        button.userData = (void *)new char[strlen((char *)userData) + 1];
        strcpy((char *)button.userData, (char *)userData);
    }
    if (parentMenuName)
        VRPinboard::instance()->addButtonToNamedMenu(&button, (char *)parentMenuName);
    else
        VRPinboard::instance()->addButtonToMainMenu(&button);

    // as the constructor of buttonspeccell doesn't call the callback
    // if state =1 we do it here
    if (state)
        cover->setButtonState(button.name, (int)button.state);
}

void
coVRPluginSupport::addSubmenuButton(const char *buttonName, const char *parentMenuName, const char *subMenuName,
                                    int state, ButtonCallback callback, int groupId, void *classPtr)
{
    START("coVRPluginSupport::addSubmenuButton");
    if (debugLevel(3))
    {
        fprintf(stderr, "\ncoVRPluginSupport::addSubmenuButton\n");
        fprintf(stderr, "\t buttonName=[%s]\n", buttonName);
        if (parentMenuName)
            fprintf(stderr, "\t parentMenuName=[%s]\n", parentMenuName);
        fprintf(stderr, "\t title=[%s]\n", subMenuName);
        fprintf(stderr, "\t state=[%d]\n", state);
        fprintf(stderr, "\t groupId=[%d]\n", groupId);
    }
    buttonSpecCell button;

    strcpy(button.name, buttonName);
    strcpy(button.subMenuName, subMenuName);
    button.actionType = BUTTON_SUBMENU;
    button.callback = callback;
    button.calledClass = classPtr;
    button.state = state;
    button.dashed = false;
    button.group = groupId;
    if (parentMenuName)
        VRPinboard::instance()->addButtonToNamedMenu(&button, (char *)parentMenuName);
    else
        VRPinboard::instance()->addButtonToMainMenu(&button);

    if (state)
        cover->setButtonState(button.name, (int)button.state);
}

void
coVRPluginSupport::addFunctionButton(const char *buttonName, const char *parentMenuName,
                                     ButtonCallback callback, void *classPtr, void *userData)
{
    START("coVRPluginSupport::addFunctionButton");
    if (debugLevel(3))
    {
        fprintf(stderr, "\ncoVRPluginSupport::addFunctionButton\n");
        fprintf(stderr, "\t buttonName=[%s]\n", buttonName);
        if (parentMenuName)
            fprintf(stderr, "\t parentMenuName=[%s]\n", parentMenuName);
    }
    buttonSpecCell button;

    strcpy(button.name, buttonName);
    button.subMenu = NULL;
    button.actionType = BUTTON_FUNCTION;
    button.callback = callback;
    button.calledClass = classPtr;
    button.dashed = false;
    if (userData)
    {
        button.userData = (void *)new char[strlen((char *)userData) + 1];
        strcpy((char *)button.userData, (char *)userData);
    }
    if (parentMenuName)
        VRPinboard::instance()->addButtonToNamedMenu(&button, parentMenuName);
    else
        VRPinboard::instance()->addButtonToMainMenu(&button);
}

void
coVRPluginSupport::addSliderButton(const char *buttonName, const char *parentMenuName,
                                   float min, float max, float value, ButtonCallback callback, void *classPtr, void *userData)
{
    START("coVRPluginSupport::addSliderButton");
    if (debugLevel(3))
    {
        fprintf(stderr, "\ncoVRPluginSupport::addSliderButton\n");
        fprintf(stderr, "\t buttonName=[%s]\n", buttonName);
        if (parentMenuName)
            fprintf(stderr, "\t parentMenuName=[%s]\n", parentMenuName);
        fprintf(stderr, "\t state=[%f]\n", min);
        fprintf(stderr, "\t state=[%f]\n", max);
        fprintf(stderr, "\t state=[%f]\n", value);
    }

    buttonSpecCell button;

    strcpy(button.name, buttonName);
    button.actionType = BUTTON_SLIDER;
    button.callback = callback;
    button.calledClass = classPtr;
    button.state = value;
    button.dashed = false;
    button.group = -1;
    button.sliderMin = min;
    button.sliderMax = max;
    if (userData)
    {
        button.userData = (void *)new char[strlen((char *)userData) + 1];
        strcpy((char *)button.userData, (char *)userData);
    }

    if (parentMenuName)
        VRPinboard::instance()->addButtonToNamedMenu(&button, (char *)parentMenuName);
    else
        VRPinboard::instance()->addButtonToMainMenu(&button);
}

// remove menu buttons

void
coVRPluginSupport::removeButton(const char *buttonName, const char *parentMenuName)
{
    START("coVRPluginSupport::removeButton");
    if (debugLevel(3))
    {
        fprintf(stderr, "\ncoVRPluginSupport::removeButton\n");
        fprintf(stderr, "\t buttonName=[%s]\n", buttonName);
        if (parentMenuName)
            fprintf(stderr, "\t parentMenuName=[%s]\n", parentMenuName);
    }
    if (parentMenuName)
        VRPinboard::instance()->removeButtonFromNamedMenu(buttonName, parentMenuName);
    else
        VRPinboard::instance()->removeButtonFromMainMenu(buttonName);
}

// set the state/value of menu buttons

bool
coVRPluginSupport::setButtonState(const char *buttonName, int state)
{
    START("coVRPluginSupport::setButtonState");
    if (debugLevel(3))
    {
        fprintf(stderr, "\ncoVRPluginSupport::setButtonState\n");
        fprintf(stderr, "\tbutton = [%s] state=[%d]\n", buttonName, state);
    }
    return VRPinboard::instance()->setButtonState((char *)buttonName, state);
}

bool
coVRPluginSupport::setSliderValue(const char *buttonName, float value)
{
    START("coVRPluginSupport::setSliderValue");
    if (debugLevel(3))
    {
        fprintf(stderr, "\ncoVRPluginSupport::setSliderValue\n");
        fprintf(stderr, "\t buttonName=[%s]\n", buttonName);
        fprintf(stderr, "\t value=[%f]\n", value);
    }
    return VRPinboard::instance()->setButtonState((char *)buttonName, value);
}

// get state/value of menu buttons

int
coVRPluginSupport::getButtonState(const char *buttonName, const char *menuName)
{
    //START("coVRPluginSupport::getButtonState");
    int s = (int)VRPinboard::instance()->namedMenu((char *)menuName)->namedButton((char *)buttonName)->spec.state;

    if (debugLevel(3))
    {
        fprintf(stderr, "\ncoVRPluginSupport::getButtonState\n");
        fprintf(stderr, "\t buttonName=[%s] menuName=[%s]\n", buttonName, menuName);
        fprintf(stderr, "\t state=[%d]\n", s);
    }
    return (s);
}

float
coVRPluginSupport::getSliderValue(const char *buttonName, const char *menuName)
{
    START("coVRPluginSupport::getSliderValue");
    float v = VRPinboard::instance()->namedMenu((char *)menuName)->namedButton((char *)buttonName)->spec.state;

    if (debugLevel(3))
    {
        fprintf(stderr, "\ncoVRPluginSupport::getSliderValue\n");
        fprintf(stderr, "\t buttonName=[%s] menuName=[%s]\n", buttonName, menuName);
        fprintf(stderr, "\t value=[%f]\n", v);
    }
    return (v);
}

// simulate a button press/release/slider drag

void
coVRPluginSupport::callButtonCallback(const char *buttonName)
{
    START("coVRPluginSupport::callButtonCallback");
    if (debugLevel(3))
    {
        fprintf(stderr, "\ncoVRPluginSupport::callButtonCallback\n");
    }

    VRPinboard::instance()->callButtonCallback((char *)buttonName);
}

// set the state/value of a COVER function

void
coVRPluginSupport::setBuiltInFunctionState(const char *functionName, int state)
{
    START("coVRPluginSupport::setBuiltInFunctionState");
    if (debugLevel(3))
    {
        fprintf(stderr, "\ncoVRPluginSupport::setBuiltInFunctionState\n");
        fprintf(stderr, "\t functionName=[%s]\n", functionName);
        fprintf(stderr, "\t state=[%d]\n", state);
    }

    // check if this function is in pinboard
    if (isBuiltInFunctionInPinboard(functionName)) // name is ok
    {
        int i = VRPinboard::instance()->getIndex(functionName);
        // check if the function is a function button
        if ((VRPinboard::instance()->functions[i].functionType == VRPinboard::BTYPE_TOGGLE)
            || (VRPinboard::instance()->functions[i].functionType == VRPinboard::BTYPE_NAVGROUP)
                   // type ok
            || (VRPinboard::instance()->functions[i].functionType == VRPinboard::BTYPE_SYNCGROUP))
        {
            char *buttonName = VRPinboard::instance()->functions[i].customButtonName;
            // val is ignored
            VRPinboard::instance()->setButtonState(buttonName, state);
        }
        else // function not configured
            fprintf(stderr, "ERROR: coVRPluginSupport::setFunctionValue - function %s is not a group or toggle button", functionName);
    }
    else
    {
        fprintf(stderr, "\ncoVRPluginSupport::setBuiltInFunctionState: %s not in pinboard\n", functionName);
    }
}

void
coVRPluginSupport::setBuiltInFunctionValue(const char *functionName, float val)
{
    START("coVRPluginSupport::setBuiltInFunctionValue");
    if (debugLevel(3))
    {
        fprintf(stderr, "\ncoVRPluginSupport::setBuiltInFunctionValue\n");
        fprintf(stderr, "\t functionName=[%s]\n", functionName);
        fprintf(stderr, "\t value=[%f]\n", val);
    }

    // check if this function is in pinboard
    if (isBuiltInFunctionInPinboard(functionName)) // name is ok
    {
        int i = VRPinboard::instance()->getIndex(functionName);
        // check if the function is a function button
        // type ok
        if (VRPinboard::instance()->functions[i].functionType == VRPinboard::BTYPE_SLIDER)
        {
            char *buttonName = VRPinboard::instance()->functions[i].customButtonName;
            // val is ignored
            VRPinboard::instance()->setButtonState((char *)buttonName, val);
        }
        else // function not configured
            fprintf(stderr, "ERROR: coVRPluginSupport::setFunctionValue - function %s is not a slider button", functionName);
    }
    else
    {
        fprintf(stderr, "\ncoVRPluginSupport::setBuiltInFunctionValue: %s not in pinboard\n", functionName);
    }
}

// get the state/value of a COVER function

int
coVRPluginSupport::getBuiltInFunctionState(const char *functionName, int *state)
{
    START("coVRPluginSupport::getBuiltInFunctionState");
    if (debugLevel(3))
    {
        fprintf(stderr, "\ncoVRPluginSupport::getFunctionState\n");
        fprintf(stderr, "\t functionName=[%s]\n", functionName);
    }

    if (isBuiltInFunctionInPinboard(functionName))
    {
        int i = VRPinboard::instance()->getIndex(functionName);
        if ((VRPinboard::instance()->functions[i].customMenuName) && (VRPinboard::instance()->functions[i].customButtonName))
            *state = (int)VRPinboard::instance()
                         ->namedMenu(VRPinboard::instance()->functions[i].customMenuName)
                         ->namedButton(VRPinboard::instance()->functions[i].customButtonName)
                         ->spec.state;
        else
            return (false);
        if (debugLevel(3))
            fprintf(stderr, "\t state=[%d]\n", *state);

        return (true);
    }
    else
        return (false);
}

int
coVRPluginSupport::getBuiltInFunctionValue(const char *functionName, float *val)
{
    START("coVRPluginSupport::getBuiltInFunctionValue");
    if (debugLevel(4))
    {
        fprintf(stderr, "\ncoVRPluginSupport::getFunctionValue\n");
        fprintf(stderr, "\t functionName=[%s]\n", functionName);
    }

    int i;

    if (isBuiltInFunctionInPinboard(functionName))
    {
        i = VRPinboard::instance()->getIndex(functionName);

        if ((VRPinboard::instance()->functions[i].customMenuName) && (VRPinboard::instance()->functions[i].customButtonName))
            *val = VRPinboard::instance()->namedMenu(VRPinboard::instance()->functions[i].customMenuName)->namedButton(VRPinboard::instance()->functions[i].customButtonName)->spec.state;
        else
            return (false);
        if (debugLevel(4))
            fprintf(stderr, "\t value=[%f]\n", *val);
        return (true);
    }
    else
        return (false);
}

void
coVRPluginSupport::callBuiltInFunctionCallback(const char *functionName)
{
    START("coVRPluginSupport::callBuiltInFunctionCallback");
    if (debugLevel(3))
        fprintf(stderr, "\ncoVRPluginSupport::callFunctionCallback\n");

    int i = VRPinboard::instance()->getIndex(functionName);
    if (i != -1)
    {
        char *buttonName = VRPinboard::instance()->functions[i].customButtonName;
        VRPinboard::instance()->callButtonCallback((char *)buttonName);
    }
}

// check if a COVER function is in the Pinboard

int
coVRPluginSupport::isBuiltInFunctionInPinboard(const char *functionName)
{
    START("coVRPluginSupport::isBuiltInFunctionInPinboard");
    if (debugLevel(4))
        fprintf(stderr, "\ncoVRPluginSupport::isFunction(%s)\n", functionName);

    // check if this function is a built in function

    int i = VRPinboard::instance()->getIndex(functionName);

    if (i != -1) // the function is a built in function
    {
        if (VRPinboard::instance()->functions[i].isInPinboard)
        {
            if (debugLevel(4))
                fprintf(stderr, "\t function is in pinboard\n");

            return (true);
        }
        else
        {
            if (debugLevel(4))
                fprintf(stderr, "\t function is NOT in pinboard\n");
            return false;
        }
    }

    if (debugLevel(4))
        fprintf(stderr, "\t function is NOT a built in function\n");
    return false;
}

coMenuItem *
coVRPluginSupport::getBuiltInFunctionMenuItem(const char *functionName)
{
    if (isBuiltInFunctionInPinboard(functionName))
    {
        // return index of builtin function
        //fprintf(stderr,"function=[%s] name=[%s]\n", functionName, VRPinboard::instance()->getCustomName(functionName));
        // get name correspondig to this index and get vrui menu item for this name
        return ((coMenuItem *)VRPinboard::instance()->getMenuItem(VRPinboard::instance()->getCustomName(functionName)));
    }
    else
    {
        return (NULL);
    }
}

osg::ClipNode *coVRPluginSupport::getObjectsRoot() const
{
    START("coVRPluginSupport::getObjectsRoot");
    return (VRSceneGraph::instance()->objectsRoot());
    return NULL;
}

osg::Group *coVRPluginSupport::getScene() const
{
    //START("coVRPluginSupport::getScene");
    return (VRSceneGraph::instance()->getScene());
}

bool
coVRPluginSupport::removeNode(osg::Node *node, bool isGroup)
{
    (void)isGroup;

    if (!node)
        return false;

    if (node->getNumParents() == 0)
        return false;

    while (node->getNumParents() > 0)
    {
        osg::Group *parent = node->getParent(0);
        osg::Node *child = node;

        while (coVRSelectionManager::instance()->isHelperNode(parent))
        {
            parent->removeChild(child);
            child = parent;
            parent = parent->getNumParents() > 0 ? parent->getParent(0) : NULL;
        }
        if (parent)
            parent->removeChild(child);
        else
            break;
    }
    return true;
}

osg::Group *
coVRPluginSupport::getMenuGroup() const
{
    return (VRSceneGraph::instance()->getMenuGroup());
}

osg::MatrixTransform *coVRPluginSupport::getPointer() const
{
    //START("coVRPluginSupport::getPointer");
    return (VRSceneGraph::instance()->getHandTransform());
}

osg::MatrixTransform *coVRPluginSupport::getObjectsXform() const
{
    START("coVRPluginSupport::getObjectsXform");
    return (VRSceneGraph::instance()->getTransform());
}

osg::MatrixTransform *coVRPluginSupport::getObjectsScale() const
{
    START("coVRPluginSupport::getObjectsScale");
    return (VRSceneGraph::instance()->getScaleTransform());
}

coPointerButton *coVRPluginSupport::getMouseButton() const
{
    if (mouseButton == NULL)
    {
        mouseButton = new coPointerButton("mouse");
    }
    return mouseButton;
}

const osg::Matrix &coVRPluginSupport::getViewerMat() const
{
    START("coVRPluginSupport::getViewerMat");
    return (VRViewer::instance()->getViewerMat());
}

const osg::Matrix &coVRPluginSupport::getXformMat() const
{
    START("coVRPluginSupport::getXformMat");
    static osg::Matrix transformMatrix;
    transformMatrix = VRSceneGraph::instance()->getTransform()->getMatrix();
    return transformMatrix;
}

const osg::Matrix &coVRPluginSupport::getMouseMat() const
{
    START("coVRPluginSupport::getMouseMat");
    return Input::instance()->mouse()->getMatrix();
}

const osg::Matrix &coVRPluginSupport::getPointerMat() const
{
    //START("coVRPluginSupport::getPointerMat");

    return Input::instance()->getHandMat();
}

void coVRPluginSupport::setXformMat(const osg::Matrix &transformMatrix)
{
    START("coVRPluginSupport::setXformMat");
    VRSceneGraph::instance()->getTransform()->setMatrix(transformMatrix);
    coVRCollaboration::instance()->SyncXform();
    coVRNavigationManager::instance()->wasJumping();
}

float coVRPluginSupport::getSceneSize() const
{
    START("coVRPluginSupport::getSceneSize");
    return coVRConfig::instance()->getSceneSize();
}

double coVRPluginSupport::currentTime() const
{
    START("coVRPluginSupport::currentTime");
    timeval currentTime;
    gettimeofday(&currentTime, NULL);
    return (currentTime.tv_sec + (double)currentTime.tv_usec / 1000000.0);
}

double coVRPluginSupport::frameTime() const
{
    START("coVRPluginSupport::frameTime");
    return frameStartTime;
}

double coVRPluginSupport::frameDuration() const
{
    return frameStartTime - lastFrameStartTime;
}

double coVRPluginSupport::frameRealTime() const
{
    return frameStartRealTime;
}

coMenu *coVRPluginSupport::getMenu() const
{
    START("coVRPluginSupport::getMenu");
    return VRPinboard::instance()->mainMenu->getCoMenu();
}

void coVRPluginSupport::addedNode(osg::Node *node, coVRPlugin *addingPlugin)
{
    START("coVRPluginSupport::addedNode");
    coVRPluginList::instance()->addNode(node, NULL, addingPlugin);
}

coPointerButton *coVRPluginSupport::getPointerButton() const
{
    //START("coVRPluginSupport::getPointerButton");
    if (pointerButton == NULL) // attach Button status as userdata to handTransform
    {
        if (coVRConfig::instance()->mouseTracking())
        {
            return getMouseButton();
        }
        pointerButton = new coPointerButton("pointer");
    }
    return pointerButton;
}

void coVRPluginSupport::setFrameTime(double ft)
{
    frameStartTime = ft;
}

void coVRPluginSupport::setFrameRealTime(double ft)
{
    frameStartRealTime = ft;
}

void coVRPluginSupport::updateTime()
{
#ifdef DOTIMING
    MARK0("COVER plugin support update time");
#endif

    lastFrameStartTime = frameStartTime;
    if (coVRMSController::instance()->isMaster())
    {
        frameStartRealTime = currentTime();
        if (coVRConfig::instance()->constantFrameRate)
        {
            frameStartTime += coVRConfig::instance()->constFrameTime;
        }
        else
        {
            frameStartTime = frameStartRealTime;
        }
    }
#ifdef DOTIMING
    MARK0("done");
#endif
}

void coVRPluginSupport::update()
{
    /// START("coVRPluginSupport::update");
    if (debugLevel(5))
        fprintf(stderr, "coVRPluginSupport::update\n");

    getMouseButton()->setWheel(Input::instance()->mouse()->wheel());
    getMouseButton()->setState(Input::instance()->mouse()->buttonState());
#if 0
   if (getMouseButton()->wasPressed() || getMouseButton()->wasReleased() || getMouseButton()->getState())
      std::cerr << "mouse pressed: " << getMouseButton()->wasPressed() << ", released: " << getMouseButton()->wasReleased() << ", state: " << getMouseButton()->getState() << std::endl;
#endif
    // don't overwrite mouse button state if only mouseTracking is used
    if (getPointerButton() != getMouseButton())
    {
        getPointerButton()->setWheel(0);
        getPointerButton()->setState(Input::instance()->getButtonState());
#if 0
   if (getPointerButton()->wasPressed() || getPointerButton()->wasReleased() || getPointerButton()->getState())
      std::cerr << "pointer pressed: " << getPointerButton()->wasPressed() << ", released: " << getPointerButton()->wasReleased() << ", state: " << getPointerButton()->getState() << std::endl;
#endif
    }

    size_t currentPerson = Input::instance()->getActivePerson();
    if ((getMouseButton()->wasPressed() & vruiButtons::PERSON_NEXT)
            || (getPointerButton()->wasPressed() & vruiButtons::PERSON_NEXT))
    {
        ++currentPerson;
        currentPerson %= Input::instance()->getNumPersons();
        Input::instance()->setActivePerson(currentPerson);
    }
    if ((getMouseButton()->wasPressed() & vruiButtons::PERSON_PREV)
            || (getPointerButton()->wasPressed() & vruiButtons::PERSON_PREV))
    {
        if (currentPerson == 0)
            currentPerson = Input::instance()->getNumPersons();
        --currentPerson;
        Input::instance()->setActivePerson(currentPerson);
    }

#ifdef DOTIMING
    MARK0("COVER update matrices and button status in plugin support class");
#endif

    osg::Matrix old = baseMatrix;
    baseMatrix = VRSceneGraph::instance()->getScaleTransform()->getMatrix();
    osg::Matrix transformMatrix = VRSceneGraph::instance()->getTransform()->getMatrix();
    baseMatrix.postMult(transformMatrix);

    if (old != baseMatrix && coVRMSController::instance()->isMaster())
    {
        coGRKeyWordMsg keyWordMsg("VIEW_CHANGED", false);
        Message grmsg;
        grmsg.type = Message::UI;
        grmsg.data = (char *)(keyWordMsg.c_str());
        grmsg.length = strlen(grmsg.data) + 1;
        sendVrbMessage(&grmsg);
    }

#ifdef DOTIMING
    MARK0("done");
#endif

    invCalculated = 0;
    sensors.update();
    updateManager->update();
    coTabletUI::instance()->update();
    //get rotational part of Xform only
    osg::Matrix frontRot(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);
    if (VRViewer::instance()->isMatrixOverwriteOn())
    {
        envCorrectMat.makeIdentity();
    }
    else if (coVRConfig::instance()->m_envMapMode == coVRConfig::FIXED_TO_VIEWER)
    {
        osg::Quat rot;
        rot.set(getXformMat());
        osg::Matrix rotMat;
        rotMat.makeRotate(rot);
        envCorrectMat = rotMat;
    }
    else if (coVRConfig::instance()->m_envMapMode == coVRConfig::FIXED_TO_OBJROOT)
    {
        envCorrectMat = getXformMat();
    }
    else if (coVRConfig::instance()->m_envMapMode == coVRConfig::FIXED_TO_VIEWER_FRONT)
    {
        osg::Quat rot;
        rot.set(getXformMat());
        osg::Matrix rotMat;
        rotMat.makeRotate(rot);
        envCorrectMat = frontRot * rotMat;
    }
    else if (coVRConfig::instance()->m_envMapMode == coVRConfig::FIXED_TO_OBJROOT_FRONT)
    {
        envCorrectMat = frontRot * getXformMat();
    }
    invEnvCorrectMat.invert(envCorrectMat);
    //draw.baseMat = getBaseMat();
    //draw.invBaseMat = getInvBaseMat();

    //START("coVRPluginSupport::getInteractorScale");
    if (coVRMSController::instance()->isMaster())
    {
        // check front screen
        const screenStruct screen = coVRConfig::instance()->screens[0];

        frontScreenCenter = screen.xyz;
        frontHorizontalSize = screen.hsize;
        frontVerticalSize = screen.vsize;

        const windowStruct window = coVRConfig::instance()->windows[0];
        frontWindowHorizontalSize = window.sx;
        frontWindowVerticalSize = window.sy;

        // get interactor position
        const osg::Vec3d eyePos = cover->getViewerMat().getTrans();

        // save the eye to screen vector
        eyeToScreen = frontScreenCenter - eyePos;

        // calculate distance between interactor and screen
        viewerDist = eyeToScreen.length();

        // use horizontal screen size as normalization factor
        scaleFactor = viewerDist / frontHorizontalSize;

        coVRMSController::instance()->sendSlaves((char *)&scaleFactor, sizeof(scaleFactor));
        coVRMSController::instance()->sendSlaves((char *)&viewerDist, sizeof(viewerDist));
        coVRMSController::instance()->sendSlaves((char *)&eyeToScreen, sizeof(eyeToScreen));

        coVRMSController::instance()->sendSlaves((char *)&frontScreenCenter, sizeof(frontScreenCenter));
        coVRMSController::instance()->sendSlaves((char *)&frontHorizontalSize, sizeof(frontHorizontalSize));
        coVRMSController::instance()->sendSlaves((char *)&frontVerticalSize, sizeof(frontVerticalSize));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&scaleFactor, sizeof(scaleFactor));
        coVRMSController::instance()->readMaster((char *)&viewerDist, sizeof(viewerDist));
        coVRMSController::instance()->readMaster((char *)&eyeToScreen, sizeof(eyeToScreen));

        coVRMSController::instance()->readMaster((char *)&frontScreenCenter, sizeof(frontScreenCenter));
        coVRMSController::instance()->readMaster((char *)&frontHorizontalSize, sizeof(frontHorizontalSize));
        coVRMSController::instance()->readMaster((char *)&frontVerticalSize, sizeof(frontVerticalSize));
    }
}

coVRPlugin *coVRPluginSupport::addPlugin(const char *name)
{
    START("coVRPluginSupport::addPlugin");
    return coVRPluginList::instance()->addPlugin(name);
}

coVRPlugin *coVRPluginSupport::getPlugin(const char *name)
{
    START("coVRPluginSupport::getPlugin");
    return coVRPluginList::instance()->getPlugin(name);
}

int coVRPluginSupport::removePlugin(const char *name)
{
    START("coVRPluginSupport::removePlugin");
    coVRPlugin *m = coVRPluginList::instance()->getPlugin(name);
    if (m)
    {
        coVRPluginList::instance()->unload(m);
    }
    return 0;
}

const osg::Matrix &coVRPluginSupport::getInvBaseMat() const
{
    START("coVRPluginSupport::getInvBaseMat");
    if (!invCalculated)
    {
        invBaseMatrix.invert(baseMatrix);
        //fprintf(stderr,"coVRPluginSupport::getInvBaseMat baseMatrix is singular\n");
        invCalculated = 1;
    }
    return invBaseMatrix;
}

void coVRPluginSupport::removePlugin(coVRPlugin *m)
{
    START("coVRPluginSupport::unload");
    coVRPluginList::instance()->unload(m);
}

int coVRPluginSupport::createUniqueButtonGroupId()
{
    START("coVRPluginSupport::uniqueButtonGroup");
    buttonGroup++;
    return buttonGroup;
}

int coVRPluginSupport::isPointerLocked()
{
    //START("coVRPluginSupport::isPointerLocked");
    bool isLocked;
    isLocked = coInteractionManager::the()->isOneActive(coInteraction::ButtonA);
    if (isLocked)
        return true;
    isLocked = coInteractionManager::the()->isOneActive(coInteraction::ButtonB);
    if (isLocked)
        return true;
    isLocked = coInteractionManager::the()->isOneActive(coInteraction::ButtonC);
    if (isLocked)
        return true;
    return (false);
}

bool coVRPluginSupport::isNavigating()
{
    START("coVRPluginSupport::isNavigating");
    return (coInteractionManager::the()->isOneActive(coInteraction::ButtonA));
}

float coVRPluginSupport::getSqrDistance(osg::Node *n, osg::Vec3 &p,
                                        osg::MatrixTransform **path = NULL, int pathLength = 0) const
{
    START("coVRPluginSupport::getSqrDistance");
    osg::Matrix mat;
    mat.makeIdentity();
    osg::BoundingSphere bsphere = n->getBound();
    if (pathLength > 0)
    {
        for (int i = 0; i < pathLength; i++)
        {
            osg::MatrixTransform *mt = path[i]->asMatrixTransform();
            if (mt)
            {
                mat.postMult(mt->getMatrix());
            }
        }
    }
    else
    {
        osg::Node *node = n;
        osg::Transform *trans = NULL;
        while (node)
        {
            if ((trans = node->asTransform()))
                break;
            node = node->getParent(0);
        }
        if (trans)
        {
            trans->asTransform()->computeLocalToWorldMatrix(mat, NULL);
        }
    }
    osg::Vec3 svec(mat(0, 0), mat(0, 1), mat(0, 2));
    float Scale = svec * svec; // Scale ^2
    osg::Vec3 v;
    if (n->asTransform())
        v = mat.getTrans();
    else
        v = mat.preMult(bsphere.center());
    osg::Vec3 dist = v - p;
    return (dist.length2() / Scale);
}

void coVRPluginSupport::setScale(float s)
{
    START("coVRPluginSupport::setScale");
    VRSceneGraph::instance()->setScaleFactor(s);
    coVRNavigationManager::instance()->wasJumping();
}

float coVRPluginSupport::getScale() const
{
    START("coVRPluginSupport::getScale");
    return VRSceneGraph::instance()->scaleFactor();
}

float coVRPluginSupport::getInteractorScale(osg::Vec3 &pos) // pos in World coordinates
{
    const osg::Vec3 eyePos = cover->getViewerMat().getTrans();
    const osg::Vec3 eyeToPos = pos - eyePos;
    float eyeToPosDist = eyeToPos.length();
    float scaleVal;
    if (eyeToPosDist < 2000)
    {
        scaleVal = 1.0;
    }
    else
    {
        //float oeffnungswinkel = frontHorizontalSize/(frontScreenCenter[1]-eyePos[1]);
        //scaleVal = ((eyeToPosDist) * oeffnungswinkel/(oeffnungswinkel*2000));

        scaleVal = eyeToPosDist / (2000);
    }

    return scaleVal / cover->getScale();
}

float coVRPluginSupport::getViewerScreenDistance()
{
    return viewerDist;
}

osg::BoundingBox coVRPluginSupport::getBBox(osg::Node *node)
{
    osg::ComputeBoundsVisitor cbv;
    cbv.setTraversalMask(Isect::Visible);
    node->accept(cbv);
    const osg::NodePath path = cbv.getNodePath();
    //fprintf(stderr, "!!!! nodepath %d\n", (int)path.size());
    //for (osg::NodePath::const_iterator it = path.begin(); it != path.end(); it++)
    //fprintf(stderr, "    node - %s(%s)\n", (*it)->getName(), (*it)->className());
    return cbv.getBoundingBox();
}

coVRPluginSupport::coVRPluginSupport()
    : scaleFactor(0.0)
    , viewerDist(0.0)
    , updateManager(0)
    , activeClippingPlane(0)
{
    START("coVRPluginSupport::coVRPluginSupport");
    /// path for the viewpoint file: initialized by 1st param() call
    intersectedNode = NULL;

    toolBar = NULL;
    numClipPlanes = coCoviseConfig::getInt("COVER.NumClipPlanes", 3);
    for (unsigned int i = 0; i < numClipPlanes; i++)
    {
        clipPlanes[i] = new osg::ClipPlane();
        clipPlanes[i]->setClipPlaneNum(i);
    }
    NoFrameBuffer = new osg::ColorMask(false, false, false, false);
    cover = this;
    player = NULL;

    pointerButton = NULL;
    mouseButton = NULL;
    buttonGroup = 40;
    baseMatrix.makeIdentity();

    invCalculated = false;
    frameStartTime = 0.0;
    frameStartRealTime = 0.0;

    // Init Joystick data
    numJoysticks = 0;
    for (int i = 0; i < MAX_NUMBER_JOYSTICKS; i++)
    {
        number_buttons[i] = 0;
        number_sliders[i] = 0;
        number_axes[i] = 0;
        number_POVs[i] = 0;
        buttons[i] = NULL;
        sliders[i] = NULL;
        axes[i] = NULL;
        POVs[i] = NULL;
    }

    currentCursor = osgViewer::GraphicsWindow::LeftArrowCursor;
    setCurrentCursor(currentCursor);

    resOn = coVRConfig::instance()->restrictOn();

    frontWindowHorizontalSize = 0;

    if (debugLevel(2))
        fprintf(stderr, "\nnew coVRPluginSupport\n");
}

coVRPluginSupport::~coVRPluginSupport()
{
    START("coVRPluginSupport::~coVRPluginSupport");
    if (debugLevel(2))
        fprintf(stderr, "delete coVRPluginSupport\n");
}

int coVRPluginSupport::getNumClipPlanes()
{
    return numClipPlanes;
}

void coVRPluginSupport::releaseKeyboard(coVRPlugin *plugin)
{
    if (coVRPluginList::instance()->keyboardGrabber() == plugin)
    {
        coVRPluginList::instance()->grabKeyboard(NULL);
    }
}

bool coVRPluginSupport::isKeyboardGrabbed()
{
    return (coVRPluginList::instance()->keyboardGrabber() != NULL);
}

bool coVRPluginSupport::grabKeyboard(coVRPlugin *plugin)
{
    if (coVRPluginList::instance()->keyboardGrabber() != NULL
        && coVRPluginList::instance()->keyboardGrabber() != plugin)
    {
        return false;
    }
    coVRPluginList::instance()->grabKeyboard(plugin);
    return true;
}

// get the active cursor number
osgViewer::GraphicsWindow::MouseCursor coVRPluginSupport::getCurrentCursor() const
{
    return currentCursor;
}

// set the active cursor number
void coVRPluginSupport::setCurrentCursor(osgViewer::GraphicsWindow::MouseCursor cursor)
{
    if (currentCursor == cursor)
        return;
    currentCursor = cursor;
    for (int i = 0; i < coVRConfig::instance()->numWindows(); i++)
    {
        if (coVRConfig::instance()->windows[i].window)
            coVRConfig::instance()->windows[i].window->setCursor(currentCursor);
    }
}

// make the cursor visible or invisible
void coVRPluginSupport::setCursorVisible(bool visible)
{
    static bool isVisible = true;
    if (visible && (!isVisible))
    {
        for (int i = 0; i < coVRConfig::instance()->numWindows(); i++)
        {
            if (coVRConfig::instance()->windows[i].window)
            {
                coVRConfig::instance()->windows[i].window->useCursor(true);
                coVRConfig::instance()->windows[i].window->setCursor(currentCursor);
            }
        }
    }
    if (!visible && (isVisible))
    {
        for (int i = 0; i < coVRConfig::instance()->numWindows(); i++)
        {
            if (coVRConfig::instance()->windows[i].window)
                coVRConfig::instance()->windows[i].window->useCursor(false);
        }
    }
    isVisible = visible;
}

//-----
void coVRPluginSupport::sendMessage(coVRPlugin *sender, int toWhom, int type, int len, const void *buf)
{
    START("coVRPluginSupport::sendMessage");
    Message *message;
    message = new Message();

    int size = len + 2 * sizeof(int);

    if (toWhom == coVRPluginSupport::TO_SAME)
        sender->message(type, len, buf);
    if (toWhom == coVRPluginSupport::TO_ALL)
        coVRPluginList::instance()->message(type, len, buf);

    if ((toWhom == coVRPluginSupport::TO_SAME) || (toWhom == coVRPluginSupport::TO_SAME_OTHERS))
    {
        size += strlen(sender->getName()) + 1;
        size += 8 - ((strlen(sender->getName()) + 1) % 8);
    }
    message->data = new char[size];
    memcpy(&message->data[size - len], buf, len);
    if ((toWhom == coVRPluginSupport::TO_SAME) || (toWhom == coVRPluginSupport::TO_SAME_OTHERS))
    {
        strcpy((char *)(message->data + 2 * sizeof(int)), sender->getName());
    }
#ifdef BYTESWAP
    int tmp = toWhom;
    byteSwap(tmp);
    ((int *)message->data)[0] = tmp;
    tmp = type;
    byteSwap(tmp);
    ((int *)message->data)[1] = tmp;
#else
    ((int *)message->data)[0] = toWhom;
    ((int *)message->data)[1] = type;
#endif

    message->type = COVISE_MESSAGE_RENDER_MODULE;
    message->length = size;

    if (!coVRMSController::instance()->isSlave())
    {
        cover->sendVrbMessage(message);
    }
    delete[] message -> data;
    message->data = NULL;
    delete message;
}

void coVRPluginSupport::sendMessage(coVRPlugin * /*sender*/, const char *destination, int type, int len, const void *buf, bool localonly)
{
    START("coVRPluginSupport::sendMessage");

    //fprintf(stderr,"coVRPluginSupport::sendMessage dest=%s\n",destination);

    int size = len + 2 * sizeof(int);
    coVRPlugin *dest = coVRPluginList::instance()->getPlugin(destination);
    if (dest)
    {
        dest->message(type, len, buf);
    }
    else if (strcmp(destination, "AKToolbar") != 0)
    {
        cerr << "did not find Plugin " << destination << " in coVRPluginSupport::sendMessage" << endl;
    }

    if (!localonly)
    {
        Message *message;
        message = new Message();

        int namelen = strlen(destination) + 1;
        namelen += 8 - ((strlen(destination) + 1) % 8);
        size += namelen;
        message->data = new char[size];
        memcpy(&message->data[size - len], buf, len);
        memset(message->data + 2 * sizeof(int), '\0', namelen);
        strcpy(message->data + 2 * sizeof(int), destination);

#ifdef BYTESWAP
        int tmp = coVRPluginSupport::TO_SAME;
        byteSwap(tmp);
        ((int *)message->data)[0] = tmp;
        tmp = type;
        byteSwap(tmp);
        ((int *)message->data)[1] = tmp;
#else
        ((int *)message->data)[0] = coVRPluginSupport::TO_SAME;
        ((int *)message->data)[1] = type;
#endif

        message->type = COVISE_MESSAGE_RENDER_MODULE;
        message->length = size;

        if (!coVRMSController::instance()->isSlave())
        {
            cover->sendVrbMessage(message);
        }
        delete[] message -> data;
        delete message;
    }
}

int coVRPluginSupport::sendBinMessage(const char *keyword, const char *data, int len)
{
    START("coVRPluginSupport::sendBinMessage");
    if (!coVRMSController::instance()->isSlave())
    {
        int size = strlen(keyword) + 2;
        size += len;

        Message message;
        message.data = new char[size];
        message.data[0] = 0;
        strcpy(&message.data[1], keyword);
        memcpy(&message.data[strlen(keyword) + 2], data, len);
        message.type = Message::RENDER;
        message.length = size;

        bool ret = sendVrbMessage(&message);

        delete[] message.data;
        message.data = NULL;

        return ret ? 1 : 0;
    }

    return 1;
}

void
coVRPluginSupport::enableNavigation(const char *str)
{
    START("coVRPluginSupport::enableNavigation");
    setBuiltInFunctionState(str, 1);
}

void
coVRPluginSupport::disableNavigation(const char *str)
{
    START("coVRPluginSupport::disableNavigation");
    setBuiltInFunctionState(str, 0);
}

void
coVRPluginSupport::setNavigationValue(const char *str, float value)
{
    START("coVRPluginSupport::setNavigationValue");
    buttonSpecCell buttonSpecifier;

    strcpy(buttonSpecifier.name, str);
    buttonSpecifier.state = value;
    VRSceneGraph::instance()->manipulate(&buttonSpecifier);
    if (strcmp(str, "DriveSpeed") == 0)
    {
        setBuiltInFunctionValue(str, value);
        callBuiltInFunctionCallback(str);
    }
    else
    {
        VRPinboard::instance()->setButtonState(str, value);
    }
}

osg::Node *coVRPluginSupport::getIntersectedNode() const
{
    return intersectedNode.get();
}

const osg::NodePath &coVRPluginSupport::getIntersectedNodePath() const
{
    return intersectedNodePath;
}

const osg::Vec3 &coVRPluginSupport::getIntersectionHitPointWorld() const
{
    return intersectionHitPointWorld;
}

const osg::Vec3 &coVRPluginSupport::getIntersectionHitPointWorldNormal() const
{
    return intersectionHitPointWorldNormal;
}

coPointerButton::coPointerButton(const std::string &name)
    : m_name(name)
{
    START("coPointerButton::coPointerButton");
    buttonStatus = 0;
    lastStatus = 0;
    wheelCount = 0;
}

coPointerButton::~coPointerButton()
{
    START("coPointerButton::~coPointerButton");
}

const std::string &coPointerButton::name() const
{

    return m_name;
}

unsigned int coPointerButton::getState()
{
    //START("coPointerButton::getButtonStatus")
    return buttonStatus;
}

unsigned int coPointerButton::oldState()
{
    START("coPointerButton::oldButtonStatus");
    return lastStatus;
}

bool coPointerButton::notPressed()
{
    START("coPointerButton::notPressed");
    return (buttonStatus == 0);
}

unsigned int coPointerButton::wasPressed()
{
    return (getState() ^ oldState()) & getState();
}

unsigned int coPointerButton::wasReleased()
{
    return (getState() ^ oldState()) & oldState();
}

void coPointerButton::setState(unsigned int newButton) // called from
{
    //START("coPointerButton::setButtonStatus");
    lastStatus = buttonStatus;
    buttonStatus = newButton;
}

int coPointerButton::getWheel()
{
    return wheelCount;
}

void coPointerButton::setWheel(int count)
{
    wheelCount = count;
}

int coVRPluginSupport::registerPlayer(vrml::Player *player)
{
    if (this->player)
        return -1;

    this->player = player;
    return 0;
}

int coVRPluginSupport::unregisterPlayer(vrml::Player *player)
{
    if (this->player != player)
        return -1;

    for (list<void (*)()>::const_iterator it = playerUseList.begin();
         it != playerUseList.end();
         it++)
    {
        if (*it)
            (*it)();
    }
    player = NULL;

    return 0;
}

vrml::Player *coVRPluginSupport::usePlayer(void (*playerUnavailableCB)())
{
    list<void (*)()>::const_iterator it = find(playerUseList.begin(),
                                               playerUseList.end(), playerUnavailableCB);
    if (it != playerUseList.end())
        return NULL;

    playerUseList.push_back(playerUnavailableCB);
    return this->player;
}

int coVRPluginSupport::unusePlayer(void (*playerUnavailableCB)())
{
    list<void (*)()>::const_iterator it = find(playerUseList.begin(),
                                               playerUseList.end(), playerUnavailableCB);
    if (it == playerUseList.end())
        return -1;

    playerUseList.remove(playerUnavailableCB);
    return 0;
}

coUpdateManager *coVRPluginSupport::getUpdateManager() const
{
    if (updateManager == 0)
        updateManager = new coUpdateManager();
    return updateManager;
}

bool coVRPluginSupport::isClippingOn() const
{
    return getObjectsRoot()->getNumClipPlanes() > 0;
}

int coVRPluginSupport::getActiveClippingPlane() const
{
    return activeClippingPlane;
}

void coVRPluginSupport::setActiveClippingPlane(int plane)
{
    activeClippingPlane = plane;
}

class IsectVisitor : public osg::NodeVisitor
{
public:
    IsectVisitor(bool isect)
        : osg::NodeVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN)
    {
        isect_ = isect;
    }
    virtual void apply(osg::Node &node)
    {
        if (isect_)
            node.setNodeMask(node.getNodeMask() | (Isect::Intersection));
        else
            node.setNodeMask(node.getNodeMask() & (~Isect::Intersection));
        traverse(node);
    }

private:
    bool isect_;
};
void coVRPluginSupport::setNodesIsectable(osg::Node *n, bool isect)
{
    IsectVisitor iv(isect);
    if (n)
    {
        n->accept(iv);
    }
}
/* see http://www.nps.navy.mil/cs/sullivan/osgtutorials/osgGetWorldCoords.htm */

// Visitor to return the world coordinates of a node.
// It traverses from the starting node to the parent.
// The first time it reaches a root node, it stores the world coordinates of
// the node it started from.  The world coordinates are found by concatenating all
// the matrix transforms found on the path from the start node to the root node.

class GetWorldCoordOfNodeVisitor : public osg::NodeVisitor
{
public:
    GetWorldCoordOfNodeVisitor()
        : osg::NodeVisitor(NodeVisitor::TRAVERSE_PARENTS)
        , done(false)
    {
        wcMatrix = new osg::Matrix();
    }
    virtual void apply(osg::Node &node)
    {
        if (!done)
        {
            if (0 == node.getNumParents()
                         // no parents
                || &node == cover->getObjectsRoot())
            {
                wcMatrix->set(osg::computeLocalToWorld(this->getNodePath()));
                done = true;
            }
            else
            {
                traverse(node);
            }
        }
    }
    osg::Matrix *giveUpDaMat()
    {
        return wcMatrix;
    }

private:
    bool done;
    osg::Matrix *wcMatrix;
};

// Given a valid node placed in a scene under a transform, return the
// world coordinates in an osg::Matrix.
// Creates a visitor that will update a matrix representing world coordinates
// of the node, return this matrix.
// (This could be a class member for something derived from node also.

osg::Matrix *coVRPluginSupport::getWorldCoords(osg::Node *node) const
{
    GetWorldCoordOfNodeVisitor ncv;
    if (node)
    {
        node->accept(ncv);
        return ncv.giveUpDaMat();
    }
    else
    {
        return NULL;
    }
}

bool coVRPluginSupport::isVRBconnected()
{
    return vrbc->isConnected();
}

void coVRPluginSupport::protectScenegraph()
{
    VRSceneGraph::instance()->protectScenegraph();
}

bool coVRPluginSupport::sendVrbMessage(const covise::Message *msg) const
{
    if (coVRPluginList::instance()->sendVisMessage(msg))
    {
        return true;
    }
    else if (vrbc)
    {
        vrbc->sendMessage(msg);
        return true;
    }

    return false;
}

covise::TokenBuffer &opencover::operator<<(covise::TokenBuffer &buffer, const osg::Matrixd &matrix)
{
    for (int ctr = 0; ctr < 16; ++ctr)
    {
        buffer << matrix.ptr()[ctr];
    }
    return buffer;
}

covise::TokenBuffer &opencover::operator>>(covise::TokenBuffer &buffer, osg::Matrixd &matrix)
{
    double array[16];
    for (int ctr = 0; ctr < 16; ++ctr)
    {
        buffer >> array[ctr];
    }
    matrix.set(array);
    return buffer;
}

covise::TokenBuffer &opencover::operator<<(covise::TokenBuffer &buffer, const osg::Vec3f &vec)
{
    for (int ctr = 0; ctr < 3; ++ctr)
    {
        buffer << vec[ctr];
    }
    return buffer;
}

covise::TokenBuffer &opencover::operator>>(covise::TokenBuffer &buffer, osg::Vec3f &vec)
{
    for (int ctr = 0; ctr < 3; ++ctr)
    {
        buffer >> vec[ctr];
    }
    return buffer;
}
