/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//**************************************************************************
//
// * Description    : Inventor examiner viewer (RE version)
//
// * Class(es)      : InvExaminerViewer
//
// * inherited from : InvFullViewer InvViewer
//
// * Author  : Dirk Rantzau
//
// * History : 29.03.94 V 1.0
//
//**************************************************************************

#include <covise/covise.h>
#include <config/CoviseConfig.h>
#include <X11/Intrinsic.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>

#include <Xm/LabelG.h>
#include <Xm/Text.h>
#include <Xm/ToggleB.h>
#include <Xm/ToggleBG.h>
#include <Xm/Form.h>

#include <stdlib.h>

#ifndef __sgi
#include "ThumbWheel.h"
#else
#include <Sgm/ThumbWheel.h>
#endif

#include <Inventor/SoDB.h>
#include <Inventor/SoInput.h>
#include <Inventor/sensors/SoFieldSensor.h>
#include <Inventor/fields/SoSFTime.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/projectors/SbSphereSheetProjector.h>
#include <Inventor/actions/SoSearchAction.h>
#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/SoXtCursors.h>
#include <Inventor/Xt/SoXtResource.h>
#include <Inventor/Xt/SoXtIcons.h>
#include "InvExaminerViewer.h"
#include "InvPixmapButton.h"
#include <GL/gl.h>

#include "BAWCursors.h"

#ifdef DEBUG
#include <Inventor/errors/SoDebugError.h>
#include <SoDebug.h>
#endif

#include "InvAnnotationManager.h"

extern int smoothNormalsEnabled;
/*
 *  Defines
 */

enum ViewerModes
{
    PICK_MODE,
    VIEW_MODE,
    SPIN_MODE_ACTIVE,
    PAN_MODE,
    PAN_MODE_ACTIVE,
    DOLLY_MODE_ACTIVE,
    SEEK_MODE
};

// list of custom push buttons
enum
{
    CAM_PUSH = 0,
    PUSH_NUM
};

// size of the rotation buffer, which is used to animate the spinning ball.
#define ROT_BUFF_SIZE 3

//
// The point of interest geometry description
//
const char *InvExaminerViewer::geometryBuffer = "\
#Inventor V2.0 ascii\n\
Separator { \
    PickStyle { style UNPICKABLE } \
    LightModel { model BASE_COLOR } \
    MaterialBinding { value PER_PART } \
    DrawStyle { lineWidth 2 } \
    Coordinate3 { point [0 0 0, 1 0 0, 0 1 0, 0 0 1] } \
    BaseColor { rgb [1 0 0, 0 1 0, 0 0 1] } \
    IndexedLineSet { coordIndex [1, 0, 2, -1, 0, 3] } \
     \
    LightModel { model PHONG } \
    MaterialBinding { value OVERALL } \
    Complexity { value .1 } \
    Separator { \
    	Material { \
	    diffuseColor    [ 0.5 0 0 ] \
	    emissiveColor   [ 0.5 0 0 ] \
	} \
	Translation { translation 1 0 0 } \
    	RotationXYZ { axis Z angle -1.570796327 } \
    	Cone { bottomRadius .2 height .3 } \
    } \
    Separator { \
    	Material { \
	    diffuseColor    [ 0 0.5 0 ] \
	    emissiveColor   [ 0 0.5 0 ] \
	} \
	Translation { translation 0 1 0 } \
    	Cone { bottomRadius .2 height .3 } \
    } \
    Material { \
	diffuseColor    [ 0 0 0.5 ] \
	emissiveColor   [ 0 0 0.5 ] \
    } \
    Translation { translation 0 0 1 } \
    RotationXYZ { axis X angle 1.570796327 } \
    Cone { bottomRadius .2 height .3 } \
} ";

/* This is the structure returned from the SERVER_OVERLAY_VISUaAL property. */
typedef struct
{
    VisualID visualID; /* The VisualID of the overlay visual */
    long transparentType; /* Can be None, TransparentPixel or TransparentMask */
    long value; /* Pixel value */
    long layer; /* Overlay planes will always be in layer 1 */
} OverlayVisualPropertyRec;

/* This is one of the possible values of the "transparentType" above. */
#ifndef TransparentPixel
#define TransparentPixel 1
#endif

static const char *thisClassName = "InvExaminerViewer";

////////////////////////////////////////////////////////////////////////
//
// Public constructor - build the widget right now
//
InvExaminerViewer::InvExaminerViewer(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    InvFullViewer::BuildFlag b,
    InvViewer::Type t)
    : InvFullViewer(
          parent,
          name,
          buildInsideParent,
          b,
          t,
          FALSE)
    , // tell GLWidget not to build just yet
    presentationCursor_(false)
//
////////////////////////////////////////////////////////////////////////
{
    // In this case, render area is what the app wants, so buildNow = TRUE
    constructorCommon(TRUE);
}

////////////////////////////////////////////////////////////////////////
//
// SoEXTENDER constructor - the subclass tells us whether to build or not
//
InvExaminerViewer::InvExaminerViewer(Widget parent,
                                     const char *name,
                                     SbBool buildInsideParent,
                                     InvFullViewer::BuildFlag b,
                                     InvViewer::Type t,
                                     SbBool buildNow)
    : InvFullViewer(parent,
                    name,
                    buildInsideParent,
                    b,
                    t,
                    FALSE)
    , // tell GLWidget not to build just yet
    presentationCursor_(false)

//
////////////////////////////////////////////////////////////////////////
{
    // In this case, render area may be what the app wants,
    // or it may want a subclass of render area. Pass along buildNow
    // as it was passed to us.
    constructorCommon(buildNow);
}

////////////////////////////////////////////////////////////////////////
//
// Called by the constructors
//
// private
//
void
InvExaminerViewer::constructorCommon(SbBool buildNow)
//
////////////////////////////////////////////////////////////////////////
{
    // init local vars
    addVisibilityChangeCallback(visibilityChangeCB, this);
    mode = isViewing() ? VIEW_MODE : PICK_MODE;
    createdCursors = FALSE;
    spinCursor = panCursor = dollyCursor = seekCursor = 0;
    firstBuild = TRUE;
    setSize(SbVec2s(500, 420)); // default size
    setClassName(thisClassName);

    // axis of rotation feedback vars
    feedbackFlag = FALSE;
    feedbackRoot = NULL;
    feedbackSwitch = NULL;
    feedbackSize = 20.0;
    feedbackSizeWheel = NULL;

    // init animation variables
    animationEnabled = TRUE;
    animatingFlag = FALSE;
    rotBuffer = new SbRotation[ROT_BUFF_SIZE];
    lastMotionTime = 0;
    animationSensor = new SoFieldSensor(InvExaminerViewer::animationSensorCB, this);
#ifdef DEBUG
    if (SoDebug::GetEnv("IV_DEBUG_SENSORS"))
    {
        SoDebug::NamePtr("examinerSpinSensor", animationSensor);
    }
#endif

    // init the projector class
    SbViewVolume vv;
    vv.ortho(-1, 1, -1, 1, -10, 10);
    sphereSheet = new SbSphereSheetProjector;
    sphereSheet->setViewVolume(vv);
    sphereSheet->setSphere(SbSphere(SbVec3f(0, 0, 0), .7));

    // assign decoration names
    setPopupMenuString("COVISE Renderer");
    setBottomWheelString("Roty");
    setLeftWheelString("Rotx");
    setPrefSheetString("COVISE Renderer Preference Sheet");
    smoothNormalsEnabled = coCoviseConfig::isOn("Renderer.GenerateSmoothNormals", false);

    for (int i = 0; i < PUSH_NUM; i++)
        buttonList[i] = NULL;

    // Build the widget tree, and let SoXtComponent know about our base widget.
    if (buildNow)
    {
        Widget w = buildWidget(getParentWidget());
        setBaseWidget(w);
    }

    // set overlay background index to transparent
    setupOverlays();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Destructor.
//
// Use: public

InvExaminerViewer::~InvExaminerViewer()
//
////////////////////////////////////////////////////////////////////////
{
    delete animationSensor;

    for (int i = 0; i < PUSH_NUM; i++)
        delete buttonList[i];

    delete sphereSheet;
    if (feedbackRoot)
        feedbackRoot->unref();

    // free the viewer cursors
    if (getDisplay())
    {
        Display *display = getDisplay();
        if (spinCursor)
            XFreeCursor(display, spinCursor);
        if (panCursor)
            XFreeCursor(display, panCursor);
        if (dollyCursor)
            XFreeCursor(display, dollyCursor);
        if (seekCursor)
            XFreeCursor(display, seekCursor);
    }

    delete[] rotBuffer;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Shows/hide the point of rotation feedback geometry
//
// Use: public
void
InvExaminerViewer::setFeedbackVisibility(SbBool insertFlag)
//
////////////////////////////////////////////////////////////////////////
{
    // check for trivial return
    if (camera == NULL || feedbackFlag == insertFlag)
    {
        feedbackFlag = insertFlag;
        return;
    }

    //
    // find the camera parent to insert/remove the feedback root
    //
    SoSearchAction sa;
    if (insertFlag)
        sa.setNode(camera);
    else
    {
        sa.setNode(feedbackRoot);
        sa.setSearchingAll(TRUE); // find under OFF switches for removal
    }
    sa.apply(sceneRoot);
    SoFullPath *fullPath = (SoFullPath *)sa.getPath();
    if (!fullPath)
    {
#if DEBUG
        SoDebugError::post("InvExaminerViewer::setFeedbackVisibility",
                           insertFlag ? "ERROR: cannot find camera in graph" : "ERROR: cannot find axis feedback in graph");
#endif
        return;
    }
    SoGroup *parent = (SoGroup *)fullPath->getNodeFromTail(1);

    feedbackFlag = insertFlag;

    // make sure the feedback has been built
    if (!feedbackRoot)
        createFeedbackNodes();

    //
    // inserts/remove the feedback axis group
    //

    if (feedbackFlag)
    {
        int camIndex;

        // check to make sure that the camera parent is not a switch node
        // (VRML camera viewpoints are kept under a switch node). Otherwise
        // we will insert the feedback after the switch node.
        if (parent->isOfType(SoSwitch::getClassTypeId()))
        {
            SoNode *switchNode = parent;
            parent = (SoGroup *)fullPath->getNodeFromTail(2);
            camIndex = parent->findChild(switchNode);
        }
        else
            camIndex = parent->findChild(camera);

        // return if feedback is already there (this should be an error !)
        if (parent->findChild(feedbackRoot) >= 0)
            return;

        // insert the feedback right after the camera+headlight (index+2)
        if (camIndex >= 0)
        {
            if (isHeadlight())
                parent->insertChild(feedbackRoot, camIndex + 2);
            else
                parent->insertChild(feedbackRoot, camIndex + 1);
        }

        // make sure the feedback switch is turned to the correct state now
        // that the feedback root has been inserted in the scene
        feedbackSwitch->whichChild.setValue(viewingFlag ? SO_SWITCH_ALL : SO_SWITCH_NONE);
    }
    else
    {
        if (parent->findChild(feedbackRoot) >= 0)
            parent->removeChild(feedbackRoot);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Sets the feedback size.
//
// Use: public
void
InvExaminerViewer::setFeedbackSize(int newSize)
//
////////////////////////////////////////////////////////////////////////
{
    if (feedbackSize == newSize)
        return;

    // assign new value and redraw (since it is not a field in the scene)
    feedbackSize = newSize;
    if (isFeedbackVisible() && isViewing())
        scheduleRedraw();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Remove the extra geometry when doing a viewAll.
//
// Use: virtual public
void
InvExaminerViewer::viewAll()
//
////////////////////////////////////////////////////////////////////////
{
    // stop spinning
    if (isAnimating())
        stopAnimating();

    float oPos = 0.0;
    if (camera->isOfType(SoOrthographicCamera::getClassTypeId()))
    {
        // change the ortho camera height
        //SoOrthographicCamera *cam = (SoOrthographicCamera *) camera;
        //cam->height = cam->height.getValue() * powf(2.0, newVal - rightWheelVal);
    }
    else
    {
        oPos = camera->position.getValue().length();
    }

    // temporarily remove the feedback geometry
    if (feedbackFlag && isViewing() && feedbackSwitch)
        feedbackSwitch->whichChild.setValue(SO_SWITCH_NONE);

    // call the base class
    InvFullViewer::viewAll();

    float nPos = 1.0;
    if (camera->isOfType(SoOrthographicCamera::getClassTypeId()))
    {
        // change the ortho camera height
        //SoOrthographicCamera *cam = (SoOrthographicCamera *) camera;
        //cam->height = cam->height.getValue() * powf(2.0, newVal - rightWheelVal);
    }
    else
    {
        nPos = camera->position.getValue().length();
    }

    if (oPos != 0.0)
    {
        float scl = nPos / oPos;
        Annotations->reScale(scl);
    }
    else
    {
        cerr << "InvExaminerViewer::viewAll(..) ERROR focalDistance == 0" << endl;
    }

    // now add the geometry back in
    if (feedbackFlag && isViewing() && feedbackSwitch)
        feedbackSwitch->whichChild.setValue(SO_SWITCH_ALL);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Call the parent class and stop animation if any
//
// Use: virtual public
void
InvExaminerViewer::resetToHomePosition()
//
////////////////////////////////////////////////////////////////////////
{
    // stop spinning
    if (isAnimating())
        stopAnimating();

    // call the base class
    InvFullViewer::resetToHomePosition();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Call the parent class and insert/remove the feedback root
//
// Use: virtual public
void
InvExaminerViewer::setCamera(SoCamera *newCamera)
//
////////////////////////////////////////////////////////////////////////
{
    if (camera == newCamera)
        return;

    // set the right thumbwheel label and toggle button image based on
    // the camera type
    if (newCamera != NULL && (camera == NULL || newCamera->getTypeId() != camera->getTypeId()))
    {
        if (newCamera->isOfType(SoOrthographicCamera::getClassTypeId()))
        {
            if (buttonList[CAM_PUSH])
                buttonList[CAM_PUSH]->setIcon(so_xt_ortho_bits,
                                              so_xt_icon_width, so_xt_icon_height);
            setRightWheelString("Zoom");
        }
        else
        {
            if (buttonList[CAM_PUSH])
                buttonList[CAM_PUSH]->setIcon(so_xt_persp_bits,
                                              so_xt_icon_width, so_xt_icon_height);
            setRightWheelString("Dolly");
        }
    }

    // detach feedback which depends on camera
    if (feedbackFlag)
    {
        setFeedbackVisibility(FALSE);
        feedbackFlag = TRUE; // can later be turned on
    }

    // call parent class
    InvFullViewer::setCamera(newCamera);

    // attach feedback back on
    if (feedbackFlag)
    {
        feedbackFlag = FALSE; // enables routine to be called
        setFeedbackVisibility(TRUE);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Call the base class and sets the correct cursors on the window,
//  plus insert/remove the feedback geometry.
//
// Use: virtual public
void
InvExaminerViewer::setViewing(SbBool flag)
//
////////////////////////////////////////////////////////////////////////
{
    if (flag == viewingFlag)
        return;

    // call the parent class
    InvFullViewer::setViewing(flag);

    switchMode(isViewing() ? VIEW_MODE : PICK_MODE);

    // show/hide the feedback geometry based on the viewing state
    if (feedbackFlag && feedbackSwitch)
        feedbackSwitch->whichChild.setValue(viewingFlag ? SO_SWITCH_ALL : SO_SWITCH_NONE);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Enables/Disable the viewer cursor on the window.
//
// Use: virtual public
void
InvExaminerViewer::setCursorEnabled(SbBool flag)
//
////////////////////////////////////////////////////////////////////////
{
    if (flag == cursorEnabledFlag)
        return;

    cursorEnabledFlag = flag;

    if (!isViewing())
        return;

    updateCursor();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Process the given event to do viewing stuff
//
// Use: virtual protected
void
InvExaminerViewer::processEvent(XAnyEvent *xe)
//
////////////////////////////////////////////////////////////////////////
{
    if (processCommonEvents(xe))
        return;

    if (!createdCursors)
        updateCursor();

    XButtonEvent *be;
    XMotionEvent *me;
    XKeyEvent *ke;
    XCrossingEvent *ce;
    KeySym keysym;

    SbVec2s raSize = getGlxSize();

    switch (xe->type)
    {
    case ButtonPress:
    case ButtonRelease:
        be = (XButtonEvent *)xe;
        if (be->button != Button1 && be->button != Button2)
            break;

        locator[0] = be->x;
        locator[1] = raSize[1] - be->y;

        if (mode == SEEK_MODE)
        {
            if (xe->type == ButtonPress)
                seekToPoint(locator);
        }
        else
        {
            if (xe->type == ButtonPress)
                interactiveCountInc();
            else //... ButtonRelease
            {

                // check if we need to start spinning
                if (mode == SPIN_MODE_ACTIVE && animationEnabled
                    && lastMotionTime == (int32_t)be->time)
                {
                    animatingFlag = TRUE;
                    computeAverage = TRUE;
                    animationSensor->attach(viewerRealTime);
                    interactiveCountInc();
                }

                interactiveCountDec();
            }
            updateViewerMode(be->state);
        }
        if (xe->type == ButtonPress)
            stopAnimating();
        break;

    case KeyPress:
    case KeyRelease:
        ke = (XKeyEvent *)xe;
        keysym = XLookupKeysym(ke, 0);

        locator[0] = ke->x;
        locator[1] = raSize[1] - ke->y;
        if (keysym == XK_Control_L || keysym == XK_Control_R)
            updateViewerMode(ke->state);
        break;

    case MotionNotify:
        me = (XMotionEvent *)xe;
        switch (mode)
        {
        case SPIN_MODE_ACTIVE:
            lastMotionTime = me->time;
            spinCamera(SbVec2f(me->x / float(raSize[0]), (raSize[1] - me->y) / float(raSize[1])));
            break;
        case PAN_MODE_ACTIVE:
            panCamera(SbVec2f(me->x / float(raSize[0]), (raSize[1] - me->y) / float(raSize[1])));
            break;
        case DOLLY_MODE_ACTIVE:
            dollyCamera(SbVec2s(me->x, raSize[1] - me->y));
            break;
        }
        break;

    case LeaveNotify:
    case EnterNotify:
        ce = (XCrossingEvent *)xe;
        //
        // because the application might use Ctrl-key for motif menu
        // accelerators we might not receive a key-up event, so make sure
        // to reset any Ctrl mode if we loose focus, but don't do anything
        // if Ctrl-key is not down (nothing to do) or if a mouse button
        // is down (we will get another leaveNotify).
        //
        if (!(ce->state & ControlMask))
            break;
        if (ce->state & Button1Mask || ce->state & Button2Mask)
            break;
        if (xe->type == LeaveNotify)
            switchMode(VIEW_MODE);
        else
            updateViewerMode(ce->state);
        break;
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    sets the viewer mode based on what keys and buttons are being pressed
//
// Use: private
void
InvExaminerViewer::updateViewerMode(unsigned int state)
//
////////////////////////////////////////////////////////////////////////
{
    // ??? WARNING - this routine ONLY works because of
    // ??? InvViewer::updateEventState() which is called for us
    // ??? by InvViewer::processCommonEvents().
    // ??? (XEvent state for button and modifier keys is not updated
    // ??? until after the event is received. WEIRD)

    // LEFT+MIDDLE down
    if (state & Button1Mask && state & Button2Mask)
    {
        switchMode(DOLLY_MODE_ACTIVE);
    }

    // LEFT down
    else if (state & Button1Mask)
    {
        if (state & ControlMask)
            switchMode(PAN_MODE_ACTIVE);
        else
            switchMode(SPIN_MODE_ACTIVE);
    }

    // MIDDLE DOWN
    else if (state & Button2Mask)
    {
        if (state & ControlMask)
            switchMode(DOLLY_MODE_ACTIVE);
        else
            switchMode(PAN_MODE_ACTIVE);
    }

    // no buttons down...
    else
    {
        if (state & ControlMask)
            switchMode(PAN_MODE);
        else
            switchMode(VIEW_MODE);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    switches to the specified viewer mode
//
// Use: private
void
InvExaminerViewer::switchMode(int newMode)
//
////////////////////////////////////////////////////////////////////////
{
    Widget raWidget = getRenderAreaWidget();
    SbVec2s raSize = getGlxSize();

    // assing new mode
    int prevMode = mode;
    mode = newMode;

    // update the cursor
    updateCursor();

    // switch to new viewer mode
    switch (newMode)
    {
    case PICK_MODE:
        if (raWidget && XtWindow(raWidget))
        {
            // ???? is if are going into PICK mode and some of our
            // mouse buttons are still down, make sure to decrement
            // interactive count correctly (correct draw style). One
            // for the LEFT and one for MIDDLE mouse.
            Window root_return, child_return;
            int root_x_return, root_y_return;
            int win_x_return, win_y_return;
            unsigned int mask_return;
            XQueryPointer(XtDisplay(raWidget), XtWindow(raWidget),
                          &root_return, &child_return,
                          &root_x_return, &root_y_return, &win_x_return,
                          &win_y_return, &mask_return);
            if (mask_return & Button1Mask && prevMode != SEEK_MODE)
                interactiveCountDec();
            if (mask_return & Button2Mask && prevMode != SEEK_MODE)
                interactiveCountDec();
        }
        stopAnimating();
        break;

    case SPIN_MODE_ACTIVE:
        // set the sphere sheet starting point
        sphereSheet->project(
            SbVec2f(locator[0] / float(raSize[0]), locator[1] / float(raSize[1])));

        // reset the animation queue
        firstIndex = 0;
        lastIndex = -1;
        break;

    case PAN_MODE_ACTIVE:
    {
        // Figure out the focal plane
        SbMatrix mx;
        mx = camera->orientation.getValue();
        SbVec3f forward(-mx[2][0], -mx[2][1], -mx[2][2]);
        SbVec3f fp = camera->position.getValue() + forward * camera->focalDistance.getValue();
        focalplane = SbPlane(forward, fp);

        // map mouse starting position onto the panning plane
        SbViewVolume cameraVolume;
        SbLine line;
        cameraVolume = camera->getViewVolume(raSize[0] / float(raSize[1]));
        cameraVolume.projectPointToLine(
            SbVec2f(locator[0] / float(raSize[0]), locator[1] / float(raSize[1])), line);
        focalplane.intersect(line, locator3D);
    }
    break;
    }
}

void
InvExaminerViewer::setPresentationCursor(bool onOrOff)
{
    presentationCursor_ = onOrOff;
    updateCursor();
}

bool
InvExaminerViewer::getPresentationCursor()
{
    return presentationCursor_;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    updates the viewer cursor
//
// Use: private
void
InvExaminerViewer::updateCursor()
//
////////////////////////////////////////////////////////////////////////
{
    Widget w = getRenderAreaWidget();
    Display *display = w ? XtDisplay(w) : NULL;
    Window window = w ? XtWindow(w) : 0;

    if (!window)
        return;

    if (!createdCursors)
        defineCursors();

    // the viewer cursor are not enabled, then we don't set a new cursor.
    // Instead erase the old viewer cursor.
    if (!cursorEnabledFlag)
    {
        XUndefineCursor(display, window);
        return;
    }

    // ...else set the right cursor for the viewer mode....
    switch (mode)
    {
    case PICK_MODE:
        if (presentationCursor_)
        {
            XDefineCursor(display, window, presArrowCursor_);
        }
        else
        {
            XUndefineCursor(display, window);
        }
        break;

    case VIEW_MODE:
    case SPIN_MODE_ACTIVE:
        if (presentationCursor_)
        {
            XDefineCursor(display, window, presSpinCursor_);
        }
        else
        {
            XDefineCursor(display, window, spinCursor);
        }

        break;

    case DOLLY_MODE_ACTIVE:
        if (presentationCursor_)
        {
            XDefineCursor(display, window, presDollyCursor_);
            //XDefineCursor(display, window, dollyCursor);
        }
        else
        {
            XDefineCursor(display, window, dollyCursor);
        }
        break;

    case PAN_MODE:
    case PAN_MODE_ACTIVE:
        XDefineCursor(display, window, panCursor);
        break;

    case SEEK_MODE:
        XDefineCursor(display, window, seekCursor);
        break;
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    draws viewer feedback during a render area redraw of the scene.
//
// Use: virtual protected
void
InvExaminerViewer::actualRedraw()
//
////////////////////////////////////////////////////////////////////////
{
    // place the feedback at the focal point
    // ??? we really only need to do this when the camera changes
    if (isViewing() && feedbackFlag && camera != NULL && feedbackRoot)
    {

        // adjust the position to be at the focal point
        SbMatrix mx;
        mx = camera->orientation.getValue();
        SbVec3f forward(-mx[2][0], -mx[2][1], -mx[2][2]);
        feedbackTransNode->translation = camera->position.getValue() + camera->focalDistance.getValue() * forward;

        // adjust the size to be constant on the screen
        float height;
        if (camera->isOfType(SoPerspectiveCamera::getClassTypeId()))
        {
            float angle = ((SoPerspectiveCamera *)camera)->heightAngle.getValue();
            height = camera->focalDistance.getValue() * tanf(angle / 2);
        }
        else if (camera->isOfType(SoOrthographicCamera::getClassTypeId()))
            height = ((SoOrthographicCamera *)camera)->height.getValue() / 2;
        else
        {
            height = 1.0;
            fprintf(stderr, "InvExaminerViewer::actualRedraw(): height uninitialized\n");
        }

        // ??? getGlxSize[1] == 0 the very first time, so return in that case
        // ??? else the redraws are 3 times slower from now on !! (alain)
        if (getGlxSize()[1] != 0)
        {
            float size = 2.0 * height * feedbackSize / float(getGlxSize()[1]);
            feedbackScaleNode->scaleFactor.setValue(size, size, size);
        }
    }

    // have the base class draw the scene
    InvFullViewer::actualRedraw();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Enable/disable the animation feature of the Examiner
//
// Use: public
void
InvExaminerViewer::setAnimationEnabled(SbBool flag)
//
////////////////////////////////////////////////////////////////////////
{
    if (animationEnabled == flag)
        return;

    animationEnabled = flag;
    if (!animationEnabled && isAnimating())
        stopAnimating();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Stops ongoing animation (if any)
//
// Use: public
void
InvExaminerViewer::stopAnimating()
//
////////////////////////////////////////////////////////////////////////
{
    if (animatingFlag)
    {
        animatingFlag = FALSE;
        animationSensor->detach();
        animationSensor->unschedule();
        interactiveCountDec();
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Call the base class and sets the correct cursors on the window
//
// Use: virtual protected
void
InvExaminerViewer::setSeekMode(SbBool flag)
//
////////////////////////////////////////////////////////////////////////
{
    if (!isViewing())
        return;

    // stop spinning
    if (isAnimating())
        stopAnimating();

    // call the base class
    InvFullViewer::setSeekMode(flag);

    mode = isSeekMode() ? SEEK_MODE : VIEW_MODE;

    updateCursor();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Redefine this routine to add some viewer specific stuff.
//
// Use: virtual protected
void
InvExaminerViewer::createPrefSheet()
//
////////////////////////////////////////////////////////////////////////
{
    // create the preference sheet shell and form widget
    Widget shell, form;
    createPrefSheetShellAndForm(shell, form);

    // create all of the parts
    Widget widgetList[20];
    int num = 0;
    widgetList[num++] = createSeekPrefSheetGuts(form);
    widgetList[num++] = createZoomPrefSheetGuts(form);
    widgetList[num++] = createSamplingPrefSheetGuts(form);
    widgetList[num++] = InvFullViewer::createClippingPrefSheetGuts(form);
    widgetList[num++] = createStereoPrefSheetGuts(form);
    widgetList[num++] = createExamPrefSheetGuts(form);
    widgetList[num++] = InvFullViewer::createColorsPrefSheetGuts(form);

    layoutPartsAndMapPrefSheet(widgetList, num, form, shell);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Creates the viewer extra pref sheet stuff
//
// Use: private
Widget
InvExaminerViewer::createExamPrefSheetGuts(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    Widget toggles[3], labels[3];
    Arg args[12];
    int n;

    // create a form to hold everything together
    Widget form = XtCreateWidget("", xmFormWidgetClass, parent, NULL, 0);

    // create all the parts
    n = 0;
    XtSetArg(args[n], XmNset, animationEnabled);
    n++;
    XtSetArg(args[n], XmNspacing, 0);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    toggles[0] = XtCreateWidget("", xmToggleButtonGadgetClass, form, args, n);
    labels[0] = XtCreateWidget("Enable spin animation",
                               xmLabelGadgetClass, form, NULL, 0);
    XtAddCallback(toggles[0], XmNvalueChangedCallback,
                  (XtCallbackProc)InvExaminerViewer::animPrefSheetToggleCB,
                  (XtPointer) this);

    n = 0;
    XtSetArg(args[n], XmNsensitive, camera != NULL);
    n++;
    XtSetArg(args[n], XmNset, feedbackFlag);
    n++;
    XtSetArg(args[n], XmNspacing, 0);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    toggles[1] = XtCreateWidget("", xmToggleButtonGadgetClass, form, args, n);
    n = 0;
    XtSetArg(args[n], XmNsensitive, camera != NULL);
    n++;
    labels[1] = XtCreateWidget("Show point of rotation axes",
                               xmLabelGadgetClass, form, args, n);
    XtAddCallback(toggles[1], XmNvalueChangedCallback,
                  (XtCallbackProc)InvExaminerViewer::feedbackPrefSheetToggleCB,
                  (XtPointer) this);

    // create all the parts
    n = 0;
    XtSetArg(args[n], XmNset, smoothNormalsEnabled);
    n++;
    XtSetArg(args[n], XmNspacing, 0);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    toggles[2] = XtCreateWidget("", xmToggleButtonGadgetClass, form, args, n);
    labels[2] = XtCreateWidget("Generate smooth normals",
                               xmLabelGadgetClass, form, NULL, 0);
    XtAddCallback(toggles[2], XmNvalueChangedCallback,
                  (XtCallbackProc)InvExaminerViewer::normalPrefSheetToggleCB,
                  (XtPointer) this);
    // layout
    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, toggles[0]);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, toggles[0]);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, toggles[0]);
    n++;
    XtSetValues(labels[0], args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, toggles[0]);
    n++;
    XtSetArg(args[n], XmNtopOffset, 10);
    n++;
    XtSetValues(toggles[1], args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, toggles[1]);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, toggles[1]);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, toggles[1]);
    n++;
    XtSetValues(labels[1], args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, toggles[1]);
    n++;
    XtSetArg(args[n], XmNtopOffset, 10);
    n++;
    XtSetValues(toggles[2], args, n);
    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, toggles[2]);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, toggles[2]);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, toggles[2]);
    n++;
    XtSetValues(labels[2], args, n);

    // manage children
    XtManageChildren(toggles, 3);
    XtManageChildren(labels, 3);

    if (feedbackFlag && camera)
        toggleFeedbackWheelSize(toggles[1]);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Brings the viewer help card (called by "?" push button)
//
// Use: virtual protected
void
InvExaminerViewer::openViewerHelpCard()
//
////////////////////////////////////////////////////////////////////////
{
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Call the base class and stop animating
//
// Use: virtual protected

void
InvExaminerViewer::bottomWheelStart()
//
////////////////////////////////////////////////////////////////////////
{
    InvFullViewer::bottomWheelStart();
    stopAnimating();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Call the base class and stop animating
//
// Use: virtual protected

void
InvExaminerViewer::leftWheelStart()
//
////////////////////////////////////////////////////////////////////////
{
    InvFullViewer::bottomWheelStart();
    stopAnimating();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Rotates the object around the screen x axis (called by thumb wheel).
//
// Use: virtual protected

void
InvExaminerViewer::bottomWheelMotion(float newVal)
//
////////////////////////////////////////////////////////////////////////
{
    // get rotation and apply to camera
    SbVec3f axis(0, 1, 0);
    SbRotation rot(axis, bottomWheelVal - newVal);
    rotateCamera(rot);

    bottomWheelVal = newVal;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Rotates the object around the screen y axis (called by thumb wheel).
//
// Use: virtual protected

void
InvExaminerViewer::leftWheelMotion(float newVal)
//
////////////////////////////////////////////////////////////////////////
{
    // get rotation and apply to camera
    SbVec3f axis(1, 0, 0);
    SbRotation rot(axis, newVal - leftWheelVal);
    rotateCamera(rot);

    leftWheelVal = newVal;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//   	Moves the camera closer/further away from the plane of interest
//  (perspective camera case), else change the camera height (orthographic
//  camera case).
//
// Use: virtual protected

void
InvExaminerViewer::rightWheelMotion(float newVal)
//
////////////////////////////////////////////////////////////////////////
{
    if (camera == NULL)
        return;

    if (camera->isOfType(SoOrthographicCamera::getClassTypeId()))
    {
        // change the ortho camera height
        SoOrthographicCamera *cam = (SoOrthographicCamera *)camera;
        cam->height = cam->height.getValue() * powf(2.0, newVal - rightWheelVal);
    }
    else
    {
        // shorter/grow the focal distance given the wheel rotation
        float focalDistance = camera->focalDistance.getValue();
        float newFocalDist = focalDistance;
        newFocalDist *= powf(2.0, newVal - rightWheelVal);

        // finally reposition the camera
        SbMatrix mx;
        mx = camera->orientation.getValue();
        SbVec3f forward(-mx[2][0], -mx[2][1], -mx[2][2]);
        SbVec3f oldCamPos = camera->position.getValue();
        camera->position = camera->position.getValue() + (focalDistance - newFocalDist) * forward;
        if (focalDistance != 0.0)
        {
            //	    float scl = newFocalDist / focalDistance;
            float scl = camera->position.getValue().length() / oldCamPos.length();
            Annotations->reScale(scl);
        }
        else
        {
            cerr << "InvExaminerViewer::dollyCamera(..) ERROR focalDistance == 0" << endl;
        }

        camera->focalDistance = newFocalDist;
    }

    rightWheelVal = newVal;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    This routine is used to define cursors (can only be called after
//  window has been realized).
//
// Use: private

void
InvExaminerViewer::defineCursors()
//
////////////////////////////////////////////////////////////////////////
{
    XColor foreground;
    XColor background;
    Pixmap source;
    Display *display = getDisplay();
    Drawable d = DefaultRootWindow(display);

    // set color
    foreground.red = 65535;
    foreground.green = foreground.blue = 0;

    background.red = 65535;
    background.green = background.blue = 0;

    // spin cursor
    source = XCreateBitmapFromData(display, d, so_xt_curved_hand_bits,
                                   so_xt_curved_hand_width, so_xt_curved_hand_height);

    //     XWriteBitmapFile(display, "hand.bmp", source, so_xt_curved_hand_width,so_xt_curved_hand_height,
    // 		     so_xt_curved_hand_x_hot, so_xt_curved_hand_y_hot);

    spinCursor = XCreatePixmapCursor(display, source, source,
                                     &foreground, &background, so_xt_curved_hand_x_hot, so_xt_curved_hand_y_hot);
    XFreePixmap(display, source);

    // panning cursor
    source = XCreateBitmapFromData(display, d,
                                   so_xt_flat_hand_bits, so_xt_flat_hand_width, so_xt_flat_hand_height);

    //     XWriteBitmapFile(display, "flat_hand.bmp", source, so_xt_curved_hand_width,so_xt_curved_hand_height,
    //   		     so_xt_curved_hand_x_hot, so_xt_curved_hand_y_hot);

    panCursor = XCreatePixmapCursor(display, source, source,
                                    &foreground, &foreground, so_xt_flat_hand_x_hot, so_xt_flat_hand_y_hot);
    XFreePixmap(display, source);

    // dolly cursor
    source = XCreateBitmapFromData(display, d,
                                   so_xt_pointing_hand_bits, so_xt_pointing_hand_width, so_xt_pointing_hand_height);

    //     XWriteBitmapFile(display, "pointing_hand.bmp", source,
    // 		     so_xt_pointing_hand_width,so_xt_pointing_hand_height,
    //   		     so_xt_curved_hand_x_hot, so_xt_curved_hand_y_hot);

    dollyCursor = XCreatePixmapCursor(display, source, source,
                                      &foreground, &foreground, so_xt_pointing_hand_x_hot, so_xt_pointing_hand_y_hot);
    XFreePixmap(display, source);

    // seek cursor
    source = XCreateBitmapFromData(display, d,
                                   so_xt_target_bits, so_xt_target_width, so_xt_target_height);

    //     XWriteBitmapFile(display, "target.bmp", source,  so_xt_target_width, so_xt_target_height,
    // 		     so_xt_target_x_hot, so_xt_target_y_hot);

    seekCursor = XCreatePixmapCursor(display, source, source,
                                     &foreground, &foreground, so_xt_target_x_hot, so_xt_target_y_hot);
    XFreePixmap(display, source);

    // presentation spin cursor
    foreground.red = 65535;
    foreground.green = foreground.blue = 65535;

    source = XCreateBitmapFromData(display, d, reinterpret_cast<char *>(hand_pres_bits),
                                   hand_pres_width, hand_pres_height);

    presSpinCursor_ = XCreatePixmapCursor(display, source, source, &foreground, &background,
                                          hand_pres_x_hot, hand_pres_y_hot);

    XFreePixmap(display, source);

    // presentation dolly cursor
    source = XCreateBitmapFromData(display, d, reinterpret_cast<char *>(pointing_hand_pres_bits),
                                   pointing_hand_pres_width, pointing_hand_pres_height);

    presDollyCursor_ = XCreatePixmapCursor(display, source, source, &foreground, &background,
                                           pointing_hand_pres_x_hot, pointing_hand_pres_y_hot);
    XFreePixmap(display, source);

    // presentation arrow
    source = XCreateBitmapFromData(display, d, reinterpret_cast<char *>(arrow_pres_bits),
                                   arrow_pres_width, arrow_pres_height);

    presArrowCursor_ = XCreatePixmapCursor(display, source, source, &foreground, &background,
                                           arrow_pres_x_hot, arrow_pres_y_hot);
    XFreePixmap(display, source);

    createdCursors = TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Rotates the camera around pt of interest by given rotation
//
// Use: private

void
InvExaminerViewer::rotateCamera(const SbRotation &rot)
//
////////////////////////////////////////////////////////////////////////
{
    if (camera == NULL)
        return;

    // get center of rotation
    SbRotation camRot = camera->orientation.getValue();
    float radius = camera->focalDistance.getValue();
    SbMatrix mx;
    mx = camRot;
    SbVec3f forward(-mx[2][0], -mx[2][1], -mx[2][2]);
    SbVec3f center = camera->position.getValue()
                     + radius * forward;

    // apply new rotation to the camera
    camRot = rot * camRot;
    camera->orientation = camRot;

    // reposition camera to look at pt of interest
    mx = camRot;
    forward.setValue(-mx[2][0], -mx[2][1], -mx[2][2]);
    camera->position = center - radius * forward;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Moves the camera into the plane defined by the camera forward vector
//  and the focal point to follow the new mouse location.
//
// Use: private

void
InvExaminerViewer::panCamera(const SbVec2f &newLocator)
//
////////////////////////////////////////////////////////////////////////
{
    if (camera == NULL)
        return;

    // map new mouse location into the camera focal plane
    SbViewVolume cameraVolume;
    SbLine line;
    SbVec3f newLocator3D;
    SbVec2s raSize = getGlxSize();
    cameraVolume = camera->getViewVolume(raSize[0] / float(raSize[1]));
    cameraVolume.projectPointToLine(newLocator, line);
    focalplane.intersect(line, newLocator3D);

    // move the camera by the delta 3D position amount
    camera->position = camera->position.getValue() + (locator3D - newLocator3D);

    // You would think we would have to set locator3D to
    // newLocator3D here.  But we don't, because moving the camera
    // essentially makes locator3D equal to newLocator3D in the
    // transformed space, and we will project the next newLocator3D in
    // this transformed space.
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    rotates the camera using the sheet sphere projector, given the new
//  mouse location.
//
// Use: private

void
InvExaminerViewer::spinCamera(const SbVec2f &newLocator)
//
////////////////////////////////////////////////////////////////////////
{
    // find rotation and rotate camera
    SbRotation rot;
    sphereSheet->projectAndGetRotation(newLocator, rot);
    rot.invert();

    rotateCamera(rot);

    // save rotation for animation
    lastIndex = ((lastIndex + 1) % ROT_BUFF_SIZE);
    rotBuffer[lastIndex] = rot;

    // check if queue is full
    if (((lastIndex + 1) % ROT_BUFF_SIZE) == firstIndex)
        firstIndex = ((firstIndex + 1) % ROT_BUFF_SIZE);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Moves the camera forward/backward based on the new mouse potion.
//  (perspective camera), else change the camera height (orthographic
//  camera case).
//
// Use: private

void
InvExaminerViewer::dollyCamera(const SbVec2s &newLocator)
//
////////////////////////////////////////////////////////////////////////
{
    if (camera == NULL)
        return;

    // moving the mouse up/down will move the camera futher/closer.
    // moving the camera sideway will not move the camera at all
    float d = (newLocator[1] - locator[1]) / 40.0;

    if (camera->isOfType(SoOrthographicCamera::getClassTypeId()))
    {
        // change the ortho camera height
        SoOrthographicCamera *cam = (SoOrthographicCamera *)camera;
        cam->height = cam->height.getValue() * powf(2.0, d);
    }
    else
    {
        // shorter/grow the focal distance given the mouse move
        float focalDistance = camera->focalDistance.getValue();
        ;
        float newFocalDist = focalDistance * powf(2.0, d);

        // finally reposition the camera
        SbMatrix mx;
        mx = camera->orientation.getValue();
        SbVec3f forward(-mx[2][0], -mx[2][1], -mx[2][2]);
        SbVec3f oldCamPos = camera->position.getValue();
        camera->position = camera->position.getValue() + (focalDistance - newFocalDist) * forward;

        if (focalDistance != 0.0)
        {
            float scl = camera->position.getValue().length() / oldCamPos.length();
            //	    float scl = newFocalDist / focalDistance;
            Annotations->reScale(scl);
        }
        else
        {
            cerr << "InvExaminerViewer::dollyCamera(..) ERROR focalDistance == 0" << endl;
        }
        camera->focalDistance = newFocalDist;
    }

    locator = newLocator;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Routine which animates the ball spinning (called by sensor).
//
// Use: private

void
InvExaminerViewer::doSpinAnimation()
//
////////////////////////////////////////////////////////////////////////
{
    //
    // check if average rotation needs to be computed
    //

    if (computeAverage)
    {
        float averageAngle, angle;
        SbVec3f averageAxis, axis;

        // get number of samples
        int num = (((lastIndex - firstIndex) + 1 + ROT_BUFF_SIZE) % ROT_BUFF_SIZE);

        // check for not enough samples
        if (num < 2)
        {
            stopAnimating();
            return;
        }

        // get average axis of rotation
        // ??? right now only take one sample
        rotBuffer[firstIndex].getValue(averageAxis, angle);

        // get average angle of rotation
        averageAngle = 0;
        for (int i = 0; i < num; i++)
        {
            int n = (firstIndex + i) % ROT_BUFF_SIZE;
            rotBuffer[n].getValue(axis, angle);
            averageAngle += angle;
        }
        averageAngle /= float(num);

        averageRotation.setValue(averageAxis, averageAngle);
        computeAverage = FALSE;
    }

    //
    // rotate camera by average rotation
    //
    rotateCamera(averageRotation);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Show/hide the pref sheet feedback size wheel and label.
//
// Use: private

void
InvExaminerViewer::toggleFeedbackWheelSize(Widget toggle)
//
////////////////////////////////////////////////////////////////////////
{
    if (feedbackFlag)
    {
        Widget parent = XtParent(toggle);
        Arg args[12];
        int n;

        // create the label/thumb/text/label in the toggle parent
        feedbackLabel[0] = XtCreateWidget("axes size:",
                                          xmLabelGadgetClass, parent, NULL, 0);

        n = 0;
        XtSetArg(args[n], XmNvalue, 0);
        n++;
        XtSetArg(args[n], SgNangleRange, 0);
        n++;
        XtSetArg(args[n], SgNunitsPerRotation, 360);
        n++;
        XtSetArg(args[n], SgNshowHomeButton, FALSE);
        n++;
        XtSetArg(args[n], XmNhighlightThickness, 0);
        n++;
        XtSetArg(args[n], XmNorientation, XmHORIZONTAL);
        n++;
        feedbackSizeWheel = SgCreateThumbWheel(parent, (char *)"", args, n);

        XtAddCallback(feedbackSizeWheel, XmNvalueChangedCallback,
                      (XtCallbackProc)InvExaminerViewer::feedbackSizeWheelCB, (XtPointer) this);
        XtAddCallback(feedbackSizeWheel, XmNdragCallback,
                      (XtCallbackProc)InvExaminerViewer::feedbackSizeWheelCB, (XtPointer) this);
        feedbackSizeWheelVal = 0;

        n = 0;
        char str[15];
        sprintf(str, "%d", int(feedbackSize));
        XtSetArg(args[n], XmNvalue, str);
        n++;
        XtSetArg(args[n], XmNhighlightThickness, 1);
        n++;
        XtSetArg(args[n], XmNcolumns, 3);
        n++;
        feedbackField = XtCreateWidget("", xmTextWidgetClass,
                                       parent, args, n);

        XtAddCallback(feedbackField, XmNactivateCallback,
                      (XtCallbackProc)InvExaminerViewer::feedbackSizeFieldCB,
                      (XtPointer) this);

        feedbackLabel[1] = XtCreateWidget("pixels",
                                          xmLabelGadgetClass, parent, NULL, 0);

        // layout
        n = 0;
        XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNleftOffset, 20);
        n++;
        XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[n], XmNtopWidget, toggle);
        n++;
        XtSetArg(args[n], XmNtopOffset, 5);
        n++;
        XtSetValues(feedbackLabel[0], args, n);

        n = 0;
        XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[n], XmNleftWidget, feedbackLabel[0]);
        n++;
        XtSetArg(args[n], XmNleftOffset, 5);
        n++;
        XtSetArg(args[n], XmNtopAttachment, XmATTACH_OPPOSITE_WIDGET);
        n++;
        XtSetArg(args[n], XmNtopWidget, feedbackLabel[0]);
        n++;
        XtSetValues(feedbackSizeWheel, args, n);

        n = 0;
        XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[n], XmNleftWidget, feedbackSizeWheel);
        n++;
        XtSetArg(args[n], XmNleftOffset, 3);
        n++;
        XtSetArg(args[n], XmNtopAttachment, XmATTACH_OPPOSITE_WIDGET);
        n++;
        XtSetArg(args[n], XmNtopWidget, feedbackSizeWheel);
        n++;
        XtSetArg(args[n], XmNtopOffset, -5);
        n++;
        XtSetValues(feedbackField, args, n);

        n = 0;
        XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[n], XmNleftWidget, feedbackField);
        n++;
        XtSetArg(args[n], XmNleftOffset, 5);
        n++;
        XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
        n++;
        XtSetArg(args[n], XmNbottomWidget, feedbackLabel[0]);
        n++;
        XtSetValues(feedbackLabel[1], args, n);

        // manage children
        XtManageChild(feedbackLabel[0]);
        XtManageChild(feedbackSizeWheel);
        XtManageChild(feedbackField);
        XtManageChild(feedbackLabel[1]);
    }
    else
    {
        // destroys the widgets
        XtDestroyWidget(feedbackLabel[1]);
        XtDestroyWidget(feedbackField);
        XtDestroyWidget(feedbackSizeWheel);
        XtDestroyWidget(feedbackLabel[0]);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Build the parent class widget, then register this widget.
//
// Use: protected
Widget
InvExaminerViewer::buildWidget(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    // Create the root widget and register it with a class name
    Widget w = InvFullViewer::buildWidget(parent);

    // If first build, get resource values
    if (firstBuild)
    {
        // Full viewer registered the widget for us
        SoXtResource xr(w);
        SbBool flag;
        short val;

        //if (xr.getResource((char *)"spinAnimation", (char *)"SpinAnimation", flag))
        //  setAnimationEnabled(flag);
        if (xr.getResource((char *)"pointOfRotationAxes", (char *)"PointOfRotationAxes", flag))
            setFeedbackVisibility(flag);
        if (xr.getResource((char *)"axesSize", (char *)"AxesSize", val))
            feedbackSize = val;

        firstBuild = FALSE;
    }

    return w;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	add our own button to the existing list
//
// Use: virtual protected
void
InvExaminerViewer::createViewerButtons(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    // get the default buttons
    InvFullViewer::createViewerButtons(parent);

    // allocate our buttons
    buttonList[CAM_PUSH] = new InvPixmapButton(parent, FALSE);
    buttonList[CAM_PUSH]->setIcon(so_xt_persp_bits, so_xt_icon_width, so_xt_icon_height);
    Widget w = buttonList[CAM_PUSH]->getWidget();
    XtAddCallback(w, XmNactivateCallback,
                  (XtCallbackProc)InvExaminerViewer::camPushCB, (XtPointer) this);

    // add this button to the list...
    viewerButtonWidgets->append(w);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	read the point of interest geometry, which will be placed
//  right after the camera node (or right after the headlight
//  if the headlight is turned on).
//
// Use: private
void
InvExaminerViewer::createFeedbackNodes()
//
////////////////////////////////////////////////////////////////////////
{
    // make sure we havn't built this yet...
    if (feedbackRoot)
        return;

    feedbackRoot = new SoSeparator(1);
    feedbackSwitch = new SoSwitch(3);
    feedbackTransNode = new SoTranslation;
    feedbackScaleNode = new SoScale;
    feedbackRoot->ref();
    feedbackRoot->addChild(feedbackSwitch);
    feedbackSwitch->addChild(feedbackTransNode);
    feedbackSwitch->addChild(feedbackScaleNode);
    SoInput in;
    in.setBuffer((void *)geometryBuffer, (size_t)strlen(geometryBuffer));
    SoNode *node;
    SbBool ok = SoDB::read(&in, node);
    if (ok && node != NULL)
        feedbackSwitch->addChild(node);
#ifdef DEBUG
    else
        SoDebugError::post("InvExaminerViewer::createFeedbackNodes",
                           "couldn't read feedback axis geometry");
#endif
}

//
// redefine those generic virtual functions
//
const char *
InvExaminerViewer::getDefaultWidgetName() const
{
    return thisClassName;
}

const char *
InvExaminerViewer::getDefaultTitle() const
{
    return "COVISE Renderer";
}

const char *
InvExaminerViewer::getDefaultIconTitle() const
{
    return "COVISE Renderer";
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//   	Send the decoration mode to other renderer.
//
// Use: private
void InvExaminerViewer::sendProjection()
//
// Dirk Rantzau
//
////////////////////////////////////////////////////////////////////////
{
    char message[20];

    if (rm_isMaster() && rm_isSynced())
    {

        if (getCameraType() == SoPerspectiveCamera::getClassTypeId())
            sprintf(message, "%d", 1);
        else
            sprintf(message, "%d", 0);

        rm_sendProjection(message);
    }
}

//
////////////////////////////////////////////////////////////////////////
// static callbacks stubs
////////////////////////////////////////////////////////////////////////
//

void
InvExaminerViewer::camPushCB(Widget, InvExaminerViewer *v, void *)
{
    v->toggleCameraType();
    v->sendProjection();
}

void
InvExaminerViewer::animationSensorCB(void *v, SoSensor *)
{
    ((InvExaminerViewer *)v)->doSpinAnimation();
}

void
InvExaminerViewer::animPrefSheetToggleCB(Widget toggle,
                                         InvExaminerViewer *v, void *)
{
    v->setAnimationEnabled(XmToggleButtonGetState(toggle));
}

void
InvExaminerViewer::normalPrefSheetToggleCB(Widget toggle,
                                           InvExaminerViewer *v, void *)
{
    (void)v;
    smoothNormalsEnabled = XmToggleButtonGetState(toggle);
}

void
InvExaminerViewer::feedbackPrefSheetToggleCB(Widget toggle,
                                             InvExaminerViewer *v, void *)
{
    // show/hide the feedback
    v->setFeedbackVisibility(XmToggleButtonGetState(toggle));

    // show/hide the extra size setting
    v->toggleFeedbackWheelSize(toggle);
}

void
InvExaminerViewer::feedbackSizeWheelCB(Widget, InvExaminerViewer *v, XtPointer *d)
{
    static SbBool firstDrag = TRUE;
    SgThumbWheelCallbackStruct *data = (SgThumbWheelCallbackStruct *)d;

    if (data->reason == XmCR_DRAG)
    {
        // for the first move, invoke the start callbacks
        if (firstDrag)
        {
            v->interactiveCountInc();
            firstDrag = FALSE;
        }

        // grow/shrink the feedback based on the wheel rotation
        v->feedbackSize *= powf(12, (data->value - v->feedbackSizeWheelVal) / 360.0);
        v->feedbackSizeWheelVal = data->value;

        // update the text field
        char str[15];
        sprintf(str, "%d", int(v->feedbackSize));
        XmTextSetString(v->feedbackField, str);

        // redraw since the wheel size isn't a field in the scene
        if (v->isViewing())
            v->scheduleRedraw();
    }
    else
    {
        // reason = XmCR_VALUE_CHANGED, invoke the finish callbacks
        v->interactiveCountDec();
        firstDrag = TRUE;
    }
}

void
InvExaminerViewer::feedbackSizeFieldCB(Widget field, InvExaminerViewer *v, void *)
{
    // get text value from the field
    char *str = XmTextGetString(field);
    int val;
    if (sscanf(str, "%d", &val) && val > 0)
        v->setFeedbackSize(val);
    else
        val = int(v->feedbackSize);
    XtFree(str);

    // reformat text field
    char valStr[15];
    sprintf(valStr, "%d", val);
    XmTextSetString(field, valStr);

    // make the text field loose the focus
    XmProcessTraversal(SoXt::getShellWidget(field), XmTRAVERSE_CURRENT);
}

// called when the viewer becomes visible/hidden - when hidden, make
// sure to temporary stop any ongoing animation (and restart it as soon
// as we become visible).
//
void
InvExaminerViewer::visibilityChangeCB(void *pt, SbBool visible)
{
    InvExaminerViewer *p = (InvExaminerViewer *)pt;

    // only do this if we are/were spinning....
    if (!p->animatingFlag)
        return;

    if (visible)
    {
        // we now are visible again so attach the field sensor
        p->animationSensor->attach(viewerRealTime);
    }
    else
    {
        // if hidden, detach the field sensor, but don't change the
        // animatingFlag var to let us know we need to turn it back on
        // when we become visible....
        p->animationSensor->detach();
        p->animationSensor->unschedule();
    }
}

//
// get transparent color of overlay planes
//
void
InvExaminerViewer::setupOverlays()
{
    Display *display;
    XVisualInfo tpl; /* Requirements for XGetVisualInfo */
    XVisualInfo *pVisuals;
    XVisualInfo *pVis;
    OverlayVisualPropertyRec *pOverlayVisuals;
    OverlayVisualPropertyRec *pOVis;
    int nVisuals, numOverlayVisuals, nOVisuals;
    Atom overlayVisualsAtom; /* Parameters for XGetWindowProperty */
    Atom actualType;
    unsigned long numItems, bytesAfter;
    int actualFormat;
    int overlayVisual;
    int transparentPixel;

    display = XOpenDisplay("");

    /* First, get the list of visuals for this screen. */
    tpl.screen = DefaultScreen(display);
    pVisuals = XGetVisualInfo(display, VisualScreenMask, &tpl, &nVisuals);

    /* Now, get the overlay visual information for this screen.  To obtain
    * this information, get the SERVER_OVERLAY_VISUALS property.
    */
    overlayVisualsAtom = XInternAtom(display, "SERVER_OVERLAY_VISUALS", True);
    if (overlayVisualsAtom != None)
    {
        /* Since the Atom exists, we can request the property's contents.
       * Accept information on up to 200 overlay visuals.
       */
        XGetWindowProperty(display,
                           DefaultRootWindow(display), /* window for property */
                           overlayVisualsAtom, /* Property name */
                           0, /* offset in longs */
                           /* max size in longs */
                           200 * sizeof(OverlayVisualPropertyRec) / sizeof(long),
                           False, /* don't delete property */
                           overlayVisualsAtom, /* type required */
                           &actualType, /* returned actual type */
                           &actualFormat, /* returned actual format */
                           (unsigned long *)&numItems, /* number of items returned */
                           (unsigned long *)&bytesAfter, /* number of unread bytes */
                           (unsigned char **)(void *)&pOverlayVisuals);

        /* Calculate the number of overlay visuals in the list. */
        numOverlayVisuals = (numItems * sizeof(long)) / sizeof(OverlayVisualPropertyRec);
    }
    else
    {
        /* This screen doesn't have overlay planes. */
        numOverlayVisuals = 0;
        pOverlayVisuals = NULL;
    }

    /* Process the pVisuals array. */
    for (pVis = pVisuals; --nVisuals >= 0; pVis++)
    {
        nOVisuals = numOverlayVisuals;
        overlayVisual = False;
        transparentPixel = -1; /* no transparent pixel by default. */
        for (pOVis = pOverlayVisuals; --nOVisuals >= 0; pOVis++)
        {
            if (pVis->visualid == pOVis->visualID)
            {
                overlayVisual = True;
                if (pOVis->transparentType == TransparentPixel)
                {
                    transparentPixel = pOVis->value;
                }
                break;
            }
        }

        if (overlayVisual)
        {
            switch (transparentPixel)
            {
            case -1:
                //                 printf("overlay visual, id: 0x%x, no transparent color\n",
                // 		       pVis->visualid);
                break;
            default:
                setOverlayBackgroundIndex(transparentPixel);
                break;
            }
        }
    }

    /* Free any allocated memory */
    XFree((char *)pVisuals);
    if (pOverlayVisuals)
        XFree((char *)pOverlayVisuals);
}
