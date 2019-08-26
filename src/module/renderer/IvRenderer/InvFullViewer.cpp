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
// * Description    : Inventor full viewer
//
// * Class(es)      : InvFullViewer
//
// * inherited from :  InvViewer
//
// * Author  : Dirk Rantzau
//
// * History : 29.03.94 V 1.0
//
//**************************************************************************

// Define MENUS_IN_POPUP to get menus in the popup window
//#ifndef __hpux
#include <covise/covise.h>
#define MENUS_IN_POPUP
//#endif

#include <X11/StringDefs.h>
#include <X11/Intrinsic.h>
#include <X11/Shell.h>

#include <Xm/Xm.h>
#include <Xm/LabelG.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>
#include <Xm/SeparatoG.h>
#include <Xm/CascadeB.h>
#include <Xm/CascadeBG.h>
#include <Xm/Form.h>
#include <Xm/ToggleB.h>
#include <Xm/ToggleBG.h>
#include <Xm/RowColumn.h>
#include <Xm/Scale.h>
#include <Xm/Text.h>

#ifndef __sgi
#include "ThumbWheel.h"
#else
#include <Sgm/ThumbWheel.h>
#endif

#ifdef __sgi
#include <invent.h>
#endif

#include <Xm/MessageB.h>

#include <Inventor/SoPickedPoint.h>
#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/SoXtResource.h>
#include "InvFullViewer.h"
#include <Inventor/Xt/SoXtIcons.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/sensors/SoFieldSensor.h>
#include <GL/gl.h>
#include "InvPixmapButton.h"
#include <Inventor/errors/SoDebugError.h>

#include "InvRenderer.h"

/* Uwe Woessner (Tracking) */
#include <Inventor/sensors/SoTimerSensor.h>
#include "head.xbm"
#include <config/CoviseConfig.h>
/* Ende Uwe Woessner (Tracking)*/

#include <net/covise_connect.h>
#include <covise/covise_msg.h>
#include <covise/covise_appproc.h>
#include <unistd.h>
#include <pwd.h>
#include <covise/Covise_Util.h>

// Annotation singleton
#include "InvAnnotationManager.h"

/*
 * Defines
 */
extern char *STRDUP(const char *s);
extern ApplicationProcess *appmod;
extern Widget MasterRequest;

// specifies the sizes for the decoration
#define DECOR_SIZE 28
#define LABEL_SPACE 3
#define LABEL_SEPARATION 12
#define THUMB_SPACE 4
#define WHEEL_DOLLY_FACTOR 0.5

// list of the different popup choices
enum popupChoices
{
    VIEW_ALL = 20, // enables the same menu routine to be used
    SET_HOME, // as the draw style entrie (can't overlap
    HOME, // with InvViewerDrawStyle values)
    HEADLIGHT,
    SEEK,
    PREF,
    VIEWING,
    DECORATION,
    NORMAL_CURSOR,
    PRESENTATION_CURSOR,
    ANNOTATION_CREATE,
    ANNOTATION_EDIT,
    ANNOTATION_DELETE,
    COPY_VIEW,
    MASTER_REQUEST,
    // changed 11.04.94
    // D.Rantzau
    ///    PASTE_VIEW,
    HELP
};

enum drawChoices
{
    AS_IS,
    HIDDEN_LINE,
    MESH,
    NO_TXT,
    LOW_RES,
    LINE,
    POINT,
    BBOX,
    LOW_VOLUME,

    MOVE_SAME_AS,
    MOVE_NO_TXT,
    MOVE_LOW_RES,
    MOVE_LINE,
    MOVE_LOW_LINE,
    MOVE_POINT,
    MOVE_LOW_POINT,
    MOVE_BBOX,
    MOVE_LOW_VOLUME,

    DRAW_STYLE_NUM // specify the length
};

// list of the toggle buttons in the popumenu
enum popupToggles
{
    HEADLIGHT_WIDGET = 0, // very convenient to start at 0
    VIEWING_WIDGET,
    DECORATION_WIDGET,

    POPUP_TOGGLE_NUM // specify the length
};

/* Uwe Woessner (Added TRACK_PUSH)*/
// list of custom push buttons
enum ViewerPushButtons
{
    PICK_PUSH,
    VIEW_PUSH,
    TRACK_PUSH,
    HELP_PUSH,
    HOME_PUSH,
    SET_HOME_PUSH,
    VIEW_ALL_PUSH,
    SEEK_PUSH,

    PUSH_NUM
};

enum ZoomSliderVars
{
    ZOOM_LABEL,
    ZOOM_SLIDER,
    ZOOM_FIELD,
    ZOOM_RANGE_LAB1,
    ZOOM_RANGE_FIELD1,
    ZOOM_RANGE_LAB2,
    ZOOM_RANGE_FIELD2,

    ZOOM_NUM
};

/*
 * Macros
 */

#define TOGGLE_ON(BUTTON) \
    XmToggleButtonSetState((Widget)BUTTON, True, False)
#define TOGGLE_OFF(BUTTON) \
    XmToggleButtonSetState((Widget)BUTTON, False, False)

static const char *thisClassName = "InvFullViewer";
static const char *stereoErrorTitle = "Stereo Error Dialog";
static const char *stereoError = "Stereo Viewing can't be set on this machine.";

////////////////////////////////////////////////////////////////////////
//
//  Constructor.
//
// Use: protected

InvFullViewer::InvFullViewer(Widget parent,
                             const char *name,
                             SbBool buildInsideParent,
                             InvFullViewer::BuildFlag buildFlag,
                             InvViewer::Type t,
                             SbBool buildNow)
    : InvViewer(parent,
                name,
                buildInsideParent,
                t,
                FALSE)
    , // buildNow
    renderer_(NULL)
//
////////////////////////////////////////////////////////////////////////
{
    int i;
    numberFormatForm_ = NULL;
    numberFormat_ = NULL;
    useNumberFormat_ = false;
    setClassName(thisClassName);

    setSize(SbVec2s(500, 390)); // default size

    firstBuild = TRUE; // used to get pref sheet resources only once

    // init decoration vars
    decorationFlag = (buildFlag & BUILD_DECORATION) != 0;
    mgrWidget = NULL;
    leftTrimForm = bottomTrimForm = rightTrimForm = NULL;
    rightWheelStr = bottomWheelStr = leftWheelStr = NULL;
    rightWheelLabel = bottomWheelLabel = leftWheelLabel = NULL;
    zoomSldRange.setValue(1, 140);

    // init pref sheet vars
    prefSheetShellWidget = NULL;
    prefSheetStr = NULL;
    zoomWidgets = new Widget[ZOOM_NUM];
    for (i = 0; i < ZOOM_NUM; i++)
        zoomWidgets[i] = NULL;

    // init popup menu vars
    popupWidget = NULL;
    popupEnabled = (buildFlag & BUILD_POPUP) != 0;
    popupTitle = NULL;
    popupToggleWidgets = new Widget[POPUP_TOGGLE_NUM];
    for (i = 0; i < POPUP_TOGGLE_NUM; i++)
        popupToggleWidgets[i] = NULL;
    drawStyleWidgets = new Widget[DRAW_STYLE_NUM];
    for (i = 0; i < DRAW_STYLE_NUM; i++)
        drawStyleWidgets[i] = NULL;
    bufferStyleWidgets = new Widget[3];
    for (i = 0; i < 3; i++)
        bufferStyleWidgets[i] = NULL;

    // init buttons stuff
    for (i = 0; i < PUSH_NUM; i++)
        buttonList[i] = NULL;
    viewerButtonWidgets = new SbPList(PUSH_NUM);
    appButtonForm = NULL;
    appButtonList = new SbPList;

    // Build the widget tree, and let SoXtComponent know about our base widget.
    if (buildNow)
    {
        Widget w = buildWidget(getParentWidget());
        setBaseWidget(w);
    }

    // Uwe Woessner
    // init tracking variables
    trackingWheelVal = 0.1;
    trackingDevice = 2;
    trackingFlag = FALSE; // (Tracking default off)
    trackingSensor = new SoTimerSensor(trackingSensorCB, this);
    trackingSensor->setInterval(trackingWheelVal);

    samplingWheelVal = 1.0; // Initialize volume rendering quality
}

////////////////////////////////////////////////////////////////////////
//
//    Destructor.
//
// Use: protected

InvFullViewer::~InvFullViewer()
//
////////////////////////////////////////////////////////////////////////
{
    // unregister the widget
    unregisterWidget(mgrWidget);

    // delete sensor

    delete trackingSensor; // Uwe Woessner

    // delete decoration stuff
    if (rightWheelStr != NULL)
        delete[] rightWheelStr;
    if (bottomWheelStr != NULL)
        delete[] bottomWheelStr;
    if (leftWheelStr != NULL)
        delete[] leftWheelStr;

    // delete  popup stuff
    if (popupTitle != NULL)
        delete[] popupTitle;
    delete[] popupToggleWidgets;
    delete[] drawStyleWidgets;
    delete[] bufferStyleWidgets;

    // delete push button stuff
    for (int i = 0; i < PUSH_NUM; i++)
        delete buttonList[i];
    delete viewerButtonWidgets;
    delete appButtonList;

    // delete pref sheet stuff
    delete[] zoomWidgets;
    if (prefSheetStr != NULL)
        delete[] prefSheetStr;
    if (prefSheetShellWidget != NULL)
    {
        XtRemoveCallback(prefSheetShellWidget, XtNdestroyCallback,
                         (XtCallbackProc)InvFullViewer::prefSheetDestroyCB,
                         (XtPointer) this);
        XtDestroyWidget(prefSheetShellWidget);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//   	Send the drawing style to other renderer.
//
// Use: private
void InvFullViewer::sendViewing(int onoroff)
//
// Dirk Rantzau
//
////////////////////////////////////////////////////////////////////////
{
    char message[20];

    if (rm_isMaster() && rm_isSynced())
    {
        sprintf(message, "%d", onoroff);
        rm_sendViewing(message);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//   	Send the decoration mode to other renderer.
//
// Use: private
void InvFullViewer::sendDecoration(int onoroff)
//
// Dirk Rantzau
//
////////////////////////////////////////////////////////////////////////
{
    char message[20];

    if (rm_isMaster() && rm_isSynced())
    {
        sprintf(message, "%d", onoroff);
        rm_sendDecoration(message);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    redefine this to also update the UI
//
// Use: virtual public
void
InvFullViewer::setViewing(SbBool flag)
//
// Dirk Rantzau
//
////////////////////////////////////////////////////////////////////////
{
    if (flag == viewingFlag)
        return;

    // call the base class
    InvViewer::setViewing(flag);

    // update the push buttons
    if (buttonList[VIEW_PUSH])
        buttonList[VIEW_PUSH]->select(viewingFlag);
    if (buttonList[PICK_PUSH])
        buttonList[PICK_PUSH]->select(!viewingFlag);

    // update the popup menu entry
    if (popupToggleWidgets[VIEWING_WIDGET])
        XmToggleButtonSetState(popupToggleWidgets[VIEWING_WIDGET],
                               viewingFlag, False);

    InvFullViewer::sendViewing(flag);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    redefine this to also update the popup menu
//
// Use: virtual public
void
InvFullViewer::setHeadlight(SbBool flag)
//
////////////////////////////////////////////////////////////////////////
{
    if (flag == isHeadlight())
        return;

    // call base class routine
    InvViewer::setHeadlight(flag);

    // update the popup menu entry
    if (popupToggleWidgets[HEADLIGHT_WIDGET])
        XmToggleButtonSetState(popupToggleWidgets[HEADLIGHT_WIDGET],
                               isHeadlight(), False);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Sets the tracking mode.
//
// Uwe Woessner
//
// Use: public
void
InvFullViewer::tougleTracking(SbBool flag)
//
////////////////////////////////////////////////////////////////////////
{
    if (flag == trackingFlag)
        return;
    if (trackingFlag)
    {
        trackingFlag = FALSE;
        /// 2 lines commented out for release 4.5 (DRA) -> tracking support disabled
        ///	closeTracking();
        ///	trackingSensor->unschedule();
    }
    else if (initTracking())
    {
        trackingFlag = TRUE;
        /// 1 line commented out for release 4.5 (DRA) -> tracking support disabled
        ///	trackingSensor->schedule();
    }
    buttonList[TRACK_PUSH]->select(trackingFlag);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    redefine this to also update the popup menu
//
// Use: virtual public
void
InvFullViewer::setDrawStyle(InvViewer::DrawType type,
                            InvViewer::DrawStyle style)
//
////////////////////////////////////////////////////////////////////////
{
    // call base class routine
    InvViewer::setDrawStyle(type, style);

    // update the popup menu entries
    if (drawStyleWidgets[0])
    {
        for (int i = 0; i < DRAW_STYLE_NUM; i++)
            TOGGLE_OFF(drawStyleWidgets[i]);
        switch (getDrawStyle(InvViewer::STILL))
        {
        case InvViewer::VIEW_AS_IS:
            TOGGLE_ON(drawStyleWidgets[AS_IS]);
            break;
        case InvViewer::VIEW_HIDDEN_LINE:
            TOGGLE_ON(drawStyleWidgets[HIDDEN_LINE]);
            break;
        case InvViewer::VIEW_MESH:
            TOGGLE_ON(drawStyleWidgets[MESH]);
            break;
        case InvViewer::VIEW_NO_TEXTURE:
            TOGGLE_ON(drawStyleWidgets[NO_TXT]);
            break;
        case InvViewer::VIEW_LOW_COMPLEXITY:
            TOGGLE_ON(drawStyleWidgets[LOW_RES]);
            break;
        case InvViewer::VIEW_LINE:
            TOGGLE_ON(drawStyleWidgets[LINE]);
            break;
        case InvViewer::VIEW_POINT:
            TOGGLE_ON(drawStyleWidgets[POINT]);
            break;
        case InvViewer::VIEW_BBOX:
            TOGGLE_ON(drawStyleWidgets[BBOX]);
            break;
        case InvViewer::VIEW_LOW_VOLUME:
            TOGGLE_ON(drawStyleWidgets[LOW_VOLUME]);
            break;
        default:
            break;
        }
        switch (getDrawStyle(InvViewer::INTERACTIVE))
        {
        case InvViewer::VIEW_SAME_AS_STILL:
            TOGGLE_ON(drawStyleWidgets[MOVE_SAME_AS]);
            break;
        case InvViewer::VIEW_LOW_VOLUME:
            TOGGLE_ON(drawStyleWidgets[LOW_VOLUME]);
            break;
        case InvViewer::VIEW_NO_TEXTURE:
            TOGGLE_ON(drawStyleWidgets[MOVE_NO_TXT]);
            break;
        case InvViewer::VIEW_LOW_COMPLEXITY:
            TOGGLE_ON(drawStyleWidgets[MOVE_LOW_RES]);
            break;
        case InvViewer::VIEW_LINE:
            TOGGLE_ON(drawStyleWidgets[MOVE_LINE]);
            break;
        case InvViewer::VIEW_LOW_RES_LINE:
            TOGGLE_ON(drawStyleWidgets[MOVE_LOW_LINE]);
            break;
        case InvViewer::VIEW_POINT:
            TOGGLE_ON(drawStyleWidgets[MOVE_POINT]);
            break;
        case InvViewer::VIEW_LOW_RES_POINT:
            TOGGLE_ON(drawStyleWidgets[MOVE_LOW_POINT]);
            break;
        case InvViewer::VIEW_BBOX:
            TOGGLE_ON(drawStyleWidgets[MOVE_BBOX]);
            break;
        default:
            break;
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    redefine this to also update the popup menu
//
// Use: virtual public
void
InvFullViewer::setBufferingType(InvViewer::BufferType type)
//
////////////////////////////////////////////////////////////////////////
{
    // call base class routine
    InvViewer::setBufferingType(type);

    // update the popup menu entries
    if (bufferStyleWidgets[0])
    {
        for (int i = 0; i < 3; i++)
            TOGGLE_OFF(bufferStyleWidgets[i]);
        TOGGLE_ON(bufferStyleWidgets[getBufferingType()]);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Sets the camera to use (done in base class) and makes sure to
// detach and re-attach() the zoom slider sensor.
//
// Use: virtual public
void
InvFullViewer::setCamera(SoCamera *newCamera)
//
////////////////////////////////////////////////////////////////////////
{
    // call base class routine
    InvViewer::setCamera(newCamera);

    // check if the zoom slider needs to be enabled
    if (zoomWidgets[ZOOM_SLIDER] != NULL)
    {

        SbBool enable = camera != NULL && camera->isOfType(SoPerspectiveCamera::getClassTypeId());
        for (int i = 0; i < ZOOM_NUM; i++)
            XtVaSetValues(zoomWidgets[i], XmNsensitive, enable, NULL);

        // update the UI if enabled
        if (enable)
        {
            float zoom = getCameraZoom();
            setZoomSliderPosition(zoom);
            setZoomFieldString(zoom);
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    shows/hides the decoration.
//
// Use: public
void
InvFullViewer::setDecoration(SbBool flag)
//
////////////////////////////////////////////////////////////////////////
{
    if (mgrWidget == NULL || flag == decorationFlag)
    {
        decorationFlag = flag;
        return;
    }

    int n;
    Arg args[12];

    decorationFlag = flag;

    if (decorationFlag)
    {

        // set renderArea offset
        n = 0;
        XtSetArg(args[n], XmNbottomOffset, DECOR_SIZE);
        n++;
        XtSetArg(args[n], XmNleftOffset, DECOR_SIZE);
        n++;
        XtSetArg(args[n], XmNrightOffset, DECOR_SIZE);
        n++;
        XtSetValues(raWidget, args, n);

        // check if decoration needs to be built
        // ??? just need to check one the decoration form widget ?
        if (leftTrimForm == NULL)
            buildDecoration(mgrWidget);

        // show the decoration
        XtManageChild(leftTrimForm);
        XtManageChild(bottomTrimForm);
        XtManageChild(rightTrimForm);

        if (renderer_)
            renderer_->manageObjs();
    }
    else
    {

        // hide the decoration, making sure it was first built
        // (just need to check one the decoration form widget)
        if (leftTrimForm != NULL)
        {
            XtUnmanageChild(leftTrimForm);
            XtUnmanageChild(bottomTrimForm);
            XtUnmanageChild(rightTrimForm);

            if (renderer_)
                renderer_->unmanageObjs();
        }

        // set renderArea offset
        n = 0;
        XtSetArg(args[n], XmNbottomOffset, 0);
        n++;
        XtSetArg(args[n], XmNleftOffset, 0);
        n++;
        XtSetArg(args[n], XmNrightOffset, 0);
        n++;
        XtSetValues(raWidget, args, n);
    }
    // update the popup menu entry
    if (popupToggleWidgets[DECORATION_WIDGET])
        XmToggleButtonSetState(popupToggleWidgets[DECORATION_WIDGET], decorationFlag, False);

    InvFullViewer::sendDecoration(flag);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    enables/disables the popup menu.
//
// Use: virtual public
void
InvFullViewer::setPopupMenuEnabled(SbBool flag)
//
////////////////////////////////////////////////////////////////////////
{
    // chech for trivial return
    if (mgrWidget == NULL || flag == popupEnabled)
    {
        popupEnabled = flag;
        return;
    }

    popupEnabled = flag;

    if (popupEnabled)
        buildPopupMenu();
    else
        destroyPopupMenu();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Add a button to the end of the application list
//
// Use: public
void
InvFullViewer::addAppPushButton(Widget newButton)
//
////////////////////////////////////////////////////////////////////////
{
    // add the button to the end of the list
    appButtonList->append(newButton);

    // redo the layout again
    doAppButtonLayout(appButtonList->getLength() - 1);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Insert a button in the application list at the given index
//
// Use: public
void
InvFullViewer::insertAppPushButton(Widget newButton, int index)
//
////////////////////////////////////////////////////////////////////////
{
    // add the button at the specified index
    appButtonList->insert(newButton, index);

    // redo the layout again
    doAppButtonLayout(appButtonList->find(newButton));
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Remove a button from the application list
//
// Use: public
void
InvFullViewer::removeAppPushButton(Widget oldButton)
//
////////////////////////////////////////////////////////////////////////
{
    // find the index where the button is
    int index = appButtonList->find(oldButton);
    if (index == -1)
        return;

    // remove from the list and redo the layout
    int lastIndex = appButtonList->getLength() - 1;
    appButtonList->remove(index);
    if (index != lastIndex)
        doAppButtonLayout(index);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Redefines this to also hide the preference sheet.
//
// Use: virtual public
void
InvFullViewer::hide()
//
////////////////////////////////////////////////////////////////////////
{
    // call the parent class
    InvViewer::hide();

    // destroy the pref sheet if it is currently on the screen
    if (prefSheetShellWidget != NULL)
        XtDestroyWidget(prefSheetShellWidget);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Sets the popup menu string
//
// use: protected
void
InvFullViewer::setPopupMenuString(const char *str)
//
////////////////////////////////////////////////////////////////////////
{
    if (popupTitle != NULL)
        delete[] popupTitle;
    popupTitle = (str != NULL) ? STRDUP(str) : NULL;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Sets the decoration bottom wheel string
//
// use: protected
void
InvFullViewer::setBottomWheelString(const char *str)
//
////////////////////////////////////////////////////////////////////////
{
    if (bottomWheelStr != NULL)
        delete[] bottomWheelStr;
    bottomWheelStr = (str != NULL) ? STRDUP(str) : NULL;
    if (bottomWheelStr != NULL && bottomWheelLabel != NULL)
    {
        Arg args[1];
        XmString xmstr = XmStringCreate(bottomWheelStr, (XmStringCharSet)XmSTRING_DEFAULT_CHARSET);
        XtSetArg(args[0], XmNlabelString, xmstr);
        XtSetValues(bottomWheelLabel, args, 1);
        XmStringFree(xmstr);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Sets the decoration left wheel string
//
// use: protected
void
InvFullViewer::setLeftWheelString(const char *str)
//
////////////////////////////////////////////////////////////////////////
{
    if (leftWheelStr != NULL)
        delete[] leftWheelStr;
    leftWheelStr = (str != NULL) ? STRDUP(str) : NULL;
    if (leftWheelStr != NULL && leftWheelLabel != NULL)
    {
        Arg args[1];
        XmString xmstr = XmStringCreate(leftWheelStr, (XmStringCharSet)XmSTRING_DEFAULT_CHARSET);
        XtSetArg(args[0], XmNlabelString, xmstr);
        XtSetValues(leftWheelLabel, args, 1);
        XmStringFree(xmstr);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Sets the decoration right wheel string
//
// use: protected
void
InvFullViewer::setRightWheelString(const char *str)
//
////////////////////////////////////////////////////////////////////////
{
    if (rightWheelStr != NULL)
        delete[] rightWheelStr;
    rightWheelStr = (str != NULL) ? STRDUP(str) : NULL;
    if (rightWheelStr != NULL && rightWheelLabel != NULL)
    {
        Arg args[1];
        XmString xmstr = XmStringCreate(rightWheelStr, (XmStringCharSet)XmSTRING_DEFAULT_CHARSET);
        XtSetArg(args[0], XmNlabelString, xmstr);
        XtSetValues(rightWheelLabel, args, 1);
        XmStringFree(xmstr);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Builds the viewer popup menu.
//
// Use: virtual protected

void
InvFullViewer::buildPopupMenu()
//
////////////////////////////////////////////////////////////////////////
{
    int n, butnum = 0;
    Arg args[12];
    Widget buttons[15];

    // create popup and register routine to pop the menu
    n = 0;
#ifdef MENUS_IN_POPUP
    SoXt::getPopupArgs(XtDisplay(mgrWidget), 0, args, &n);
#endif
    popupWidget = XmCreatePopupMenu(mgrWidget, (char *)"menu", args, n);

#ifdef MENUS_IN_POPUP
    // register callbacks to load/unload the pulldown colormap when the
    // pulldown menu is posted.
    SoXt::registerColormapLoad(popupWidget, SoXt::getShellWidget(mgrWidget));
#endif

    XtAddEventHandler(mgrWidget, ButtonPressMask, FALSE,
                      (XtEventHandler)&InvFullViewer::popMenuCallback, (XtPointer) this);

    // make a title label for the popup menu
    if (popupTitle == NULL)
        popupTitle = STRDUP("Viewer Menu");
    buttons[butnum++] = XtCreateWidget(popupTitle, xmLabelGadgetClass, popupWidget, NULL, 0);
    buttons[butnum++] = XtCreateWidget("sep", xmSeparatorGadgetClass, popupWidget, NULL, 0);

    //
    // create the submenus
    //
    buttons[butnum++] = buildFunctionsSubmenu(popupWidget);
    buttons[butnum++] = buildDrawStyleSubmenu(popupWidget);

#ifdef _BAW
    buttons[butnum++] = buildCursorSubmenu(popupWidget);
#endif

    //
    // add the toggle buttons
    //
    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;

#define ADD_TOGGLE(NAME, W, ID, STATE)                                                                                     \
    XtSetArg(args[n], XmNset, STATE);                                                                                      \
    buttons[butnum++] = popupToggleWidgets[W] = XtCreateWidget(NAME, xmToggleButtonGadgetClass, popupWidget, args, n + 1); \
    XtAddCallback(popupToggleWidgets[W], XmNvalueChangedCallback,                                                          \
                  (XtCallbackProc)InvFullViewer::menuPick, (XtPointer)ID);

    ADD_TOGGLE("Viewing", VIEWING_WIDGET, VIEWING, isViewing())
    ADD_TOGGLE("Decoration", DECORATION_WIDGET, DECORATION, isDecoration())
    ADD_TOGGLE("Headlight", HEADLIGHT_WIDGET, HEADLIGHT, isHeadlight())
#undef ADD_TOGGLE

//
// add some more regular buttons
//
#define ADD_ENTRY(NAME, ID)                                                                                      \
    buttons[butnum] = XtCreateWidget(NAME, xmPushButtonGadgetClass, popupWidget, args, n);                       \
    XtAddCallback(buttons[butnum], XmNactivateCallback, (XtCallbackProc)InvFullViewer::menuPick, (XtPointer)ID); \
    butnum++;

    ADD_ENTRY("Preferences...", PREF)
    ADD_ENTRY("new Annotation", ANNOTATION_CREATE)
    ADD_ENTRY("edit Annotation", ANNOTATION_EDIT)

    XtSetArg(args[0], XmNsensitive, False);
    XtSetValues(buttons[butnum - 1], args, 1);
    editAnnoWidget = buttons[butnum - 1];

    ADD_ENTRY("delete Annotation", ANNOTATION_DELETE)

    XtSetArg(args[0], XmNsensitive, False);
    XtSetValues(buttons[butnum - 1], args, 1);
    deleteAnnoWidget = buttons[butnum - 1];

    buttons[butnum++] = XtCreateWidget("sep2", xmSeparatorGadgetClass, popupWidget, NULL, 0);
    ADD_ENTRY("MasterRequest", MASTER_REQUEST)
    MasterRequest = buttons[butnum - 1];

#undef ADD_ENTRY

    // manage children
    XtManageChildren(buttons, butnum);
}

//
// Build Cursor submenu
//
Widget
InvFullViewer::buildCursorSubmenu(Widget popup)
//
////////////////////////////////////////////////////////////////////////
{
    int n, butnum = 0;
    Arg args[12];
    Widget buttons[15];

    // create a cascade menu entry which will bring the submenu
    Widget cascade = XtCreateWidget("Cursor", xmCascadeButtonGadgetClass,
                                    popup, NULL, 0);

    // create the submenu widget
    n = 0;
#ifdef MENUS_IN_POPUP
    SoXt::getPopupArgs(XtDisplay(popup), 0, args, &n);
#endif
    Widget submenu = XmCreatePulldownMenu(popup, (char *)"cursorss", args, n);

    XtSetArg(args[0], XmNsubMenuId, submenu);
    XtSetValues(cascade, args, 1);

    //
    // create the menu entries
    //
    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;

#define ADD_ENTRY(NAME, ID)                                                                                      \
    buttons[butnum] = XtCreateWidget(NAME, xmPushButtonGadgetClass, submenu, args, n);                           \
    XtAddCallback(buttons[butnum], XmNactivateCallback, (XtCallbackProc)InvFullViewer::menuPick, (XtPointer)ID); \
    butnum++;

    ADD_ENTRY("Normal", NORMAL_CURSOR)
    XtSetArg(args[n], XmNsensitive, False);
    n++;
    XtSetValues(buttons[butnum - 1], args, 1);
    normCursorWidget = buttons[butnum - 1];

    ADD_ENTRY("Presenation", PRESENTATION_CURSOR);
    XtSetArg(args[n], XmNsensitive, True);
    n++;
    XtSetValues(buttons[butnum - 1], args, 1);
    presCursorWidget = buttons[butnum - 1];

#undef ADD_ENTRY

    // manage children
    XtManageChildren(buttons, butnum);

    return cascade;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Builds function submenu - this include all of the viewer push
//  buttons plus any useful entries.
//
// Use: protected

Widget
InvFullViewer::buildFunctionsSubmenu(Widget popup)
//
////////////////////////////////////////////////////////////////////////
{
    int n, butnum = 0;
    Arg args[12];
    Widget buttons[15];

    // create a cascade menu entry which will bring the submenu
    Widget cascade = XtCreateWidget("Functions", xmCascadeButtonGadgetClass,
                                    popup, NULL, 0);

    // create the submenu widget
    n = 0;
#ifdef MENUS_IN_POPUP
    SoXt::getPopupArgs(XtDisplay(popup), 0, args, &n);
#endif
    Widget submenu = XmCreatePulldownMenu(popup, (char *)"functions", args, n);

    XtSetArg(args[0], XmNsubMenuId, submenu);
    XtSetValues(cascade, args, 1);

    //
    // create the menu entries
    //
    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;

#define ADD_ENTRY(NAME, ID)                                                                                      \
    buttons[butnum] = XtCreateWidget(NAME, xmPushButtonGadgetClass, submenu, args, n);                           \
    XtAddCallback(buttons[butnum], XmNactivateCallback, (XtCallbackProc)InvFullViewer::menuPick, (XtPointer)ID); \
    butnum++;

    //ADD_ENTRY("Help", HELP)
    ADD_ENTRY("Home", HOME)
    ADD_ENTRY("Set Home", SET_HOME)
    ADD_ENTRY("View All", VIEW_ALL)
    ADD_ENTRY("Seek", SEEK)

    buttons[butnum++] = XtCreateWidget("sep", xmSeparatorGadgetClass, submenu, NULL, 0);

    ADD_ENTRY("Copy View", COPY_VIEW)
//
// changed 11.04.94 since it is not acceptable to paste anything
// in the PAGEIN viewer
// D. Rantzau
// ( commented out by /// )
///    ADD_ENTRY("Paste View", PASTE_VIEW)
#undef ADD_ENTRY

    // manage children
    XtManageChildren(buttons, butnum);

    return cascade;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Builds drawing style submenu
//
// Use: protected

Widget
InvFullViewer::buildDrawStyleSubmenu(Widget popup)
//
////////////////////////////////////////////////////////////////////////
{
    int n, butnum = 0;
    Arg args[12];
    Widget buttons[30];

    // create a cascade menu entry which will bring the submenu
    Widget cascade = XtCreateWidget("Draw Style", xmCascadeButtonGadgetClass,
                                    popup, NULL, 0);

    // create the submenu widget
    n = 0;
#ifdef MENUS_IN_POPUP
    SoXt::getPopupArgs(XtDisplay(popup), 0, args, &n);
#endif
    Widget submenu = XmCreatePulldownMenu(popup, (char *)"draw style", args, n);

    XtSetArg(args[0], XmNsubMenuId, submenu);
    XtSetValues(cascade, args, 1);

    //
    // create the first part of this sub menu
    //
    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;
    XtSetArg(args[n], XmNindicatorType, XmONE_OF_MANY);
    n++;

#define ADD_ENTRY(NAME, ID, STATE)                                                                                    \
    XtSetArg(args[n], XmNset, STATE);                                                                                 \
    buttons[butnum++] = drawStyleWidgets[ID] = XtCreateWidget(NAME, xmToggleButtonGadgetClass, submenu, args, n + 1); \
    XtAddCallback(drawStyleWidgets[ID], XmNvalueChangedCallback,                                                      \
                  (XtCallbackProc)InvFullViewer::drawStyleMenuPick, (XtPointer)ID);

    int drawType = getDrawStyle(InvViewer::STILL);
    ADD_ENTRY("as is", AS_IS, drawType == InvViewer::VIEW_AS_IS)
    ADD_ENTRY("hidden line", HIDDEN_LINE, drawType == InvViewer::VIEW_HIDDEN_LINE)
    ADD_ENTRY("discrete mesh", MESH, drawType == InvViewer::VIEW_MESH)
    ADD_ENTRY("no texture", NO_TXT, drawType == InvViewer::VIEW_NO_TEXTURE)
    ADD_ENTRY("low volume", LOW_VOLUME, drawType == InvViewer::VIEW_LOW_VOLUME)
    ADD_ENTRY("low resolution", LOW_RES, drawType == InvViewer::VIEW_LOW_COMPLEXITY)
    ADD_ENTRY("wireframe", LINE, drawType == InvViewer::VIEW_LINE)
    ADD_ENTRY("points", POINT, drawType == InvViewer::VIEW_POINT)
    ADD_ENTRY("bounding box (no depth)", BBOX, drawType == InvViewer::VIEW_BBOX)

    buttons[butnum++] = XtCreateWidget("sep", xmSeparatorGadgetClass, submenu, NULL, 0);

    drawType = getDrawStyle(InvViewer::INTERACTIVE);
    ADD_ENTRY("move same as still", MOVE_SAME_AS, drawType == InvViewer::VIEW_SAME_AS_STILL)
    ADD_ENTRY("move low volume", MOVE_LOW_VOLUME, drawType == InvViewer::VIEW_LOW_VOLUME)
    ADD_ENTRY("move no texture", MOVE_NO_TXT, drawType == InvViewer::VIEW_NO_TEXTURE)
    ADD_ENTRY("move low res", MOVE_LOW_RES, drawType == InvViewer::VIEW_LOW_COMPLEXITY)
    ADD_ENTRY("move wireframe", MOVE_LINE, drawType == InvViewer::VIEW_LINE)
    ADD_ENTRY("move low res wireframe (no depth)", MOVE_LOW_LINE, drawType == InvViewer::VIEW_LOW_RES_LINE)
    ADD_ENTRY("move points", MOVE_POINT, drawType == InvViewer::VIEW_POINT)
    ADD_ENTRY("move low res points (no depth)", MOVE_LOW_POINT, drawType == InvViewer::VIEW_LOW_RES_POINT)
    ADD_ENTRY("move bounding box (no depth)", MOVE_BBOX, drawType == InvViewer::VIEW_BBOX)
#undef ADD_ENTRY

    buttons[butnum++] = XtCreateWidget("sep", xmSeparatorGadgetClass, submenu, NULL, 0);

//
// create the second part of this sub menu
//
#define ADD_ENTRY(NAME, ID)                                                                                             \
    XtSetArg(args[n], XmNset, bufType == ID);                                                                           \
    buttons[butnum++] = bufferStyleWidgets[ID] = XtCreateWidget(NAME, xmToggleButtonGadgetClass, submenu, args, n + 1); \
    XtAddCallback(bufferStyleWidgets[ID], XmNvalueChangedCallback,                                                      \
                  (XtCallbackProc)InvFullViewer::bufferStyleMenuPick, (XtPointer)ID);

    int bufType = getBufferingType();
    ADD_ENTRY("single buffer", InvViewer::BUFFER_SINGLE)
    ADD_ENTRY("double buffer", InvViewer::BUFFER_DOUBLE)
    ADD_ENTRY("interactive buffer", InvViewer::BUFFER_INTERACTIVE)
#undef ADD_ENTRY

    // manage children
    XtManageChildren(buttons, butnum);

    return cascade;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Deletes the viewer popup menu.
//
// Use: protected

void
InvFullViewer::destroyPopupMenu()
//
////////////////////////////////////////////////////////////////////////
{

    cerr << "InvFullViewer::destroyPopupMenu() called" << endl;

    int i;

    // remove callback to pop it up
    XtRemoveEventHandler(mgrWidget, ButtonPressMask, FALSE,
                         (XtEventHandler)&InvFullViewer::popMenuCallback, (XtPointer) this);

    // destroy the popup menu and reset the menu variables...
    XtDestroyWidget(popupWidget);
    popupWidget = NULL;
    for (i = 0; i < POPUP_TOGGLE_NUM; i++)
        popupToggleWidgets[i] = NULL;
    for (i = 0; i < DRAW_STYLE_NUM; i++)
        drawStyleWidgets[i] = NULL;
    for (i = 0; i < 3; i++)
        bufferStyleWidgets[i] = NULL;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Get X resources for the widget.
//
// Use: private
void
InvFullViewer::getResources(SoXtResource *xr)
//
////////////////////////////////////////////////////////////////////////
{
    // Decoration
    xr->getResource((char *)"decoration", (char *)"Decoration", decorationFlag);

    // Get resources for preference sheet items.
    float val;
    SbBool flag;
    char *str;

    // seek...
    if (xr->getResource((char *)"seekAnimationTime", (char *)"SeekAnimationTime", val))
        setSeekTime(val);
    if (xr->getResource((char *)"seekTo", (char *)"SeekTo", str))
    {
        if (strcasecmp(str, "point") == 0)
            setDetailSeek(TRUE);
        else if (strcasecmp(str, "object") == 0)
            setDetailSeek(FALSE);
    }
    if (xr->getResource((char *)"seekDistanceUsage", (char *)"SeekDistanceUsage", str))
    {
        if (strcasecmp(str, "percentage") == 0)
            seekDistAsPercentage = TRUE;
        else if (strcasecmp(str, "absolute") == 0)
            seekDistAsPercentage = FALSE;
    }

    // zoom slider...
    if (xr->getResource((char *)"zoomMin", (char *)"ZoomMin", val))
        zoomSldRange[0] = val;
    if (xr->getResource((char *)"zoomMax", (char *)"ZoomMax", val))
        zoomSldRange[1] = val;

    // auto clipping planes...
    if (xr->getResource((char *)"autoClipping", (char *)"AutoClipping", flag))
        setAutoClipping(flag);

    // manual clipping planes...
    //??? what if camera is NULL? should we save the values somewhere?
    if (camera != NULL)
    {
        if (xr->getResource((char *)"nearDistance", (char *)"NearDistance", val))
            camera->nearDistance = val;
        if (xr->getResource((char *)"farDistance", (char *)"FarDistance", val))
            camera->farDistance = val;
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Builds the basic Viewer Component widget, complete with
// functionality of a InvFullViewerManip, pop-up menu, sliders, etc.
// Builds all subwidgets, and does layout using motif
//
// Use: protected
Widget
InvFullViewer::buildWidget(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{

    //
    // create a top level form to hold everything together
    //

    Arg args[8];
    int n = 0;

    SbVec2s size = getSize();
    if ((size[0] != 0) && (size[1] != 0))
    {
        XtSetArg(args[n], XtNwidth, size[0]);
        n++;
        XtSetArg(args[n], XtNheight, size[1]);
        n++;
    }

    // ??? don't listen to resize request by children - because the
    // ??? form widget layout will force the size down. This will prevent
    // ??? the RenderArea to pop to 400x400 size (default ) after the user
    // ??? set an explicit smaller size.
    XtSetArg(args[n], XmNresizePolicy, XmRESIZE_NONE);
    n++;

    // Create the root widget and register it with a class name
    mgrWidget = XtCreateWidget(getWidgetName(), xmFormWidgetClass, parent, args, n);
    registerWidget(mgrWidget);

    // Get widget resources
    if (firstBuild)
    {
        SoXtResource xr(mgrWidget);
        getResources(&xr);
        firstBuild = FALSE;
    }

    // build the components
    raWidget = SoXtRenderArea::buildWidget(mgrWidget);
    if (decorationFlag)
        buildDecoration(mgrWidget);

    //
    // Layout
    //
    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetValues(raWidget, args, n);

    // manage children
    decorationFlag = !decorationFlag; // enable routine to be called
    setDecoration(!decorationFlag);
    XtManageChild(raWidget);

    // build the popup menu
    if (popupEnabled)
        buildPopupMenu();

    return mgrWidget;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Builds the Decoration trim (thumbwheel, text, slider, buttons, ..).
//
// Use: virtual protected
void
InvFullViewer::buildDecoration(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    int n;
    Arg args[12];

    // build the trim sides
    leftTrimForm = buildLeftTrim(parent);
    bottomTrimForm = buildBottomTrim(parent);
    rightTrimForm = buildRightTrim(parent);

    //
    // layout
    //

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmNONE);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNheight, DECOR_SIZE);
    n++;
    XtSetValues(bottomTrimForm, args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomOffset, DECOR_SIZE);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmNONE);
    n++;
    XtSetArg(args[n], XmNwidth, DECOR_SIZE);
    n++;
    XtSetValues(leftTrimForm, args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomOffset, DECOR_SIZE);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmNONE);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNwidth, DECOR_SIZE);
    n++;
    XtSetValues(rightTrimForm, args, n);

    // ??? children are managed by setDecoration()
    // ??? which is called after this routine by buildWidget()
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Builds the left thumbwheel
//
// Use: protected
void
InvFullViewer::buildLeftWheel(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    int n;
    Arg args[12];

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
    XtSetArg(args[n], XmNorientation, XmVERTICAL);
    n++;
    leftWheel = SgCreateThumbWheel(parent, (char *)"", args, n);
    XtAddCallback(leftWheel, XmNvalueChangedCallback,
                  (XtCallbackProc)InvFullViewer::leftWheelCB, (XtPointer) this);
    XtAddCallback(leftWheel, XmNdragCallback,
                  (XtCallbackProc)InvFullViewer::leftWheelCB, (XtPointer) this);
    leftWheelVal = 0;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Builds the left trim decoration
//
// Use: virtual protected
Widget
InvFullViewer::buildLeftTrim(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    int n;
    Arg args[12];

    // create a form to hold all the parts
    Widget form = XtCreateWidget("LeftTrimForm", xmFormWidgetClass, parent, NULL, 0);

    // create all the parts
    buildLeftWheel(form);
    Widget butForm = buildAppButtons(form);

    //
    // layout
    //

    n = 0;
    XtSetArg(args[n], XmNrightAttachment, XmNONE);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftOffset, THUMB_SPACE);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmNONE);
    n++;
    XtSetValues(leftWheel, args, n);

    n = 0;
    XtSetArg(args[n], XmNrightAttachment, XmNONE);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, leftWheel);
    n++;
    XtSetValues(butForm, args, n);

    // manage children
    XtManageChild(leftWheel);
    XtManageChild(butForm);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Builds the bottom trim decoration
//
// Use: virtual protected
Widget
InvFullViewer::buildBottomTrim(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    int n;
    Arg args[12];

    // create a form to hold all the parts
    Widget form = XtCreateWidget("BottomTrimForm", xmFormWidgetClass, parent, NULL, 0);

    // create all the parts
    if (rightWheelStr == NULL)
        rightWheelStr = STRDUP("Motion Z");
    rightWheelLabel = XtCreateWidget(rightWheelStr, xmLabelGadgetClass, form, NULL, 0);
    if (bottomWheelStr == NULL)
        bottomWheelStr = STRDUP("Motion X");
    bottomWheelLabel = XtCreateWidget(bottomWheelStr, xmLabelGadgetClass, form, NULL, 0);
    if (leftWheelStr == NULL)
        leftWheelStr = STRDUP("Motion Y");
    leftWheelLabel = XtCreateWidget(leftWheelStr, xmLabelGadgetClass, form, NULL, 0);

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
    bottomWheel = SgCreateThumbWheel(form, (char *)"", args, n);

    XtAddCallback(bottomWheel, XmNvalueChangedCallback,
                  (XtCallbackProc)InvFullViewer::bottomWheelCB, (XtPointer) this);
    XtAddCallback(bottomWheel, XmNdragCallback,
                  (XtCallbackProc)InvFullViewer::bottomWheelCB, (XtPointer) this);

    bottomWheelVal = 0;
    //
    // layout
    //

    // left corner
    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmNONE);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomOffset, 5);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftOffset, THUMB_SPACE);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmNONE);
    n++;
    XtSetValues(leftWheelLabel, args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmNONE);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomOffset, 5);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, leftWheelLabel);
    n++;
    XtSetArg(args[n], XmNleftOffset, LABEL_SEPARATION);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmNONE);
    n++;
    XtSetValues(bottomWheelLabel, args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmNONE);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomOffset, THUMB_SPACE);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, bottomWheelLabel);
    n++;
    XtSetArg(args[n], XmNleftOffset, LABEL_SPACE);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmNONE);
    n++;
    XtSetValues(bottomWheel, args, n);

    // right corner
    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmNONE);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomOffset, 5);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightOffset, THUMB_SPACE);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmNONE);
    n++;
    XtSetValues(rightWheelLabel, args, n);

    // manage children (order important)
    XtManageChild(leftWheelLabel);
    XtManageChild(bottomWheelLabel);
    XtManageChild(bottomWheel);
    XtManageChild(rightWheelLabel);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Builds the right trim decoration
//
// Use: virtual protected
Widget
InvFullViewer::buildRightTrim(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    int n;
    Arg args[12];

    // create a form to hold all the parts
    Widget form = XtCreateWidget("RightTrimForm", xmFormWidgetClass, parent, NULL, 0);

    // create all the parts
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
    XtSetArg(args[n], XmNorientation, XmVERTICAL);
    n++;
    rightWheel = SgCreateThumbWheel(form, (char *)"", args, n);

    XtAddCallback(rightWheel, XmNvalueChangedCallback,
                  (XtCallbackProc)InvFullViewer::rightWheelCB, (XtPointer) this);
    XtAddCallback(rightWheel, XmNdragCallback,
                  (XtCallbackProc)InvFullViewer::rightWheelCB, (XtPointer) this);

    rightWheelVal = 0;

    Widget buttonForm = buildViewerButtons(form);

    //
    // layout
    //

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmNONE);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNrightOffset, THUMB_SPACE);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmNONE);
    n++;
    XtSetValues(rightWheel, args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, rightWheel);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmNONE);
    n++;
    XtSetValues(buttonForm, args, n);

    // manage children
    XtManageChild(rightWheel);
    XtManageChild(buttonForm);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Builds the viewer buttons (all within a form)
//
// Use: protected
Widget
InvFullViewer::buildViewerButtons(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    int i, num;
    Widget form, *list;
    Arg args[12];

    // create a form to hold everything
    form = XtCreateWidget(NULL, xmFormWidgetClass, parent, NULL, 0);

    createViewerButtons(form);

    // get all the button widgets
    num = viewerButtonWidgets->getLength();
    list = new Widget[num];
    for (i = 0; i < num; i++)
        list[i] = (Widget)((*viewerButtonWidgets)[i]);

    //
    // layout
    //
    int n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmNONE);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmNONE);
    n++;

    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    XtSetValues(list[0], args, n + 1);

    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    for (i = 1; i < num; i++)
    {
        XtSetArg(args[n], XmNtopWidget, list[i - 1]);
        XtSetValues(list[i], args, n + 1);
    }

    // manage children
    XtManageChildren(list, num);
    delete[] list;

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	creates the default viewer buttons
//
// Use: virtual protected
void
InvFullViewer::createViewerButtons(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    // allocate the custom buttons
    for (long i = 0; i < PUSH_NUM; i++)
    {
        buttonList[i] = new InvPixmapButton(parent, (i == 0 || i == 1 || i == 2));
        Widget w = buttonList[i]->getWidget();
        XtVaSetValues(w, XmNuserData, this, NULL);
        XtAddCallback(w, XmNactivateCallback,
                      (XtCallbackProc)InvFullViewer::pushButtonCB, (XtPointer)i);

        // add this button to the list...
        viewerButtonWidgets->append(w);
    }

    // set the button images
    buttonList[PICK_PUSH]->setIcon(so_xt_pick_bits, so_xt_icon_width, so_xt_icon_height);
    buttonList[VIEW_PUSH]->setIcon(so_xt_view_bits, so_xt_icon_width, so_xt_icon_height);
    buttonList[TRACK_PUSH]->setIcon(reinterpret_cast<char *>(head_bits), head_width, head_height);
    /* Uwe Woessner (Added TRACK_PUSH)*/
    buttonList[HELP_PUSH]->setIcon(so_xt_help_bits, so_xt_icon_width, so_xt_icon_height);
    buttonList[HOME_PUSH]->setIcon(so_xt_home_bits, so_xt_icon_width, so_xt_icon_height);
    buttonList[SET_HOME_PUSH]->setIcon(so_xt_set_home_bits, so_xt_icon_width, so_xt_icon_height);
    buttonList[VIEW_ALL_PUSH]->setIcon(so_xt_see_all_bits, so_xt_icon_width, so_xt_icon_height);
    buttonList[SEEK_PUSH]->setIcon(so_xt_seek_bits, so_xt_icon_width, so_xt_icon_height);

    // show the pick/view state
    if (isViewing())
        buttonList[VIEW_PUSH]->select(TRUE);
    else
        buttonList[PICK_PUSH]->select(TRUE);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Builds the app buttons form and any putton the application supplied
//
// Use: protected
Widget
InvFullViewer::buildAppButtons(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    // create a form to hold the buttons
    appButtonForm = XtCreateWidget("AppButtForm", xmFormWidgetClass, parent, NULL, 0);

    // build all the buttons
    if (appButtonList->getLength() > 0)
        doAppButtonLayout(0);

    return appButtonForm;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Do the app push button build/layout/show withing the button's
//  form widget. Start at the given index
//
// Use: private
void
InvFullViewer::doAppButtonLayout(int start)
//
////////////////////////////////////////////////////////////////////////
{
    int i, num;
    Arg args[12];
    Widget *widgetList, prevWidget;

    num = appButtonList->getLength() - start;
    widgetList = new Widget[num];

    // build all the buttons
    for (i = 0; i < num; i++)
        widgetList[i] = (Widget)((*appButtonList)[i + start]);

    // unmage any managed widget before the new layout,
    // starting from the end of the list
    for (i = num - 1; i >= 0; i--)
    {
        if (XtIsManaged(widgetList[i]))
            XtUnmanageChild(widgetList[i]);
    }

    if (start != 0)
        prevWidget = (Widget)((*appButtonList)[start - 1]);
    else
        prevWidget = 0;

    //
    // layout
    //
    XtSetArg(args[0], XmNrightAttachment, XmNONE);
    XtSetArg(args[1], XmNleftAttachment, XmATTACH_FORM);
    XtSetArg(args[2], XmNbottomAttachment, XmNONE);

    for (i = 0; i < num; i++)
    {
        if (i == 0 && start == 0)
        {
            XtSetArg(args[3], XmNtopAttachment, XmATTACH_FORM);
            XtSetValues(widgetList[i], args, 4);
        }
        else
        {
            XtSetArg(args[3], XmNtopAttachment, XmATTACH_WIDGET);
            if (i == 0)
                XtSetArg(args[4], XmNtopWidget, prevWidget);
            else
                XtSetArg(args[4], XmNtopWidget, widgetList[i - 1]);
            XtSetValues(widgetList[i], args, 5);
        }
    }

    // manage all children
    XtManageChildren(widgetList, num);
    delete[] widgetList;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	This creates the preference sheet in a separate window. It
//  calls other routines to create the actual content of the sheet.
//
// Use: virtual protected
void
InvFullViewer::createPrefSheet()
//
////////////////////////////////////////////////////////////////////////
{
    // create the preference sheet shell and form widget
    Widget shell, form;
    createPrefSheetShellAndForm(shell, form);

    // create all of the default parts
    Widget widgetList[11];
    int num = 0;
    createDefaultPrefSheetParts(widgetList, num, form);

    layoutPartsAndMapPrefSheet(widgetList, num, form, shell);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Sets the pref sheet string
//
// use: protected
void
InvFullViewer::setPrefSheetString(const char *str)
//
////////////////////////////////////////////////////////////////////////
{
    if (prefSheetStr != NULL)
        delete[] prefSheetStr;
    prefSheetStr = (str != NULL) ? STRDUP(str) : NULL;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	This creates the preference sheet outer shell.
//
// Use: protected
void
InvFullViewer::createPrefSheetShellAndForm(Widget &shell, Widget &form)
//
////////////////////////////////////////////////////////////////////////
{
    Arg args[12];
    int n;

    if (prefSheetStr == NULL)
        prefSheetStr = STRDUP("Viewer Preference Sheet");

    // create a top level shell widget
    n = 0;
    XtSetArg(args[n], XtNtitle, prefSheetStr);
    n++;
    XtSetArg(args[n], XmNiconName, "Pref Sheet");
    n++;
    XtSetArg(args[n], XmNallowShellResize, TRUE);
    n++;
    prefSheetShellWidget = shell = XtCreatePopupShell("preferenceSheet",
                                                      topLevelShellWidgetClass, SoXt::getShellWidget(mgrWidget),
                                                      args, n);

    // create a form to hold all the parts
    n = 0;
    XtSetArg(args[n], XmNmarginHeight, 10);
    n++;
    XtSetArg(args[n], XmNmarginWidth, 10);
    n++;
    form = XtCreateWidget("", xmFormWidgetClass, shell, args, n);

    // register destroy callback to init pref sheet pointers
    XtAddCallback(prefSheetShellWidget, XtNdestroyCallback,
                  (XtCallbackProc)InvFullViewer::prefSheetDestroyCB,
                  (XtPointer) this);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	This simply creates the default parts of the pref sheet.
//
// Use: protected
void
InvFullViewer::createDefaultPrefSheetParts(Widget widgetList[],
                                           int &num, Widget form)
//
////////////////////////////////////////////////////////////////////////
{
    widgetList[num++] = createSeekPrefSheetGuts(form);
    widgetList[num++] = createSeekDistPrefSheetGuts(form);
    widgetList[num++] = createZoomPrefSheetGuts(form);
    widgetList[num++] = createClippingPrefSheetGuts(form);
    widgetList[num++] = createStereoPrefSheetGuts(form);
    widgetList[num++] = createColorsPrefSheetGuts(form);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Given a widget list for the preference sheet and it's lenght
//  lay them out one after the other and manage them all. The dialog
//  is them mapped onto the screen.
//
// Use: protected
void
InvFullViewer::layoutPartsAndMapPrefSheet(Widget widgetList[],
                                          int num, Widget form, Widget shell)
//
////////////////////////////////////////////////////////////////////////
{
    Arg args[12];
    int n;

    // layout
    for (int i = 0; i < num; i++)
    {
        n = 0;
        XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
        n++;
        if (i == 0)
        {
            XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
            n++;
        }
        else
        {
            XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
            n++;
            XtSetArg(args[n], XmNtopWidget, widgetList[i - 1]);
            n++;
            XtSetArg(args[n], XmNtopOffset, 10);
            n++;
        }
        if (i == (num - 1))
        {
            XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
            n++;
        }
        XtSetValues(widgetList[i], args, n);
    }

    XtManageChildren(widgetList, num);

    // pop the pref sheet window on the screen
    XtManageChild(form);
    XtRealizeWidget(shell);
    XMapWindow(XtDisplay(shell), XtWindow(shell));
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	This creates the seek preference sheet stuff.
//
// Use: protected
Widget
InvFullViewer::createSeekPrefSheetGuts(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    Widget widgetList[6];
    Arg args[12];
    int n;

    // create a form to hold verything together
    Widget form = XtCreateWidget("", xmFormWidgetClass,
                                 parent, NULL, 0);

    // create the first line
    widgetList[0] = XtCreateWidget("Seek animation time:",
                                   xmLabelGadgetClass, form, NULL, 0);

    n = 0;
    XtSetArg(args[n], XmNhighlightThickness, 1);
    n++;
    XtSetArg(args[n], XmNcolumns, 5);
    n++;
    char str[10];
    sprintf(str, "%.2f", getSeekTime());
    XtSetArg(args[n], XmNvalue, str);
    n++;
    widgetList[1] = XtCreateWidget("", xmTextWidgetClass,
                                   form, args, n);
    XtAddCallback(widgetList[1], XmNactivateCallback,
                  (XtCallbackProc)InvFullViewer::seekPrefSheetFieldCB, (XtPointer) this);

    widgetList[2] = XtCreateWidget("seconds",
                                   xmLabelGadgetClass, form, NULL, 0);

    // create the second line
    widgetList[3] = XtCreateWidget("Seek to:",
                                   xmLabelGadgetClass, form, NULL, 0);

    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;
    XtSetArg(args[n], XmNindicatorType, XmONE_OF_MANY);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    widgetList[4] = XtCreateWidget("point",
                                   xmToggleButtonGadgetClass, form, args, n);
    widgetList[5] = XtCreateWidget("object",
                                   xmToggleButtonGadgetClass, form, args, n);
    XmToggleButtonSetState(widgetList[4], isDetailSeek(), FALSE);
    XmToggleButtonSetState(widgetList[5], !isDetailSeek(), FALSE);
    XtAddCallback(widgetList[4], XmNvalueChangedCallback,
                  (XtCallbackProc)InvFullViewer::seekPrefSheetToggle1CB,
                  (XtPointer)widgetList[5]);
    XtAddCallback(widgetList[5], XmNvalueChangedCallback,
                  (XtCallbackProc)InvFullViewer::seekPrefSheetToggle2CB,
                  (XtPointer)widgetList[4]);

    // layout
    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNtopOffset, 5);
    n++;
    XtSetValues(widgetList[0], args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, widgetList[0]);
    n++;
    XtSetArg(args[n], XmNleftOffset, 10);
    n++;
    XtSetValues(widgetList[1], args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, widgetList[1]);
    n++;
    XtSetArg(args[n], XmNleftOffset, 5);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, widgetList[0]);
    n++;
    XtSetValues(widgetList[2], args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, widgetList[1]);
    n++;
    XtSetArg(args[n], XmNtopOffset, 10);
    n++;
    XtSetValues(widgetList[3], args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[1], XmNleftWidget, widgetList[3]);
    n++;
    XtSetArg(args[n], XmNleftOffset, 10);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, widgetList[3]);
    n++;
    XtSetArg(args[n], XmNbottomOffset, -2);
    n++;
    XtSetValues(widgetList[4], args, n);
    XtSetArg(args[1], XmNleftWidget, widgetList[4]);
    XtSetValues(widgetList[5], args, n);

    XtManageChildren(widgetList, 6);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	This creates the seek distance setting preference sheet stuff.
//
// Use: protected
Widget
InvFullViewer::createSeekDistPrefSheetGuts(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    Widget text, thumb, label, toggles[2];
    Arg args[12];
    int n;

    // create a form to hold everything together
    Widget form = XtCreateWidget("", xmFormWidgetClass,
                                 parent, NULL, 0);

    // create the first line
    label = XtCreateWidget("Seek distance:",
                           xmLabelGadgetClass, form, NULL, 0);

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
    thumb = SgCreateThumbWheel(form, (char *)"", args, n);

    XtAddCallback(thumb, XmNdragCallback,
                  (XtCallbackProc)InvFullViewer::seekDistWheelCB, (XtPointer) this);
    seekDistWheelVal = 0;

    n = 0;
    char str[15];
    sprintf(str, "%f", seekDistance);
    XtSetArg(args[0], XmNvalue, str);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 1);
    n++;
    XtSetArg(args[n], XmNcolumns, 8);
    n++;
    seekDistField = text = XtCreateWidget("", xmTextWidgetClass,
                                          form, args, n);
    XtAddCallback(text, XmNactivateCallback,
                  (XtCallbackProc)InvFullViewer::seekDistFieldCB,
                  (XtPointer) this);

    // create the second line
    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;
    XtSetArg(args[n], XmNindicatorType, XmONE_OF_MANY);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    toggles[0] = XtCreateWidget("percentage",
                                xmToggleButtonGadgetClass, form, args, n);
    toggles[1] = XtCreateWidget("absolute",
                                xmToggleButtonGadgetClass, form, args, n);

    XmToggleButtonSetState(toggles[0], seekDistAsPercentage, FALSE);
    XmToggleButtonSetState(toggles[1], !seekDistAsPercentage, FALSE);
    XtAddCallback(toggles[0], XmNvalueChangedCallback,
                  (XtCallbackProc)InvFullViewer::seekDistPercPrefSheetToggleCB,
                  (XtPointer)toggles[1]);
    XtAddCallback(toggles[1], XmNvalueChangedCallback,
                  (XtCallbackProc)InvFullViewer::seekDistAbsPrefSheetToggleCB,
                  (XtPointer)toggles[0]);

    // layout
    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNtopOffset, 5);
    n++;
    XtSetValues(label, args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, label);
    n++;
    XtSetArg(args[n], XmNleftOffset, 5);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, label);
    n++;
    XtSetValues(thumb, args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, thumb);
    n++;
    XtSetArg(args[n], XmNleftOffset, 3);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetValues(text, args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftOffset, 30);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, text);
    n++;
    XtSetArg(args[n], XmNtopOffset, 2);
    n++;
    XtSetValues(toggles[0], args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, toggles[0]);
    n++;
    XtSetArg(args[n], XmNleftOffset, 10);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, toggles[0]);
    n++;
    XtSetValues(toggles[1], args, n);

    // manage children
    XtManageChild(label);
    XtManageChild(thumb);
    XtManageChild(text);
    XtManageChildren(toggles, 2);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	This creates the zoom slider preference sheet stuff.
//
// Use: protected
Widget
InvFullViewer::createZoomPrefSheetGuts(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    Arg args[12];
    int n;

    // create a form to hold verything together
    Widget form = XtCreateWidget("ZoomForm", xmFormWidgetClass,
                                 parent, NULL, 0);

    // create all the parts
    zoomWidgets[ZOOM_LABEL] = XtCreateWidget("Camera zoom:",
                                             xmLabelGadgetClass, form, NULL, 0);
    zoomWidgets[ZOOM_RANGE_LAB1] = XtCreateWidget("Zoom slider ranges from:",
                                                  xmLabelGadgetClass, form, NULL, 0);
    zoomWidgets[ZOOM_RANGE_LAB2] = XtCreateWidget("to:",
                                                  xmLabelGadgetClass, form, NULL, 0);

    n = 0;
    XtSetArg(args[n], XmNwidth, 130);
    n++;
    XtSetArg(args[n], XmNminimum, 0);
    n++;
    XtSetArg(args[n], XmNmaximum, 1000);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    XtSetArg(args[n], XmNorientation, XmHORIZONTAL);
    n++;
    zoomWidgets[ZOOM_SLIDER] = XtCreateWidget("ZoomSlider",
                                              xmScaleWidgetClass, form, args, n);

    XtAddCallback(zoomWidgets[ZOOM_SLIDER], XmNvalueChangedCallback,
                  (XtCallbackProc)&InvFullViewer::zoomSliderCB, (XtPointer) this);
    XtAddCallback(zoomWidgets[ZOOM_SLIDER], XmNdragCallback,
                  (XtCallbackProc)&InvFullViewer::zoomSliderCB, (XtPointer) this);

    n = 0;
    char str[15];
    XtSetArg(args[n], XmNhighlightThickness, 1);
    n++;
    XtSetArg(args[n], XmNcolumns, 5);
    n++;

    zoomWidgets[ZOOM_FIELD] = XtCreateWidget("ZoomField",
                                             xmTextWidgetClass, form, args, n);

    sprintf(str, "%.1f", zoomSldRange[0]);
    XtSetArg(args[n], XmNvalue, str);
    zoomWidgets[ZOOM_RANGE_FIELD1] = XtCreateWidget("zoomFrom",
                                                    xmTextWidgetClass, form, args, n + 1);

    sprintf(str, "%.1f", zoomSldRange[1]);
    XtSetArg(args[n], XmNvalue, str);
    zoomWidgets[ZOOM_RANGE_FIELD2] = XtCreateWidget("zoomTo",
                                                    xmTextWidgetClass, form, args, n + 1);

    XtAddCallback(zoomWidgets[ZOOM_FIELD], XmNactivateCallback,
                  (XtCallbackProc)&InvFullViewer::zoomFieldCB,
                  (XtPointer) this);
    XtAddCallback(zoomWidgets[ZOOM_RANGE_FIELD1], XmNactivateCallback,
                  (XtCallbackProc)InvFullViewer::zoomPrefSheetMinFieldCB,
                  (XtPointer) this);
    XtAddCallback(zoomWidgets[ZOOM_RANGE_FIELD2], XmNactivateCallback,
                  (XtCallbackProc)InvFullViewer::zoomPrefSheetMaxFieldCB,
                  (XtPointer) this);

    // layout
    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNtopOffset, 5);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetValues(zoomWidgets[ZOOM_LABEL], args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNtopOffset, 8);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, zoomWidgets[ZOOM_LABEL]);
    n++;
    XtSetArg(args[n], XmNleftOffset, 5);
    n++;
    XtSetValues(zoomWidgets[ZOOM_SLIDER], args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, zoomWidgets[ZOOM_SLIDER]);
    n++;
    XtSetArg(args[n], XmNleftOffset, 5);
    n++;
    XtSetValues(zoomWidgets[ZOOM_FIELD], args, n);

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, zoomWidgets[ZOOM_LABEL]);
    n++;
    XtSetArg(args[n], XmNtopOffset, 15);
    n++;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetValues(zoomWidgets[ZOOM_RANGE_LAB1], args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, zoomWidgets[ZOOM_RANGE_LAB1]);
    n++;
    XtSetArg(args[n], XmNleftOffset, 5);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, zoomWidgets[ZOOM_RANGE_LAB1]);
    n++;
    XtSetArg(args[n], XmNbottomOffset, -5);
    n++;
    XtSetValues(zoomWidgets[ZOOM_RANGE_FIELD1], args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, zoomWidgets[ZOOM_RANGE_FIELD1]);
    n++;
    XtSetArg(args[n], XmNleftOffset, 5);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, zoomWidgets[ZOOM_RANGE_LAB1]);
    n++;
    XtSetValues(zoomWidgets[ZOOM_RANGE_LAB2], args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, zoomWidgets[ZOOM_RANGE_LAB2]);
    n++;
    XtSetArg(args[n], XmNleftOffset, 5);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, zoomWidgets[ZOOM_RANGE_FIELD1]);
    n++;
    XtSetValues(zoomWidgets[ZOOM_RANGE_FIELD2], args, n);

    XtManageChildren(zoomWidgets, ZOOM_NUM);

    //
    // finally update the UI
    //
    float zoom = getCameraZoom();
    setZoomSliderPosition(zoom);
    setZoomFieldString(zoom);
    XtSetArg(args[0], XmNsensitive, (camera != NULL && camera->isOfType(SoPerspectiveCamera::getClassTypeId())));
    for (int i = 0; i < ZOOM_NUM; i++)
        XtSetValues(zoomWidgets[i], args, 1);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	This creates the clipping plane preference sheet stuff.
//
// Use: protected
Widget
InvFullViewer::createClippingPrefSheetGuts(Widget dialog)
//
////////////////////////////////////////////////////////////////////////
{
    Arg args[12];
    int n;

    // create a form to hold everything together
    Widget form = XtCreateWidget("", xmFormWidgetClass, dialog, NULL, 0);

    // create all the parts
    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;
    XtSetArg(args[n], XmNsensitive, camera != NULL);
    n++;
    XtSetArg(args[n], XmNset, isAutoClipping());
    n++;
    XtSetArg(args[n], XmNspacing, 0);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    Widget toggle = XtCreateWidget("",
                                   xmToggleButtonGadgetClass, form, args, n);
    n = 0;
    XtSetArg(args[n], XmNsensitive, camera != NULL);
    n++;
    Widget label = XtCreateWidget("Auto clipping planes",
                                  xmLabelGadgetClass, form, args, n);
    XtAddCallback(toggle, XmNvalueChangedCallback,
                  (XtCallbackProc)InvFullViewer::clipPrefSheetToggleCB,
                  (XtPointer)form);

    if (!isAutoClipping() && camera)
        InvFullViewer::clipPrefSheetToggleCB(toggle, form, NULL);

    // layout
    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, toggle);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, toggle);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, toggle);
    n++;
    XtSetValues(label, args, n);

    // manage children
    XtManageChild(toggle);
    XtManageChild(label);

    return form;
}

Widget
InvFullViewer::createColorsPrefSheetGuts(Widget dialog)
//
////////////////////////////////////////////////////////////////////////
{
    Arg args[12];
    int n;

    // create a form to hold everything together
    Widget form = XtCreateWidget("", xmFormWidgetClass, dialog, NULL, 0);

    // create all the parts
    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;
    XtSetArg(args[n], XmNsensitive, camera != NULL);
    n++;
    XtSetArg(args[n], XmNset, isAutoClipping());
    n++;
    XtSetArg(args[n], XmNspacing, 0);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    Widget toggle = XtCreateWidget("",
                                   xmToggleButtonGadgetClass, form, args, n);
    n = 0;
    XtSetArg(args[n], XmNsensitive, camera != NULL);
    n++;
    Widget label = XtCreateWidget("Use alternative number format for Colormaps",
                                  xmLabelGadgetClass, form, args, n);
    XmToggleButtonSetState(toggle, false, false);
    XtAddCallback(toggle, XmNvalueChangedCallback,
                  (XtCallbackProc)InvFullViewer::colorsPrefSheetToggleCB,
                  (XtPointer)form);

    //    if ( !isAutoClipping() && camera)
    //	InvFullViewer::clipPrefSheetToggleCB(toggle, form, NULL);

    // layout
    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, toggle);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, toggle);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, toggle);
    n++;
    XtSetValues(label, args, n);

    // manage children
    XtManageChild(toggle);
    XtManageChild(label);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	This creates the stereo viewing preference sheet stuff.
//
// Use: protected
Widget
InvFullViewer::createStereoPrefSheetGuts(Widget dialog)
//
////////////////////////////////////////////////////////////////////////
{
    Arg args[12];
    int n;

    // create a form to hold everything together
    Widget form = XtCreateWidget("", xmFormWidgetClass, dialog, NULL, 0);

    // create the toggle
    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;
    XtSetArg(args[n], XmNset, isStereoViewing());
    n++;
    XtSetArg(args[n], XmNspacing, 0);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    Widget toggle = XtCreateWidget("",
                                   xmToggleButtonGadgetClass, form, args, n);
    XtAddCallback(toggle, XmNvalueChangedCallback,
                  (XtCallbackProc)InvFullViewer::stereoPrefSheetToggleCB,
                  (XtPointer)form);

    // toggle text
    stereoLabel = XtCreateWidget("Stereo Viewing",
                                 xmLabelGadgetClass, form, NULL, 0);

    // layout
    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, toggle);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, toggle);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, toggle);
    n++;
    XtSetValues(stereoLabel, args, n);

    // manage children
    XtManageChild(toggle);
    XtManageChild(stereoLabel);

    // create the toggle
    n = 0;
    XtSetArg(args[n], XmNuserData, this);
    n++;
    XtSetArg(args[n], XmNset, isStrippleStereoViewing());
    n++;
    XtSetArg(args[n], XmNspacing, 0);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, toggle);
    n++;
    stippToggle = XtCreateWidget("",
                                 xmToggleButtonGadgetClass, form, args, n);
    XtAddCallback(stippToggle, XmNvalueChangedCallback,
                  (XtCallbackProc)InvFullViewer::stereoPrefSheetToggleCB,
                  (XtPointer)form);

    // toggle text
    stereoLabel = XtCreateWidget("StippleStereo Viewing",
                                 xmLabelGadgetClass, form, NULL, 0);

    // layout
    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNleftWidget, stippToggle);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, stippToggle);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, stippToggle);
    n++;
    XtSetValues(stereoLabel, args, n);

    // manage children
    XtManageChild(stippToggle);
    XtManageChild(stereoLabel);

    // call this routine to bring the additional UI (making it look like
    // the user pressed the toggle).
    stereoWheelForm = NULL;
    if (isStereoViewing())
        InvFullViewer::stereoPrefSheetToggleCB(toggle, form, NULL);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	Creates the viewer speed pref sheet stuff
//
// Use: protected
Widget
InvFullViewer::createSpeedPrefSheetGuts(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    Widget widgetList[3];
    Arg args[12];
    int n;

    // create a form to hold everything together
    Widget form = XtCreateWidget("", xmFormWidgetClass,
                                 parent, NULL, 0);

    // create all the parts
    widgetList[0] = XtCreateWidget("Viewer speed:",
                                   xmLabelGadgetClass, form, NULL, 0);

    n = 0;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    widgetList[1] = XtCreateWidget(" increase ", xmPushButtonGadgetClass,
                                   form, args, n);
    widgetList[2] = XtCreateWidget(" decrease ", xmPushButtonGadgetClass,
                                   form, args, n);
    XtAddCallback(widgetList[1], XmNactivateCallback,
                  (XtCallbackProc)InvFullViewer::speedIncPrefSheetButtonCB,
                  (XtPointer) this);
    XtAddCallback(widgetList[2], XmNactivateCallback,
                  (XtCallbackProc)InvFullViewer::speedDecPrefSheetButtonCB,
                  (XtPointer) this);

    // layout
    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetValues(widgetList[0], args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[1], XmNleftWidget, widgetList[0]);
    n++;
    XtSetArg(args[n], XmNleftOffset, 10);
    n++;
    XtSetValues(widgetList[1], args, n);
    XtSetArg(args[1], XmNleftWidget, widgetList[1]);
    XtSetValues(widgetList[2], args, n);

    XtManageChildren(widgetList, 3);

    return form;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Sets the camera given zoom value (in degree for perspective cameras).
//
// Use: private

void
InvFullViewer::setCameraZoom(float zoom)
//
////////////////////////////////////////////////////////////////////////
{
    if (camera == NULL)
        return;

    if (camera->isOfType(SoPerspectiveCamera::getClassTypeId()))
        ((SoPerspectiveCamera *)camera)->heightAngle = zoom * M_PI / 180.0;
    else if (camera->isOfType(SoOrthographicCamera::getClassTypeId()))
        ((SoOrthographicCamera *)camera)->height = zoom;
#if DEBUG
    else
        SoDebugError::post("InvFullViewer::setCameraZoom",
                           "unknown camera type");
#endif
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Gets the camera current zoom value. The value is returned in degrees
//  for a perspective camera.
//
// Use: private

float
InvFullViewer::getCameraZoom()
//
////////////////////////////////////////////////////////////////////////
{
    if (camera == NULL)
        return 0;

    if (camera->isOfType(SoPerspectiveCamera::getClassTypeId()))
        return ((SoPerspectiveCamera *)camera)->heightAngle.getValue() * 180.0 / M_PI;
    else if (camera->isOfType(SoOrthographicCamera::getClassTypeId()))
        return ((SoOrthographicCamera *)camera)->height.getValue();
    else
    {
#if DEBUG
        SoDebugError::post("InvFullViewer::getCameraZoom",
                           "unknown camera type");
#endif
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Sets the zoom slider position based on the camera values using
//  the square root for the actual position.
//
// Use: private

void
InvFullViewer::setZoomSliderPosition(float zoom)
//
////////////////////////////////////////////////////////////////////////
{
    if (zoomWidgets[ZOOM_SLIDER] == NULL)
        return;

    // find the slider position, using a square root distance to make the
    // slider smoother and less sensitive when close to zero.
    float f = (zoom - zoomSldRange[0]) / (zoomSldRange[1] - zoomSldRange[0]);
    f = (f < 0) ? 0 : ((f > 1) ? 1 : f);
    f = sqrtf(f);

    // finally position the slider
    Arg args[1];
    int val = int(f * 1000);
    XtSetArg(args[0], XmNvalue, val);
    XtSetValues(zoomWidgets[ZOOM_SLIDER], args, 1);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Sets the zoom field value based on the current camera zoom value.
//
// Use: private

void
InvFullViewer::setZoomFieldString(float zoom)
//
////////////////////////////////////////////////////////////////////////
{
    if (zoomWidgets[ZOOM_FIELD] == NULL)
        return;

    Arg args[1];
    char str[15];
    sprintf(str, "%.1f", zoom);
    XtSetArg(args[0], XmNvalue, str);
    XtSetValues(zoomWidgets[ZOOM_FIELD], args, 1);
}

//
// Virtual thumb wheels methods which subclasses can redefine
//
void InvFullViewer::rightWheelStart() { interactiveCountInc(); }
void InvFullViewer::bottomWheelStart() { interactiveCountInc(); }
void InvFullViewer::leftWheelStart() { interactiveCountInc(); }
void InvFullViewer::rightWheelFinish() { interactiveCountDec(); }
void InvFullViewer::bottomWheelFinish() { interactiveCountDec(); }
void InvFullViewer::leftWheelFinish() { interactiveCountDec(); }

void InvFullViewer::rightWheelMotion(float) {}
void InvFullViewer::bottomWheelMotion(float) {}
void InvFullViewer::leftWheelMotion(float) {}
void InvFullViewer::openViewerHelpCard() {}

//
////////////////////////////////////////////////////////////////////////
// static callbacks stubs
////////////////////////////////////////////////////////////////////////
//

//
// This static variable is used to detect start/finish callbacks
// for the SgThumbwheel and the XmScale widgets.
//
static SbBool firstDrag = TRUE;

// thumb wheel static value changed callbacks
void InvFullViewer::rightWheelCB(Widget, InvFullViewer *v, XtPointer *d)
{
    SgThumbWheelCallbackStruct *data = (SgThumbWheelCallbackStruct *)d;

    if (data->reason == XmCR_DRAG)
    {
        // for the first move, invoke the start callbacks
        if (firstDrag)
        {
            v->rightWheelStart();
            firstDrag = FALSE;
        }

        v->rightWheelMotion(-data->value * M_PI / 180.0);
    }
    else
    {
        // reason = XmCR_VALUE_CHANGED, invoke the finish callbacks
        v->rightWheelFinish();
        firstDrag = TRUE;
    }
}

void InvFullViewer::bottomWheelCB(Widget, InvFullViewer *v, XtPointer *d)
{
    SgThumbWheelCallbackStruct *data = (SgThumbWheelCallbackStruct *)d;

    if (data->reason == XmCR_DRAG)
    {
        // for the first move, invoke the start callbacks
        if (firstDrag)
        {
            v->bottomWheelStart();
            firstDrag = FALSE;
        }

        v->bottomWheelMotion(data->value * M_PI / 180.0);
    }
    else
    {
        // reason = XmCR_VALUE_CHANGED, invoke the finish callbacks
        v->bottomWheelFinish();
        firstDrag = TRUE;
    }
}

void InvFullViewer::leftWheelCB(Widget, InvFullViewer *v, XtPointer *d)
{
    SgThumbWheelCallbackStruct *data = (SgThumbWheelCallbackStruct *)d;

    if (data->reason == XmCR_DRAG)
    {
        // for the first move, invoke the start callbacks
        if (firstDrag)
        {
            v->leftWheelStart();
            firstDrag = FALSE;
        }

        v->leftWheelMotion(-data->value * M_PI / 180.0);
    }
    else
    {
        // reason = XmCR_VALUE_CHANGED, invoke the finish callbacks
        v->leftWheelFinish();
        firstDrag = TRUE;
    }
}

//
// viewer push button callbacks
//
void
InvFullViewer::pushButtonCB(Widget w, int id, void *)
{
    InvFullViewer *v;
    XtVaGetValues(w, XmNuserData, &v, NULL);

    switch (id)
    {
    case PICK_PUSH:
        v->setViewing(FALSE);
        break;
    case VIEW_PUSH:
        v->setViewing(TRUE);
        break;
    // Uwe Woessner (Added TRACK_PUSH)
    case TRACK_PUSH:
        v->tougleTracking(!v->trackingFlag);
        break;
    case HELP_PUSH:
        v->openViewerHelpCard();
        break;
    case HOME_PUSH:
        v->resetToHomePosition();
        break;
    case SET_HOME_PUSH:
        v->saveHomePosition();
        break;
    case VIEW_ALL_PUSH:
        v->viewAll();
        break;
    case SEEK_PUSH:
        v->setSeekMode(!v->isSeekMode());
        break;
    }
}

void
InvFullViewer::prefSheetDestroyCB(Widget, InvFullViewer *v, void *)
{
    // reset our vars when the dialog gets destroyed....
    v->prefSheetShellWidget = NULL;
    for (int i = 0; i < ZOOM_NUM; i++)
        v->zoomWidgets[i] = NULL;
}

void
InvFullViewer::seekPrefSheetFieldCB(Widget field, InvFullViewer *v, void *)
{
    // get text value from the label
    char *str = XmTextGetString(field);
    float val;
    if (sscanf(str, "%f", &val))
    {
        if (val < 0)
            val = 0;
        v->setSeekTime(val);
    }
    XtFree(str);

    // reformat text field
    char valStr[10];
    sprintf(valStr, "%.2f", v->getSeekTime());
    XmTextSetString(field, valStr);

    // make the text field loose the focus
    XmProcessTraversal(XtParent(field), XmTRAVERSE_CURRENT);
}

void
InvFullViewer::seekPrefSheetToggle1CB(Widget tog1, Widget tog2, void *)
{
    XmToggleButtonSetState(tog2, !XmToggleButtonGetState(tog1), FALSE);

    // get viewer pointer and set seek detail state
    InvFullViewer *v;
    Arg args[1];
    XtSetArg(args[0], XmNuserData, &v);
    XtGetValues(tog1, args, 1);
    v->setDetailSeek(XmToggleButtonGetState(tog1));
}

void
InvFullViewer::seekPrefSheetToggle2CB(Widget tog2, Widget tog1, void *)
{
    XmToggleButtonSetState(tog1, !XmToggleButtonGetState(tog2), FALSE);

    // get viewer pointer and set seek detail state
    InvFullViewer *v;
    Arg args[1];
    XtSetArg(args[0], XmNuserData, &v);
    XtGetValues(tog1, args, 1);
    v->setDetailSeek(XmToggleButtonGetState(tog1));
}

void
InvFullViewer::seekDistPercPrefSheetToggleCB(Widget tog1, Widget tog2, void *)
{
    XmToggleButtonSetState(tog2, !XmToggleButtonGetState(tog1), FALSE);

    // get viewer pointer and set seek distance state
    InvFullViewer *v;
    Arg args[1];
    XtSetArg(args[0], XmNuserData, &v);
    XtGetValues(tog1, args, 1);
    v->seekDistAsPercentage = XmToggleButtonGetState(tog1);
}

void
InvFullViewer::seekDistAbsPrefSheetToggleCB(Widget tog2, Widget tog1, void *)
{
    XmToggleButtonSetState(tog1, !XmToggleButtonGetState(tog2), FALSE);

    // get viewer pointer and set seek distance state
    InvFullViewer *v;
    Arg args[1];
    XtSetArg(args[0], XmNuserData, &v);
    XtGetValues(tog1, args, 1);
    v->seekDistAsPercentage = XmToggleButtonGetState(tog1);
}

////////////////////////////////////////////////////////////////////////
//
//  Called whenever the user changes the zoom slider position.
//
//  Use: static private
//
void
InvFullViewer::zoomSliderCB(Widget, InvFullViewer *v, XtPointer *d)
//
////////////////////////////////////////////////////////////////////////
{
    XmScaleCallbackStruct *data = (XmScaleCallbackStruct *)d;

    // for the first move, invoke the start callbacks
    if (data->reason == XmCR_DRAG && firstDrag)
    {
        v->interactiveCountInc();
        firstDrag = FALSE;
    }

    // if the slider is being dragged OR the slider jumps around
    // (user clicked left mouse on the side which causes the slider
    // to animate) update the camera zoom value.
    if (data->reason == XmCR_DRAG || (data->reason == XmCR_VALUE_CHANGED && firstDrag))
    {

        // get the slider zoom value, taking the square value since we
        // are using the square root to make the slider smoother to use.
        float f = data->value / 1000.0;
        f *= f;
        float zoom = v->zoomSldRange[0] + f * (v->zoomSldRange[1] - v->zoomSldRange[0]);

        // now update the camera and text field
        v->setCameraZoom(zoom);
        v->setZoomFieldString(zoom);
    }

    // reason = XmCR_VALUE_CHANGED, invoke the finish callbacks
    if (data->reason == XmCR_VALUE_CHANGED && !firstDrag)
    {
        v->interactiveCountDec();
        firstDrag = TRUE;
    }
}

////////////////////////////////////////////////////////////////////////
//
//  Called whenever the zoom slider field has a new value typed in.
//
//  Use: static private
//
void
InvFullViewer::zoomFieldCB(Widget field, InvFullViewer *v, XtPointer *)
//
////////////////////////////////////////////////////////////////////////
{
    // get value from the label
    char *str = XmTextGetString(field);
    float zoom;
    if (sscanf(str, "%f", &zoom) && zoom > 0)
    {

        // check for valid perspective camera range
        if (v->camera != NULL && v->camera->isOfType(SoPerspectiveCamera::getClassTypeId()))
        {
            zoom = (zoom < 0.01) ? 0.01 : ((zoom > 179.99) ? 179.99 : zoom);
        }

        // check if the newly typed value changed the slider range
        if (zoom < v->zoomSldRange[0])
            v->zoomSldRange[0] = zoom;
        else if (zoom > v->zoomSldRange[1])
            v->zoomSldRange[1] = zoom;

        // update the slider and camera zoom values.
        v->setCameraZoom(zoom);
        v->setZoomSliderPosition(zoom);
    }
    else
        zoom = v->getCameraZoom();
    XtFree(str);

    // always reformat text field
    v->setZoomFieldString(zoom);

    // make the text field loose the focus
    XmProcessTraversal(SoXt::getShellWidget(field), XmTRAVERSE_CURRENT);
}

////////////////////////////////////////////////////////////////////////
//
//  This routine opens up the popup menu.
//
//  Use: static private
//
void
InvFullViewer::popMenuCallback(Widget, InvFullViewer *v, XEvent *event, Boolean *)
//
////////////////////////////////////////////////////////////////////////
{
    Arg args[1];
    int button;

    if (Annotations->getNumFlags() > 0)
    {
        XtSetSensitive(v->deleteAnnoWidget, TRUE);
        XtSetSensitive(v->editAnnoWidget, TRUE);
    }
    else
    {
        XtSetSensitive(v->deleteAnnoWidget, FALSE);
        XtSetSensitive(v->editAnnoWidget, FALSE);
    }

#ifdef _BAW
    if (v->getPresentationCursor())
    {
        XtSetSensitive(v->normCursorWidget, TRUE);
        XtSetSensitive(v->presCursorWidget, FALSE);
    }
    else
    {
        XtSetSensitive(v->normCursorWidget, FALSE);
        XtSetSensitive(v->presCursorWidget, TRUE);
    }
#endif

    XtSetArg(args[0], XmNwhichButton, &button);
    XtGetValues(v->popupWidget, args, 1);
    if (event->xbutton.button == (unsigned int)button)
    {
        XmMenuPosition(v->popupWidget, (XButtonPressedEvent *)event);
        XtManageChild(v->popupWidget);
    }
}

////////////////////////////////////////////////////////////////////////
//
//  Called by Xt when a main menu item is picked.
//
//  Use: static private
//
void
InvFullViewer::menuPick(Widget w, int id, XmAnyCallbackStruct *cb)
//
////////////////////////////////////////////////////////////////////////
{
    Time eventTime = cb->event->xbutton.time;
    InvFullViewer *v;
    Arg args[1];
    char Buffer[300];
    char hostname[200];
    char username[100];

    XtSetArg(args[0], XmNuserData, &v);
    XtGetValues(w, args, 1);
    Message *message;

    switch (id)
    {
        // HELP: 	v->openViewerHelpCard(); break;
    case VIEW_ALL:
        v->viewAll();
        break;
    case SET_HOME:
        v->saveHomePosition();
        break;
    case HOME:
        v->resetToHomePosition();
        break;
    case SEEK:
        v->setSeekMode(!v->isSeekMode());
        break;
    case PREF:
        if (v->prefSheetShellWidget == NULL)
            v->createPrefSheet();
        else
            SoXt::show(v->prefSheetShellWidget);
        break;
    case NORMAL_CURSOR:
        v->setPresentationCursor(false);
        break;

    case PRESENTATION_CURSOR:
        v->setPresentationCursor(true);
        break;

    case ANNOTATION_CREATE:
        Annotations->activate(InvAnnoManager::MAKE);
        break;
    case ANNOTATION_DELETE:
        Annotations->activate(InvAnnoManager::REMOVE);
        break;
    case ANNOTATION_EDIT:
        Annotations->activate(InvAnnoManager::EDIT);
        break;
    case HEADLIGHT:
        v->setHeadlight(!v->isHeadlight());
        break;
    case VIEWING:
        v->setViewing(!v->isViewing());
        break;
    case DECORATION:
        v->setDecoration(!v->decorationFlag);
        break;
    case COPY_VIEW:
        v->copyView(eventTime);
        break;
    case MASTER_REQUEST:
        struct passwd* pwd = getpwuid(getuid());
        strcpy(username, pwd->pw_name);
        strcpy(hostname, appmod->get_hostname());

        strcpy(Buffer, "MASTERREQ\n");
        strcat(Buffer, hostname);
        strcat(Buffer, "\n");
        strcat(Buffer, username);
        strcat(Buffer, "\n");
        message = new Message{ COVISE_MESSAGE_UI, DataHandle{Buffer, strlen(Buffer) + 1, false } };
        appmod->send_ctl_msg(message);
        delete message;
        break;
        // changed 11.04.94
        // D. Rantzau
        //
        ///	case PASTE_VIEW: v->pasteView(eventTime); break;
    }
}

////////////////////////////////////////////////////////////////////////
//
//  Called by Xt when a menu item is picked in the drawStyle menu.
//
//  Use: static private
//
void
InvFullViewer::drawStyleMenuPick(Widget w, int id, void *)
//
////////////////////////////////////////////////////////////////////////
{
    InvFullViewer *v;
    XtVaGetValues(w, XmNuserData, &v, NULL);

    switch (id)
    {
    case AS_IS:
        v->setDrawStyle(InvViewer::STILL, InvViewer::VIEW_AS_IS);
        break;
    case HIDDEN_LINE:
        v->setDrawStyle(InvViewer::STILL, InvViewer::VIEW_HIDDEN_LINE);
        break;
    case MESH:
        v->setDrawStyle(InvViewer::STILL, InvViewer::VIEW_MESH);
        break;
    case NO_TXT:
        v->setDrawStyle(InvViewer::STILL, InvViewer::VIEW_NO_TEXTURE);
        break;
    case LOW_RES:
        v->setDrawStyle(InvViewer::STILL, InvViewer::VIEW_LOW_COMPLEXITY);
        break;
    case LINE:
        v->setDrawStyle(InvViewer::STILL, InvViewer::VIEW_LINE);
        break;
    case POINT:
        v->setDrawStyle(InvViewer::STILL, InvViewer::VIEW_POINT);
        break;
    case BBOX:
        v->setDrawStyle(InvViewer::STILL, InvViewer::VIEW_BBOX);
        break;
    case LOW_VOLUME:
        v->setDrawStyle(InvViewer::STILL, InvViewer::VIEW_LOW_VOLUME);
        break;

    case MOVE_SAME_AS:
        v->setDrawStyle(InvViewer::INTERACTIVE, InvViewer::VIEW_SAME_AS_STILL);
        break;
    case MOVE_NO_TXT:
        v->setDrawStyle(InvViewer::INTERACTIVE, InvViewer::VIEW_NO_TEXTURE);
        break;
    case MOVE_LOW_RES:
        v->setDrawStyle(InvViewer::INTERACTIVE, InvViewer::VIEW_LOW_COMPLEXITY);
        break;
    case MOVE_LINE:
        v->setDrawStyle(InvViewer::INTERACTIVE, InvViewer::VIEW_LINE);
        break;
    case MOVE_LOW_LINE:
        v->setDrawStyle(InvViewer::INTERACTIVE, InvViewer::VIEW_LOW_RES_LINE);
        break;
    case MOVE_POINT:
        v->setDrawStyle(InvViewer::INTERACTIVE, InvViewer::VIEW_POINT);
        break;
    case MOVE_LOW_POINT:
        v->setDrawStyle(InvViewer::INTERACTIVE, InvViewer::VIEW_LOW_RES_POINT);
        break;
    case MOVE_BBOX:
        v->setDrawStyle(InvViewer::INTERACTIVE, InvViewer::VIEW_BBOX);
        break;
    case MOVE_LOW_VOLUME:
        v->setDrawStyle(InvViewer::INTERACTIVE, InvViewer::VIEW_LOW_VOLUME);
        break;
    }
}

////////////////////////////////////////////////////////////////////////
//
//  Called by Xt when a menu item in the buffer style menu is picked.
//
//  Use: static private
//
void
InvFullViewer::bufferStyleMenuPick(Widget w, int id, void *)
//
////////////////////////////////////////////////////////////////////////
{
    InvFullViewer *v;
    XtVaGetValues(w, XmNuserData, &v, NULL);

    v->setBufferingType((InvViewer::BufferType)id);
}

void InvFullViewer::colorsPrefSheetToggleCB(Widget toggle, Widget parent, void *)
{
    InvFullViewer *v;
    Arg args[10];
    XtSetArg(args[0], XmNuserData, &v);
    XtGetValues(toggle, args, 1);

    v->setUseNumberFormat(XmToggleButtonGetState(toggle));
    if (!(v->getUseNumberFormat()))
    {
        XtDestroyWidget(v->numberFormatForm_);
    }
    else
    {
        XtSetArg(args[0], XmNleftAttachment, XmATTACH_WIDGET);
        XtSetArg(args[1], XmNleftOffset, 2);
        XtSetArg(args[2], XmNleftWidget, toggle);
        XtSetArg(args[3], XmNrightAttachment, XmATTACH_FORM);
        XtSetArg(args[4], XmNtopAttachment, XmATTACH_WIDGET);
        XtSetArg(args[5], XmNtopWidget, toggle);
        XtSetArg(args[6], XmNtopOffset, 2);

        v->numberFormatForm_ = XmCreateText(parent, (char *)"Entry", args, 7);
        XtManageChild(v->numberFormatForm_);

        XtManageChild(v->numberFormatForm_);

        XtAddCallback(v->numberFormatForm_, XmNvalueChangedCallback,
                      (XtCallbackProc)InvFullViewer::colorFormatEntryCB,
                      (XtPointer)v);

        XtManageChild(parent);
    }
}

void InvFullViewer::colorFormatEntryCB(Widget entry, void *viewer, XmAnyCallbackStruct *call_data)
{

    InvFullViewer *v = (InvFullViewer *)viewer;
    (void)entry;
    (void)call_data;
    char *value = XmTextGetString(v->numberFormatForm_);
    v->setNumberFormat(value);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Called by the auto clipping preference sheet to toggle auto clipping
//  and show/hide the extra manual set thumbwheels.
//
// Use: static private

void
InvFullViewer::clipPrefSheetToggleCB(Widget toggle, Widget parent, void *)
//
////////////////////////////////////////////////////////////////////////
{
    // get the viewer pointer
    InvFullViewer *v;
    Arg args[1];
    XtSetArg(args[0], XmNuserData, &v);
    XtGetValues(toggle, args, 1);

    v->setAutoClipping(XmToggleButtonGetState(toggle));

    // check if toggle button is on or off
    if (v->isAutoClipping())
    {
        XtDestroyWidget(v->clipWheelForm);
    }
    else
    {
        Widget label[2], thumb[2], text[2];
        Arg args[12];
        int n;

        // create a form to hold everything together
        Widget form = XtCreateWidget("", xmFormWidgetClass,
                                     parent, NULL, 0);
        v->clipWheelForm = form;

        // create the labels
        label[0] = XtCreateWidget("near plane:", xmLabelGadgetClass,
                                  form, NULL, 0);
        label[1] = XtCreateWidget("far plane:", xmLabelGadgetClass,
                                  form, NULL, 0);

        // allocate the thumbwheels
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

        thumb[0] = SgCreateThumbWheel(form, (char *)"", args, n);
        thumb[1] = SgCreateThumbWheel(form, (char *)"", args, n);

        XtAddCallback(thumb[0], XmNvalueChangedCallback,
                      (XtCallbackProc)InvFullViewer::clipNearWheelCB, (XtPointer)v);
        XtAddCallback(thumb[0], XmNdragCallback,
                      (XtCallbackProc)InvFullViewer::clipNearWheelCB, (XtPointer)v);
        XtAddCallback(thumb[1], XmNvalueChangedCallback,
                      (XtCallbackProc)InvFullViewer::clipFarWheelCB, (XtPointer)v);
        XtAddCallback(thumb[1], XmNdragCallback,
                      (XtCallbackProc)InvFullViewer::clipFarWheelCB, (XtPointer)v);
        v->clipNearWheelVal = 0;
        v->clipFarWheelVal = 0;

        // allocate the text fields
        n = 0;
        char str[15];
        float val = (v->camera != NULL) ? v->camera->nearDistance.getValue() : 0;
        sprintf(str, "%f", val);
        XtSetArg(args[0], XmNvalue, str);
        n++;
        XtSetArg(args[n], XmNhighlightThickness, 1);
        n++;
        XtSetArg(args[n], XmNcolumns, 11);
        n++;
        v->clipNearField = text[0] = XtCreateWidget("", xmTextWidgetClass,
                                                    form, args, n);
        val = (v->camera != NULL) ? v->camera->farDistance.getValue() : 0;
        sprintf(str, "%f", val);
        XtSetArg(args[0], XmNvalue, str);
        v->clipFarField = text[1] = XtCreateWidget("", xmTextWidgetClass,
                                                   form, args, n);
        XtAddCallback(text[0], XmNactivateCallback,
                      (XtCallbackProc)InvFullViewer::clipFieldCB,
                      (XtPointer)v);
        XtAddCallback(text[1], XmNactivateCallback,
                      (XtCallbackProc)InvFullViewer::clipFieldCB,
                      (XtPointer)v);

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
        XtSetArg(args[n], XmNtopOffset, 2);
        n++;
        XtSetValues(form, args, n);

        n = 0;
        XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
        n++;
        XtSetValues(text[0], args, n);
        n = 0;
        XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[n], XmNtopWidget, text[0]);
        n++;
        XtSetValues(text[1], args, n);

        n = 0;
        XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
        n++;
        XtSetArg(args[1], XmNbottomWidget, text[0]);
        n++;
        XtSetArg(args[n], XmNbottomOffset, 3);
        n++;
        XtSetArg(args[n], XmNrightAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[4], XmNrightWidget, text[0]);
        n++;
        XtSetArg(args[n], XmNrightOffset, 3);
        n++;
        XtSetValues(thumb[0], args, n);
        XtSetArg(args[1], XmNbottomWidget, text[1]);
        XtSetArg(args[4], XmNrightWidget, text[1]);
        XtSetValues(thumb[1], args, n);

        n = 0;
        XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
        n++;
        XtSetArg(args[1], XmNbottomWidget, thumb[0]);
        n++;
        XtSetArg(args[n], XmNrightAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[3], XmNrightWidget, thumb[0]);
        n++;
        XtSetArg(args[n], XmNrightOffset, 5);
        n++;
        XtSetValues(label[0], args, n);
        XtSetArg(args[1], XmNbottomWidget, thumb[1]);
        XtSetArg(args[3], XmNrightWidget, thumb[1]);
        XtSetValues(label[1], args, n);

        // manage children
        XtManageChild(form);
        XtManageChildren(text, 2);
        XtManageChildren(thumb, 2);
        XtManageChildren(label, 2);
    }
}

void
InvFullViewer::clipNearWheelCB(Widget, InvFullViewer *v, XtPointer *d)
{
    if (v->camera == NULL)
        return;

    SgThumbWheelCallbackStruct *data = (SgThumbWheelCallbackStruct *)d;

    if (data->reason == XmCR_DRAG)
    {
        // for the first move, invoke the start callbacks
        if (firstDrag)
        {
            v->interactiveCountInc();
            firstDrag = FALSE;
        }

        // shorter/grow the near plane distance given the wheel rotation
        float dist = v->camera->nearDistance.getValue();
        dist *= powf(80.0, (data->value - v->clipNearWheelVal) / 360.0);
        v->clipNearWheelVal = data->value;

        // change the camera and update the text field
        v->camera->nearDistance = dist;
        char str[15];
        sprintf(str, "%f", dist);
        XmTextSetString(v->clipNearField, str);
    }
    else
    {
        // reason = XmCR_VALUE_CHANGED, invoke the finish callbacks
        v->interactiveCountDec();
        firstDrag = TRUE;
    }
}

void
InvFullViewer::clipFarWheelCB(Widget, InvFullViewer *v, XtPointer *d)
{
    if (v->camera == NULL)
        return;

    SgThumbWheelCallbackStruct *data = (SgThumbWheelCallbackStruct *)d;

    if (data->reason == XmCR_DRAG)
    {
        // for the first move, invoke the start callbacks
        if (firstDrag)
        {
            v->interactiveCountInc();
            firstDrag = FALSE;
        }

        // shorter/grow the near plane distance given the wheel rotation
        float dist = v->camera->farDistance.getValue();
        dist *= powf(80.0, (data->value - v->clipFarWheelVal) / 360.0);
        v->clipFarWheelVal = data->value;

        // change the camera and update the text field
        v->camera->farDistance = dist;
        char str[15];
        sprintf(str, "%f", dist);
        XmTextSetString(v->clipFarField, str);
    }
    else
    {
        // reason = XmCR_VALUE_CHANGED, invoke the finish callbacks
        v->interactiveCountDec();
        firstDrag = TRUE;
    }
}

void
InvFullViewer::clipFieldCB(Widget field, InvFullViewer *v, void *)
{
    if (v->camera == NULL)
        return;

    // get text value from the label and update camera
    char *str = XmTextGetString(field);
    float val;
    if (sscanf(str, "%f", &val) && (val > 0 || v->camera->isOfType(SoOrthographicCamera::getClassTypeId())))
    {
        if (field == v->clipNearField)
            v->camera->nearDistance = val;
        else
            v->camera->farDistance = val;
    }
    else
    {
        if (field == v->clipNearField)
            val = v->camera->nearDistance.getValue();
        else
            val = v->camera->farDistance.getValue();
    }
    XtFree(str);

    // reformat text field
    char valStr[10];
    sprintf(valStr, "%f", val);
    XmTextSetString(field, valStr);

    // make the text field loose the focus
    XmProcessTraversal(SoXt::getShellWidget(field), XmTRAVERSE_CURRENT);
}

#ifdef __sgi
static void destroyStereoInfoDialogCB(Widget dialog, void *, void *)
{
    XtDestroyWidget(dialog);
}

static char *str1 = "Please refer to the setmon man pages to set and restore the";
static char *str2 = "monitor stereo mode.";
static char *str3 = "On RealityEngine, try '/usr/gfx/setmon -n 1025x768_96s'";
static char *str4 = "On Indy/Indigo, try '/usr/gfx/setmon -n STR_TOP' (or STR_BOT, ";
static char *str5 = "depending on which half of the screen the viewer is).";
static char *str6 = "To restore the monitor try '/usr/gfx/setmon -n 72HZ' (or 60HZ).";

static void createStereoInfoDialog(Widget shell)
{
    Arg args[5];
    XmString xmstr = XmStringCreateSimple(str1);
    xmstr = XmStringConcat(xmstr, XmStringSeparatorCreate());
    xmstr = XmStringConcat(xmstr, XmStringCreateSimple(str2));
    xmstr = XmStringConcat(xmstr, XmStringSeparatorCreate());
    xmstr = XmStringConcat(xmstr, XmStringSeparatorCreate());
    xmstr = XmStringConcat(xmstr, XmStringCreateSimple(str3));
    xmstr = XmStringConcat(xmstr, XmStringSeparatorCreate());
    xmstr = XmStringConcat(xmstr, XmStringSeparatorCreate());
    xmstr = XmStringConcat(xmstr, XmStringCreateSimple(str4));
    xmstr = XmStringConcat(xmstr, XmStringSeparatorCreate());
    xmstr = XmStringConcat(xmstr, XmStringCreateSimple(str5));
    xmstr = XmStringConcat(xmstr, XmStringSeparatorCreate());
    xmstr = XmStringConcat(xmstr, XmStringSeparatorCreate());
    xmstr = XmStringConcat(xmstr, XmStringCreateSimple(str6));

    int n = 0;
    XtSetArg(args[n], XmNautoUnmanage, FALSE);
    n++;
    XtSetArg(args[n], XtNtitle, "Stereo Usage Dialog");
    n++;
    XtSetArg(args[n], XmNmessageString, xmstr);
    n++;
    Widget dialog = XmCreateWarningDialog(shell, "Stereo Dialog", args, n);
    XmStringFree(xmstr);

    XtUnmanageChild(XmMessageBoxGetChild(dialog, XmDIALOG_CANCEL_BUTTON));
    XtUnmanageChild(XmMessageBoxGetChild(dialog, XmDIALOG_HELP_BUTTON));

    // register callback to destroy (and not just unmap) the dialog
    XtAddCallback(dialog, XmNokCallback,
                  (XtCallbackProc)destroyStereoInfoDialogCB, (XtPointer)NULL);

    XtManageChild(dialog);
}
#endif

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Called by the stereo preference sheet to toggle stereo viewing
//  and show/hide the extra offset thumbwheel.
//
// Use: static private

void
InvFullViewer::stereoPrefSheetToggleCB(Widget toggle, Widget parent, void *)
//
////////////////////////////////////////////////////////////////////////
{
    // get the viewer pointer
    InvFullViewer *v;
    XtVaGetValues(toggle, XmNuserData, &v, NULL);

    //
    // checks to make sure stereo viewing can be set, else
    // grey the UI and bring and error message.
    //
    SbBool toggleState = XmToggleButtonGetState(toggle);
    SbBool sameState = (toggleState == v->isStereoViewing());
    if (!sameState)
    {
        if (toggle == v->stippToggle)
            v->setStippleStereoViewing(toggleState);
        else
            v->setStereoViewing(toggleState);
    }
    if (toggleState && !v->isStereoViewing())
    {
        TOGGLE_OFF(toggle);
        XtVaSetValues(toggle, XmNsensitive, FALSE, NULL);
        XtVaSetValues(v->stereoLabel, XmNsensitive, FALSE, NULL);
        SoXt::createSimpleErrorDialog(toggle, (char *)stereoErrorTitle, (char *)stereoError);
        return;
    }

    // show/hide the spacing thumbwheel
    if (!v->isStereoViewing())
    {
        if (v->stereoWheelForm != NULL)
        {
            XtDestroyWidget(v->stereoWheelForm);
            v->stereoWheelForm = NULL;
            /// added by D. Rantzau
            ///

            std::string line = coCoviseConfig::getEntry("Renderer.MonoCommand");
            if (line.length() > 3)
            {
                if (system(line.c_str()) == -1)
                {
                    fprintf(stderr, "Mono command %s failed\n", line.c_str());
                }
            }
        }
    }
    else
    {
        if (v->stereoWheelForm != NULL)
            return;
        Widget label, thumb, text;
        Arg args[12];
        int n;

        // create a form to hold everything together
        Widget form = XtCreateWidget("Stereo thumb form", xmFormWidgetClass,
                                     parent, NULL, 0);
        v->stereoWheelForm = form;

        // create the label
        label = XtCreateWidget("camera rotation:", xmLabelGadgetClass,
                               form, NULL, 0);

        // allocate the thumbwheel
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
        thumb = SgCreateThumbWheel(form, (char *)"", args, n);

        XtAddCallback(thumb, XmNvalueChangedCallback,
                      (XtCallbackProc)InvFullViewer::stereoWheelCB, (XtPointer)v);
        XtAddCallback(thumb, XmNdragCallback,
                      (XtCallbackProc)InvFullViewer::stereoWheelCB, (XtPointer)v);
        v->stereoWheelVal = 0;

        // allocate the text field
        n = 0;
        char str[15];
        sprintf(str, "%.4f", v->getStereoOffset());
        XtSetArg(args[n], XmNvalue, str);
        n++;
        XtSetArg(args[n], XmNhighlightThickness, 1);
        n++;
        XtSetArg(args[n], XmNcolumns, 6);
        n++;
        v->stereoField = text = XtCreateWidget("", xmTextWidgetClass,
                                               form, args, n);
        XtAddCallback(text, XmNactivateCallback,
                      (XtCallbackProc)InvFullViewer::stereoFieldCB,
                      (XtPointer)v);

        // layout
        n = 0;
        XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNleftOffset, 20);
        n++;
        XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[n], XmNtopWidget, v->stippToggle);
        n++;
        XtSetArg(args[n], XmNtopOffset, 2);
        n++;
        XtSetValues(form, args, n);

        n = 0;
        XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
        n++;
        XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
        n++;
        XtSetValues(text, args, n);

        n = 0;
        XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
        n++;
        XtSetArg(args[n], XmNbottomWidget, text);
        n++;
        XtSetArg(args[n], XmNbottomOffset, 3);
        n++;
        XtSetArg(args[n], XmNrightAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[n], XmNrightWidget, text);
        n++;
        XtSetArg(args[n], XmNrightOffset, 3);
        n++;
        XtSetValues(thumb, args, n);

        n = 0;
        XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
        n++;
        XtSetArg(args[n], XmNbottomWidget, thumb);
        n++;
        XtSetArg(args[n], XmNrightAttachment, XmATTACH_WIDGET);
        n++;
        XtSetArg(args[n], XmNrightWidget, thumb);
        n++;
        XtSetArg(args[n], XmNrightOffset, 5);
        n++;
        XtSetValues(label, args, n);

        // manage children
        XtManageChild(form);
        XtManageChild(text);
        XtManageChild(thumb);
        XtManageChild(label);

        // bring a dialog to tell the user to look at setmon to set
        // the monitor to stereo mode
        // createStereoInfoDialog(SoXt::getShellWidget(toggle));
        ///
        std::string line = coCoviseConfig::getEntry("Renderer.StereoCommand");
        if (line.length() > 3)
        {
            if (system(line.c_str()) == -1)
            {
                fprintf(stderr, "Mono command %s failed\n", line.c_str());
            }
        }
    }
}

void
InvFullViewer::stereoWheelCB(Widget, InvFullViewer *v, XtPointer *d)
{

    SgThumbWheelCallbackStruct *data = (SgThumbWheelCallbackStruct *)d;

    if (data->reason == XmCR_DRAG)
    {
        // for the first move, invoke the start callbacks
        if (firstDrag)
        {
            v->interactiveCountInc();
            firstDrag = FALSE;
        }

        // shorter/grow the stereo camera offset (linear characteristic)
        float inc(0.02);
        v->setStereoOffset(v->getStereoOffset() + copysign(inc, (data->value - v->stereoWheelVal)));
        v->stereoWheelVal = data->value;

        // update the text field
        char str[15];
        sprintf(str, "%.4f", v->getStereoOffset());
        XmTextSetString(v->stereoField, str);

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
InvFullViewer::stereoFieldCB(Widget field, InvFullViewer *v, void *)
{
    // get text value from the label and update camera
    char *str = XmTextGetString(field);
    float val;
    if (sscanf(str, "%f", &val))
    {
        v->setStereoOffset(val);
        v->scheduleRedraw();
    }
    XtFree(str);

    // reformat text field
    char valStr[10];
    sprintf(valStr, "%.4f", v->getStereoOffset());
    XmTextSetString(field, valStr);

    // make the text field loose the focus
    XmProcessTraversal(SoXt::getShellWidget(field), XmTRAVERSE_CURRENT);
}

void
InvFullViewer::seekDistWheelCB(Widget, InvFullViewer *v, XtPointer *d)
{
    SgThumbWheelCallbackStruct *data = (SgThumbWheelCallbackStruct *)d;

    // shorter/grow the seek distance given the wheel rotation
    v->seekDistance *= powf(80.0, (data->value - v->seekDistWheelVal) / 360.0);
    v->seekDistWheelVal = data->value;

    // update the text field
    char str[15];
    sprintf(str, "%f", v->seekDistance);
    XmTextSetString(v->seekDistField, str);
}

void
InvFullViewer::seekDistFieldCB(Widget field, InvFullViewer *v, void *)
{
    // get text value from the label
    char *str = XmTextGetString(field);
    float val;
    if (sscanf(str, "%f", &val) && val > 0)
        v->seekDistance = val;
    else
        val = v->seekDistance;
    XtFree(str);

    // reformat text field
    char valStr[15];
    sprintf(valStr, "%f", val);
    XmTextSetString(field, valStr);

    // make the text field loose the focus
    XmProcessTraversal(SoXt::getShellWidget(field), XmTRAVERSE_CURRENT);
}

void
InvFullViewer::zoomPrefSheetMinFieldCB(Widget field, InvFullViewer *v, void *)
{
    // get text value from the label
    char *str = XmTextGetString(field);
    float val;
    if (sscanf(str, "%f", &val) && val >= 0)
    {

        // check for valid perspective camera range
        if (v->camera != NULL && v->camera->isOfType(SoPerspectiveCamera::getClassTypeId()))
        {
            val = (val < 0.01) ? 0.01 : ((val > 178.99) ? 178.99 : val);
        }

        // finally update the slider to reflect the changes
        v->zoomSldRange[0] = val;
        v->setZoomSliderPosition(v->getCameraZoom());
    }
    else
        val = v->zoomSldRange[0];
    XtFree(str);

    // reformat text field
    char valStr[15];
    sprintf(valStr, "%.1f", val);
    XmTextSetString(field, valStr);

    // make the text field loose the focus
    XmProcessTraversal(SoXt::getShellWidget(field), XmTRAVERSE_CURRENT);
}

void
InvFullViewer::zoomPrefSheetMaxFieldCB(Widget field, InvFullViewer *v, void *)
{
    // get text value from the field
    char *str = XmTextGetString(field);
    float val;
    if (sscanf(str, "%f", &val) && val >= 0)
    {

        // check for valid perspective camera range
        if (v->camera != NULL && v->camera->isOfType(SoPerspectiveCamera::getClassTypeId()))
        {
            val = (val < 1.01) ? 1.01 : ((val > 179.99) ? 179.99 : val);
        }

        // finally update the slider to reflect the changes
        v->zoomSldRange[1] = val;
        v->setZoomSliderPosition(v->getCameraZoom());
    }
    else
        val = v->zoomSldRange[1];
    XtFree(str);

    // reformat text field
    char valStr[15];
    sprintf(valStr, "%.1f", val);
    XmTextSetString(field, valStr);

    // make the text field loose the focus
    XmProcessTraversal(SoXt::getShellWidget(field), XmTRAVERSE_CURRENT);
}

void
InvFullViewer::speedIncPrefSheetButtonCB(Widget, InvFullViewer *p, void *)
{
    p->viewerSpeed *= 2.0;
}

void
InvFullViewer::speedDecPrefSheetButtonCB(Widget, InvFullViewer *p, void *)
{
    p->viewerSpeed /= 2.0;
}

Widget
InvFullViewer::createSamplingPrefSheetGuts(Widget parent)
{
    Widget label, thumb, text;
    Arg args[12];
    int n;

    // create a form to hold everything together
    Widget form = XtCreateWidget("", xmFormWidgetClass, parent, NULL, 0);

    // create the label
    label = XtCreateWidget("Volume sampling rate:", xmLabelGadgetClass, form, NULL, 0);

    // allocate the thumbwheel
    n = 0;
    XtSetArg(args[n], XmNvalue, 1000);
    n++;
    XtSetArg(args[n], SgNangleRange, 5340);
    n++;
    XtSetArg(args[n], XmNmaximum, 100000);
    n++;
    XtSetArg(args[n], XmNminimum, 0);
    n++;
    XtSetArg(args[n], SgNshowHomeButton, FALSE);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    XtSetArg(args[n], XmNorientation, XmHORIZONTAL);
    n++;
    samplingWheel = thumb = SgCreateThumbWheel(form, (char *)"", args, n);

    XtAddCallback(thumb, XmNvalueChangedCallback,
                  (XtCallbackProc)InvFullViewer::samplingWheelCB, (XtPointer) this);

    // allocate the text field
    n = 0;
    char str[15];
    sprintf(str, "%.6f", samplingWheelVal);
    XtSetArg(args[n], XmNvalue, str);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 1);
    n++;
    XtSetArg(args[n], XmNcolumns, 8);
    n++;
    samplingField = text = XtCreateWidget("", xmTextWidgetClass, form, args, n);
    XtAddCallback(text, XmNactivateCallback,
                  (XtCallbackProc)InvFullViewer::samplingFieldCB,
                  (XtPointer) this);

    // layout

    n = 0;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetValues(text, args, n);

    n = 0;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, text);
    n++;
    XtSetArg(args[n], XmNbottomOffset, 3);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNrightWidget, text);
    n++;
    XtSetArg(args[n], XmNrightOffset, 3);
    n++;
    XtSetValues(thumb, args, n);

    n = 0;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, thumb);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNrightWidget, thumb);
    n++;
    XtSetArg(args[n], XmNrightOffset, 5);
    n++;
    XtSetValues(label, args, n);

    // manage children
    XtManageChild(text);
    XtManageChild(thumb);
    XtManageChild(label);

    return form;
}

void
InvFullViewer::samplingWheelCB(Widget, InvFullViewer *v, XtPointer *d)
{
    SgThumbWheelCallbackStruct *data = (SgThumbWheelCallbackStruct *)d;

    // reason = XmCR_VALUE_CHANGED
    v->samplingWheelVal = data->value / 1000.0;

    // update the text field
    char str[15];
    sprintf(str, "%.6f", v->samplingWheelVal);
    XmTextSetString(v->samplingField, str);
    v->redraw();
}

void
InvFullViewer::samplingFieldCB(Widget field, InvFullViewer *v, void *)
{
    Arg args[12];
    int n;
    // get text value from the label and update camera
    char *str = XmTextGetString(field);
    float val;
    if (sscanf(str, "%f", &val) && val > 0)
    {
        v->samplingWheelVal = val;
        v->redraw();
    }
    XtFree(str);

    // update the thumbwheel
    n = 0;
    XtSetArg(args[n], XmNvalue, int(v->samplingWheelVal * 1000));
    n++;
    XtSetValues(v->samplingWheel, args, n);

    // reformat text field
    char valStr[10];
    sprintf(valStr, "%.6f", v->samplingWheelVal);
    XmTextSetString(field, valStr);

    // make the text field lose the focus
    XmProcessTraversal(SoXt::getShellWidget(field), XmTRAVERSE_CURRENT);
}

float InvFullViewer::getSamplingRate()
{
    return samplingWheelVal;
}

void
InvFullViewer::setPresentationCursor(bool onOrOff)
{
    (void)onOrOff;
    cerr << "InvFullViewer::setPresentationCursor(..) called" << endl;
}

bool
InvFullViewer::getPresentationCursor()
{
    return true;
}
