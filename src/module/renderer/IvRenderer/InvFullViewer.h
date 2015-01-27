/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_FULL_VIEWER_
#define _INV_FULL_VIEWER_

/* $Id: InvFullViewer.h /main/vir_main/1 30-Jan-2002.17:21:09 we_te $ */

/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

// collaborative enhancements
extern void rm_sendViewing(char *message);
extern void rm_sendDecoration(char *message);
extern int rm_isMaster();
extern int rm_isSynced();

#include <Xm/Xm.h>
#include "InvViewer.h"
#include "InvRenderer.h"
#include <Inventor/SbPList.h>

// classes used
class SoXtResource;
class InvPixmapButton;

//////////////////////////////////////////////////////////////////////////////
//
//  Class: InvFullViewer
//
//	The InvFullViewer component class is the abstract base class for all
//  viewers which include a decoration around the rendering area. The
//  decoration is made of thumbwheels, sliders and push/toggle buttons. There
//  is also a popup menu which includes all of the viewing options and methods.
//
//////////////////////////////////////////////////////////////////////////////

// C-api: abstract
// C-api: prefix=SoXtFullVwr
class InvFullViewer : public InvViewer
{
public:
    // This specifies what should be build by default in the constructor
    enum BuildFlag
    {
        BUILD_NONE = 0x00,
        BUILD_DECORATION = 0x01,
        BUILD_POPUP = 0x02,
        BUILD_ALL = 0xff
    };

    //
    // Show/hide the viewer component trims (default ON)
    //
    // C-api: name=setDecor
    void setDecoration(SbBool onOrOff);
    // C-api: name=isDecor
    SbBool isDecoration()
    {
        return decorationFlag;
    }

    //
    // Enable/disable the popup menu (default enabled)
    //
    // C-api: name=SetPopupEnabled
    void setPopupMenuEnabled(SbBool trueOrFalse);
    // C-api: name=IsPopupEnabled
    SbBool isPopupMenuEnabled()
    {
        return popupEnabled;
    }

    //
    // Add/remove push buttons for the application, which will be placed
    // in the left hand side decoration trim.
    // The add() method appends the button to the end of the list, while
    // insert() places the button at the specified index (starting at 0).
    //
    // C-api: name=getAppPushBtnParent
    // returns the parent widget, which is needed when creating new buttons
    // NOTE that this will be NULL if the decoration is NOT created in the
    // constructor (see the BuildFlag) until it is shown.
    Widget getAppPushButtonParent() const
    {
        return appButtonForm;
    }
    // C-api: name=addAppPushBtn
    void addAppPushButton(Widget newButton);
    // C-api: name=insertAppPushBtn
    void insertAppPushButton(Widget newButton, int index);
    // C-api: name=removeAppPushBtn
    void removeAppPushButton(Widget oldButton);
    // C-api: name=findAppPushBtn
    int findAppPushButton(Widget oldButton)
    {
        return appButtonList->find(oldButton);
    }
    // C-api: name=lengthAppPushBtn
    int lengthAppPushButton()
    {
        return appButtonList->getLength();
    }

    // C-api: name=getRAWidget
    Widget getRenderAreaWidget()
    {
        return raWidget;
    }

    // redefine these from the base class
    virtual void setViewing(SbBool onOrOff);
    virtual void setHeadlight(SbBool onOrOff);
    virtual void setDrawStyle(InvViewer::DrawType type,
                              InvViewer::DrawStyle style);
    virtual void setBufferingType(InvViewer::BufferType type);
    virtual void setCamera(SoCamera *cam);
    virtual void hide();

    //Colormap Stuff
    static void colorsPrefSheetToggleCB(Widget toggle, Widget parent, void *);
    static void colorFormatEntryCB(Widget entry, void *viewer, XmAnyCallbackStruct *call_data);
    void setUseNumberFormat(bool format)
    {
        useNumberFormat_ = format;
    }
    bool getUseNumberFormat()
    {
        return useNumberFormat_;
    }
    void setNumberFormat(char *format)
    {
        delete numberFormat_;
        numberFormat_ = new char[1 + strlen(format)];
        strcpy(numberFormat_, format);
    }
    char *getNumberFormat()
    {
        return numberFormat_;
    }
    char *numberFormat_;
    bool useNumberFormat_;
    Widget numberFormatForm_;

    // Uwe Woessner (tracking vars)
    SoTimerSensor *trackingSensor;
    static void trackingSensorCB(void *p, SoSensor *);
    static void trackingWheelCB(Widget, InvFullViewer *v, XtPointer *d);
    static void trackingFieldCB(Widget field, InvFullViewer *v, void *);
    static void trackingDevice1CB(Widget field, InvFullViewer *v, void *);
    static void trackingDevice2CB(Widget field, InvFullViewer *v, void *);
    static void trackingDevice3CB(Widget field, InvFullViewer *v, void *);
    static void trackingDevice4CB(Widget field, InvFullViewer *v, void *);
    static void trackingDevice5CB(Widget field, InvFullViewer *v, void *);
    static void trackingDevice6CB(Widget field, InvFullViewer *v, void *);
    void tougleTracking(SbBool onOrOff);
    SbBool trackingFlag; // FALSE when the tracking is off
    SbBool initTracking(void); // True when init successfull
    void closeTracking(void);
    int trackingFd; // Filedescriptor for headtracking device
    float trackingWheelVal; // Tracking rate (0-1.0)
    int trackingDevice;
    Widget trackingField;
    Widget trackingWheel;

    float samplingWheelVal; // volume rendering quality (1.0 = one texture slice per voxel slice)
    static void samplingWheelCB(Widget, InvFullViewer *v, XtPointer *d);
    static void samplingFieldCB(Widget field, InvFullViewer *v, void *);
    Widget samplingField;
    Widget samplingWheel;

    void setRenderer(InvRenderer *rd)
    {
        renderer_ = rd;
    };
    float getSamplingRate();

protected:
    // Constructor/Destructor
    InvFullViewer(
        Widget parent,
        const char *name,
        SbBool buildInsideParent,
        InvFullViewer::BuildFlag flag,
        InvViewer::Type type,
        SbBool buildNow);
    ~InvFullViewer();

    // general decoration vars
    SbBool decorationFlag;
    Widget mgrWidget; // form which manages all other widgets
    Widget raWidget; // render area widget
    Widget leftTrimForm, bottomTrimForm, rightTrimForm;

    // thumb wheel vars
    Widget rightWheel, bottomWheel, leftWheel;
    char *rightWheelStr, *bottomWheelStr, *leftWheelStr;
    float rightWheelVal, bottomWheelVal, leftWheelVal;
    Widget rightWheelLabel, bottomWheelLabel, leftWheelLabel;

    Widget deleteAnnoWidget;
    Widget editAnnoWidget;
    Widget normCursorWidget;
    Widget presCursorWidget;

    // widget list of viewer buttons
    SbPList *viewerButtonWidgets;

    // The button widget should be used as the parent widget
    // when creating new buttons
    Widget getButtonWidget() const
    {
        return appButtonForm;
    }

    // popup menu vars
    SbBool popupEnabled;
    Widget popupWidget, *popupToggleWidgets;
    Widget *drawStyleWidgets, *bufferStyleWidgets;
    char *popupTitle;

    //
    // Build/destroy decoration routines
    //
    Widget buildWidget(Widget parent);
    void buildLeftWheel(Widget parent);

    virtual void buildDecoration(Widget parent);
    virtual Widget buildLeftTrim(Widget parent);
    virtual Widget buildBottomTrim(Widget parent);
    virtual Widget buildRightTrim(Widget parent);
    Widget buildAppButtons(Widget parent);
    Widget buildViewerButtons(Widget parent);
    virtual void createViewerButtons(Widget parent);

    //
    // popup menu build routines
    //
    virtual void buildPopupMenu();
    virtual void destroyPopupMenu();
    void setPopupMenuString(const char *name);
    Widget buildFunctionsSubmenu(Widget popup);
    Widget buildDrawStyleSubmenu(Widget popup);
    Widget buildCursorSubmenu(Widget popup);
    virtual void setPresentationCursor(bool onOrOff);
    virtual bool getPresentationCursor();

    //
    // Preference sheet build routines
    //
    void setPrefSheetString(const char *name);
    virtual void createPrefSheet();
    void createPrefSheetShellAndForm(Widget &shell, Widget &form);
    void createDefaultPrefSheetParts(Widget widgetList[],
                                     int &num, Widget form);
    void layoutPartsAndMapPrefSheet(Widget widgetList[],
                                    int num, Widget form, Widget shell);
    Widget createSeekPrefSheetGuts(Widget parent);
    Widget createSeekDistPrefSheetGuts(Widget parent);
    Widget createZoomPrefSheetGuts(Widget parent);
    Widget createColorsPrefSheetGuts(Widget parent);
    Widget createClippingPrefSheetGuts(Widget parent);
    Widget createStereoPrefSheetGuts(Widget parent);
    Widget createSpeedPrefSheetGuts(Widget parent);
    // Uwe Woessner (tracking)
    Widget createTrackingPrefSheetGuts(Widget parent);
    Widget createSamplingPrefSheetGuts(Widget parent);

    // Subclasses SHOULD redefine these to do viewer specific tasks
    // since these do nothing in the base class. those routines are
    // called by the thumb wheels whenever they rotate.
    virtual void rightWheelMotion(float);
    virtual void bottomWheelMotion(float);
    virtual void leftWheelMotion(float);

    // Subclasses can redefine these to add viewer functionality. They
    // by default call start and finish interactive viewing methods.
    // (defined in the base class).
    virtual void rightWheelStart();
    virtual void bottomWheelStart();
    virtual void leftWheelStart();
    virtual void rightWheelFinish();
    virtual void bottomWheelFinish();
    virtual void leftWheelFinish();

    // Subclasses SHOULD set those wheel string to whatever functionality
    // name they are redefining the thumb wheels to do. Default names are
    // "Motion X, "Motion Y" and "Motion Z" for bottom, left and right wheels.
    void setBottomWheelString(const char *name);
    void setLeftWheelString(const char *name);
    void setRightWheelString(const char *name);

    // Subclasses SHOULD redefine the openViewerHelpCard() routine to bring
    // their help card (called by push button and popup menu entry).
    // They can simply call SoXtComponent::openHelpCard() to open their
    // specific help card.
    virtual void openViewerHelpCard();

private:
    SbBool firstBuild; // set FALSE after buildWidget called once

    // collaborative enhancements
    void sendViewing(int onoroff);
    void sendDecoration(int onoroff);

    // app button vars
    Widget appButtonForm;
    SbPList *appButtonList;
    void doAppButtonLayout(int start);

    // this is called the first time the widget is built
    void getResources(SoXtResource *xr);

    // popup menu callbacks
    static void popMenuCallback(Widget, InvFullViewer *, XEvent *, Boolean *);
    static void drawStyleMenuPick(Widget, int id, void *);
    static void bufferStyleMenuPick(Widget, int id, void *);
    static void menuPick(Widget, int id, XmAnyCallbackStruct *);
    static void menuDisplay(Widget, InvFullViewer *, XtPointer *);

    // push buttons vars and callbacks
    InvPixmapButton *buttonList[10];
    static void pushButtonCB(Widget, int id, void *);

    // pref sheet variables
    Widget prefSheetShellWidget;
    char *prefSheetStr;
    static void prefSheetDestroyCB(Widget, InvFullViewer *, void *);

    // seek pref sheet callbacks
    static void seekPrefSheetFieldCB(Widget, InvFullViewer *, void *);
    static void seekPrefSheetToggle1CB(Widget, Widget, void *);
    static void seekPrefSheetToggle2CB(Widget, Widget, void *);

    // zoom pref sheet callbacks
    Widget *zoomWidgets;
    SbVec2f zoomSldRange;
    void setCameraZoom(float zoom);
    float getCameraZoom();
    void setZoomSliderPosition(float zoom);
    void setZoomFieldString(float zoom);
    static void zoomSliderCB(Widget, InvFullViewer *, XtPointer *);
    static void zoomFieldCB(Widget, InvFullViewer *, XtPointer *);
    static void zoomPrefSheetMinFieldCB(Widget, InvFullViewer *, void *);
    static void zoomPrefSheetMaxFieldCB(Widget, InvFullViewer *, void *);

    // seek dist pref sheet vars and callbacks
    int seekDistWheelVal;
    Widget seekDistField;
    static void seekDistWheelCB(Widget, InvFullViewer *v, XtPointer *d);
    static void seekDistFieldCB(Widget, InvFullViewer *, void *);
    static void seekDistPercPrefSheetToggleCB(Widget, Widget, void *);
    static void seekDistAbsPrefSheetToggleCB(Widget, Widget, void *);

    // clipping plane pref sheet callbacks and variables
    Widget clipWheelForm;
    int clipNearWheelVal, clipFarWheelVal;
    Widget clipNearField, clipFarField;
    static void clipPrefSheetToggleCB(Widget, Widget, void *);
    static void clipNearWheelCB(Widget, InvFullViewer *v, XtPointer *d);
    static void clipFarWheelCB(Widget, InvFullViewer *v, XtPointer *d);
    static void clipFieldCB(Widget, InvFullViewer *, void *);

    // stereo viewing pref sheet callbacks and variables
    Widget stereoWheelForm, stereoField, stereoLabel, stippToggle;
    int stereoWheelVal;
    static void stereoPrefSheetToggleCB(Widget, Widget, void *);
    static void stereoWheelCB(Widget, InvFullViewer *, XtPointer *);
    static void stereoFieldCB(Widget, InvFullViewer *, void *);

    // dolly speed pref sheet callbacks
    static void speedIncPrefSheetButtonCB(Widget, InvFullViewer *, void *);
    static void speedDecPrefSheetButtonCB(Widget, InvFullViewer *, void *);

    // callback functions for thumbwheels
    static void rightWheelCB(Widget, InvFullViewer *v, XtPointer *d);
    static void bottomWheelCB(Widget, InvFullViewer *v, XtPointer *d);
    static void leftWheelCB(Widget, InvFullViewer *v, XtPointer *d);

    InvRenderer *renderer_; // we should know it in order to establish
    //popup-menu -> renderer window interaction
};
#endif /* _INV_FULL_VIEWER_ */
