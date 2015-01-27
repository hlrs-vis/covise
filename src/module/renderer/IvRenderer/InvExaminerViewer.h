/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_RE_EXAMINER_VIEWER_
#define _INV_RE_EXAMINER_VIEWER_

//**************************************************************************
//
// * Description    : Inventor examiner viewer adoption (RE version)
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

#include "InvFullViewer.h"
#include <Inventor/SbLinear.h>

// classes used
class SbSphereSheetProjector;
class SoFieldSensor;
class SoSeparator;
class SoSwitch;
class SoTranslation;
class SoScale;
class InvPixmapButton;

//////////////////////////////////////////////////////////////////////////////
//
//  Class: InvExaminerViewer
//
//	Examiner viewer - allows the user to change the camera position
//  by spinning a sphere in front of the viewpoint.
//
//
//	Keys used by this viewer:
//	-------------------------
//
//	Left Mouse: Tumbles the virtual trackball.
//
//	Middle Mouse:
//	Ctrl + Left Mouse: Translate up, down, left and right.
//
//	Ctrl + Middle Mouse:
//	Left + Middle Mouse: Dolly in and out (gets closer to and further
//			    away from the object).
//
//	<s> + click: Alternative to the Seek button. Press (but do not hold
//		    down) the <s> key, then click on a target object.
//
//	Right Mouse: Open the popup menu.
//
//////////////////////////////////////////////////////////////////////////////

// collaborative enhancements
extern void rm_sendProjection(char *message);

// C-api: prefix=SoXtExamVwr
class InvExaminerViewer : public InvFullViewer
{
public:
    InvExaminerViewer(
        Widget parent = NULL,
        const char *name = NULL,
        SbBool buildInsideParent = TRUE,
        InvFullViewer::BuildFlag flag = BUILD_ALL,
        InvViewer::Type type = BROWSER);
    // C-api: interpret #define SoXtExamVwrCreateStd(parent,name) SoXtExamVwrCreate(parent,name,TRUE,SO_XT_FULL_VWR_BUILD_ALL,SO_XT_VWR_BROWSER)
    ~InvExaminerViewer();

    //
    // Show/hide the point of rotation feedback, which only appears while
    // in Viewing mode. (default OFF)
    //
    // C-api: name=setFeedbackVis
    void setFeedbackVisibility(SbBool onOrOff);
    // C-api: name=isFeedbackVis
    SbBool isFeedbackVisible() const
    {
        return feedbackFlag;
    }

    //
    // Set/get the point of rotation feeedback size in pixels (default 20).
    //
    void setFeedbackSize(int newSize);
    int getFeedbackSize() const
    {
        return (int)feedbackSize;
    }

    //
    // Enable/disable the animation feature of the viewer.
    // (enabled by default)
    //
    // C-api: name=SetAnimEnabled
    void setAnimationEnabled(SbBool onOrOff);
    // C-api: name=IsAnimEnabled
    SbBool isAnimationEnabled()
    {
        return animationEnabled;
    }

    //
    // Stop animation, if it is occurring, and queuery if the viewer is
    // currently animating.
    //
    // C-api: name=StopAnim
    void stopAnimating();
    // C-api: name=IsAnim
    SbBool isAnimating()
    {
        return animatingFlag;
    }

    //
    // redefine these to add Examiner viewer functionality
    //
    virtual void viewAll();
    virtual void resetToHomePosition();
    virtual void setCamera(SoCamera *cam);
    virtual void setViewing(SbBool onOrOff);
    virtual void setCursorEnabled(SbBool onOrOff);

    virtual void setPresentationCursor(bool onOrOff);
    virtual bool getPresentationCursor();

protected:
    // This constructor takes a boolean whether to build the widget now.
    // Subclasses can pass FALSE, then call InvExaminerViewer::buildWidget()
    // when they are ready for it to be built.
    SoEXTENDER
    InvExaminerViewer(
        Widget parent,
        const char *name,
        SbBool buildInsideParent,
        InvFullViewer::BuildFlag flag,
        InvViewer::Type type,
        SbBool buildNow);

    // redefine these
    virtual const char *getDefaultWidgetName() const;
    virtual const char *getDefaultTitle() const;
    virtual const char *getDefaultIconTitle() const;

    // redefine those routines to do viewer specific stuff
    virtual void processEvent(XAnyEvent *anyevent);
    virtual void setSeekMode(SbBool onOrOff);
    virtual void actualRedraw();

    // Get X resources for the widget.
    Widget buildWidget(Widget parent);

    // Define those thumb wheels to rotate the object
    virtual void bottomWheelMotion(float newVal);
    virtual void leftWheelMotion(float newVal);
    virtual void rightWheelMotion(float newVal);

    // redefine those routines to also stop animation if any
    virtual void bottomWheelStart();
    virtual void leftWheelStart();

    // add viewer preference stuff
    virtual void createPrefSheet();

    // add some viewer buttons
    virtual void createViewerButtons(Widget parent);

    // Define this to bring the viewer help card
    virtual void openViewerHelpCard();

private:
    // collaborative enhancements
    void sendProjection();

    // viewer state variables
    int mode;
    SbBool createdCursors;
    bool presentationCursor_;
    Cursor spinCursor, panCursor, dollyCursor, seekCursor;
    Cursor presSpinCursor_, presPanCursor_, presDollyCursor_;
    Cursor presSeekCursor_, presArrowCursor_;
    SbSphereSheetProjector *sphereSheet;
    SbVec2s locator; // mouse position
    SbBool firstBuild; // set FALSE after buildWidget called once

    // point of rotation feeedback vars
    SbBool feedbackFlag;
    float feedbackSize;
    SoSeparator *feedbackRoot;
    SoSwitch *feedbackSwitch;
    SoTranslation *feedbackTransNode;
    SoScale *feedbackScaleNode;
    static const char *geometryBuffer;
    void createFeedbackNodes();

    // variables used for doing spinning animation
    SbBool animationEnabled, animatingFlag;
    SoFieldSensor *animationSensor;
    SbRotation *rotBuffer;
    int firstIndex, lastIndex;
    int32_t lastMotionTime;
    SbRotation averageRotation;
    SbBool computeAverage;
    static void animationSensorCB(void *v, SoSensor *s);
    static void visibilityChangeCB(void *pt, SbBool visible);
    void doSpinAnimation();

    // camera panning vars
    SbVec3f locator3D;
    SbPlane focalplane;

    void updateViewerMode(unsigned int state);
    void switchMode(int newMode);
    void updateCursor();
    void defineCursors();
    void rotateCamera(const SbRotation &rot);
    void panCamera(const SbVec2f &newLocator);
    void spinCamera(const SbVec2f &newLocator);
    void dollyCamera(const SbVec2s &newLocator);

    // preference sheet stuff
    Widget createExamPrefSheetGuts(Widget parent);
    static void animPrefSheetToggleCB(Widget, InvExaminerViewer *, void *);
    static void normalPrefSheetToggleCB(Widget, InvExaminerViewer *, void *);

    // point of rotation pref sheet stuff
    int feedbackSizeWheelVal;
    Widget feedbackLabel[2], feedbackField, feedbackSizeWheel;
    static void feedbackSizeWheelCB(Widget, InvExaminerViewer *p, XtPointer *d);
    static void feedbackSizeFieldCB(Widget, InvExaminerViewer *, void *);
    static void feedbackPrefSheetToggleCB(Widget, InvExaminerViewer *, void *);
    void toggleFeedbackWheelSize(Widget toggle);

    // push button vars and callbacks
    InvPixmapButton *buttonList[10];
    static void camPushCB(Widget, InvExaminerViewer *, void *);

    // this is called by both constructors
    void constructorCommon(SbBool buildNow);

    // setup of overlay planes - find the transparent pixel
    void setupOverlays();
};
#endif /* _INV_RE_EXAMINER_VIEWER_ */
