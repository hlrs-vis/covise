/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_XT_COLOR_EDITOR_
#define _INV_XT_COLOR_EDITOR_

/* $Id: InvColorEditor.h,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $ */

/* $Log: InvColorEditor.h,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include <Xm/Xm.h>
#include <Inventor/SbColor.h>
#include <Inventor/misc/SoCallbackList.h>
#include <Inventor/Xt/SoXtComponent.h>

class SoBase;
class SoNodeSensor;
class SoMFColor;
class SoPathList;
class SoSFColor;
class SoSensor;
class SoXtClipboard;
class MyColorPatch;
class MyColorWheel;
class MyColorSlider;

struct ColorEditorCBData;

// callback function prototypes
typedef void MyColorEditorCB(void *userData, const SbColor *color);

//////////////////////////////////////////////////////////////////////////////
//
//  Class: MyColorEditor
//
//	MyColorEditor class which lets you interactively edit a color.
//
//////////////////////////////////////////////////////////////////////////////

// C-api: prefix=SoXtColEd
class MyColorEditor : public SoXtComponent
{

public:
    //
    // list of possible slider combinations, which is used to specify
    // which sliders should be displayed at any time.
    //
    enum Sliders
    {
        NONE,
        INTENSITY, // default
        RGB,
        HSV,
        RGB_V,
        RGB_HSV
    };

    // UpdateFrequency is how often new values should be sent
    // to the node or the callback routine.
    enum UpdateFrequency
    {
        CONTINUOUS, // send updates with every mouse motion
        AFTER_ACCEPT // only send updates after user hits accept button
    };

    // Constructor/Destructor
    MyColorEditor(
        Widget parent = NULL,
        const char *name = NULL,
        SbBool buildInsideParent = TRUE);
    ~MyColorEditor();

    //
    // Routines for attaching/detaching the editor to a color field. It uses a
    // sensor on the color field to automatically update itself when the
    // color is changed externally.
    //
    // NOTE: the node containing the field needs to also be passed to attach
    // the sensor to it (since field sensors are not yet suported).
    //
    // NOTE: it can only be attached to either a single field or a multiple
    // field at any given time.
    //
    // C-api: name=attachSF
    void attach(SoSFColor *color, SoBase *node);
    // C-api: name=attachMF
    void attach(SoMFColor *color, int index, SoBase *node);
    void detach();
    SbBool isAttached()
    {
        return attached;
    }

    //
    // Additional way of using the color editor, by registering a callback
    // and setting the color. At the time dictated by setUpdateFrequency
    // the callbacks will be called with the new color.
    //
    // NOTE: this is independent to the attach/detach routines, and
    // therefore can be used in conjunction.
    //
    // C-api: name=addColChangedCB
    inline void addColorChangedCallback(
        MyColorEditorCB *f,
        void *userData = NULL);
    // C-api: name=removeColChangedCB
    inline void removeColorChangedCallback(
        MyColorEditorCB *f,
        void *userData = NULL);

    //
    // Sets/gets the color displayed by the color editor.
    //
    // NOTE: setColor() will call colorChanged callbacks if the color
    // differs.
    //
    // C-api: name=setCol
    void setColor(const SbColor &color);
    // C-api: name=getCol
    const SbColor &getColor()
    {
        return baseRGB;
    }

    //
    // Sets/gets the WYSIWYG mode. (default OFF).
    //
    void setWYSIWYG(SbBool trueOrFalse);
    SbBool isWYSIWYG()
    {
        return WYSIWYGmode;
    }

    //
    // Sets/gets which slider should be displayed. (default INTENSITY)
    //
    // C-api: name=setCurSldrs
    void setCurrentSliders(MyColorEditor::Sliders whichSliders);
    // C-api: name=getCurSldrs
    MyColorEditor::Sliders getCurrentSliders()
    {
        return whichSliders;
    }

    //
    // Set/get the update frequency of when colorChanged callbacks should
    // be called. (default CONTINUOUS).
    //
    // C-api: name=setUpdateFreq
    void setUpdateFrequency(MyColorEditor::UpdateFrequency freq);
    // C-api: name=getUpdateFreq
    MyColorEditor::UpdateFrequency getUpdateFrequency()
    {
        return updateFreq;
    }

protected:
    // This constructor takes a boolean whether to build the widget now.
    // Subclasses can pass FALSE, then call MyColorEditor::buildWidget()
    // when they are ready for it to be built.
    SoEXTENDER
    MyColorEditor(
        Widget parent,
        const char *name,
        SbBool buildInsideParent,
        SbBool buildNow);

    // redefine these
    virtual const char *getDefaultWidgetName() const;
    virtual const char *getDefaultTitle() const;
    virtual const char *getDefaultIconTitle() const;

private:
    // redefine these to do colorEditor specific things
    Widget buildWidget(Widget parent);
    static void visibilityChangeCB(void *pt, SbBool visible);

    // local variables
    Widget mgrWidget; // form manages all child widgets
    SbBool WYSIWYGmode;
    Sliders whichSliders;
    SbColor baseRGB;
    float baseHSV[3];
    SbBool ignoreCallback;
    MyColorSlider *sliders[6];
    MyColorWheel *wheel;
    MyColorPatch *current, *previous;
    ColorEditorCBData *dataId;
    SbPList menuItems; // Widgets
    MyColorEditor::UpdateFrequency updateFreq;

    // attach/detach variables
    SbBool attached;
    SoBase *editNode;
    SoSFColor *colorSF;
    SoMFColor *colorMF;
    SoNodeSensor *colorSensor;
    SoCallbackList *callbackList;
    int index;

    // copy/paste support
    SoXtClipboard *clipboard;
    void copy(Time eventTime);
    void paste(Time eventTime);
    void pasteDone(SoPathList *pathList);
    static void pasteDoneCB(void *userData, SoPathList *pathList);

    // list of widgets which need to be accessed
    Widget acceptButton, slidersForm, buttonsForm, wheelForm;

    // build/destroy routines
    Widget buildPulldownMenu(Widget parent);
    Widget buildControls(Widget parent);
    Widget buildSlidersForm(Widget parent);

    void doSliderLayout();
    void doDynamicTopLevelLayout();
    int numberOfSliders(MyColorEditor::Sliders slider);

    // do the updates - if attached, update the node; if callback, call it.
    void doUpdates();

    // color field sensor callback and routine
    void fieldChanged();
    static void fieldChangedCB(void *, SoSensor *);

    // callbacks and actual routine from sliders, wheel, buttons, menu...
    static void wheelCallback(void *, const float hsv[3]);
    void wheelChanged(const float hsv[3]);
    static void sliderCallback(void *, float);
    void sliderChanged(short id, float value);
    static void buttonsCallback(Widget, ColorEditorCBData *, XtPointer);
    void buttonPressed(short id);
    static void editMenuCallback(Widget, ColorEditorCBData *, XmAnyCallbackStruct *);
    static void sliderMenuCallback(Widget, ColorEditorCBData *, XtPointer);

    static void menuDisplay(Widget, MyColorEditor *editor, XtPointer);

    // this is called by both constructors
    void constructorCommon(SbBool buildNow);
};

// Inline functions
void
MyColorEditor::addColorChangedCallback(
    MyColorEditorCB *f,
    void *userData)
{
    callbackList->addCallback((SoCallbackListCB *)f, userData);
}

void
MyColorEditor::removeColorChangedCallback(
    MyColorEditorCB *f,
    void *userData)
{
    callbackList->removeCallback((SoCallbackListCB *)f, userData);
}
#endif /* _INV_XT_COLOR_EDITOR_ */
