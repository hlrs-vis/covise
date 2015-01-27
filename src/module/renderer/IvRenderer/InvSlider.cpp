/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/* $Log: InvSlider.C,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include <covise/covise.h>

#include <X11/StringDefs.h>
#include <X11/Intrinsic.h>
#include <Xm/Form.h>
#include <Xm/Text.h>
#include <Xm/Label.h>
#include <Xm/LabelG.h>

#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/devices/SoXtMouse.h>
#include <Inventor/errors/SoDebugError.h>
#include "InvUIRegion.h"
#include "InvFloatCallbackList.h"
#include "InvSlider.h"

#include <GL/gl.h>

/*
 * Defines
 */

#define BOX_H 12 // thum height
#define BOX_L 7 // thum width/2
#define SPACE 2
#define THUMB_SPACE 1
#define SIDE (UI_THICK + SPACE + UI_THICK)

// pixels between the slider and the label
#define TEXT_OFFSET 6

////////////////////////////////////////////////////////////////////////
//
// Public constructor - build the widget right now
//
MySlider::MySlider(
    Widget parent,
    const char *name,
    SbBool buildInsideParent)
    : SoXtGLWidget(
          parent,
          name,
          buildInsideParent,
          SO_GLX_RGB,
          FALSE) // tell GLWidget not to build just yet
//
////////////////////////////////////////////////////////////////////////
{
    // In this case, this component is what the app wants, so buildNow = TRUE
    constructorCommon(TRUE);
}

////////////////////////////////////////////////////////////////////////
//
// SoEXTENDER constructor - the subclass tells us whether to build or not
//
MySlider::MySlider(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    SbBool buildNow)
    : SoXtGLWidget(
          parent,
          name,
          buildInsideParent,
          SO_GLX_RGB,
          FALSE) // tell GLWidget not to build just yet
//
////////////////////////////////////////////////////////////////////////
{
    // In this case, this component may be what the app wants,
    // or it may want a subclass of this component. Pass along buildNow
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
MySlider::constructorCommon(SbBool buildNow)
//
//////////////////////////////////////////////////////////////////////
{
    mouse = new SoXtMouse(ButtonPressMask | ButtonMotionMask | ButtonReleaseMask);

    // init local vars
    startCallbacks = new MyFloatCallbackList;
    changedCallbacks = new MyFloatCallbackList;
    finishCallbacks = new MyFloatCallbackList;
    interactive = FALSE;
    value = 0.0;
    sliderWidget = labelWidget = numberWidget = mgrWidget = NULL;
    labelStr = NULL;
    numberVisible = TRUE; // show the number by default

    // slider geometry vars which do not depend on window size
    slx1 = sly1 = SIDE;
    thumy1 = THUMB_SPACE;
    thumy2 = thumy1 + BOX_H;

    sliderSize.setValue(110, 25); // default slider size

    // Build the widget tree, and let SoXtComponent know about our base widget.
    if (buildNow)
    {
        Widget w = buildWidget(getParentWidget());
        setBaseWidget(w);
    }
}

////////////////////////////////////////////////////////////////////////
//
//    Destructor.
//

MySlider::~MySlider()
//
////////////////////////////////////////////////////////////////////////
{
    delete startCallbacks;
    delete changedCallbacks;
    delete finishCallbacks;
    if (labelStr != NULL)
        delete[] labelStr;
    delete mouse;
}

////////////////////////////////////////////////////////////////////////
//
//	This routine draws the entire slider region.
//
// Use: virtual public

void
MySlider::redraw()
//
////////////////////////////////////////////////////////////////////////
{
    if (!isVisible())
        return;
    glXMakeCurrent(getDisplay(), getNormalWindow(), getNormalContext());

    // compute thum position
    position = slx1 + short(value * (slx2 - slx1));
    thumx1 = position - BOX_L;
    thumx2 = position + BOX_L;

    // draw slider surrounding
    SbVec2s size = getGlxSize();
    drawDownUIRegion(0, 0, size[0] - 1, size[1] - 1);

    // now call the routine which does drawing inside the slider top region
    drawSliderTopRegion();
}

////////////////////////////////////////////////////////////////////////
//
//  Sets the label string.
//
// Use: public

void
MySlider::setLabel(const char *newlabel)
//
////////////////////////////////////////////////////////////////////////
{
    // out with the old...
    if (labelStr != NULL)
        delete[] labelStr;
    labelStr = NULL;

    // in with the new...
    if (newlabel != NULL)
    {
        labelStr = new char[strlen(newlabel) + 1];
        strcpy(labelStr, newlabel);
    }

    if (getWidget() != NULL)
        doLabelLayout();
}

////////////////////////////////////////////////////////////////////////
//
//  Sets visibility of the numeric field
//
// Use: public

void
MySlider::setNumericFieldVisible(SbBool setVisible)
//
////////////////////////////////////////////////////////////////////////
{
    if (setVisible)
    {
        if (numberVisible)
            return; // already visible
        else
        {
            // make the number visible
            numberVisible = TRUE;
            if (getWidget() != NULL)
                doNumberLayout();
        }
    }
    else
    {
        if (!numberVisible)
            return; // already not visible
        else
        {
            // make the number not visible
            numberVisible = FALSE;
            if (getWidget() != NULL)
                doNumberLayout();
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
//  lays out the motif label (if any) and slider relative to it.
//
// Use: private

void
MySlider::doLabelLayout()
//
////////////////////////////////////////////////////////////////////////
{
    int n;
    Arg args[10];

    if (labelStr == NULL)
    {

        XtSetArg(args[0], XmNleftAttachment, XmATTACH_FORM);
        XtSetValues(sliderWidget, args, 1);

        if (labelWidget != 0)
        {
            //??? XtUnmanage first?
            XtDestroyWidget(labelWidget);
            labelWidget = 0;
        }
    }
    else
    {
        if (labelWidget == 0)
        {

            labelWidget = XtCreateWidget(labelStr, xmLabelGadgetClass, mgrWidget,
                                         NULL, 0);

            n = 0;
            XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
            n++;
            XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
            n++;
            XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
            n++;
            XtSetArg(args[n], XmNrightAttachment, XmATTACH_NONE);
            n++;
            XtSetValues(labelWidget, args, n);

            n = 0;
            XtSetArg(args[n], XmNleftAttachment, XmATTACH_WIDGET);
            n++;
            XtSetArg(args[n], XmNleftWidget, labelWidget);
            n++;
            XtSetArg(args[n], XmNleftOffset, TEXT_OFFSET);
            n++;
            XtSetValues(sliderWidget, args, n);

            XtManageChild(labelWidget);
        }
        else
        {
            XtSetArg(args[0], XmNname, labelStr);
            XtSetValues(labelWidget, args, 1);
        }
    }
}

////////////////////////////////////////////////////////////////////////
//
//  lays out the motif numeric text field (if any) and slider relative to it.
//
// Use: private

void
MySlider::doNumberLayout()
//
////////////////////////////////////////////////////////////////////////
{
    int n;
    Arg args[10];

    if (!numberVisible)
    {

        XtSetArg(args[0], XmNrightAttachment, XmATTACH_FORM);
        XtSetValues(sliderWidget, args, 1);

        if (numberWidget != NULL)
        {
            //??? XtUnmanage first?
            XtDestroyWidget(numberWidget);
            numberWidget = NULL;
        }
    }
    else
    {
        if (numberWidget == NULL)
        {

            n = 0;
            XtSetArg(args[n], XmNhighlightThickness, 1);
            n++;
            XtSetArg(args[n], XmNcolumns, 4);
            n++;
            numberWidget = XtCreateWidget("sliderText", xmTextWidgetClass,
                                          mgrWidget, args, n);
            XtAddCallback(numberWidget, XmNactivateCallback,
                          (XtCallbackProc)MySlider::textFieldCB, (XtPointer) this);

            n = 0;
            XtSetArg(args[n], XmNtopAttachment, XmATTACH_NONE);
            n++;
            XtSetArg(args[n], XmNleftAttachment, XmATTACH_NONE);
            n++;
            XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
            n++;
            XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
            n++;
            XtSetValues(numberWidget, args, n);

            n = 0;
            XtSetArg(args[n], XmNrightAttachment, XmATTACH_WIDGET);
            n++;
            XtSetArg(args[n], XmNrightWidget, numberWidget);
            n++;
            XtSetValues(sliderWidget, args, n);

            // now set the string to the slider value
            char valStr[6];
            sprintf(valStr, "%.2f", getValue());
            XmTextSetString(numberWidget, valStr);

            XtManageChild(numberWidget);
        }
#ifdef DEBUG
        else
        {
            SoDebugError::post("MySlider::doNumberLayout::menuPick",
                               "INTERNAL ERROR");
        }
#endif
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	This builds the parent Glx widget, then registers interest
// in mouse events.
//
// Use: virtual protected
Widget
MySlider::buildWidget(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    int n;
    Arg args[10];

    //
    // create a top form to hold everything together
    //
    mgrWidget = XtCreateWidget(getWidgetName(), xmFormWidgetClass, parent, NULL, 0);
    // ??? do we need to register this widget class?

    //
    // create MySlider and text widget
    //
    sliderWidget = SoXtGLWidget::buildWidget(mgrWidget);
    SoXt::setWidgetSize(sliderWidget, sliderSize);
    mouse->enable(getNormalWidget(),
                  (XtEventHandler)SoXtGLWidget::eventHandler,
                  (XtPointer) this);

    //
    // do layout
    //

    n = 0;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_FORM);
    n++;
    XtSetValues(sliderWidget, args, n);

    doLabelLayout();
    doNumberLayout();

    // manage that slider
    XtManageChild(sliderWidget);

    return mgrWidget;
}

////////////////////////////////////////////////////////////////////////
//
//  Resize the slider (only) widget
//
// Use: virtual public
void
MySlider::setSliderSize(const SbVec2s &newSize)
//
////////////////////////////////////////////////////////////////////////
{
    if (sliderWidget != NULL)
        SoXt::setWidgetSize(sliderWidget, newSize);
    sliderSize = newSize;
}

////////////////////////////////////////////////////////////////////////
//
//  Returns the slider (only) widget size
//
// Use: virtual public
SbVec2s
MySlider::getSliderSize()
//
////////////////////////////////////////////////////////////////////////
{
    return sliderSize;
}

////////////////////////////////////////////////////////////////////////
//
// Process the passed X event.
//
// Use: virtual public

void
MySlider::processEvent(XAnyEvent *xe)
//
////////////////////////////////////////////////////////////////////////
{
    short x, y;
    XButtonEvent *be;
    XMotionEvent *me;
    SbVec2s size = getGlxSize();

    switch (xe->type)
    {
    case ButtonPress:
        be = (XButtonEvent *)xe;
        if (be->button == Button1)
        {

            startCallbacks->invokeCallbacks(value);
            interactive = TRUE;

            // check for thumb picking region
            x = short(be->x);
            y = short(size[1] - be->y);
            if (x > thumx1 && x < thumx2 && y > thumy1 && y < thumy2)
                posdiff = x - position;
            else
            {
                posdiff = 0;
                setValue((x - slx1 - posdiff) / float(slx2 - slx1));
            }
        }
        break;

    case ButtonRelease:
        be = (XButtonEvent *)xe;
        if (be->button == Button1)
        {
            interactive = FALSE;
            finishCallbacks->invokeCallbacks(value);
        }
        break;

    case MotionNotify:
        me = (XMotionEvent *)xe;
        if (me->state & Button1Mask)
            setValue((me->x - slx1 - posdiff) / float(slx2 - slx1));
        break;
    }
}

////////////////////////////////////////////////////////////////////////
//
//  Sets the slider to this value..
//
// Use: public

void
MySlider::setValue(float v)
//
////////////////////////////////////////////////////////////////////////
{
    // assign the clipped value and check for redraw needs
    float old = value;
    value = (v > 1.0) ? 1.0 : ((v < 0.0) ? 0.0 : v);

    if (value == old)
        return;

    redraw();

    // update text field
    if (numberWidget != NULL)
    {
        char valStr[6];
        sprintf(valStr, "%.2f", value);
        XmTextSetString(numberWidget, valStr);
    }

    changedCallbacks->invokeCallbacks(value);
}

////////////////////////////////////////////////////////////////////////
//
//  This routine is when the window has changed size
//
// Use: virtual protected

void
MySlider::sizeChanged(const SbVec2s &newSize)
//
////////////////////////////////////////////////////////////////////////
{
    // calculate slider parameters which depends on window size
    slx2 = newSize[0] - SIDE - 1;
    sly2 = newSize[1] - SIDE - 1;

    // reset projection
    glXMakeCurrent(getDisplay(), getNormalWindow(), getNormalContext());
    glViewport(0, 0, newSize[0], newSize[1]);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, newSize[0], 0, newSize[1], -1, 1);
}

////////////////////////////////////////////////////////////////////////
//
//  Draws the little white maker. subclasses can redefine this routine to
//  make the slider top region look like anything they want.
//
// Use: virtual protected

void
MySlider::drawSliderTopRegion()
//
////////////////////////////////////////////////////////////////////////
{
    MAIN_UI_COLOR;
    glRects(slx1, sly1, slx2, sly2);
    drawThumbUIRegion(thumx1, thumy1, thumx2, thumy2);
}

void
MySlider::addStartCallback(MySliderCB *f, void *userData)
{
    startCallbacks->addCallback((MyFloatCallbackListCB *)f, userData);
}

void
MySlider::addValueChangedCallback(MySliderCB *f, void *userData)
{
    changedCallbacks->addCallback((MyFloatCallbackListCB *)f, userData);
}

void
MySlider::addFinishCallback(MySliderCB *f, void *userData)
{
    finishCallbacks->addCallback((MyFloatCallbackListCB *)f, userData);
}

void
MySlider::removeStartCallback(MySliderCB *f, void *userData)
{
    startCallbacks->removeCallback((MyFloatCallbackListCB *)f, userData);
}

void
MySlider::removeValueChangedCallback(MySliderCB *f, void *userData)
{
    changedCallbacks->removeCallback((MyFloatCallbackListCB *)f, userData);
}

void
MySlider::removeFinishCallback(MySliderCB *f, void *userData)
{
    finishCallbacks->removeCallback((MyFloatCallbackListCB *)f, userData);
}

//
////////////////////////////////////////////////////////////////////////
// static callbacks stubs
////////////////////////////////////////////////////////////////////////
//

//
// called whenever the user types a new value in the text field.
//
void
MySlider::textFieldCB(Widget w, MySlider *p, XtPointer)
{
    // get text value from the label
    char *str = XmTextGetString(w);
    float val;
    if (sscanf(str, "%f", &val))
        p->setValue(val);
    else
    {
        // reformat the text field
        char valStr[6];
        sprintf(valStr, "%.2f", p->getValue());
        XmTextSetString(w, valStr);
    }
    XtFree(str);

    // make the text field loose the focus
    XmProcessTraversal(SoXt::getShellWidget(w), XmTRAVERSE_CURRENT);
}
