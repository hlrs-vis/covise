/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_XT_COLOR_WHEEL_
#define _INV_XT_COLOR_WHEEL_

/* $Id: InvColorWheel.h,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $ */

/* $Log: InvColorWheel.h,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include <Inventor/SbColor.h>
#include <Inventor/SbLinear.h>
#include <Inventor/misc/SoCallbackList.h>
#include <Inventor/Xt/SoXtGLWidget.h>

class SoXtMouse;

// callback function prototypes
typedef void MyColorWheelCB(void *userData, const float hsv[3]);

//////////////////////////////////////////////////////////////////////////////
//
//  Class: MyColorWheel
//
//	Lets you interactively select colors using a color wheel. User register
//  callback(s) to be notified when a new color has been selected. There is
//  also a call to tell the color wheel what the current color is when it is
//  changed externally.
//
//////////////////////////////////////////////////////////////////////////////

// C-api: prefix=SoXtColWhl
class MyColorWheel : public SoXtGLWidget
{

public:
    MyColorWheel(
        Widget parent = NULL,
        const char *name = NULL,
        SbBool buildInsideParent = TRUE);
    ~MyColorWheel();

    //
    // Routine to tell the color wheel what the current HSV color is.
    //
    // NOTE: if calling setBaseColor() changes the marker position the
    // valueChanged callbacks will be called with the new hsv color.
    //
    // C-api: name=setBaseCol
    void setBaseColor(const float hsv[3]);
    // C-api: name=getBaseCol
    const float *getBaseColor()
    {
        return hsvColor;
    }

    //
    // This routine sets the WYSIWYG (What You See Is What You Get) mode.
    // When WYSIWYG is on the colors on the wheel will reflect the current
    // color intensity (i.e. get darker and brighter)
    //
    void setWYSIWYG(SbBool trueOrFalse); // default FALSE
    SbBool isWYSIWYG()
    {
        return WYSIWYGmode;
    }

    //
    // Those routines are used to register callbacks for the different
    // color wheel actions.
    //
    // NOTE: the start and finish callbacks are only to signal when the mouse
    // goes down and up. No valid callback data is passed (NULL passed).
    //
    // C-api: name=addStartCB
    void addStartCallback(
        MyColorWheelCB *f,
        void *userData = NULL)
    {
        startCallbacks->addCallback((SoCallbackListCB *)f, userData);
    }

    // C-api: name=addValueChangedCB
    void addValueChangedCallback(
        MyColorWheelCB *f,
        void *userData = NULL)
    {
        changedCallbacks->addCallback((SoCallbackListCB *)f, userData);
    }

    // C-api: name=addFinishCB
    void addFinishCallback(
        MyColorWheelCB *f,
        void *userData = NULL)
    {
        finishCallbacks->addCallback((SoCallbackListCB *)f, userData);
    }

    // C-api: name=removeStartCB
    void removeStartCallback(
        MyColorWheelCB *f,
        void *userData = NULL)
    {
        startCallbacks->removeCallback((SoCallbackListCB *)f, userData);
    }

    // C-api: name=removeValueChangedCB
    void removeValueChangedCallback(
        MyColorWheelCB *f,
        void *userData = NULL)
    {
        changedCallbacks->removeCallback((SoCallbackListCB *)f, userData);
    }

    // C-api: name=removeFinishCB
    void removeFinishCallback(
        MyColorWheelCB *f,
        void *userData = NULL)
    {
        finishCallbacks->removeCallback((SoCallbackListCB *)f, userData);
    }

    // true while the color is changing interactively
    SbBool isInteractive()
    {
        return interactive;
    }

protected:
    // This constructor takes a boolean whether to build the widget now.
    // Subclasses can pass FALSE, then call MyColorWheel::buildWidget()
    // when they are ready for it to be built.
    SoEXTENDER
    MyColorWheel(
        Widget parent,
        const char *name,
        SbBool buildInsideParent,
        SbBool buildNow);

    Widget buildWidget(Widget parent);

private:
    // redefine these to do color wheel specific things
    virtual void redraw();
    virtual void redrawOverlay();
    virtual void processEvent(XAnyEvent *anyevent);
    virtual void initOverlayGraphic();
    virtual void sizeChanged(const SbVec2s &newSize);

    // color wheels local variables
    SbBool WYSIWYGmode;
    SbBool blackMarker;
    float hsvColor[3];
    short cx, cy, radius;
    SbColor *defaultColors, *colors;
    SbVec2f *geometry;
    SoXtMouse *mouse;

    // callback variables
    SoCallbackList *startCallbacks;
    SoCallbackList *changedCallbacks;
    SoCallbackList *finishCallbacks;
    SbBool interactive;

    // routines to make the wheel geometry, colors, draw it....
    void makeWheelGeometry();
    void makeWheelColors(SbColor *col, float intensity);
    void drawWheelSurrounding();
    void drawWheelColors();
    void checkMarkerColor();
    void drawWheelMarker();
    void moveWheelMarker(short x, short y);

    // this is called by both constructors
    void constructorCommon(SbBool buildNow);
};
#endif /* _INV_XT_COLOR_WHEEL_ */
