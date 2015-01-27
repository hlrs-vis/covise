/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_XT_THUMB_WHEEL_
#define _INV_XT_THUMB_WHEEL_

/* $Id: InvThumbWheel.h,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $ */

/* $Log: InvThumbWheel.h,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include <Inventor/Xt/SoXtGLWidget.h>
#include <Inventor/misc/SoCallbackList.h>

class SoXtMouse;
class MyFloatCallbackList;

// callback function prototypes
typedef void MyThumbWheelCB(void *userData, float val);

//////////////////////////////////////////////////////////////////////////////
//
//  Class: MyThumbWheel
//
//
//////////////////////////////////////////////////////////////////////////////

// C-api: prefix=MyThumbWhl
class MyThumbWheel : public SoXtGLWidget
{

public:
    MyThumbWheel(SbBool horizontal = TRUE);
    MyThumbWheel(
        Widget parent = NULL,
        const char *name = NULL,
        SbBool buildInsideParent = TRUE,
        SbBool horizontal = TRUE);
    ~MyThumbWheel();

    //
    // Routines to specify the wheel value (a rotation given in radians) and get
    // the current wheel value.
    //
    // NOTE: setValue() will call valueChanged callbacks if the value differs.
    //
    void setValue(float radians);
    float getValue()
    {
        return value;
    }

    //
    // Those routines are used to register callbacks for the different thumb wheel
    // actions.
    //
    // NOTE: the start and finish callbacks are only to signal when the mouse
    // goes down and up. No valid callback data is passed (NULL passed).
    //
    // C-api: name=addStartCB
    void addStartCallback(
        MyThumbWheelCB *f,
        void *userData = NULL);
    // C-api: name=addValueChangedCB
    void addValueChangedCallback(
        MyThumbWheelCB *f,
        void *userData = NULL);
    // C-api: name=addFinishCB
    void addFinishCallback(
        MyThumbWheelCB *f,
        void *userData = NULL);

    // C-api: name=removeStartCB
    void removeStartCallback(
        MyThumbWheelCB *f,
        void *userData = NULL);
    // C-api: name=removeValueChangedCB
    void removeValueChangedCallback(
        MyThumbWheelCB *f,
        void *userData = NULL);
    // C-api: name=removeFinishCB
    void removeFinishCallback(
        MyThumbWheelCB *f,
        void *userData = NULL);

    // true while the value is changing interactively
    SbBool isInteractive()
    {
        return interactive;
    }

protected:
    // This constructor takes a boolean whether to build the widget now.
    // Subclasses can pass FALSE, then call SoXtPushButton::buildWidget()
    // when they are ready for it to be built.
    SoEXTENDER
    MyThumbWheel(
        Widget parent,
        const char *name,
        SbBool buildInsideParent,
        SbBool horizontal,
        SbBool buildNow);

    Widget buildWidget(Widget parent);

private:
    // redefine these to do thumb wheel specific things
    virtual void redraw();
    virtual void processEvent(XAnyEvent *anyevent);
    virtual void sizeChanged(const SbVec2s &newSize);

    // local variables
    SbBool horizontal;
    float value;
    int lastPosition;
    SoXtMouse *mouse;

    // callback variables
    MyFloatCallbackList *startCallbacks;
    MyFloatCallbackList *changedCallbacks;
    MyFloatCallbackList *finishCallbacks;
    SbBool interactive;

    // this is called by both constructors
    void constructorCommon(SbBool horizontal, SbBool buildNow);
};
#endif /* _INV_XT_THUMB_WHEEL_ */
