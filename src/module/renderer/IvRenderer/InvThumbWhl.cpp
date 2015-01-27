/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/* $Log: InvThumbWhl.C,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include <math.h>

#include <X11/StringDefs.h>
#include <X11/Intrinsic.h>

#include <Inventor/Xt/devices/SoXtMouse.h>
#include "InvFloatCallbackList.h"
#include "InvThumbWheel.h"
#include "InvUIRegion.h"
#include <GL/gl.h>

/*
 * Defines
 */

#define TICK_NUM 21
#define PART1 4
#define PART2 5
#define PART3 7
#define PART4 9
#define UI_THICK 3

#define RECT(x1, y1, x2, y2) \
    glBegin(GL_LINE_LOOP);   \
    glVertex2s(x1, y1);      \
    glVertex2s(x1, y2);      \
    glVertex2s(x2, y2);      \
    glVertex2s(x2, y1);      \
    glEnd();

////////////////////////////////////////////////////////////////////////
//
// Public constructor - build the widget right now
//
MyThumbWheel::MyThumbWheel(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    SbBool horiz)
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
    constructorCommon(horiz, TRUE);
}

////////////////////////////////////////////////////////////////////////
//
// SoEXTENDER constructor - the subclass tells us whether to build or not
//
MyThumbWheel::MyThumbWheel(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    SbBool horiz,
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
    constructorCommon(horiz, buildNow);
}

////////////////////////////////////////////////////////////////////////
//
// Called by the constructors
//
// private
//
void
MyThumbWheel::constructorCommon(SbBool horiz, SbBool buildNow)
//
//////////////////////////////////////////////////////////////////////
{
    mouse = new SoXtMouse(ButtonPressMask | ButtonReleaseMask | ButtonMotionMask);

    // init local vars
    startCallbacks = new MyFloatCallbackList;
    changedCallbacks = new MyFloatCallbackList;
    finishCallbacks = new MyFloatCallbackList;
    interactive = FALSE;
    value = 0.0;
    horizontal = horiz;

    // default size
    if (horizontal)
        setGlxSize(SbVec2s(120, 22));
    else
        setGlxSize(SbVec2s(22, 120));

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
MyThumbWheel::~MyThumbWheel()
//
////////////////////////////////////////////////////////////////////////
{
    delete startCallbacks;
    delete changedCallbacks;
    delete finishCallbacks;
    delete mouse;
}

////////////////////////////////////////////////////////////////////////
//
//  This routine draws the entire thumb wheel region.
//
// Use: virtual protected
//
void
MyThumbWheel::redraw()
//
////////////////////////////////////////////////////////////////////////
{
    if (!isVisible())
        return;

    glXMakeCurrent(getDisplay(), getNormalWindow(), getNormalContext());

    int i, n;
    short x, y, x1, y1, x2, y2;
    short mid, rad, l, d;
    float angle, ang_inc;

    SbVec2s size = getGlxSize();
    x1 = y1 = 0;
    x2 = size[0] - 1;
    y2 = size[1] - 1;

    drawDownUIBorders(x1, y1, x2, y2);
    x1 += UI_THICK;
    y1 += UI_THICK;
    x2 -= UI_THICK;
    y2 -= UI_THICK;

    LIGHT1_UI_COLOR;
    RECT(x1, y1, x2, y2);
    x1++;
    y1++;
    x2--;
    y2--;
    RECT(x1, y1, x2, y2);
    x1++;
    y1++;
    x2--;
    y2--;

    glBegin(GL_LINES);

    if (horizontal)
    {
        DARK3_UI_COLOR;
        glVertex2s(x1, y2);
        glVertex2s(x2, y2);
        BLACK_UI_COLOR;
        glVertex2s(x1, y1);
        glVertex2s(x2, y1);
        y1++;
        y2--;

        l = x2 - x1;
        d = 0;
        BLACK_UI_COLOR;
        glVertex2s(x1 + d, y1);
        glVertex2s(x2 - d, y1);
        glVertex2s(x1 + d, y2);
        glVertex2s(x2 - d, y2);
        d = (short)(l * .06);
        DARK2_UI_COLOR;
        glVertex2s(x1 + d, y1);
        glVertex2s(x2 - d, y1);
        glVertex2s(x1 + d, y2);
        glVertex2s(x2 - d, y2);
        d = (short)(l * .12);
        DARK1_UI_COLOR;
        glVertex2s(x1 + d, y1);
        glVertex2s(x2 - d, y1);
        glVertex2s(x1 + d, y2);
        glVertex2s(x2 - d, y2);
        d = (short)(l * .20);
        MAIN_UI_COLOR;
        glVertex2s(x1 + d, y1);
        glVertex2s(x2 - d, y1);
        glVertex2s(x1 + d, y2);
        glVertex2s(x2 - d, y2);
        d = (short)(l * .30);
        LIGHT1_UI_COLOR;
        glVertex2s(x1 + d, y1);
        glVertex2s(x2 - d, y1);
        glVertex2s(x1 + d, y2);
        glVertex2s(x2 - d, y2);
        d = (short)(l * .40);
        WHITE_UI_COLOR;
        glVertex2s(x1 + d, y1);
        glVertex2s(x2 - d, y1);
        glVertex2s(x1 + d, y2);
        glVertex2s(x2 - d, y2);
        x1++;
        y1++;
        x2--;
        y2--;
    }
    else
    {
        DARK3_UI_COLOR;
        glVertex2s(x1, y1);
        glVertex2s(x1, y2);
        BLACK_UI_COLOR;
        glVertex2s(x2, y1);
        glVertex2s(x2, y2);
        x1++;
        x2--;

        l = y2 - y1;
        d = 0;
        BLACK_UI_COLOR;
        glVertex2s(x1, y1 + d);
        glVertex2s(x1, y2 - d);
        glVertex2s(x2, y1 + d);
        glVertex2s(x2, y2 - d);
        d = (short)(l * .06);
        DARK2_UI_COLOR;
        glVertex2s(x1, y1 + d);
        glVertex2s(x1, y2 - d);
        glVertex2s(x2, y1 + d);
        glVertex2s(x2, y2 - d);
        d = (short)(l * .12);
        DARK1_UI_COLOR;
        glVertex2s(x1, y1 + d);
        glVertex2s(x1, y2 - d);
        glVertex2s(x2, y1 + d);
        glVertex2s(x2, y2 - d);
        d = (short)(l * .20);
        MAIN_UI_COLOR;
        glVertex2s(x1, y1 + d);
        glVertex2s(x1, y2 - d);
        glVertex2s(x2, y1 + d);
        glVertex2s(x2, y2 - d);
        d = (short)(l * .30);
        LIGHT1_UI_COLOR;
        glVertex2s(x1, y1 + d);
        glVertex2s(x1, y2 - d);
        glVertex2s(x2, y1 + d);
        glVertex2s(x2, y2 - d);
        d = (short)(l * .40);
        WHITE_UI_COLOR;
        glVertex2s(x1, y1 + d);
        glVertex2s(x1, y2 - d);
        glVertex2s(x2, y1 + d);
        glVertex2s(x2, y2 - d);
        x1++;
        y1++;
        x2--;
        y2--;
    }

    glEnd();

    MAIN_UI_COLOR;
    glRecti(x1, y1, x2, y2);

    //
    // draw the tick marks
    //

    angle = value;
    ang_inc = M_PI / TICK_NUM;
    n = (int)(floorf(angle / ang_inc));
    angle -= n * ang_inc;

    glBegin(GL_LINES);

    if (horizontal)
    {
        mid = size[0] / 2;
        rad = mid - UI_THICK - 2;

        for (i = 0; i < TICK_NUM; i++)
        {
            x = mid - (short)(cosf(angle) * rad) - 1;

            if (i < PART1 || i > (TICK_NUM - PART1))
            {
                BLACK_UI_COLOR;
                glVertex2s(x, y1);
                glVertex2s(x, y2);
                x++;
                DARK1_UI_COLOR;
                glVertex2s(x, y1);
                glVertex2s(x, y2);
            }
            else if (i < PART2 || i > (TICK_NUM - PART2))
            {
                DARK3_UI_COLOR;
                glVertex2s(x, y1);
                glVertex2s(x, y2);
            }
            else if (i < PART3 || i > (TICK_NUM - PART3))
            {
                LIGHT1_UI_COLOR;
                glVertex2s(x, y1);
                glVertex2s(x, y2);
                x++;
                DARK2_UI_COLOR;
                glVertex2s(x, y1);
                glVertex2s(x, y2);
            }
            else if (i < PART4 || i > (TICK_NUM - PART4))
            {
                WHITE_UI_COLOR;
                glVertex2s(x, y1);
                glVertex2s(x, y2);
                x++;
                DARK2_UI_COLOR;
                glVertex2s(x, y1);
                glVertex2s(x, y2);
            }
            else
            {
                WHITE_UI_COLOR;
                x--;
                glVertex2s(x, y1);
                glVertex2s(x, y2);
                x++;
                glVertex2s(x, y1);
                glVertex2s(x, y2);
                DARK2_UI_COLOR;
                x++;
                glVertex2s(x, y1);
                glVertex2s(x, y2);
            }

            angle += ang_inc;
        }
    }
    else
    {
        mid = size[1] / 2;
        rad = mid - UI_THICK - 2;

        for (i = 0; i < TICK_NUM; i++)
        {
            y = mid - (short)(cosf(angle) * rad);

            if (i < PART1 || i > (TICK_NUM - PART1))
            {
                BLACK_UI_COLOR;
                glVertex2s(x1, y);
                glVertex2s(x2, y);
                y--;
                DARK1_UI_COLOR;
                glVertex2s(x1, y);
                glVertex2s(x2, y);
            }
            else if (i < PART2 || i > (TICK_NUM - PART2))
            {
                DARK3_UI_COLOR;
                glVertex2s(x1, y);
                glVertex2s(x2, y);
            }
            else if (i < PART3 || i > (TICK_NUM - PART3))
            {
                LIGHT1_UI_COLOR;
                glVertex2s(x1, y);
                glVertex2s(x2, y);
                y--;
                DARK2_UI_COLOR;
                glVertex2s(x1, y);
                glVertex2s(x2, y);
            }
            else if (i < PART4 || i > (TICK_NUM - PART4))
            {
                WHITE_UI_COLOR;
                glVertex2s(x1, y);
                glVertex2s(x2, y);
                y--;
                DARK2_UI_COLOR;
                glVertex2s(x1, y);
                glVertex2s(x2, y);
            }
            else
            {
                WHITE_UI_COLOR;
                y++;
                glVertex2s(x1, y);
                glVertex2s(x2, y);
                y--;
                glVertex2s(x1, y);
                glVertex2s(x2, y);
                DARK2_UI_COLOR;
                y--;
                glVertex2s(x1, y);
                glVertex2s(x2, y);
            }

            angle += ang_inc;
        }
    }

    glEnd();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//  	This builds the parent Glx widget, then registers interest
// in mouse events.
//
// Use: protected
Widget
MyThumbWheel::buildWidget(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    Widget w = SoXtGLWidget::buildWidget(parent);

    mouse->enable(getNormalWidget(),
                  (XtEventHandler)SoXtGLWidget::eventHandler,
                  (XtPointer) this);

    return w;
}

////////////////////////////////////////////////////////////////////////
//
//  Process the passed X event.
//
// Use: virtual protected
//
void
MyThumbWheel::processEvent(XAnyEvent *xe)
//
////////////////////////////////////////////////////////////////////////
{
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

            // get starting point
            lastPosition = (horizontal) ? be->x : (size[1] - be->y);
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
        {
            float r;
            if (horizontal)
                r = (me->x - lastPosition) / float(size[0] - 2 * UI_THICK);
            else
                r = (size[1] - me->y - lastPosition) / float(size[1] - 2 * UI_THICK);

            // now rotate wheel
            if (r != 0.0)
            {
                value += r * M_PI;
                changedCallbacks->invokeCallbacks(value);
                redraw();
            }

            lastPosition = (horizontal) ? me->x : (size[1] - me->y);
        }
        break;
    }
}

////////////////////////////////////////////////////////////////////////
//
//  Sets the thumb wheel to this value..
//
// Use: public
//
void
MyThumbWheel::setValue(float v)
//
////////////////////////////////////////////////////////////////////////
{
    if (value == v)
        return;

    value = v;

    // call the callbacks
    changedCallbacks->invokeCallbacks(value);

    redraw();
}

////////////////////////////////////////////////////////////////////////
//
//  This routine is when the window has changed size
//
// Use: virtual protected
//
void
MyThumbWheel::sizeChanged(const SbVec2s &newSize)
//
////////////////////////////////////////////////////////////////////////
{
    glXMakeCurrent(getDisplay(), getNormalWindow(), getNormalContext());
    glViewport(0, 0, newSize[0], newSize[1]);

    // reset projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, newSize[0], 0, newSize[1], -1, 1);
}

void
MyThumbWheel::addStartCallback(MyThumbWheelCB *f, void *userData)
{
    startCallbacks->addCallback((MyFloatCallbackListCB *)f, userData);
}

void
MyThumbWheel::addValueChangedCallback(MyThumbWheelCB *f, void *userData)
{
    changedCallbacks->addCallback((MyFloatCallbackListCB *)f, userData);
}

void
MyThumbWheel::addFinishCallback(MyThumbWheelCB *f, void *userData)
{
    finishCallbacks->addCallback((MyFloatCallbackListCB *)f, userData);
}

void
MyThumbWheel::removeStartCallback(MyThumbWheelCB *f, void *userData)
{
    startCallbacks->removeCallback((MyFloatCallbackListCB *)f, userData);
}

void
MyThumbWheel::removeValueChangedCallback(MyThumbWheelCB *f, void *userData)
{
    changedCallbacks->removeCallback((MyFloatCallbackListCB *)f, userData);
}

void
MyThumbWheel::removeFinishCallback(MyThumbWheelCB *f, void *userData)
{
    finishCallbacks->removeCallback((MyFloatCallbackListCB *)f, userData);
}
