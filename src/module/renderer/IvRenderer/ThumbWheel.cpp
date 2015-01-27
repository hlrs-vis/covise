/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#if defined(__hpux) || defined(__linux)

/*
 *
 *  Copyright (C) 2000 Silicon Graphics, Inc.  All Rights Reserved.
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  Further, this software is distributed without any warranty that it is
 *  free of the rightful claim of any third person regarding infringement
 *  or the like.  Any license provided herein, whether implied or
 *  otherwise, applies only to this software file.  Patent licenses, if
 *  any, provided herein do not apply to combinations of this program with
 *  other software, or any other product whatsoever.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *  Contact information: Silicon Graphics, Inc., 1600 Amphitheatre Pkwy,
 *  Mountain View, CA  94043, or:
 *
 *  http://www.sgi.com
 *
 *  For further information regarding this notice, see:
 *
 *  http://oss.sgi.com/projects/GenInfo/NoticeExplan/
 *
 */

/*
 * ThumbWheel.c : Thumbwheel ("infinite scrollbar") widget.
 */

#include <Xm/XmP.h>
#include <X11/StringDefs.h>

#include <stdio.h>

#include "ThumbWheelP.h"

#ifndef MAX
#define MAX(a, b) ((a) > (b)) ? (a) : (b)
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#endif

typedef Arg *Arglist;

#define VIEWABLE_ANGLE 150
#define WHEEL_LONG_DIMENSION 122
#define WHEEL_NARROW_DIMENSION 16
#define BUTTON_SIZE 16

#define PixelsToAngleFactor(w) ((float)VIEWABLE_ANGLE / (float)((w)->thumbWheel.viewable_pixels))
#define AngleToUserUnitFactor(w) ((w)->thumbWheel.infinite ? ((float)((w)->thumbWheel.angle_factor) / (float)360) : ((float)((w)->thumbWheel.upper_bound - (w)->thumbWheel.lower_bound) \
                                                                                                                     / (float)((w)->thumbWheel.angle_range)))
#define DimensionToSigned(num) ((sizeof(int) > sizeof(Dimension)) && ((num) > (1 << (sizeof(Dimension) - 1)) - 1)) ? (int)(num) - (1 << sizeof(Dimension)) : (int)(num)
#define WheelDrawLine(thumb, dpy, pix, gc, x1, y1, x2, y2) \
    (((thumb)->thumbWheel.orientation == XmHORIZONTAL) ? XDrawLine(dpy, pix, gc, y1, x1, y2, x2) : XDrawLine(dpy, pix, gc, x1, y1, x2, y2))

#ifdef __ICC
#define OFFSET(field) XtOffset(SgThumbWheelRec *, field)
#else
#define OFFSET(field) XtOffsetOf(SgThumbWheelRec, field)
#endif

#ifdef NO_PROTO
static void ValueDefaultProc();
#else
static void ValueDefaultProc(SgThumbWheelWidget w, int offset, XrmValue *v);
#endif /* _NO_PROTO */

static XtResource resources[] = {
    {
      XmNminimum,
      XmCMinimum,
      XmRInt,
      sizeof(int),
      OFFSET(thumbWheel.lower_bound),
      XmRImmediate,
      (XtPointer)0,
    },
    {
      XmNmaximum,
      XmCMaximum,
      XmRInt,
      sizeof(int),
      OFFSET(thumbWheel.upper_bound),
      XmRImmediate,
      (XtPointer)100,
    },
    {
      SgNhomePosition,
      SgCHomePosition,
      XmRInt,
      sizeof(int),
      OFFSET(thumbWheel.home_position),
      XmRImmediate,
      (XtPointer)50,
    },
    {
      SgNangleRange,
      SgCAngleRange,
      XmRInt,
      sizeof(int),
      OFFSET(thumbWheel.angle_range),
      XmRImmediate,
      (XtPointer)VIEWABLE_ANGLE,
    },
    {
      SgNunitsPerRotation,
      SgCUnitsPerRotation,
      XmRInt,
      sizeof(int),
      OFFSET(thumbWheel.angle_factor),
      XmRImmediate,
      (XtPointer)240, /* it matches the other defaults, 1.5 degrees per unit */
    },
    {
      XmNvalue,
      XmCValue,
      XmRInt,
      sizeof(int),
      OFFSET(thumbWheel.value),
      XmRCallProc,
      (XtPointer)ValueDefaultProc,
    },
    {
      XmNorientation,
      XmCOrientation,
      XmROrientation,
      sizeof(unsigned char),
      OFFSET(thumbWheel.orientation),
      XmRImmediate,
      (XtPointer)XmVERTICAL,
    },
    {
      SgNanimate,
      SgCAnimate,
      XmRBoolean,
      sizeof(Boolean),
      OFFSET(thumbWheel.animate),
      XmRImmediate,
      (XtPointer)False,
    },
    {
      XmNvalueChangedCallback,
      XmCCallback,
      XmRCallback,
      sizeof(XtCallbackList),
      OFFSET(thumbWheel.value_changed_callback),
      XmRPointer,
      (XtPointer)NULL,
    },
    {
      XmNdragCallback,
      XmCCallback,
      XmRCallback,
      sizeof(XtCallbackList),
      OFFSET(thumbWheel.drag_callback),
      XmRPointer,
      (XtPointer)NULL,
    },
    {
      SgNshowHomeButton,
      SgCShowHomeButton,
      XmRBoolean,
      sizeof(Boolean),
      OFFSET(thumbWheel.show_home_button),
      XmRImmediate,
      (XtPointer)TRUE,
    },
#ifndef __sgi
    {
      XmNtraversalOn,
      XmCTraversalOn,
      XmRBoolean,
      sizeof(Boolean),
      OFFSET(primitive.traversal_on),
      XmRImmediate,
      (XtPointer)FALSE,
    },
#endif
};
#undef OFFSET

/* Declaration of methods */

#ifdef _NO_PROTO

static void ClassInitialize();
static void Initialize();
static void Realize();
static void Redisplay();
static void Destroy();
static void Resize();
static Boolean SetValues();
static XtGeometryResult QueryGeometry();

static void Motion();
static void Btn1Down();
static void Btn2Down();
static void Btn3Down();
static void Btn1Motion();
static void Btn2Motion();
static void Btn3Motion();
static void Btn1Up();
static void Btn2Up();
static void Btn3Up();
static void Enter();
static void Leave();
static void PageUp();
static void PageDown();
static void Up();
static void Down();
static void Left();
static void Right();
static void Help();
static void BeginLine();
static void EndLine();

static int ConvertPixelsToUserUnits();
static int ConvertUserUnitsToPixels();
static void IssueCallback();
static int ProcessMouseEvent();
static void CreateAndRenderPixmaps();
static void RenderPixmap();
static void RenderButtonPixmaps();
static void GetForegroundGC();
static Boolean MouseIsInWheel();
static Boolean MouseIsInButton();
static void SetCurrentPixmap();
static void FreePixmaps();
static Boolean ValidateFields();
static void ArmHomeButton();
static void DisarmHomeButton();
static void RenderButtonShadows();

#else

static void ClassInitialize();
static void Initialize(Widget rw, Widget nw, ArgList args, Cardinal *num_args);
static void Realize(Widget w, XtValueMask *window_mask,
                    XSetWindowAttributes *window_attributes);
static void Redisplay(Widget wid, XEvent *event, Region region);
static void Destroy(Widget wid);
static void Resize(Widget wid);
static Boolean SetValues(Widget cw, Widget rw, Widget nw, ArgList args,
                         Cardinal *num_args);
static XtGeometryResult QueryGeometry(Widget widget,
                                      XtWidgetGeometry *intended,
                                      XtWidgetGeometry *desired);

static void Motion(Widget wid, XEvent *event, String *params,
                   Cardinal *num_params);
static void Btn1Down(Widget wid, XEvent *event, String *params,
                     Cardinal *num_params);
static void Btn2Down(Widget wid, XEvent *event, String *params,
                     Cardinal *num_params);
static void Btn3Down(Widget wid, XEvent *event, String *params,
                     Cardinal *num_params);
static void Btn1Motion(Widget wid, XEvent *event, String *params,
                       Cardinal *num_params);
static void Btn2Motion(Widget wid, XEvent *event, String *params,
                       Cardinal *num_params);
static void Btn3Motion(Widget wid, XEvent *event, String *params,
                       Cardinal *num_params);
static void Btn1Up(Widget wid, XEvent *event, String *params,
                   Cardinal *num_params);
static void Btn2Up(Widget wid, XEvent *event, String *params,
                   Cardinal *num_params);
static void Btn3Up(Widget wid, XEvent *event, String *params,
                   Cardinal *num_params);
static void Enter(Widget wid, XEvent *event, String *params,
                  Cardinal *num_params);
static void Leave(Widget wid, XEvent *event, String *params,
                  Cardinal *num_params);
static void PageUp(Widget wid, XEvent *event, String *params,
                   Cardinal *num_params);
static void PageDown(Widget wid, XEvent *event, String *params,
                     Cardinal *num_params);
static void Up(Widget wid, XEvent *event, String *params,
               Cardinal *num_params);
static void Down(Widget wid, XEvent *event, String *params,
                 Cardinal *num_params);
static void Left(Widget wid, XEvent *event, String *params,
                 Cardinal *num_params);
static void Right(Widget wid, XEvent *event, String *params,
                  Cardinal *num_params);
static void Help(Widget wid, XEvent *event, String *params,
                 Cardinal *num_params);
static void BeginLine(Widget wid, XEvent *event, String *params,
                      Cardinal *num_params);
static void EndLine(Widget wid, XEvent *event, String *params,
                    Cardinal *num_params);

static int ConvertPixelsToUserUnits(SgThumbWheelWidget w, int pixels);
static int ConvertUserUnitsToPixels(SgThumbWheelWidget w, int uu);
static void IssueCallback(SgThumbWheelWidget thumb, int reason, int value,
                          XEvent *event);
static int ProcessMouseEvent(SgThumbWheelWidget thumb, int event_x,
                             int event_y);
static void CreateAndRenderPixmaps(SgThumbWheelWidget thumb);
static void RenderPixmap(SgThumbWheelWidget thumb, int which);
static void RenderButtonPixmaps(SgThumbWheelWidget thumb);
static void GetForegroundGC(SgThumbWheelWidget thumb);
static Boolean MouseIsInWheel(SgThumbWheelWidget thumb, int event_x,
                              int event_y);
static Boolean MouseIsInButton(SgThumbWheelWidget thumb, int event_x,
                               int event_y);
static void SetCurrentPixmap(SgThumbWheelWidget thumb,
                             Boolean value_increased);
static void FreePixmaps(SgThumbWheelWidget thumb);
static Boolean ValidateFields(SgThumbWheelWidget cur_w,
                              SgThumbWheelWidget req_w,
                              SgThumbWheelWidget new_w);
static void ArmHomeButton(SgThumbWheelWidget thumb);
static void DisarmHomeButton(SgThumbWheelWidget thumb);
static void RenderButtonShadows(SgThumbWheelWidget thumb);
#endif /* _NO_PROTO */

static char defaultTranslations[] = "<Motion>:	Motion() \n\
 <Btn1Down>:	Btn1Down() \n\
 <Btn2Down>:	Btn2Down() \n\
 <Btn3Down>:	Btn3Down() \n\
 <Btn1Motion>:	Btn1Motion() \n\
 <Btn2Motion>:	Btn2Motion() \n\
 <Btn3Motion>:	Btn3Motion() \n\
 <Btn1Up>:	Btn1Up() \n\
 <Btn2Up>:	Btn2Up() \n\
 <Btn3Up>:	Btn3Up() \n\
 <EnterWindow>:	Enter() \n\
 <LeaveWindow>:	Leave() \n\
 <Key>osfUp:	Up() \n\
 <Key>osfDown:	Down() \n\
 <Key>osfLeft:	Left() \n\
 <Key>osfRight:	Right() \n\
 <Key>osfPageUp:	PageUp() \n\
 <Key>osfPageDown:	PageDown() \n\
 <Key>osfBeginLine:	BeginLine() \n\
 <Key>osfEndLine:	EndLine() \n\
 <Key>osfHelp:	Help()";

static XtActionsRec actions[] = {
    { (char *)"Motion", Motion },
    { (char *)"Btn1Down", Btn1Down },
    { (char *)"Btn2Down", Btn2Down },
    { (char *)"Btn3Down", Btn3Down },
    { (char *)"Btn1Motion", Btn1Motion },
    { (char *)"Btn2Motion", Btn2Motion },
    { (char *)"Btn3Motion", Btn3Motion },
    { (char *)"Btn1Up", Btn1Up },
    { (char *)"Btn2Up", Btn2Up },
    { (char *)"Btn3Up", Btn3Up },
    { (char *)"Enter", Enter },
    { (char *)"Leave", Leave },
    { (char *)"PageUp", PageUp },
    { (char *)"PageDown", PageDown },
    { (char *)"Up", Up },
    { (char *)"Down", Down },
    { (char *)"Left", Left },
    { (char *)"Right", Right },
    { (char *)"Help", Help },
    { (char *)"BeginLine", BeginLine },
    { (char *)"EndLine", EndLine },
};

SgThumbWheelClassRec sgThumbWheelClassRec = {
    { /* core class fields */
      (WidgetClass)&xmPrimitiveClassRec,
      (char *)"ThumbWheel",
      sizeof(SgThumbWheelRec),
      ClassInitialize,
      NULL,
      False,
      Initialize,
      NULL,
      Realize,
      actions,
      XtNumber(actions),
      resources,
      XtNumber(resources),
      NULLQUARK,
      True, /* don't compress motion events */
      XtExposeCompressMultiple,
      True,
      False,
      Destroy,
      Resize,
      Redisplay,
      SetValues,
      NULL,
      XtInheritSetValuesAlmost,
      NULL,
      NULL,
      XtVersion,
      NULL,
      defaultTranslations,
      QueryGeometry,
      XtInheritDisplayAccelerator,
      NULL,

    },
    { /* Primitive class fields */
      XmInheritBorderHighlight,
      XmInheritBorderUnhighlight,
      XtInheritTranslations,
      NULL,
      NULL,
      0,
      NULL,
    },
    {
      0,
    },
};

GC GCarray[10];

WidgetClass sgThumbWheelWidgetClass = (WidgetClass)&sgThumbWheelClassRec;

/* PUT ALL THE ACTIONS AND OTHER FUNCTIONS HERE */

static void
#ifdef _NO_PROTO
ClassInitialize()
#else
ClassInitialize(void)
#endif /* _NO_PROTO */
{
}

static void
#ifdef _NO_PROTO
    Initialize(rw, nw, args, num_args)
        Widget rw;
Widget nw;
ArgList args;
Cardinal *num_args;
#else
Initialize(Widget rw, Widget nw, ArgList args, Cardinal *num_args)
#endif /* _NO_PROTO */
{
    SgThumbWheelWidget request_w = (SgThumbWheelWidget)rw;
    SgThumbWheelWidget new_w = (SgThumbWheelWidget)nw;
    static int GCinit = 0;
    int status, i;
    int pixel[] = /*close enough */
        {
          0, 42, 85, 128, 170, 213, 255
        };
    XtGCMask value_mask = 0;
    XGCValues gcValues;

    XColor screen_def;
    (void)request_w;
    (void)args;
    (void)num_args;
    if (!GCinit)
    {
        for (i = 0; i < 7; i++)
        {
            screen_def.red = screen_def.blue = screen_def.green = pixel[i] << 8;
            status = XAllocColor(XtDisplay(nw), nw->core.colormap, &screen_def);
            value_mask = GCForeground | GCBackground;
            gcValues.foreground = screen_def.pixel;
            gcValues.background = screen_def.pixel;
            GCarray[i] = XtGetGC(nw, value_mask, &gcValues);
        }
        GCinit = 1;
    }
    {
        int hilite = new_w->primitive.highlight_thickness;
        int shadow = new_w->primitive.shadow_thickness;
        Boolean horiz = (new_w->thumbWheel.orientation == XmHORIZONTAL);

        /* Private state - where the wheel and button will be drawn. */
        new_w->thumbWheel.wheel_x = hilite + shadow;
        new_w->thumbWheel.wheel_y = hilite + shadow;
        new_w->thumbWheel.button_x = new_w->thumbWheel.wheel_x + (horiz ? WHEEL_LONG_DIMENSION : 0);
        new_w->thumbWheel.button_y = new_w->thumbWheel.wheel_y + (horiz ? 0 : WHEEL_LONG_DIMENSION);

        /* Set up a geometry. */
        if (new_w->thumbWheel.orientation == XmHORIZONTAL)
        {
            if (new_w->thumbWheel.show_home_button == TRUE)
            {
                new_w->core.width = WHEEL_LONG_DIMENSION + BUTTON_SIZE
                                    + 2 * hilite + 4 * shadow;
            }
            else
            {
                new_w->core.width = WHEEL_LONG_DIMENSION + 2 * (hilite + shadow);
            }
        }
        else
        {
            new_w->core.width = WHEEL_NARROW_DIMENSION + 2 * (hilite + shadow);
        }
        if (new_w->thumbWheel.orientation == XmHORIZONTAL)
        {
            new_w->core.height = WHEEL_NARROW_DIMENSION + 2 * (hilite + shadow);
        }
        else
        {
            if (new_w->thumbWheel.show_home_button == TRUE)
            {
                new_w->core.height = WHEEL_LONG_DIMENSION + BUTTON_SIZE
                                     + 2 * hilite + 4 * shadow;
            }
            else
            {
                new_w->core.height = WHEEL_LONG_DIMENSION + 2 * (hilite + shadow);
            }
        }
    }
    new_w->thumbWheel.infinite = FALSE;

    if (new_w->thumbWheel.lower_bound > new_w->thumbWheel.upper_bound)
    {
        int tmp = new_w->thumbWheel.lower_bound;
        new_w->thumbWheel.lower_bound = new_w->thumbWheel.upper_bound;
        new_w->thumbWheel.upper_bound = tmp;
    }
    else if (new_w->thumbWheel.upper_bound == new_w->thumbWheel.lower_bound)
    {
        new_w->thumbWheel.infinite = TRUE;
    }

    if (new_w->thumbWheel.angle_range == 0)
    {
        new_w->thumbWheel.infinite = TRUE;
    }

    if (new_w->thumbWheel.infinite != TRUE)
    {
        /* Range / values checking */
        if (new_w->thumbWheel.value < new_w->thumbWheel.lower_bound)
        {
            new_w->thumbWheel.value = new_w->thumbWheel.lower_bound;
        }
        if (new_w->thumbWheel.value > new_w->thumbWheel.upper_bound)
        {
            new_w->thumbWheel.value = new_w->thumbWheel.upper_bound;
        }
        if (new_w->thumbWheel.home_position < new_w->thumbWheel.lower_bound)
        {
            new_w->thumbWheel.home_position = new_w->thumbWheel.lower_bound;
        }
        if (new_w->thumbWheel.home_position > new_w->thumbWheel.upper_bound)
        {
            new_w->thumbWheel.home_position = new_w->thumbWheel.upper_bound;
        }
    }

    /* Set private state */
    new_w->thumbWheel.home_button_armed = FALSE;
    new_w->thumbWheel.dragging = FALSE;
    new_w->thumbWheel.last_mouse_position = 0;
    new_w->thumbWheel.pegged = FALSE;
    new_w->thumbWheel.pegged_mouse_position = 0;
    new_w->thumbWheel.viewable_pixels = WHEEL_LONG_DIMENSION;
    new_w->thumbWheel.user_pixels = ConvertUserUnitsToPixels(new_w, new_w->thumbWheel.value);

    GetForegroundGC(new_w);
    /*  _sgFindShader((Widget)new_w, &(new_w->thumbWheel.shader),
         new_w->core.background_pixel);*/

    new_w->thumbWheel.pix1 = 0;
    new_w->thumbWheel.pix2 = 0;
    new_w->thumbWheel.pix3 = 0;
    new_w->thumbWheel.pix4 = 0;
    new_w->thumbWheel.pix1_hilite = 0;
    new_w->thumbWheel.pix2_hilite = 0;
    new_w->thumbWheel.pix3_hilite = 0;
    new_w->thumbWheel.pix4_hilite = 0;
    new_w->thumbWheel.current_quiet_pixmap = 0;
    new_w->thumbWheel.current_hilite_pixmap = 0;
    new_w->thumbWheel.wheel_hilite = FALSE;

    new_w->thumbWheel.button_quiet_pixmap = 0;
    new_w->thumbWheel.button_hilite_pixmap = 0;
    new_w->thumbWheel.button_hilite = FALSE;
}

static void
#ifdef _NO_PROTO
    ValueDefaultProc(w, offset, v)
        SgThumbWheelWidget w;
int offset;
XrmValue *v;
#else
ValueDefaultProc(SgThumbWheelWidget w, int offset, XrmValue *v)
#endif /* _NO_PROTO */

{
    static int val;
    (void)offset;
    v->addr = (char *)&val;
    val = w->thumbWheel.home_position;
}

static void
#ifdef _NO_PROTO
    Realize(w, window_mask, window_attributes)
        Widget w;
XtValueMask *window_mask;
XSetWindowAttributes *window_attributes;
#else
Realize(Widget w, XtValueMask *window_mask,
        XSetWindowAttributes *window_attributes)
#endif /* _NO_PROTO */
{
    XtCreateWindow(w, InputOutput, CopyFromParent,
                   *window_mask, window_attributes);
}

static void
#ifdef _NO_PROTO
    Redisplay(wid, event, region)
        Widget wid;
XEvent *event;
Region region;
#else
Redisplay(Widget wid, XEvent *event, Region region)
#endif /* _NO_PROTO */
{
    SgThumbWheelWidget thumb = (SgThumbWheelWidget)wid;
    int hilite = thumb->primitive.highlight_thickness;
    int shadow = thumb->primitive.shadow_thickness;
    Boolean horiz = (thumb->thumbWheel.orientation == XmHORIZONTAL);
    (void)event;
    (void)region;

    if (thumb->thumbWheel.wheel_x < hilite + shadow)
    {
        thumb->thumbWheel.wheel_x = hilite + shadow;
    }
    if (thumb->thumbWheel.wheel_y < hilite + shadow)
    {
        thumb->thumbWheel.wheel_y = hilite + shadow;
    }

    if (thumb->thumbWheel.button_x < thumb->thumbWheel.wheel_x + (horiz ? WHEEL_LONG_DIMENSION + 2 * shadow : 0))
    {
        thumb->thumbWheel.button_x = thumb->thumbWheel.wheel_x + (horiz ? WHEEL_LONG_DIMENSION + 2 * shadow : 0);
    }
    if (thumb->thumbWheel.button_y < thumb->thumbWheel.wheel_y + (horiz ? 0 : WHEEL_LONG_DIMENSION + 2 * shadow))
    {
        thumb->thumbWheel.button_y = thumb->thumbWheel.wheel_y + (horiz ? 0 : WHEEL_LONG_DIMENSION + 2 * shadow);
    }

    if (thumb->thumbWheel.current_quiet_pixmap == 0)
    {
        CreateAndRenderPixmaps(thumb);
        thumb->thumbWheel.current_quiet_pixmap = thumb->thumbWheel.pix1;
        thumb->thumbWheel.current_hilite_pixmap = thumb->thumbWheel.pix1_hilite;
        /*
       * Clear the window only if pixmaps needed to be created
       * (eg foreground changed).  Otherwise, there's lots of flashing.
       */
        XClearWindow(XtDisplay(wid), XtWindow(wid));
    }

    /*
    * Render wheel.
    */
    XCopyArea(XtDisplay(wid),
              (thumb->thumbWheel.wheel_hilite == TRUE)
                  ? thumb->thumbWheel.current_hilite_pixmap
                  : thumb->thumbWheel.current_quiet_pixmap,
              XtWindow(wid),
              thumb->thumbWheel.foreground_GC,
              0, 0,
              (thumb->thumbWheel.orientation == XmHORIZONTAL)
                  ? WHEEL_LONG_DIMENSION
                  : WHEEL_NARROW_DIMENSION,
              (thumb->thumbWheel.orientation == XmHORIZONTAL)
                  ? WHEEL_NARROW_DIMENSION
                  : WHEEL_LONG_DIMENSION,
              thumb->thumbWheel.wheel_x, thumb->thumbWheel.wheel_y);

    /*
    * Render shadows around wheel area.
    */

    _XmDrawShadows(XtDisplay(wid), XtWindow(wid),
                   thumb->primitive.top_shadow_GC,
                   thumb->primitive.bottom_shadow_GC,
                   thumb->thumbWheel.wheel_x - shadow,
                   thumb->thumbWheel.wheel_y - shadow,
                   ((thumb->thumbWheel.orientation == XmHORIZONTAL) ? WHEEL_LONG_DIMENSION : WHEEL_NARROW_DIMENSION)
                   + 2 * shadow,
                   ((thumb->thumbWheel.orientation == XmHORIZONTAL) ? WHEEL_NARROW_DIMENSION : WHEEL_LONG_DIMENSION)
                   + 2 * shadow,
                   shadow, XmSHADOW_OUT);

    /*
    * Render home button.
    */
    if (thumb->thumbWheel.show_home_button == TRUE)
    {
        XCopyArea(XtDisplay(wid),
                  (thumb->thumbWheel.button_hilite == TRUE)
                      ? thumb->thumbWheel.button_hilite_pixmap
                      : thumb->thumbWheel.button_quiet_pixmap,
                  XtWindow(wid),
                  thumb->thumbWheel.foreground_GC,
                  0, 0,
                  BUTTON_SIZE, BUTTON_SIZE,
                  thumb->thumbWheel.button_x, thumb->thumbWheel.button_y);
    }

    /*
    * Render shadows around home button.
    */

    RenderButtonShadows(thumb);
}

static void
#ifdef _NO_PROTO
    Destroy(wid)
        Widget wid;
#else
Destroy(Widget wid)
#endif /* _NO_PROTO */
{
    SgThumbWheelWidget thumb = (SgThumbWheelWidget)wid;
    FreePixmaps(thumb);
}

static void
#ifdef _NO_PROTO
    Resize(wid)
        Widget wid;
#else
Resize(Widget wid)
#endif /* _NO_PROTO */
{
    SgThumbWheelWidget thumb = (SgThumbWheelWidget)wid;
    int hilite = thumb->primitive.highlight_thickness;
    int shadow = thumb->primitive.shadow_thickness;
    Boolean horiz = (thumb->thumbWheel.orientation == XmHORIZONTAL);
    Boolean home = thumb->thumbWheel.show_home_button;
    int mywidth = thumb->core.width - 2 * (hilite + shadow);
    int myheight = thumb->core.height - 2 * (hilite + shadow);

    if (mywidth <= (horiz ? WHEEL_LONG_DIMENSION + (home ? BUTTON_SIZE + 2 * shadow : 0) : WHEEL_NARROW_DIMENSION))
    {
        thumb->thumbWheel.wheel_x = hilite + shadow;
    }
    else
    {
        thumb->thumbWheel.wheel_x = horiz ? (mywidth / 2) - ((WHEEL_LONG_DIMENSION + (home ? BUTTON_SIZE + 2 * shadow : 0)) / 2) : (mywidth / 2) - (WHEEL_NARROW_DIMENSION / 2);
        thumb->thumbWheel.wheel_x += hilite + shadow;
    }
    thumb->thumbWheel.button_x = thumb->thumbWheel.wheel_x + (horiz ? WHEEL_LONG_DIMENSION + 2 * shadow : 0);

    if (myheight <= (horiz ? WHEEL_NARROW_DIMENSION : WHEEL_LONG_DIMENSION + (home ? BUTTON_SIZE + 2 * shadow : 0)))
    {
        thumb->thumbWheel.wheel_y = hilite + shadow;
    }
    else
    {
        thumb->thumbWheel.wheel_y = horiz ? (myheight / 2) - (WHEEL_NARROW_DIMENSION / 2) : (myheight / 2) - ((WHEEL_LONG_DIMENSION + (home ? BUTTON_SIZE + 2 * shadow : 0)) / 2);
        thumb->thumbWheel.wheel_y += hilite + shadow;
    }
    thumb->thumbWheel.button_y = thumb->thumbWheel.wheel_y + (horiz ? 0 : WHEEL_LONG_DIMENSION + 2 * shadow);

    /*
     fprintf(stderr, "Resize: wheel x y %d %d, button %d %d.\n",
        thumb->thumbWheel.wheel_x, thumb->thumbWheel.wheel_y,
        thumb->thumbWheel.button_x, thumb->thumbWheel.button_y);
   */
}

static Boolean
#ifdef _NO_PROTO
    SetValues(cw, rw, nw, args, num_args)
        Widget cw;
Widget rw;
Widget nw;
ArgList args;
Cardinal *num_args;
#else
SetValues(Widget cw, Widget rw, Widget nw, ArgList args, Cardinal *num_args)
#endif /* _NO_PROTO */
{

    Boolean return_flag = FALSE;
    SgThumbWheelWidget new_w = (SgThumbWheelWidget)nw;
    SgThumbWheelWidget req_w = (SgThumbWheelWidget)rw;
    SgThumbWheelWidget cur_w = (SgThumbWheelWidget)cw;
    (void)args;
    (void)num_args;

    while (!ValidateFields(cur_w, req_w, new_w))
        ;

/*
     fprintf(stderr, "SetValues\n"); fflush(stderr);
   */

#define NEQ(field) (new_w->field != cur_w->field)

    if (NEQ(thumbWheel.orientation) || NEQ(primitive.shadow_thickness) || NEQ(primitive.highlight_thickness))
    {
        FreePixmaps(new_w);
        CreateAndRenderPixmaps(new_w);
        /* Make sure the pixmaps are located properly in the window */
        (*(new_w->core.widget_class->core_class.resize))(nw);
        return_flag = TRUE;
    }
    if (NEQ(core.width) || NEQ(core.height) || NEQ(thumbWheel.show_home_button))
    {
        /* Make sure the pixmaps are located properly in the window */
        (*(new_w->core.widget_class->core_class.resize))(nw);
        return_flag = TRUE;
    }

    if (NEQ(core.background_pixel))
    {
        /* Get the foreground GC again */
        XtReleaseGC((Widget)new_w, new_w->thumbWheel.foreground_GC);
        GetForegroundGC(new_w);
        /* Get the shader again */
        /*
          _sgFindShader((Widget)new_w, &(new_w->thumbWheel.shader),
              new_w->core.background_pixel);*/
        /* Render the pixmaps again */
        FreePixmaps(new_w);
        CreateAndRenderPixmaps(new_w);
        return_flag = TRUE;
    }

    if (NEQ(thumbWheel.value))
    {
        /* Spin the wheel here. */
        /* Recompute private field "user_pixels", which tracks value. */
        new_w->thumbWheel.user_pixels = ConvertUserUnitsToPixels(new_w, new_w->thumbWheel.value);
        return_flag = TRUE;
    }

#undef NEQ

    return return_flag;
}

static XtGeometryResult
#ifdef _NO_PROTO
    QueryGeometry(widget, intended, desired)
        Widget widget;
XtWidgetGeometry *intended;
XtWidgetGeometry *desired;
#else
QueryGeometry(Widget widget, XtWidgetGeometry *intended,
              XtWidgetGeometry *desired)
#endif /* _NO_PROTO */
{
#ifdef RESIZE
#else
    SgThumbWheelWidget thumb = (SgThumbWheelWidget)widget;
    int hilite = thumb->primitive.highlight_thickness;
    int shadow = thumb->primitive.shadow_thickness;
    Boolean horiz = (thumb->thumbWheel.orientation == XmHORIZONTAL);
    Boolean home = thumb->thumbWheel.show_home_button;
    int borders = 2 * (hilite + shadow);
    (void)intended;

    /*
     fprintf(stderr,"QueryGeometry\n"); fflush(stderr);
   */

    /* Hahahaha... */
    if (horiz)
    {
        desired->width = WHEEL_LONG_DIMENSION + (home ? BUTTON_SIZE + 2 * shadow : 0) + borders;
        desired->height = WHEEL_NARROW_DIMENSION + borders;
    }
    else
    {
        desired->width = WHEEL_NARROW_DIMENSION + borders;
        desired->height = WHEEL_LONG_DIMENSION + (home ? BUTTON_SIZE + 2 * shadow : 0) + borders;
    }
    desired->request_mode = CWWidth | CWHeight;

#if 0
      return _XmGMReplyToQueryGeometry(widget, intended, desired);
#else
    return XtGeometryAlmost;
#endif
#endif
}

static void
#ifdef _NO_PROTO
    Motion(wid, event, params, num_params)
        Widget wid;
XEvent *event;
String *params;
Cardinal *num_params;
#else
Motion(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    SgThumbWheelWidget thumb = (SgThumbWheelWidget)wid;
    XMotionEvent *xmotion = (XMotionEvent *)event;

    if (xmotion->window != XtWindow(wid))
    {
        fprintf(stderr, "Windows not the same!\n");
        fflush(stderr);
    }

    /*
    * If this is a button motion event,
    * pass it along or ignore it, but do not process it here.
    */
    if (xmotion->state & Button1Mask)
    {
        Btn1Motion(wid, event, params, num_params);
        return;
    }
    else if (xmotion->state & Button2Mask)
    {
        Btn2Motion(wid, event, params, num_params);
        return;
    }
    else if (xmotion->state & Button3Mask)
    {
        Btn3Motion(wid, event, params, num_params);
        return;
    }
    else if ((xmotion->state & Button4Mask) || (xmotion->state & Button5Mask))
    {
        return;
    }

    if ((thumb->thumbWheel.wheel_hilite == FALSE) && (thumb->thumbWheel.button_hilite == FALSE))
    {
        return;
    }
    thumb->thumbWheel.wheel_hilite = FALSE;
    thumb->thumbWheel.button_hilite = FALSE;
    Redisplay((Widget)thumb, NULL, NULL);
}

static void
#ifdef _NO_PROTO
    Btn1Down(wid, event, params, num_params)
        Widget wid;
XEvent *event;
String *params;
Cardinal *num_params;
#else
Btn1Down(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{

    SgThumbWheelWidget thumb = (SgThumbWheelWidget)wid;
    XButtonPressedEvent *xbutton = (XButtonPressedEvent *)event;
    (void)params;
    (void)num_params;

    if (MouseIsInWheel(thumb, DimensionToSigned(xbutton->x),
                       DimensionToSigned(xbutton->y)) == TRUE)
    {
        /* Save this mouse position for later drag calculations */
        if (thumb->thumbWheel.orientation == XmHORIZONTAL)
        {
            thumb->thumbWheel.last_mouse_position = DimensionToSigned(xbutton->x);
        }
        else
        {
            thumb->thumbWheel.last_mouse_position = DimensionToSigned(xbutton->y);
        }

        /* Button down in the wheel area starts a drag */
        thumb->thumbWheel.dragging = TRUE;

        /*
       * Save the value now, so we can tell whether to issue
       * a value changed callback later on
       */
        thumb->thumbWheel.drag_begin_value = thumb->thumbWheel.value;

        /* Are we already pegged to the highest or lowest value? */
        if (((thumb->thumbWheel.value == thumb->thumbWheel.lower_bound) || (thumb->thumbWheel.value == thumb->thumbWheel.upper_bound)) && (thumb->thumbWheel.infinite == FALSE))
        {
            thumb->thumbWheel.pegged = TRUE;
            thumb->thumbWheel.pegged_mouse_position = thumb->thumbWheel.last_mouse_position;
        }
    }
    else if (MouseIsInButton(thumb, DimensionToSigned(xbutton->x),
                             DimensionToSigned(xbutton->y)) == TRUE)
    {
        /* home button click */

        /*
       * Arm the home button
       */
        ArmHomeButton(thumb);
    }
}

static void
#ifdef _NO_PROTO
Btn2Down()
#else
Btn2Down(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)event;
    (void)params;
    (void)num_params;
    /*  fprintf(stderr, "Btn2Down\n");*/
}

static void
#ifdef _NO_PROTO
Btn3Down()
#else
Btn3Down(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)event;
    (void)params;
    (void)num_params;
    /*  fprintf(stderr, "Btn3Down\n");*/
}

static void
#ifdef _NO_PROTO
    Btn1Motion(wid, event, params, num_params)
        Widget wid;
XEvent *event;
String *params;
Cardinal *num_params;
#else
Btn1Motion(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    SgThumbWheelWidget thumb = (SgThumbWheelWidget)wid;
    XMotionEvent *xmotion = (XMotionEvent *)event;
    int old_value = thumb->thumbWheel.value;
    int new_value;
    (void)params;
    (void)num_params;
    if (thumb->thumbWheel.dragging != TRUE)
    {
        return;
    }

    new_value = ProcessMouseEvent(thumb, DimensionToSigned(xmotion->x),
                                  DimensionToSigned(xmotion->y));

    /*
    * Change the current pixmap to reflect "spinning" the wheel.
    */
    if (new_value != old_value)
    {
        SetCurrentPixmap(thumb, (new_value > old_value));
        Redisplay((Widget)thumb, NULL, NULL);
    }

    /*
    * Issue the dragCallback with the new value, if changed.
    */
    if (new_value != old_value)
    {
        IssueCallback(thumb, XmCR_DRAG, thumb->thumbWheel.value, event);
    }
}

static void
#ifdef _NO_PROTO
Btn2Motion()
#else
Btn2Motion(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)event;
    (void)params;
    (void)num_params;
    /*  fprintf(stderr, "Btn2Motion\n");*/
}

static void
#ifdef _NO_PROTO
Btn3Motion()
#else
Btn3Motion(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)event;
    (void)params;
    (void)num_params;
    /*  fprintf(stderr, "Btn3Motion\n");*/
}

static void
#ifdef _NO_PROTO
    Btn1Up(wid, event, params, num_params)
        Widget wid;
XEvent *event;
String *params;
Cardinal *num_params;
#else
Btn1Up(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    SgThumbWheelWidget thumb = (SgThumbWheelWidget)wid;
    XButtonReleasedEvent *xbutton = (XButtonReleasedEvent *)event;
    int old_value = thumb->thumbWheel.value;
    int new_value;
    (void)params;
    (void)num_params;
    if (thumb->thumbWheel.dragging == TRUE)
    {
        thumb->thumbWheel.dragging = FALSE;

        new_value = ProcessMouseEvent(thumb, DimensionToSigned(xbutton->x),
                                      DimensionToSigned(xbutton->y));

        /*
       * Change the current pixmap to reflect "spinning" the wheel.
       */
        if (new_value != old_value)
        {
            SetCurrentPixmap(thumb, (new_value > old_value));
            Redisplay((Widget)thumb, NULL, NULL);
        }

        /*
       * Issue the valueChangedCallback with the new value,
       * if the value has changed since the beginning of the drag.
       */
        if (new_value != thumb->thumbWheel.drag_begin_value)
        {
            IssueCallback(thumb, XmCR_VALUE_CHANGED, thumb->thumbWheel.value, event);
        }
    }
    else if (thumb->thumbWheel.home_button_armed)
    {
        /*
       * Disarm the home button
       */
        DisarmHomeButton(thumb);
        /*
       * If the mouse up happened in the button, change the value.
       */
        if (MouseIsInButton(thumb, DimensionToSigned(xbutton->x),
                            DimensionToSigned(xbutton->y)) == TRUE)
        {
            /*
          * value becomes the home position value
          */
            int old_value = thumb->thumbWheel.value;
            thumb->thumbWheel.value = thumb->thumbWheel.home_position;
            thumb->thumbWheel.dragging = FALSE;
            thumb->thumbWheel.user_pixels = ConvertUserUnitsToPixels(thumb, thumb->thumbWheel.value);

            /*
          * set "pegged" if the home position happens to be one of the bounds
          */
            if (((thumb->thumbWheel.value == thumb->thumbWheel.lower_bound) || (thumb->thumbWheel.value == thumb->thumbWheel.upper_bound)) && (thumb->thumbWheel.infinite == FALSE))
            {
                thumb->thumbWheel.pegged = TRUE;
            }
            else
            {
                thumb->thumbWheel.pegged = FALSE;
            }

            /* restore original pixmap, and redisplay */
            thumb->thumbWheel.current_quiet_pixmap = thumb->thumbWheel.pix1;
            thumb->thumbWheel.current_hilite_pixmap = thumb->thumbWheel.pix1_hilite;
            Redisplay((Widget)thumb, NULL, NULL);

            /*
          * Issue the valueChangedCallback with the new value,
          * if the value has changed.
          */
            if (thumb->thumbWheel.value != old_value)
            {
                IssueCallback(thumb, XmCR_VALUE_CHANGED, thumb->thumbWheel.value, event);
            }
        }
    }
}

static void
#ifdef _NO_PROTO
Btn2Up()
#else
Btn2Up(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)event;
    (void)params;
    (void)num_params;
    /*  fprintf(stderr, "Btn2Up\n");*/
}

static void
#ifdef _NO_PROTO
Btn3Up()
#else
Btn3Up(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)params;
    (void)event;
    (void)num_params;
    /*  fprintf(stderr, "Btn3Up\n");*/
}

static void
#ifdef _NO_PROTO
    Enter(wid, event, params, num_params)
        Widget wid;
XEvent *event;
String *params;
Cardinal *num_params;
#else
Enter(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    SgThumbWheelWidget thumb = (SgThumbWheelWidget)wid;
    XCrossingEvent *xenter = (XCrossingEvent *)event;
    (void)params;
    (void)num_params;

    if (MouseIsInWheel(thumb, DimensionToSigned(xenter->x),
                       DimensionToSigned(xenter->y)) == TRUE)
    {
        if ((thumb->thumbWheel.wheel_hilite == TRUE) && (thumb->thumbWheel.button_hilite == FALSE))
        {
            return;
        }
        thumb->thumbWheel.wheel_hilite = TRUE;
        thumb->thumbWheel.button_hilite = FALSE;
        Redisplay((Widget)thumb, NULL, NULL);
    }
    else if (MouseIsInButton(thumb, DimensionToSigned(xenter->x),
                             DimensionToSigned(xenter->y)) == TRUE)
    {
        if ((thumb->thumbWheel.wheel_hilite == FALSE) && (thumb->thumbWheel.button_hilite == TRUE))
        {
            return;
        }
        thumb->thumbWheel.wheel_hilite = FALSE;
        thumb->thumbWheel.button_hilite = TRUE;
        Redisplay((Widget)thumb, NULL, NULL);
    }
    else
    {
        if ((thumb->thumbWheel.wheel_hilite == FALSE) && (thumb->thumbWheel.button_hilite == FALSE))
        {
            return;
        }
        thumb->thumbWheel.wheel_hilite = FALSE;
        thumb->thumbWheel.button_hilite = FALSE;
        Redisplay((Widget)thumb, NULL, NULL);
    }
}

static void
#ifdef _NO_PROTO
    Leave(wid, event, params, num_params)
        Widget wid;
XEvent *event;
String *params;
Cardinal *num_params;
#else
Leave(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    SgThumbWheelWidget thumb = (SgThumbWheelWidget)wid;
    (void)event;
    (void)params;
    (void)num_params;
    if ((thumb->thumbWheel.wheel_hilite == FALSE) && (thumb->thumbWheel.button_hilite == FALSE))
    {
        return;
    }
    thumb->thumbWheel.wheel_hilite = FALSE;
    thumb->thumbWheel.button_hilite = FALSE;
    Redisplay((Widget)thumb, NULL, NULL);
}

static void
#ifdef _NO_PROTO
PageUp()
#else
PageUp(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)event;
    (void)params;
    (void)num_params;
    /*  fprintf(stderr, "PageUp\n");*/
}

static void
#ifdef _NO_PROTO
PageDown()
#else
PageDown(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)event;
    (void)params;
    (void)num_params;
    /*  fprintf(stderr, "PageDown\n");*/
}

static void
#ifdef _NO_PROTO
Up()
#else
Up(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)event;
    (void)params;
    (void)num_params;
    /*  fprintf(stderr, "Up\n");*/
}

static void
#ifdef _NO_PROTO
Down()
#else
Down(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)event;
    (void)params;
    (void)num_params;
    /*  fprintf(stderr, "Down\n");*/
}

static void
#ifdef _NO_PROTO
Left()
#else
Left(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)event;
    (void)params;
    (void)num_params;
    /*  fprintf(stderr, "Left\n");*/
}

static void
#ifdef _NO_PROTO
Right()
#else
Right(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)event;
    (void)params;
    (void)num_params;
    /*  fprintf(stderr, "Right\n");*/
}

static void
#ifdef _NO_PROTO
Help()
#else
Help(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)event;
    (void)params;
    (void)num_params;
    /*  fprintf(stderr, "Help\n");*/
}

static void
#ifdef _NO_PROTO
BeginLine()
#else
BeginLine(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)event;
    (void)params;
    (void)num_params;
    /*  fprintf(stderr, "BeginLine\n");*/
}

static void
#ifdef _NO_PROTO
EndLine()
#else
EndLine(Widget wid, XEvent *event, String *params, Cardinal *num_params)
#endif /* _NO_PROTO */
{
    (void)wid;
    (void)event;
    (void)params;
    (void)num_params;
    /*  fprintf(stderr, "EndLine\n");*/
}

Widget SgCreateThumbWheel(Widget parent, char *name, Arglist arglist, Cardinal argcount)
{
    return (XtCreateWidget(name, sgThumbWheelWidgetClass, parent, arglist,
                           argcount));
}

/* Private functions */

static int
#ifdef _NO_PROTO
    ConvertPixelsToUserUnits(w, pixels)
        SgThumbWheelWidget w;
int pixels;
#else
ConvertPixelsToUserUnits(SgThumbWheelWidget w, int pixels)
#endif /* _NO_PROTO */
{
    int user_units;
    if (pixels > 0)
    {
        user_units = (int)((float)pixels * PixelsToAngleFactor(w) * AngleToUserUnitFactor(w) + 0.5);
    }
    else
    {
        user_units = (int)((float)pixels * PixelsToAngleFactor(w) * AngleToUserUnitFactor(w) - 0.5);
    }

    return user_units;
}

static int
#ifdef _NO_PROTO
    ConvertUserUnitsToPixels(w, uu)
        SgThumbWheelWidget w;
int uu;
#else
ConvertUserUnitsToPixels(SgThumbWheelWidget w, int uu)
#endif /* _NO_PROTO */
{
    if (uu > 0)
    {
        return (
            (int)((float)uu / PixelsToAngleFactor(w) / AngleToUserUnitFactor(w) + 0.5));
    }
    else
    {
        return (
            (int)((float)uu / PixelsToAngleFactor(w) / AngleToUserUnitFactor(w) - 0.5));
    }
}

static void
#ifdef _NO_PROTO
    IssueCallback(thumb, reason, value, xpixel, ypixel, event)
        SgThumbWheelWidget thumb;
int reason;
int value;
XEvent *event;
#else
IssueCallback(SgThumbWheelWidget thumb, int reason, int value, XEvent *event)
#endif /* _NO_PROTO */
{
    SgThumbWheelCallbackStruct call_value;
    call_value.reason = reason;
    call_value.event = event;
    call_value.value = value;

    switch (reason)
    {
    case XmCR_VALUE_CHANGED:
        XtCallCallbackList((Widget)thumb, thumb->thumbWheel.value_changed_callback,
                           &call_value);
        break;
    case XmCR_DRAG:
        if (thumb->thumbWheel.drag_callback)
        {
            XtCallCallbackList((Widget)thumb, thumb->thumbWheel.drag_callback,
                               &call_value);
        }
        break;
    }
}

/*
 * ProcessMouseEvent:
 *
 * Computes the new thumb wheel value, given the present thumb wheel
 * (pegged state, current value, last mouse event position) and the
 * x and y coordinates of a mouse event (button motion or button up).
 *
 * (*) Determines the pertinent mouse position from the x or y, depending
 *     on the orientation.
 * (*) If the widget is pegged at max or min value, "move" our event
 *     such that it will not exceed the pegged range.
 * (*) Determine pixels changed since last event (motion or button down).
 *     If zero, return the existing thumb wheel value.
 * (*) Add pixel change to user_pixels.
 * (*) Determine what the new thumb wheel value would be, given user_pixels.
 * (*) If new value equals the existing value, save the mouse position and
 *     return the value.
 * (*) If new value is out of bounds, peg it, set "pegged" to TRUE,
 *     and recompute user_pixels to match the pegged value.
 * (*) Else if new value is exactly max or min, set "pegged" to TRUE.
 * (*) Else set "pegged" to FALSE.
 * (*) Set the widget's value and return it.
 *
 * Issuing any callbacks is the responsibility of whoever called us.
 */

static int
#ifdef _NO_PROTO
    ProcessMouseEvent(thumb, event_x, event_y)
        SgThumbWheelWidget thumb;
int event_x, event_y;
#else
ProcessMouseEvent(SgThumbWheelWidget thumb, int event_x,
                  int event_y)
#endif /* _NO_PROTO */
{
    int current_mouse_position;
    int pixel_change;
    int new_value;

    if (thumb->thumbWheel.orientation == XmHORIZONTAL)
    {
        current_mouse_position = event_x;
    }
    else
    {
        current_mouse_position = event_y;
    }

    /*
    * If we're already pegged at our max or minimum allowable value,
    * avoid processing a mouse event outside that range.
    */
    if (thumb->thumbWheel.pegged == TRUE)
    {
        if (thumb->thumbWheel.value == thumb->thumbWheel.upper_bound)
        {
            /* we assume max-on-right here; configurable later? */
            /* we assume max-on-bottom here; configurable later? */
            if (current_mouse_position > thumb->thumbWheel.pegged_mouse_position)
            {
                current_mouse_position = thumb->thumbWheel.pegged_mouse_position;
            }
        }
        else /* value isn't upper_bound, so we must be pegged at lower_bound */
        {
            /* we assume max-on-right here; configurable later? */
            /* we assume max-on-bottom here; configurable later? */
            if (current_mouse_position < thumb->thumbWheel.pegged_mouse_position)
            {
                current_mouse_position = thumb->thumbWheel.pegged_mouse_position;
            }
        }
    }

    /*
    * Compute the number of pixels the mouse has moved
    * since the last motion (or button down) event.
    */
    if (thumb->thumbWheel.pegged == TRUE)
    {
        pixel_change = current_mouse_position - thumb->thumbWheel.pegged_mouse_position;
    }
    else
    {
        pixel_change = current_mouse_position - thumb->thumbWheel.last_mouse_position;
    }

    /*
    * If the number of pixels moved is zero
    * (which will happen often when "pegged" is true),
    * just return the existing value.
    */
    if (pixel_change == 0)
    {
        /* We don't care about saving the last_mouse_position if no change. */
        return thumb->thumbWheel.value;
    }

    /*
    * Add the pixel_change to user_pixels.
    */
    thumb->thumbWheel.user_pixels += pixel_change;

    /*
    * Figure the new value, in user units, represented by
    * our cumulative pixel value ("user_pixels").
    */
    new_value = ConvertPixelsToUserUnits(thumb, thumb->thumbWheel.user_pixels);

    /*
    * If this new value is no different from the previous one,
    * save the mouse position of this event
    * (since we already added this event in to user_pixels)
    * and return the existing value.
    */
    if (new_value == thumb->thumbWheel.value)
    {
        thumb->thumbWheel.last_mouse_position = current_mouse_position;
        return thumb->thumbWheel.value;
    }

    /*
    * If this change would bring the value out of bounds,
    * set the value to the appropriate bound,
    * set "pegged" to TRUE,
    * and recompute user_pixels (since it represented an out of bounds value).
    */
    if ((thumb->thumbWheel.infinite == FALSE) && (new_value > thumb->thumbWheel.upper_bound))
    {
        int user_units_allowed = thumb->thumbWheel.upper_bound - thumb->thumbWheel.value;
        int pixels_allowed = ConvertUserUnitsToPixels(thumb, user_units_allowed);
        thumb->thumbWheel.pegged_mouse_position = thumb->thumbWheel.last_mouse_position + pixels_allowed;
        thumb->thumbWheel.pegged = TRUE;
        thumb->thumbWheel.value = thumb->thumbWheel.upper_bound;
        thumb->thumbWheel.user_pixels = ConvertUserUnitsToPixels(thumb, thumb->thumbWheel.value);
    }
    else if ((thumb->thumbWheel.infinite == FALSE) && (new_value < thumb->thumbWheel.lower_bound))
    {
        int user_units_allowed = thumb->thumbWheel.lower_bound - thumb->thumbWheel.value;
        int pixels_allowed = ConvertUserUnitsToPixels(thumb, user_units_allowed);
        thumb->thumbWheel.pegged_mouse_position = thumb->thumbWheel.last_mouse_position + pixels_allowed;
        thumb->thumbWheel.pegged = TRUE;
        thumb->thumbWheel.value = thumb->thumbWheel.lower_bound;
        thumb->thumbWheel.user_pixels = ConvertUserUnitsToPixels(thumb, thumb->thumbWheel.value);
    }
    /*
    * Otherwise (new value not out of bounds):
    * If new value equals a bound, set "pegged" to TRUE;
    * else set it to FALSE.
    */
    else
    {
        thumb->thumbWheel.value = new_value;
        if (((thumb->thumbWheel.value == thumb->thumbWheel.upper_bound) || (thumb->thumbWheel.value == thumb->thumbWheel.lower_bound)) && (thumb->thumbWheel.infinite == FALSE))
        {
            thumb->thumbWheel.pegged = TRUE;
            thumb->thumbWheel.pegged_mouse_position = current_mouse_position;
        }
        else
        {
            thumb->thumbWheel.pegged = FALSE;
        }
    }

    /*
    * Save the current mouse position
    * for comparison with mouse motion / button-up events in future.
    */
    thumb->thumbWheel.last_mouse_position = current_mouse_position;

    return thumb->thumbWheel.value;
}

static void
#ifdef _NO_PROTO
    CreateAndRenderPixmaps(thumb)
        Widget thumb;
#else
CreateAndRenderPixmaps(SgThumbWheelWidget thumb)
#endif /* _NO_PROTO */
{
    if (thumb->thumbWheel.orientation == XmHORIZONTAL)
    {
        thumb->thumbWheel.pix1 = XCreatePixmap(XtDisplay((Widget)thumb),
                                               RootWindowOfScreen(XtScreen((Widget)thumb)),
                                               WHEEL_LONG_DIMENSION, WHEEL_NARROW_DIMENSION,
                                               thumb->core.depth);
        thumb->thumbWheel.pix2 = XCreatePixmap(XtDisplay((Widget)thumb),
                                               RootWindowOfScreen(XtScreen((Widget)thumb)),
                                               WHEEL_LONG_DIMENSION, WHEEL_NARROW_DIMENSION,
                                               thumb->core.depth);
        thumb->thumbWheel.pix3 = XCreatePixmap(XtDisplay((Widget)thumb),
                                               RootWindowOfScreen(XtScreen((Widget)thumb)),
                                               WHEEL_LONG_DIMENSION, WHEEL_NARROW_DIMENSION,
                                               thumb->core.depth);
        thumb->thumbWheel.pix4 = XCreatePixmap(XtDisplay((Widget)thumb),
                                               RootWindowOfScreen(XtScreen((Widget)thumb)),
                                               WHEEL_LONG_DIMENSION, WHEEL_NARROW_DIMENSION,
                                               thumb->core.depth);
        thumb->thumbWheel.pix1_hilite = XCreatePixmap(XtDisplay((Widget)thumb),
                                                      RootWindowOfScreen(XtScreen((Widget)thumb)),
                                                      WHEEL_LONG_DIMENSION, WHEEL_NARROW_DIMENSION,
                                                      thumb->core.depth);
        thumb->thumbWheel.pix2_hilite = XCreatePixmap(XtDisplay((Widget)thumb),
                                                      RootWindowOfScreen(XtScreen((Widget)thumb)),
                                                      WHEEL_LONG_DIMENSION, WHEEL_NARROW_DIMENSION,
                                                      thumb->core.depth);
        thumb->thumbWheel.pix3_hilite = XCreatePixmap(XtDisplay((Widget)thumb),
                                                      RootWindowOfScreen(XtScreen((Widget)thumb)),
                                                      WHEEL_LONG_DIMENSION, WHEEL_NARROW_DIMENSION,
                                                      thumb->core.depth);
        thumb->thumbWheel.pix4_hilite = XCreatePixmap(XtDisplay((Widget)thumb),
                                                      RootWindowOfScreen(XtScreen((Widget)thumb)),
                                                      WHEEL_LONG_DIMENSION, WHEEL_NARROW_DIMENSION,
                                                      thumb->core.depth);
    }
    else
    {
        thumb->thumbWheel.pix1 = XCreatePixmap(XtDisplay((Widget)thumb),
                                               RootWindowOfScreen(XtScreen((Widget)thumb)),
                                               WHEEL_NARROW_DIMENSION, WHEEL_LONG_DIMENSION,
                                               thumb->core.depth);
        thumb->thumbWheel.pix2 = XCreatePixmap(XtDisplay((Widget)thumb),
                                               RootWindowOfScreen(XtScreen((Widget)thumb)),
                                               WHEEL_NARROW_DIMENSION, WHEEL_LONG_DIMENSION,
                                               thumb->core.depth);
        thumb->thumbWheel.pix3 = XCreatePixmap(XtDisplay((Widget)thumb),
                                               RootWindowOfScreen(XtScreen((Widget)thumb)),
                                               WHEEL_NARROW_DIMENSION, WHEEL_LONG_DIMENSION,
                                               thumb->core.depth);
        thumb->thumbWheel.pix4 = XCreatePixmap(XtDisplay((Widget)thumb),
                                               RootWindowOfScreen(XtScreen((Widget)thumb)),
                                               WHEEL_NARROW_DIMENSION, WHEEL_LONG_DIMENSION,
                                               thumb->core.depth);
        thumb->thumbWheel.pix1_hilite = XCreatePixmap(XtDisplay((Widget)thumb),
                                                      RootWindowOfScreen(XtScreen((Widget)thumb)),
                                                      WHEEL_NARROW_DIMENSION, WHEEL_LONG_DIMENSION,
                                                      thumb->core.depth);
        thumb->thumbWheel.pix2_hilite = XCreatePixmap(XtDisplay((Widget)thumb),
                                                      RootWindowOfScreen(XtScreen((Widget)thumb)),
                                                      WHEEL_NARROW_DIMENSION, WHEEL_LONG_DIMENSION,
                                                      thumb->core.depth);
        thumb->thumbWheel.pix3_hilite = XCreatePixmap(XtDisplay((Widget)thumb),
                                                      RootWindowOfScreen(XtScreen((Widget)thumb)),
                                                      WHEEL_NARROW_DIMENSION, WHEEL_LONG_DIMENSION,
                                                      thumb->core.depth);
        thumb->thumbWheel.pix4_hilite = XCreatePixmap(XtDisplay((Widget)thumb),
                                                      RootWindowOfScreen(XtScreen((Widget)thumb)),
                                                      WHEEL_NARROW_DIMENSION, WHEEL_LONG_DIMENSION,
                                                      thumb->core.depth);
    }
    RenderPixmap(thumb, 1);
    RenderPixmap(thumb, 2);
    RenderPixmap(thumb, 3);
    RenderPixmap(thumb, 4);
    RenderPixmap(thumb, -1);
    RenderPixmap(thumb, -2);
    RenderPixmap(thumb, -3);
    RenderPixmap(thumb, -4);

    thumb->thumbWheel.button_quiet_pixmap = XCreatePixmap(XtDisplay((Widget)thumb),
                                                          RootWindowOfScreen(XtScreen((Widget)thumb)),
                                                          BUTTON_SIZE, BUTTON_SIZE, thumb->core.depth);
    thumb->thumbWheel.button_hilite_pixmap = XCreatePixmap(XtDisplay((Widget)thumb),
                                                           RootWindowOfScreen(XtScreen((Widget)thumb)),
                                                           BUTTON_SIZE, BUTTON_SIZE, thumb->core.depth);

    RenderButtonPixmaps(thumb);
}

static void
#ifdef _NO_PROTO
    RenderPixmap(thumb, which)
        SgThumbWheelWidget thumb;
int which;
#else
RenderPixmap(SgThumbWheelWidget thumb, int which)
#endif /* _NO_PROTO */
{
    GC darkestGC = GCarray[0];
    GC veryDarkGC = GCarray[1];
    GC darkGC = GCarray[2];
    GC mediumGC = GCarray[3];
    GC lightGC = GCarray[4];
    GC veryLightGC = GCarray[5];
    GC lightestGC = GCarray[6];

    Display *dpy = XtDisplay((Widget)thumb);
    Pixmap pix = 0;
    int off = 0;

    switch (which)
    {
    case 1:
        pix = thumb->thumbWheel.pix1;
        off = 0;
        break;
    case 2:
        pix = thumb->thumbWheel.pix2;
        off = 2;
        break;
    case 3:
        pix = thumb->thumbWheel.pix3;
        off = 4;
        break;
    case 4:
        pix = thumb->thumbWheel.pix4;
        off = -2;
        break;
    case -1:
        pix = thumb->thumbWheel.pix1_hilite;
        off = 0;
        break;
    case -2:
        pix = thumb->thumbWheel.pix2_hilite;
        off = 2;
        break;
    case -3:
        pix = thumb->thumbWheel.pix3_hilite;
        off = 4;
        break;
    case -4:
        pix = thumb->thumbWheel.pix4_hilite;
        off = -2;
        break;
    default:
        fprintf(stderr, "ThumbWheel.c:RenderPixmap(): pix and off uninitialized\n");
        break;
    }

    if (thumb->thumbWheel.orientation == XmHORIZONTAL)
    {
        XFillRectangle(dpy, pix, ((which > 0) ? lightGC : veryLightGC),
                       0, 0, WHEEL_LONG_DIMENSION, WHEEL_NARROW_DIMENSION);
    }
    else
    {
        XFillRectangle(dpy, pix, ((which > 0) ? lightGC : veryLightGC),
                       0, 0, WHEEL_NARROW_DIMENSION, WHEEL_LONG_DIMENSION);
    }
    /*
    * The macro WheelDrawLine takes arguments similar to what we would use
    * for XDrawLine if the wheel were vertical.  If the wheel is horizontal,
    * the macro flips the arguments.
    */
    /* lines to either side of wheel */
    WheelDrawLine(thumb, dpy, pix, veryDarkGC, 2, 7, 2, 116);
    WheelDrawLine(thumb, dpy, pix, darkestGC, 13, 7, 13, 116);
    /* piecemeal lines to left and right of wheel */
    /* 6 pixels */
    WheelDrawLine(thumb, dpy, pix, darkestGC, 3, 7, 3, 12);
    /* 6 pixels */
    WheelDrawLine(thumb, dpy, pix, darkestGC, 12, 7, 12, 12);
    /* 7 pixels */
    WheelDrawLine(thumb, dpy, pix, darkGC, 3, 13, 3, 19);
    /* 7 pixels */
    WheelDrawLine(thumb, dpy, pix, darkGC, 12, 13, 12, 19);
    /* 8 pixels */
    WheelDrawLine(thumb, dpy, pix, mediumGC, 3, 20, 3, 27);
    /* 8 pixels */
    WheelDrawLine(thumb, dpy, pix, mediumGC, 12, 20, 12, 27);
    /* 11 pixels */
    WheelDrawLine(thumb, dpy, pix, lightGC, 3, 28, 3, 38);
    /* 11 pixels */
    WheelDrawLine(thumb, dpy, pix, lightGC, 12, 28, 12, 38);
    /* 11 pixels */
    WheelDrawLine(thumb, dpy, pix, veryLightGC, 3, 39, 3, 49);
    /* 11 pixels */
    WheelDrawLine(thumb, dpy, pix, veryLightGC, 12, 39, 12, 49);
    /* 23 pixels */
    WheelDrawLine(thumb, dpy, pix, lightestGC, 3, 50, 3, 72);
    /* 23 pixels */
    WheelDrawLine(thumb, dpy, pix, lightestGC, 12, 50, 12, 72);
    /* 11 pixels */
    WheelDrawLine(thumb, dpy, pix, veryLightGC, 3, 73, 3, 83);
    /* 11 pixels */
    WheelDrawLine(thumb, dpy, pix, veryLightGC, 12, 73, 12, 83);
    /* 11 pixels */
    WheelDrawLine(thumb, dpy, pix, lightGC, 3, 84, 3, 94);
    /* 11 pixels */
    WheelDrawLine(thumb, dpy, pix, lightGC, 12, 84, 12, 94);
    /* 8 pixels */
    WheelDrawLine(thumb, dpy, pix, mediumGC, 3, 95, 3, 102);
    /* 8 pixels */
    WheelDrawLine(thumb, dpy, pix, mediumGC, 12, 95, 12, 102);
    /* 7 pixels */
    WheelDrawLine(thumb, dpy, pix, darkGC, 3, 103, 3, 109);
    /* 7 pixels */
    WheelDrawLine(thumb, dpy, pix, darkGC, 12, 103, 12, 109);
    /* 6 pixels? */
    WheelDrawLine(thumb, dpy, pix, darkestGC, 3, 110, 3, 116);
    /* 6 pixels? */
    WheelDrawLine(thumb, dpy, pix, darkestGC, 12, 110, 12, 116);

    /* lines across wheel */
    /* constant line across top and bottom edge of wheel */
    WheelDrawLine(thumb, dpy, pix, darkestGC, 4, 7, 11, 7);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 8, 11, 8);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 115, 11, 115);
    WheelDrawLine(thumb, dpy, pix, darkestGC, 4, 116, 11, 116);
#ifdef HALFOFFSET
    /* half-offset lines across top of wheel */
    WheelDrawLine(thumb, dpy, pix, darkestGC, 4, 9 + off / 2, 11, 9 + off / 2);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 10 + off / 2, 11, 10 + off / 2);
    WheelDrawLine(thumb, dpy, pix, darkestGC, 4, 12 + off / 2, 11, 12 + off / 2);
    /* half-offset lines across bottom of wheel */
    WheelDrawLine(thumb, dpy, pix, darkestGC, 4, 111 + off / 2, 11, 111 + off / 2);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 113 + off / 2, 11, 113 + off / 2);
    WheelDrawLine(thumb, dpy, pix, darkestGC, 4, 114 + off / 2, 11, 114 + off / 2);
#else
    /* constant lines across top of wheel */
    WheelDrawLine(thumb, dpy, pix, darkestGC, 4, 9, 11, 9);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 10, 11, 10);
    WheelDrawLine(thumb, dpy, pix, darkestGC, 4, 12, 11, 12);
    /* constant lines across bottom of wheel */
    WheelDrawLine(thumb, dpy, pix, darkestGC, 4, 111, 11, 111);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 113, 11, 113);
    WheelDrawLine(thumb, dpy, pix, darkestGC, 4, 114, 11, 114);
#endif /*HALFOFFSET */
    /* which-pixmap dependent lines across middle region of wheel */
    if (off > -2)
    {
        WheelDrawLine(thumb, dpy, pix, darkGC, 4, 13 + off, 11, 13 + off);
    }
    WheelDrawLine(thumb, dpy, pix, veryDarkGC, 4, 16 + off, 11, 16 + off);
    WheelDrawLine(thumb, dpy, pix, veryLightGC, 4, 21 + off, 11, 21 + off);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 22 + off, 11, 22 + off);
    WheelDrawLine(thumb, dpy, pix, veryLightGC, 4, 27 + off, 11, 27 + off);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 28 + off, 11, 28 + off);
    WheelDrawLine(thumb, dpy, pix, lightestGC, 4, 34 + off, 11, 34 + off);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 35 + off, 11, 35 + off);
    WheelDrawLine(thumb, dpy, pix, lightestGC, 4, 41 + off, 11, 41 + off);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 42 + off, 11, 42 + off);
    WheelDrawLine(thumb, dpy, pix, lightestGC, 4, 48 + off, 11, 48 + off);
    WheelDrawLine(thumb, dpy, pix, lightestGC, 4, 49 + off, 11, 49 + off);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 50 + off, 11, 50 + off);
    WheelDrawLine(thumb, dpy, pix, lightestGC, 4, 56 + off, 11, 56 + off);
    WheelDrawLine(thumb, dpy, pix, lightestGC, 4, 57 + off, 11, 57 + off);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 58 + off, 11, 58 + off);
    WheelDrawLine(thumb, dpy, pix, lightestGC, 4, 64 + off, 11, 64 + off);
    WheelDrawLine(thumb, dpy, pix, lightestGC, 4, 65 + off, 11, 65 + off);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 66 + off, 11, 66 + off);
    WheelDrawLine(thumb, dpy, pix, lightestGC, 4, 72 + off, 11, 72 + off);
    WheelDrawLine(thumb, dpy, pix, lightestGC, 4, 73 + off, 11, 73 + off);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 74 + off, 11, 74 + off);
    WheelDrawLine(thumb, dpy, pix, lightestGC, 4, 81 + off, 11, 81 + off);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 82 + off, 11, 82 + off);
    WheelDrawLine(thumb, dpy, pix, lightestGC, 4, 88 + off, 11, 88 + off);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 89 + off, 11, 89 + off);
    WheelDrawLine(thumb, dpy, pix, veryLightGC, 4, 95 + off, 11, 95 + off);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 96 + off, 11, 96 + off);
    WheelDrawLine(thumb, dpy, pix, veryLightGC, 4, 101 + off, 11, 101 + off);
    WheelDrawLine(thumb, dpy, pix, darkGC, 4, 102 + off, 11, 102 + off);
    if (off < 4)
    {
        WheelDrawLine(thumb, dpy, pix, veryDarkGC, 4, 107 + off, 11, 107 + off);
    }
    if (off < 2)
    {
        WheelDrawLine(thumb, dpy, pix, darkGC, 4, 110 + off, 11, 110 + off);
    }
}

static void
#ifdef _NO_PROTO
    RenderButtonPixmaps(thumb)
        SgThumbWheelWidget thumb;
#else
RenderButtonPixmaps(SgThumbWheelWidget thumb)
#endif /* _NO_PROTO */
{
    GC darkestGC = GCarray[0];
    /* GC veryDarkGC =GCarray[1]; */
    /* GC darkGC = GCarray[2]; */
    /* GC mediumGC = GCarray[3]; */
    GC lightGC = GCarray[4];
    GC veryLightGC = GCarray[5];
    /* GC lightestGC = GCarray[6]; */

    Display *dpy = XtDisplay((Widget)thumb);
    Pixmap pixq;
    Pixmap pixh;

    pixq = thumb->thumbWheel.button_quiet_pixmap;
    pixh = thumb->thumbWheel.button_hilite_pixmap;

    /*
    * Fill the background of the buttons
    */
    XFillRectangle(dpy, pixq, lightGC, 0, 0, BUTTON_SIZE, BUTTON_SIZE);
    XFillRectangle(dpy, pixh, veryLightGC, 0, 0, BUTTON_SIZE, BUTTON_SIZE);

    /*
    * Draw the outer square
    */
    XDrawLine(dpy, pixq, darkestGC, 3, 3, 12, 3);
    XDrawLine(dpy, pixh, darkestGC, 3, 3, 12, 3);
    XDrawLine(dpy, pixq, darkestGC, 12, 3, 12, 12);
    XDrawLine(dpy, pixh, darkestGC, 12, 3, 12, 12);
    XDrawLine(dpy, pixq, darkestGC, 12, 12, 3, 12);
    XDrawLine(dpy, pixh, darkestGC, 12, 12, 3, 12);
    XDrawLine(dpy, pixq, darkestGC, 3, 12, 3, 3);
    XDrawLine(dpy, pixh, darkestGC, 3, 12, 3, 3);

    /*
    * Draw the inner square
    */
    XFillRectangle(dpy, pixq, darkestGC, 6, 6, 4, 4);
    XFillRectangle(dpy, pixh, darkestGC, 6, 6, 4, 4);
}

static void
#ifdef _NO_PROTO
    GetForegroundGC(thumb)
        SgThumbWheelWidget thumb;
#else
GetForegroundGC(SgThumbWheelWidget thumb)
#endif /* _NO_PROTO */
{
    XGCValues values;
    XtGCMask valueMask;

    valueMask = GCForeground | GCBackground | GCGraphicsExposures;
    /* ?? */
    values.foreground = thumb->primitive.foreground;
    values.background = thumb->core.background_pixel;
    values.graphics_exposures = FALSE;

    thumb->thumbWheel.foreground_GC = XtGetGC((Widget)thumb, valueMask, &values);
}

static Boolean
#ifdef _NO_PROTO
    MouseIsInWheel(thumb, event_x, event_y)
        SgThumbWheelWidget thumb;
int event_x, event_y;
#else
MouseIsInWheel(SgThumbWheelWidget thumb, int event_x,
               int event_y)
#endif /* _NO_PROTO */
{
    int shadow = thumb->primitive.shadow_thickness;

    /*
    * The wheel should highlight when the mouse is over it or
    * over the shadows drawn around it.
    */
    if (thumb->thumbWheel.orientation == XmHORIZONTAL)
    {
        return (((event_x <= thumb->thumbWheel.wheel_x + WHEEL_LONG_DIMENSION - 1) && (event_x >= thumb->thumbWheel.wheel_x - shadow))
                && ((event_y <= thumb->thumbWheel.wheel_y + WHEEL_NARROW_DIMENSION - 1)
                    && (event_y >= thumb->thumbWheel.wheel_y - shadow)));
    }
    else
    {
        return (((event_y <= thumb->thumbWheel.wheel_y + WHEEL_LONG_DIMENSION - 1) && (event_y >= thumb->thumbWheel.wheel_y - shadow))
                && ((event_x <= thumb->thumbWheel.wheel_x + WHEEL_NARROW_DIMENSION - 1)
                    && (event_x >= thumb->thumbWheel.wheel_x - shadow)));
    }
}

static Boolean
#ifdef _NO_PROTO
    MouseIsInButton(thumb, event_x, event_y)
        SgThumbWheelWidget thumb;
int event_x, event_y;
#else
MouseIsInButton(SgThumbWheelWidget thumb, int event_x,
                int event_y)
#endif /* _NO_PROTO */
{
    int shadow = thumb->primitive.shadow_thickness;

    if (thumb->thumbWheel.show_home_button == FALSE)
    {
        return FALSE;
    }

    /*
    * The button should highlight when the mouse is over it or
    * over the shadows drawn around it.
    */
    if (thumb->thumbWheel.orientation == XmHORIZONTAL)
    {
        return (((event_x > thumb->thumbWheel.button_x - shadow) && (event_x <= thumb->thumbWheel.button_x + BUTTON_SIZE - 1))
                && ((event_y <= thumb->thumbWheel.button_y + BUTTON_SIZE - 1) && (event_y >= thumb->thumbWheel.button_y - shadow)));
    }
    else
    {
        return (((event_y > thumb->thumbWheel.button_y - shadow) && (event_y <= thumb->thumbWheel.button_y + BUTTON_SIZE - 1))
                && ((event_x <= thumb->thumbWheel.button_x + BUTTON_SIZE - 1) && (event_x >= thumb->thumbWheel.button_x - shadow)));
    }
}

static void
#ifdef _NO_PROTO
    SetCurrentPixmap(thumb, value_increased)
        SgThumbWheelWidget thumb;
Boolean value_increased;
#else
SetCurrentPixmap(SgThumbWheelWidget thumb, Boolean value_increased)
#endif /* _NO_PROTO */
{
    if (thumb->thumbWheel.current_quiet_pixmap == thumb->thumbWheel.pix1)
    {
        thumb->thumbWheel.current_quiet_pixmap = ((value_increased == TRUE) ? thumb->thumbWheel.pix2 : thumb->thumbWheel.pix4);
    }
    else if (thumb->thumbWheel.current_quiet_pixmap == thumb->thumbWheel.pix2)
    {
        thumb->thumbWheel.current_quiet_pixmap = ((value_increased == TRUE) ? thumb->thumbWheel.pix3 : thumb->thumbWheel.pix1);
    }
    else if (thumb->thumbWheel.current_quiet_pixmap == thumb->thumbWheel.pix3)
    {
        thumb->thumbWheel.current_quiet_pixmap = ((value_increased == TRUE) ? thumb->thumbWheel.pix4 : thumb->thumbWheel.pix2);
    }
    else if (thumb->thumbWheel.current_quiet_pixmap == thumb->thumbWheel.pix4)
    {
        thumb->thumbWheel.current_quiet_pixmap = ((value_increased == TRUE) ? thumb->thumbWheel.pix1 : thumb->thumbWheel.pix3);
    }
    if (thumb->thumbWheel.current_hilite_pixmap
        == thumb->thumbWheel.pix1_hilite)
    {
        thumb->thumbWheel.current_hilite_pixmap = ((value_increased == TRUE) ? thumb->thumbWheel.pix2_hilite : thumb->thumbWheel.pix4_hilite);
    }
    else if (thumb->thumbWheel.current_hilite_pixmap
             == thumb->thumbWheel.pix2_hilite)
    {
        thumb->thumbWheel.current_hilite_pixmap = ((value_increased == TRUE) ? thumb->thumbWheel.pix3_hilite : thumb->thumbWheel.pix1_hilite);
    }
    else if (thumb->thumbWheel.current_hilite_pixmap
             == thumb->thumbWheel.pix3_hilite)
    {
        thumb->thumbWheel.current_hilite_pixmap = ((value_increased == TRUE) ? thumb->thumbWheel.pix4_hilite : thumb->thumbWheel.pix2_hilite);
    }
    else if (thumb->thumbWheel.current_hilite_pixmap
             == thumb->thumbWheel.pix4_hilite)
    {
        thumb->thumbWheel.current_hilite_pixmap = ((value_increased == TRUE) ? thumb->thumbWheel.pix1_hilite : thumb->thumbWheel.pix3_hilite);
    }
}

static void
#ifdef _NO_PROTO
    FreePixmaps(thumb)
        SgThumbWheelWidget thumb;
#else
FreePixmaps(SgThumbWheelWidget thumb)
#endif /* _NO_PROTO */
{
#define MyFreePixmap(w, field)                                  \
    if (w->thumbWheel.field != 0)                               \
    {                                                           \
        XFreePixmap(XtDisplay((Widget)w), w->thumbWheel.field); \
        w->thumbWheel.field = 0;                                \
    }

    MyFreePixmap(thumb, pix1);
    MyFreePixmap(thumb, pix2);
    MyFreePixmap(thumb, pix3);
    MyFreePixmap(thumb, pix4);
    MyFreePixmap(thumb, pix1_hilite);
    MyFreePixmap(thumb, pix2_hilite);
    MyFreePixmap(thumb, pix3_hilite);
    MyFreePixmap(thumb, pix4_hilite);
    thumb->thumbWheel.current_quiet_pixmap = 0;
    thumb->thumbWheel.current_hilite_pixmap = 0;

    MyFreePixmap(thumb, button_quiet_pixmap);
    MyFreePixmap(thumb, button_hilite_pixmap);
#undef MyFreePixmap
}

static Boolean
#ifdef _NO_PROTO
    ValidateFields(cur_w, req_w, new_w)
        SgThumbWheelWidget cur_w;
SgThumbWheelWidget req_w;
SgThumbWheelWidget new_w;
#else
ValidateFields(SgThumbWheelWidget cur_w, SgThumbWheelWidget req_w,
               SgThumbWheelWidget new_w)
#endif /* _NO_PROTO */
{
    Boolean return_flag = True;
    (void)req_w;
    if (new_w->thumbWheel.lower_bound > new_w->thumbWheel.upper_bound)
    {
        int tmp = new_w->thumbWheel.lower_bound;
        new_w->thumbWheel.lower_bound = new_w->thumbWheel.upper_bound;
        new_w->thumbWheel.upper_bound = tmp;
        /*    return_flag = FALSE;  is there any reason for this? comment out for now*/
    }

#if 1
    /* the efficient way to code this. */
    new_w->thumbWheel.infinite = ((new_w->thumbWheel.angle_range == 0) || (new_w->thumbWheel.upper_bound == new_w->thumbWheel.lower_bound));
    if (new_w->thumbWheel.infinite == TRUE)
    {
        new_w->thumbWheel.pegged = FALSE;
    }
#else
    else if (new_w->thumbWheel.upper_bound == new_w->thumbWheel.lower_bound)
    {
        new_w->thumbWheel.infinite = TRUE;
        new_w->thumbWheel.pegged = FALSE;
    }

    if (new_w->thumbWheel.angle_range == 0)
    {
        new_w->thumbWheel.infinite = TRUE;
        new_w->thumbWheel.pegged = FALSE;
    }
    else
    {
        /* angle range is nonzero.  if upper != lower, infinite is FALSE. */
        if (new_w->thumbWheel.upper_bound != new_w->thumbWheel.lower_bound)
        {
            new_w->thumbWheel.infinite = FALSE;
        }
    }
#endif

    if (new_w->thumbWheel.infinite != TRUE)
    {
        /* Range / values checking */
        if (new_w->thumbWheel.value < new_w->thumbWheel.lower_bound)
        {
            new_w->thumbWheel.value = new_w->thumbWheel.lower_bound;
            new_w->thumbWheel.pegged = TRUE;
        }
        if (new_w->thumbWheel.value > new_w->thumbWheel.upper_bound)
        {
            new_w->thumbWheel.value = new_w->thumbWheel.upper_bound;
            new_w->thumbWheel.pegged = TRUE;
        }
        if (new_w->thumbWheel.home_position < new_w->thumbWheel.lower_bound)
        {
            new_w->thumbWheel.home_position = new_w->thumbWheel.lower_bound;
        }
        if (new_w->thumbWheel.home_position > new_w->thumbWheel.upper_bound)
        {
            new_w->thumbWheel.home_position = new_w->thumbWheel.upper_bound;
        }
    }

    if ((new_w->thumbWheel.orientation != XmHORIZONTAL) && (new_w->thumbWheel.orientation != XmVERTICAL))
    {
        new_w->thumbWheel.orientation = cur_w->thumbWheel.orientation;
    }

    return return_flag;
}

static void
#ifdef _NO_PROTO
    ArmHomeButton(thumb)
        SgThumbWheelWidget thumb;
#else
ArmHomeButton(SgThumbWheelWidget thumb)
#endif /* _NO_PROTO */
{
    if (thumb->thumbWheel.show_home_button == FALSE)
    {
        return;
    }
    thumb->thumbWheel.home_button_armed = TRUE;
    RenderButtonShadows(thumb);
}

static void
#ifdef _NO_PROTO
    DisarmHomeButton(thumb)
        SgThumbWheelWidget thumb;
#else
DisarmHomeButton(SgThumbWheelWidget thumb)
#endif /* _NO_PROTO */
{
    if (thumb->thumbWheel.show_home_button == FALSE)
    {
        return;
    }
    thumb->thumbWheel.home_button_armed = FALSE;
    RenderButtonShadows(thumb);
}

static void
#ifdef _NO_PROTO
    RenderButtonShadows(thumb)
        SgThumbWheelWidget thumb;
#else
RenderButtonShadows(SgThumbWheelWidget thumb)
#endif /* _NO_PROTO */
{
    /*
    * Render shadows around home button.
    */
    if (thumb->thumbWheel.show_home_button == TRUE)
    {
        int shadow = thumb->primitive.shadow_thickness;

        _XmDrawShadows(XtDisplay((Widget)thumb), XtWindow((Widget)thumb),
                       thumb->primitive.top_shadow_GC,
                       thumb->primitive.bottom_shadow_GC,
                       thumb->thumbWheel.button_x - shadow,
                       thumb->thumbWheel.button_y - shadow,
                       BUTTON_SIZE + 2 * shadow,
                       BUTTON_SIZE + 2 * shadow,
                       shadow,
                       (thumb->thumbWheel.home_button_armed ? XmSHADOW_IN : XmSHADOW_OUT));
    }
}
#endif
