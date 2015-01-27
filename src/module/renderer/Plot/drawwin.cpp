/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: drawwin.c,v 1.3 1994/09/29 03:37:37 pturner Exp pturner $
 *
 * Set drawing and scrolling options
 */
#include <stdio.h>
#include <math.h>

#include <Xm/Xm.h>
#include <Xm/BulletinB.h>
#include <Xm/DialogS.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/ToggleB.h>
#include <Xm/Scale.h>
#include <Xm/Separator.h>

#include "globals.h"
#include "motifinc.h"

extern int cursortype;

static Widget draw_frame;
static Widget draw_panel;

/*
 * Panel item declarations
 */
static Widget autoredraw_type_item;
static Widget autorefresh_type_item;
static Widget autoclear_type_item;
static Widget cursor_type_item;
static Widget scrollper_item;
static Widget linkscroll_item;

/*
 * scroll amount declared in graphutils.c TODO - move to globals.h
 */
extern int scrolling_islinked;
extern double scrollper;
extern double shexper;

/* suppress clear in the drivers - TODO move to globals.h */
int overlay = 0;

/*
 * Event and Notify proc declarations
 */

void update_draw(void)
{
    Arg a;
    int iv;
    if (draw_frame)
    {
        XmToggleButtonSetState(linkscroll_item, scrolling_islinked == TRUE, False);
        XmToggleButtonSetState(autoredraw_type_item, auto_redraw == TRUE, False);
        XmToggleButtonSetState(autorefresh_type_item, allow_refresh == TRUE, False);
        XmToggleButtonSetState(autoclear_type_item, overlay == TRUE, False);
        XmToggleButtonSetState(cursor_type_item, cursortype == TRUE, False);
        iv = (int)(100 * scrollper);
        XtSetArg(a, XmNvalue, iv);
        XtSetValues(scrollper_item, &a, 1);
    }
}

/*
 * define the draw options
 */
static void define_draw_proc(Widget, XtPointer, XtPointer)
{
    Arg a;
    int value, otmp = overlay;
    scrolling_islinked = XmToggleButtonGetState(linkscroll_item);
    auto_redraw = XmToggleButtonGetState(autoredraw_type_item);
    allow_refresh = XmToggleButtonGetState(autorefresh_type_item);
    overlay = XmToggleButtonGetState(autoclear_type_item);
    cursortype = XmToggleButtonGetState(cursor_type_item);
    XtSetArg(a, XmNvalue, &value);
    XtGetValues(scrollper_item, &a, 1);
    scrollper = value / 100.0;
    XtUnmanageChild(draw_frame);
    if (otmp != overlay)
    {
        drawgraph();
    }
}

/*
 * Create the draw Frame and the draw Panel
 */
void create_draw_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget buts[2];
    //Widget wlabel;

    set_wait_cursor();
    if (draw_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        draw_frame = XmCreateDialogShell(app_shell, (char *)"Draw options", NULL, 0);
        handle_close(draw_frame);
        XtVaSetValues(draw_frame,
                      XmNx, x,
                      XmNy, y,
                      NULL);
        draw_panel = XmCreateRowColumn(draw_frame, (char *)"draw_rc", NULL, 0);

        //wlabel =
        XtVaCreateManagedWidget("Scroll %:", xmLabelWidgetClass, draw_panel, NULL);
        scrollper_item = XtVaCreateManagedWidget("scroll", xmScaleWidgetClass, draw_panel,
                                                 XmNwidth, 200,
                                                 XmNminimum, 0,
                                                 XmNmaximum, 200,
                                                 XmNvalue, 0,
                                                 XmNshowValue, True,
                                                 XmNprocessingDirection, XmMAX_ON_RIGHT,
                                                 XmNorientation, XmHORIZONTAL,
                                                 NULL);
        linkscroll_item = XtVaCreateManagedWidget("Linked scrolling",
                                                  xmToggleButtonWidgetClass, draw_panel,
                                                  NULL);
        autoredraw_type_item = XtVaCreateManagedWidget("Auto redraw",
                                                       xmToggleButtonWidgetClass, draw_panel,
                                                       NULL);
        autorefresh_type_item = XtVaCreateManagedWidget("Auto refresh",
                                                        xmToggleButtonWidgetClass, draw_panel,
                                                        NULL);
        autoclear_type_item = XtVaCreateManagedWidget("Overlay",
                                                      xmToggleButtonWidgetClass, draw_panel,
                                                      NULL);
        cursor_type_item = XtVaCreateManagedWidget("Crosshair cursor",
                                                   xmToggleButtonWidgetClass, draw_panel,
                                                   NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, draw_panel, NULL);

        CreateCommandButtons(draw_panel, 2, buts, label1);
        XtAddCallback(buts[0], XmNactivateCallback,
                      (XtCallbackProc)define_draw_proc, (XtPointer)0);
        XtAddCallback(buts[1], XmNactivateCallback,
                      (XtCallbackProc)destroy_dialog, (XtPointer)draw_frame);

        XtManageChild(draw_panel);
    }
    XtRaise(draw_frame);
    update_draw();
    unset_wait_cursor();
}
