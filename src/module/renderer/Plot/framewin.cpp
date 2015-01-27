/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: framewin.c,v 1.3 1994/09/29 03:37:37 pturner Exp pturner $
 *
 * frame Panel
 *
 */

#include <stdio.h>

#include <Xm/Xm.h>
#include <Xm/BulletinB.h>
#include <Xm/DialogS.h>
#include <Xm/Frame.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/Separator.h>
#include <Xm/ToggleB.h>

#include "globals.h"
#include "motifinc.h"

static Widget frame_frame;
static Widget frame_panel;

/*
 * Widget item declarations
 */
static Widget frame_frameactive_choice_item;
static Widget *frame_framestyle_choice_item;
static Widget *frame_color_choice_item;
static Widget *frame_lines_choice_item;
static Widget *frame_linew_choice_item;
static Widget frame_fillbg_choice_item;
static Widget *frame_bgcolor_choice_item;
static Widget *frame_applyto_choice_item;

/*
 * Event and Notify proc declarations
 */
static void frame_define_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);

void update_frame_items(int gno)
{
    if (frame_frame)
    {
        XmToggleButtonSetState(frame_frameactive_choice_item,
                               g[gno].f.active == ON, False);
        SetChoice(frame_framestyle_choice_item, g[gno].f.type);
        SetChoice(frame_color_choice_item, g[gno].f.color);
        SetChoice(frame_linew_choice_item, g[gno].f.linew - 1);
        SetChoice(frame_lines_choice_item, g[gno].f.lines - 1);
        XmToggleButtonSetState(frame_fillbg_choice_item,
                               g[gno].f.fillbg == ON, False);
        SetChoice(frame_bgcolor_choice_item, g[gno].f.bgcolor);
    }
}

/*
 * Create the frame Widget and the frame Widget
 */
void create_frame_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget rc;

    set_wait_cursor();
    if (frame_frame == NULL)
    {
        Widget buts[2];
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";

        XmGetPos(app_shell, 0, &x, &y);
        frame_frame = XmCreateDialogShell(app_shell, (char *)"Frame", NULL, 0);
        handle_close(frame_frame);
        XtVaSetValues(frame_frame, XmNx, x, XmNy, y, NULL);
        frame_panel = XtVaCreateWidget("frame panel", xmRowColumnWidgetClass, frame_frame,
                                       NULL);

        frame_frameactive_choice_item = XmCreateToggleButton(frame_panel, (char *)"Display graph frame",
                                                             NULL, 0);
        XtManageChild(frame_frameactive_choice_item);

        rc = XtVaCreateWidget("rc", xmRowColumnWidgetClass, frame_panel,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 5,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        XtVaCreateManagedWidget("Frame type:", xmLabelWidgetClass, rc, NULL);
        frame_framestyle_choice_item = CreatePanelChoice(rc, " ",
                                                         7,
                                                         "Closed",
                                                         "Half open",
                                                         "Break top",
                                                         "Break bottom",
                                                         "Break left",
                                                         "Break right",
                                                         NULL,
                                                         NULL);

        XtVaCreateManagedWidget("Line color:", xmLabelWidgetClass, rc, NULL);
        frame_color_choice_item = CreateColorChoice(rc, " ", 0);

        XtVaCreateManagedWidget("Line width:", xmLabelWidgetClass, rc, NULL);
        frame_linew_choice_item = CreatePanelChoice(rc, " ",
                                                    10,
                                                    "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                                    NULL,
                                                    NULL);

        XtVaCreateManagedWidget("Line style:", xmLabelWidgetClass, rc, NULL);
        frame_lines_choice_item = CreatePanelChoice(rc, " ",
                                                    6,
                                                    "Solid line",
                                                    "Dotted line",
                                                    "Dashed line",
                                                    "Long Dashed",
                                                    "Dot-dashed",
                                                    NULL,
                                                    NULL);

        frame_fillbg_choice_item = XmCreateToggleButton(rc, (char *)"Fill graph frame",
                                                        NULL, 0);
        XtManageChild(frame_fillbg_choice_item);

        frame_bgcolor_choice_item = CreateColorChoice(rc, "Fill color:", 0);
        XtManageChild(rc);

        frame_applyto_choice_item = CreatePanelChoice(frame_panel, "Apply to:",
                                                      3,
                                                      "Current graph",
                                                      "All active graphs",
                                                      NULL,
                                                      NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, frame_panel,
                                NULL);

        CreateCommandButtons(frame_panel, 2, buts, label1);
        XtAddCallback(buts[0], XmNactivateCallback,
                      (XtCallbackProc)frame_define_notify_proc, (XtPointer)NULL);
        XtAddCallback(buts[1], XmNactivateCallback,
                      (XtCallbackProc)destroy_dialog, (XtPointer)frame_frame);

        XtManageChild(frame_panel);
    }
    XtRaise(frame_frame);
    update_frame_items(cg);
    unset_wait_cursor();
}

/*
 * Notify and event procs
 */

static void frame_define_notify_proc(Widget, XtPointer, XtPointer)
{
    int i, ming, maxg;
    int a = (int)GetChoice(frame_applyto_choice_item);
    if (a == 0)
    {
        ming = maxg = cg;
    }
    else
    {
        ming = 0;
        maxg = maxgraph - 1;
    }
    for (i = ming; i <= maxg; i++)
    {
        if (isactive_graph(i))
        {
            g[i].f.active = XmToggleButtonGetState(frame_frameactive_choice_item) ? ON : OFF;
            g[i].f.type = (int)GetChoice(frame_framestyle_choice_item);
            g[i].f.color = (int)GetChoice(frame_color_choice_item);
            g[i].f.linew = (int)GetChoice(frame_linew_choice_item) + 1;
            g[i].f.lines = (int)GetChoice(frame_lines_choice_item) + 1;
            g[i].f.fillbg = XmToggleButtonGetState(frame_fillbg_choice_item) ? ON : OFF;
            g[i].f.bgcolor = (int)GetChoice(frame_bgcolor_choice_item);
        }
    }
    drawgraph();
}
