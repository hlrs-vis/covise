/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: ptswin.c,v 1.10 1994/11/02 04:40:52 pturner Exp pturner $
 *
 * edit points, clip points to window, etc.
 *
 */

/* TODO:
   allow 'restrict to' option to operate only on a selected set
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <Xm/Xm.h>
#include <Xm/BulletinB.h>
#include <Xm/DialogS.h>
#include <Xm/Frame.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/Separator.h>
#include <Xm/Text.h>
#include <Xm/ToggleB.h>

#include "globals.h"
#include "motifinc.h"

static Widget but1[4];

extern int add_setno;
extern int add_at;
extern int move_dir;

int digitflag = 0; /* TODO debug digitize routine and remove */

int paint_skip = 0;

int track_set = -1;
int track_point = -1;

static Widget points_frame;
static Widget points_panel;

Widget locate_point_item;
static Widget locate_set_item;
static Widget locate_point_message;
XmString label_string;

static Widget paint_item;

// static Widget goto_which_item;
static Widget goto_pointx_item;
static Widget goto_pointy_item;
static SetChoiceItem goto_set_item;
static Widget goto_index_item;

static SetChoiceItem addinset_item;
static Widget *addat_item;

static void do_find_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_ptsmove_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_del_proc(Widget w, XtPointer client_data, XtPointer call_data);
// static void do_extract_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_track_proc(Widget w, XtPointer client_data, XtPointer call_data);
void create_points_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_goto_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_digit_frame(Widget w, XtPointer client_data, XtPointer call_data);

void SetLabel(Widget w, char *buf)
{
    Arg al;
    XmStringFree(label_string);
    label_string = XmStringCreateLtoR(buf, charset);
    XtSetArg(al, XmNlabelString, label_string);
    XtSetValues(w, &al, 1);
}

/*
 * set tracker
 */
static void do_track_proc(Widget, XtPointer, XtPointer)
{
    int setno;
    if (activeset(cg))
    {
        set_action(0);
        set_action(TRACKER);
        strcpy(buf, xv_getstr(locate_set_item));
        track_point = -1;
        if (strlen(buf) >= 1)
        {
            setno = atoi(buf);
            if (setno >= 0 && setno < g[cg].maxplot && isactive(cg, setno))
            {
                track_set = setno;
            }
            else
            {
                errwin("Set number out of range or not active");
                track_set = -1;
            }
        }
        else
        {
            track_set = -1;
        }
        create_points_frame((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
        SetLabel(locate_point_message, (char *)"Tracking -  Set, location, (X, Y):");
    }
    else
    {
        errwin("No active sets to track");
    }
}

/*
 * activate the add point item in the canvas event proc
 */
static void do_add_proc(Widget, XtPointer, XtPointer)
{
    int paint = XmToggleButtonGetState(paint_item);
    if (paint)
    {
        set_action(0);
        set_action(PAINT_POINTS);
    }
    else
    {
        set_action(0);
        set_action(ADD_POINT);
    }
    add_setno = GetSelectedSet(addinset_item);
    if (add_setno == SET_SELECT_NEXT)
    {
        if ((add_setno = nextset(cg)) != -1)
        {
            activateset(cg, add_setno);
        }
        else
        {
            set_action(0);
            return;
        }
    }
    add_at = (int)GetChoice(addat_item);
    create_points_frame((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
    SetLabel(locate_point_message, (char *)"Adding points to set, location, (X, Y):");
}

/*
 * activate the find point item in the canvas event proc
 */
static void do_find_proc(Widget, XtPointer, XtPointer)
{
    set_action(0);
    set_action(FIND_POINT);
    create_points_frame((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
    SetLabel(locate_point_message, (char *)"Set, location, (X, Y):");
}

/*
 * activate the delete point item in the canvas event proc
 */
static void do_del_proc(Widget, XtPointer, XtPointer)
{
    set_action(0);
    set_action(DEL_POINT);
    create_points_frame((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
    SetLabel(locate_point_message, (char *)"Delete points - set, location, (X, Y):");
}

/*
 * move a point
 */
static void do_ptsmove_proc(Widget, XtPointer, XtPointer)
{
    set_action(0);
    set_action(MOVE_POINT1ST);
    move_dir = 0;
    create_points_frame((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
    SetLabel(locate_point_message, (char *)"Move points - set, location, (X, Y):");
}

static void do_movey_proc(Widget, XtPointer, XtPointer)
{
    set_action(0);
    set_action(MOVE_POINT1ST);
    move_dir = 2;
    create_points_frame((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
    SetLabel(locate_point_message, (char *)"Move points along y - set, location, (X, Y):");
}

static void do_movex_proc(Widget, XtPointer, XtPointer)
{
    set_action(0);
    set_action(MOVE_POINT1ST);
    move_dir = 1;
    create_points_frame((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
    SetLabel(locate_point_message, (char *)"Move points along x - set, location, (X, Y):");
}

static void do_gotoxy_proc(Widget, XtPointer, XtPointer)
{
    double wx, wy;
    int sx, sy;

    wx = atof((char *)xv_getstr(goto_pointx_item));
    wy = atof((char *)xv_getstr(goto_pointy_item));
    world2deviceabs(wx, wy, &sx, &sy);
    setpointer(sx, sy);
    getpoints(sx, sy);
}

static void do_gotopt_proc(Widget, XtPointer, XtPointer)
{
    int setno, ind;
    double wx, wy;
    int sx, sy;
    double *x, *y;

    setno = GetSelectedSet(goto_set_item);
    ind = atoi((char *)xv_getstr(goto_index_item));
    if (isactive(cg, setno))
    {
        if (ind <= getsetlength(cg, setno) && ind > 0)
        {
            x = getx(cg, setno);
            y = gety(cg, setno);
            wx = x[ind - 1];
            wy = y[ind - 1];
            world2deviceabs(wx, wy, &sx, &sy);
            setpointer(sx, sy);
            getpoints(sx, sy);
        }
        else
        {
            errwin("Point index out of range");
        }
    }
}

static void do_dist_proc(Widget, XtPointer, XtPointer)
{
    set_action(0);
    set_action(DISLINE1ST);
    create_points_frame((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
    SetLabel(locate_point_message, (char *)"Distance, dx, dy, angle in degrees between 2 points:");
}

void define_points_popup(Widget, XtPointer, XtPointer)
{
    Widget wbut;
    int x, y;
    set_wait_cursor();
    if (points_frame == NULL)
    {
        XmGetPos(app_shell, 0, &x, &y);
        points_frame = XmCreateDialogShell(app_shell, (char *)"Point ops", NULL, 0);
        handle_close(points_frame);
        XtVaSetValues(points_frame, XmNx, x, XmNy, y, NULL);
        points_panel = XmCreateRowColumn(points_frame, (char *)"points_rc", NULL, 0);

        wbut = XtVaCreateManagedWidget("Find", xmPushButtonWidgetClass, points_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_find_proc, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Track", xmPushButtonWidgetClass, points_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_track_proc, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Delete", xmPushButtonWidgetClass, points_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_del_proc, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Add...", xmPushButtonWidgetClass, points_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_add_frame, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Move", xmPushButtonWidgetClass, points_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_ptsmove_proc, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Move X", xmPushButtonWidgetClass, points_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_movex_proc, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Move Y", xmPushButtonWidgetClass, points_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_movey_proc, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Distance, dx, dy, angle", xmPushButtonWidgetClass, points_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_dist_proc, (XtPointer)1);

        wbut = XtVaCreateManagedWidget("Goto...", xmPushButtonWidgetClass, points_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_goto_frame, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Digitize...", xmPushButtonWidgetClass, points_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_digit_frame, (XtPointer)0);
        wbut = XtVaCreateManagedWidget("Close", xmPushButtonWidgetClass, points_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)points_frame);

        XtManageChild(points_panel);
    }
    XtRaise(points_frame);
    unset_wait_cursor();
}

void create_goto_frame(Widget, XtPointer, XtPointer)
{
    static Widget top;
    int x, y;
    Widget rc, fr, dialog, buts[3];
    set_wait_cursor();
    if (top == NULL)
    {
        char *label1[3];
        label1[0] = (char *)"Goto X, Y";
        label1[1] = (char *)"Goto point, set";
        label1[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        top = XmCreateDialogShell(app_shell, (char *)"Goto", NULL, 0);
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(top, (char *)"dialog_rc", NULL, 0);

        fr = XmCreateFrame(dialog, (char *)"fr", NULL, 0);
        rc = XmCreateRowColumn(fr, (char *)"rc", NULL, 0);
        goto_pointx_item = CreateTextItem2(rc, 10, (char *)"X: ");
        goto_pointy_item = CreateTextItem2(rc, 10, (char *)"Y: ");
        XtManageChild(rc);
        XtManageChild(fr);

        fr = XmCreateFrame(dialog, (char *)"fr", NULL, 0);
        rc = XmCreateRowColumn(fr, (char *)"rc", NULL, 0);
        goto_index_item = CreateTextItem2(rc, 10, (char *)"Goto point: ");
        goto_set_item = CreateSetSelector(dialog, (char *)"In set:",
                                          SET_SELECT_ACTIVE,
                                          FILTER_SELECT_NONE,
                                          GRAPH_SELECT_CURRENT,
                                          SELECTION_TYPE_MULTIPLE);

        XtManageChild(rc);
        XtManageChild(fr);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 3, buts, label1);
        XtAddCallback(buts[0], XmNactivateCallback, (XtCallbackProc)do_gotoxy_proc, (XtPointer)NULL);
        XtAddCallback(buts[1], XmNactivateCallback, (XtCallbackProc)do_gotopt_proc, (XtPointer)NULL);
        XtAddCallback(buts[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)top);

        XtManageChild(dialog);
    }
    XtRaise(top);
    unset_wait_cursor();
}

void create_add_frame(Widget, XtPointer, XtPointer)
{
    static Widget top, dialog;
    int x, y;

    set_wait_cursor();
    if (top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        top = XmCreateDialogShell(app_shell, (char *)"Add points", NULL, 0);
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(top, (char *)"dialog_rc", NULL, 0);

        addinset_item = CreateSetSelector(dialog, (char *)"Add to set:",
                                          SET_SELECT_NEXT,
                                          FILTER_SELECT_NONE,
                                          GRAPH_SELECT_CURRENT,
                                          SELECTION_TYPE_SINGLE);
        addat_item = CreatePanelChoice(dialog,
                                       "Add to:", 5,
                                       "End of set",
                                       "Beginning of set",
                                       "After nearest point in set",
                                       "Before nearest point in set",
                                       NULL, 0);
        paint_item = XmCreateToggleButton(dialog, (char *)"Paint points", NULL, 0);
        XtManageChild(paint_item);
        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_add_proc, (XtPointer)NULL);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)top);

        XtManageChild(dialog);
    }
    XtRaise(top);
    unset_wait_cursor();
}

static void points_done_proc(Widget, XtPointer client_data, XtPointer)
{
    set_action(0);
    XtUnmanageChild((Widget)client_data);
}

void create_points_frame(Widget, XtPointer, XtPointer)
{
    static Widget top, dialog;
    int x, y;
    Widget wbut, rc;

    set_wait_cursor();
    if (top == NULL)
    {
        //char *label0[1];
        //label0[0] = "Close";
        XmGetPos(app_shell, 0, &x, &y);
        top = XmCreateDialogShell(app_shell, (char *)"Points", NULL, 0);
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(top, (char *)"dialog_rc", NULL, 0);

        rc = XmCreateRowColumn(dialog, (char *)"rc", NULL, 0);
        /*
          XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
      */

        label_string = XmStringCreateLtoR((char *)"Set, location, (X, Y): ", charset);
        locate_point_message = XtVaCreateManagedWidget("pointslabel", xmLabelWidgetClass, rc,
                                                       XmNlabelString, label_string,
                                                       NULL);

        locate_point_item = XtVaCreateManagedWidget("locator", xmTextWidgetClass, rc,
                                                    XmNcolumns, 60,
                                                    NULL);
        XtManageChild(rc);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);
        rc = XmCreateRowColumn(dialog, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
        XtVaCreateManagedWidget("Use set (blank for for nearest):", xmLabelWidgetClass, rc,
                                NULL);

        locate_set_item = XtVaCreateManagedWidget("setno", xmTextWidgetClass, rc,
                                                  XmNcolumns, 3,
                                                  NULL);

        XtManageChild(rc);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);
        rc = XmCreateRowColumn(dialog, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);

        wbut = XtVaCreateManagedWidget("Find", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_find_proc, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Track", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_track_proc, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Delete", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_del_proc, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Add...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_add_frame, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Move", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_ptsmove_proc, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Move X", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_movex_proc, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Move Y", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_movey_proc, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("D, dx, dy, angle", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_dist_proc, (XtPointer)1);

        wbut = XtVaCreateManagedWidget("Goto...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_goto_frame, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Digitize...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_digit_frame, (XtPointer)0);
        wbut = XtVaCreateManagedWidget("Done", xmPushButtonWidgetClass, rc,
                                       (XtPointer)NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)points_done_proc, (XtPointer)top);
        XtManageChild(rc);

        /*
          CreateCommandButtons(dialog, 1, but1, label0);
          XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc) destroy_dialog, (XtPointer) top);
      */

        XtManageChild(dialog);
    }
    XtRaise(top);
    unset_wait_cursor();
}

/*
 * Digitize points from an image
 */

static SetChoiceItem digit_set_item;
static Widget digit_p1x_item;
static Widget digit_p2x_item;
static Widget digit_p3x_item;
static Widget digit_p1y_item;
static Widget digit_p2y_item;
static Widget digit_p3y_item;

int digit_setno;

static void do_digit_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_digitpick_proc(Widget w, XtPointer client_data, XtPointer call_data);

void create_digit_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    static Widget top;
    Widget dialog;

    set_wait_cursor();
    if (top == NULL)
    {
        char *label1[4];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Pick";
        label1[2] = (char *)"Paint";
        label1[3] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        top = XmCreateDialogShell(app_shell, (char *)"Digitize points", NULL, 0);
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(top, (char *)"dialog_rc", NULL, 0);

        digit_set_item = CreateSetSelector(dialog, (char *)"Add points to set:",
                                           SET_SELECT_NEXT,
                                           FILTER_SELECT_NONE,
                                           GRAPH_SELECT_CURRENT,
                                           SELECTION_TYPE_SINGLE);

        digit_p1x_item = CreateTextItem2(dialog, 10, (char *)"P1 X:");
        digit_p2x_item = CreateTextItem2(dialog, 10, (char *)"P2 X:");
        digit_p3x_item = CreateTextItem2(dialog, 10, (char *)"P3 X:");
        digit_p1y_item = CreateTextItem2(dialog, 10, (char *)"P1 Y:");
        digit_p2y_item = CreateTextItem2(dialog, 10, (char *)"P2 Y:");
        digit_p3y_item = CreateTextItem2(dialog, 10, (char *)"P3 Y:");

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 4, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_digit_proc, (XtPointer)top);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)do_digitpick_proc, (XtPointer)top);
        XtAddCallback(but1[2], XmNactivateCallback, (XtCallbackProc)do_digit_proc, (XtPointer)top);
        XtAddCallback(but1[3], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)top);

        XtManageChild(dialog);
        xv_setstr(digit_p1x_item, (char *)"0.0");
        xv_setstr(digit_p2x_item, (char *)"0.0");
        xv_setstr(digit_p3x_item, (char *)"0.0");
        xv_setstr(digit_p1y_item, (char *)"0.0");
        xv_setstr(digit_p2y_item, (char *)"0.0");
        xv_setstr(digit_p3y_item, (char *)"0.0");
    }
    XtRaise(top);
    unset_wait_cursor();
}

/*
 * compute digit
 */
static void do_digit_proc(Widget, XtPointer, XtPointer)
{
    int i, setno;
    double angle, sx, sy, p1x, p2x, p3x, p1y, p2y, p3y, xtmp, ytmp, *x, *y;
    double sxd, syd, dxd, dyd;
    double xd1, yd1;
    double xd2, yd2;
    double xd3, yd3;
    char buf[256];

    setno = digit_setno;

    if (getsetlength(cg, setno) < 4)
    {
        errwin("Insufficient number of points to digitize");
        return;
    }
    x = getx(cg, setno);
    y = gety(cg, setno);
    strcpy(buf, (char *)xv_getstr(digit_p1x_item));
    p1x = atof(buf);
    strcpy(buf, (char *)xv_getstr(digit_p2x_item));
    p2x = atof(buf);
    strcpy(buf, (char *)xv_getstr(digit_p3x_item));
    p3x = atof(buf);
    strcpy(buf, (char *)xv_getstr(digit_p1y_item));
    p1y = atof(buf);
    strcpy(buf, (char *)xv_getstr(digit_p2y_item));
    p2y = atof(buf);
    strcpy(buf, (char *)xv_getstr(digit_p3y_item));
    p3y = atof(buf);

    if (p1x == p2x && p2x == p3x)
    {
        errwin("Control points incorrectly specified");
        return;
    }
    if (p1y == p2y && p2y == p3y)
    {
        errwin("Control points incorrectly specified");
        return;
    }
    set_wait_cursor();

    xd1 = x[0];
    xd2 = x[1];
    xd3 = x[2];
    yd1 = y[0];
    yd2 = y[1];
    yd3 = y[2];
    angle = -atan2(yd2 - yd1, xd2 - xd1);

    for (i = 0; i < getsetlength(cg, setno); i++)
    {
        xtmp = x[i] - xd1;
        ytmp = y[i] - yd1;
        x[i] = cos(angle) * xtmp - sin(angle) * ytmp;
        y[i] = sin(angle) * xtmp + cos(angle) * ytmp;
        sxd = my_hypot(xd2 - xd1, yd2 - yd1);
        syd = my_hypot(xd3 - xd2, yd3 - yd2);
        dxd = my_hypot(p2x - p1x, p2y - p1y);
        dyd = my_hypot(p3x - p2x, p3y - p2y);
        sx = dxd / sxd;
        sy = dyd / syd;
        x[i] = p1x + x[i] * sx;
        y[i] = p1y + y[i] * sy;
    }
    updatesetminmax(cg, setno);
    update_set_status(cg, setno);
    unset_wait_cursor();
    drawgraph();
}

static void do_digitpick_proc(Widget, XtPointer, XtPointer)
{
    digit_setno = GetSelectedSet(digit_set_item);
    set_action(0);
    set_action(ADD_POINT1ST);
    create_points_frame((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
    SetLabel(locate_point_message, (char *)"Set, location, (X, Y):");
}
