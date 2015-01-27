/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: locatewin.c,v 1.5 1994/09/29 03:37:37 pturner Exp pturner $
 *
 * Locator Panel
 *
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

#include "globals.h"
#include "motifinc.h"

static Widget locator_frame;
static Widget locator_panel;

/*
 * Panel item declarations
 */
static Widget *locator_onoff_item;
static Widget *delta_item;
static Widget *loc_formatx;
static Widget *loc_formaty;
static Widget *loc_precx;
static Widget *loc_precy;
static Widget locx_item;
static Widget locy_item;
static Widget *fixedp_item;

/*
 * Event and Notify proc declarations
 */

static void locator_define_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void locator_reset_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);

extern int go_locateflag;

// static int locfx = 2, locfy = 2, locpx = 6, locpy = 6;

void update_locator_items(int gno)
{
    if (locator_frame)
    {
        SetChoice(locator_onoff_item, go_locateflag == FALSE);
        SetChoice(fixedp_item, g[gno].pointset == TRUE);
        SetChoice(delta_item, g[gno].pt_type);
        SetChoice(loc_formatx, getFormat_index(g[gno].fx));
        SetChoice(loc_formaty, getFormat_index(g[gno].fy));
        SetChoice(loc_precx, g[gno].px);
        SetChoice(loc_precy, g[gno].py);
        if (g[gno].pointset)
        {
            sprintf(buf, "%lf", g[gno].dsx);
            xv_setstr(locx_item, buf);
            sprintf(buf, "%lf", g[gno].dsy);
            xv_setstr(locy_item, buf);
        }
    }
}

/*
 * Create the locator Panel
 */
void create_locator_frame(Widget, XtPointer, XtPointer)
{
    Widget rc, fr, rc2;
    int x, y;
    Widget buts[3];
    set_wait_cursor();
    if (locator_frame == NULL)
    {
        char *label1[3];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Reset";
        label1[2] = (char *)"Cancel";
        XmGetPos(app_shell, 0, &x, &y);
        locator_frame = XmCreateDialogShell(app_shell, (char *)"Locator props", NULL, 0);
        handle_close(locator_frame);
        XtVaSetValues(locator_frame, XmNx, x, XmNy, y, NULL);
        locator_panel = XmCreateRowColumn(locator_frame, (char *)"ticks_rc", NULL, 0);

        locator_onoff_item = (Widget *)CreatePanelChoice(locator_panel, "Locator:",
                                                         3,
                                                         "ON",
                                                         "OFF",
                                                         NULL,
                                                         NULL);
        delta_item = (Widget *)CreatePanelChoice(locator_panel, "Locator display type:",
                                                 7,
                                                 "[X, Y]",
                                                 "[DX, DY]",
                                                 "[DISTANCE]",
                                                 "[R, Theta]",
                                                 "[VX, VY]",
                                                 "[SX, SY]",
                                                 NULL,
                                                 NULL);
        fixedp_item = CreatePanelChoice(locator_panel, "Fixed point:",
                                        3, "OFF", "ON", NULL,
                                        NULL);

        rc2 = XmCreateRowColumn(locator_panel, (char *)"rc2", NULL, 0);
        /*
         XtVaSetValues(rc2, XmNorientation, XmHORIZONTAL, NULL);
      */
        fr = XmCreateFrame(rc2, (char *)"fr", NULL, 0);
        rc = XmCreateRowColumn(fr, (char *)"rc", NULL, 0);

        loc_formatx = CreatePanelChoice0(rc,
                                         "Format X:",
                                         4, 29,
                                         "Decimal",
                                         "Exponential",
                                         "Power (decimal)",
                                         "General",
                                         "DD-MM-YY",
                                         "MM-DD-YY",
                                         "YY-MM-DD",
                                         "MM-YY",
                                         "MM-DD",
                                         "Month-DD",
                                         "DD-Month",
                                         "Month (abrev.)",
                                         "Month",
                                         "Day of week (abrev.)",
                                         "Day of week",
                                         "Day of year",
                                         "HH:MM:SS.s",
                                         "MM-DD HH:MM:SS.s",
                                         "MM-DD-YY HH:MM:SS.s",
                                         "YY-MM-DD HH:MM:SS.s",
                                         "Degrees (lon)",
                                         "DD MM' (lon)",
                                         "DD MM' SS.s\" (lon)",
                                         "MM' SS.s\" (lon)",
                                         "Degrees (lat)",
                                         "DD MM' (lat)",
                                         "DD MM' SS.s\" (lat)",
                                         "MM' SS.s\" (lat)",
                                         NULL,
                                         NULL);
        loc_precx = CreatePanelChoice(rc, "Precision X:",
                                      12,
                                      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                                      NULL,
                                      NULL);
        locx_item = (Widget)CreateTextItem2(rc, 10, (char *)"Fixed point X:");
        XtManageChild(rc);
        XtManageChild(fr);

        fr = XmCreateFrame(rc2, (char *)"fr", NULL, 0);
        rc = XmCreateRowColumn(fr, (char *)"rc", NULL, 0);
        loc_formaty = CreatePanelChoice0(rc,
                                         "Format Y:",
                                         4, 29,
                                         "Decimal",
                                         "Exponential",
                                         "Power (decimal)",
                                         "General",
                                         "DD-MM-YY",
                                         "MM-DD-YY",
                                         "YY-MM-DD",
                                         "MM-YY",
                                         "MM-DD",
                                         "Month-DD",
                                         "DD-Month",
                                         "Month (abrev.)",
                                         "Month",
                                         "Day of week (abrev.)",
                                         "Day of week",
                                         "Day of year",
                                         "HH:MM:SS.s",
                                         "MM-DD HH:MM:SS.s",
                                         "MM-DD-YY HH:MM:SS.s",
                                         "YY-MM-DD HH:MM:SS.s",
                                         "Degrees (lon)",
                                         "DD MM' (lon)",
                                         "DD MM' SS.s\" (lon)",
                                         "MM' SS.s\" (lon)",
                                         "Degrees (lat)",
                                         "DD MM' (lat)",
                                         "DD MM' SS.s\" (lat)",
                                         "MM' SS.s\" (lat)",
                                         NULL,
                                         NULL);

        loc_precy = CreatePanelChoice(rc, "Precision Y:",
                                      12,
                                      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                                      NULL,
                                      NULL);
        locy_item = (Widget)CreateTextItem2(rc, 10, (char *)"Fixed point Y:");
        XtManageChild(rc);
        XtManageChild(fr);
        XtManageChild(rc2);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, locator_panel, NULL);

        CreateCommandButtons(locator_panel, 3, buts, label1);
        XtAddCallback(buts[0], XmNactivateCallback,
                      (XtCallbackProc)locator_define_notify_proc, (XtPointer)0);
        XtAddCallback(buts[1], XmNactivateCallback,
                      (XtCallbackProc)locator_reset_notify_proc, (XtPointer)0);
        XtAddCallback(buts[2], XmNactivateCallback,
                      (XtCallbackProc)destroy_dialog, (XtPointer)locator_frame);

        XtManageChild(locator_panel);
    }
    XtRaise(locator_frame);
    update_locator_items(cg);
    unset_wait_cursor();
} /* end create_locator_panel */

/*
 * Notify and event procs
 */

static void locator_define_notify_proc(Widget, XtPointer, XtPointer)
{
    int type;

    go_locateflag = (int)GetChoice(locator_onoff_item) == 0;
    type = g[cg].pt_type = (int)GetChoice(delta_item);
    /*locfx =*/g[cg].fx = format_types[(int)GetChoice(loc_formatx)];
    /*locfy =*/g[cg].fy = format_types[(int)GetChoice(loc_formaty)];
    /*locpx =*/g[cg].px = (int)GetChoice(loc_precx);
    /*locpy =*/g[cg].py = (int)GetChoice(loc_precy);
    g[cg].pointset = (int)GetChoice(fixedp_item);
    if (g[cg].pointset)
    {
        strcpy(buf, (char *)xv_getstr(locx_item));
        if (buf[0])
        {
            g[cg].dsx = atof(buf);
        }
        strcpy(buf, (char *)xv_getstr(locy_item));
        if (buf[0])
        {
            g[cg].dsy = atof(buf);
        }
    }
    make_format(cg);
    XtUnmanageChild(locator_frame);
}

static void locator_reset_notify_proc(Widget, XtPointer, XtPointer)
{
    g[cg].dsx = g[cg].dsy = 0.0; /* locator props */
    g[cg].pointset = FALSE;
    g[cg].pt_type = 0;
    g[cg].fx = GENERAL;
    g[cg].fy = GENERAL;
    g[cg].px = 6;
    g[cg].py = 6;
    update_locator_items(cg);
}
