/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: worldwin.c,v 1.7 1994/11/02 04:40:52 pturner Exp pturner $
 *
 * Contents:
 *     world scale popup
 *     viewport popup
 *     arrange graphs popup
 *     overlay graphs popup
 *     autoscaling popup
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

#include "globals.h"
#include "motifinc.h"

static Widget world_frame;
static Widget world_panel;

static Widget view_frame;
static Widget view_panel;

static Widget arrange_frame;
static Widget arrange_panel;

static Widget overlay_frame;
static Widget overlay_panel;

/*
 * Panel item declarations
 */
static Widget define_world_xg1;
static Widget define_world_xg2;
static Widget define_world_yg1;
static Widget define_world_yg2;
static Widget define_world_xminor;
static Widget define_world_yminor;
static Widget define_world_ymajor;
static Widget define_world_xmajor;
static Widget *world_applyto_item;

static Widget define_view_xv1;
static Widget define_view_xv2;
static Widget define_view_yv1;
static Widget define_view_yv2;
static Widget *view_applyto_item;

static Widget *arrange_rows_item;
static Widget *arrange_cols_item;
static Widget arrange_vgap_item;
static Widget arrange_hgap_item;
static Widget arrange_startx_item;
static Widget arrange_starty_item;
static Widget arrange_widthx_item;
static Widget arrange_widthy_item;
static Widget *arrange_packed_item;
// static Widget arrange_applyto_item;

static Widget *graph_overlay1_choice_item;
static Widget *graph_overlay2_choice_item;
static Widget *graph_overlaytype_item;

static Widget but1[2];

int maxmajorticks = 500;
int maxminorticks = 1000;

static void define_world_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void define_view_proc(Widget w, XtPointer client_data, XtPointer call_data);
//static void define_viewm_proc(Widget w, XtPointer client_data, XtPointer call_data);
//static void snap_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void define_arrange_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void define_overlay_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void define_autos_proc(Widget w, XtPointer client_data, XtPointer call_data);

/*
 * update the items in define world/view popup
 */
void update_world(int gno)
{
    if (world_frame)
    {
        sprintf(buf, "%.9lg", g[gno].w.xg1);
        xv_setstr(define_world_xg1, buf);
        sprintf(buf, "%.9lg", g[gno].w.xg2);
        xv_setstr(define_world_xg2, buf);
        sprintf(buf, "%.9lg", g[gno].w.yg1);
        xv_setstr(define_world_yg1, buf);
        sprintf(buf, "%.9lg", g[gno].w.yg2);
        xv_setstr(define_world_yg2, buf);
        sprintf(buf, "%.9lg", g[gno].t[0].tmajor);
        xv_setstr(define_world_xmajor, buf);
        sprintf(buf, "%.9lg", g[gno].t[0].tminor);
        xv_setstr(define_world_xminor, buf);
        sprintf(buf, "%.9lg", g[gno].t[1].tmajor);
        xv_setstr(define_world_ymajor, buf);
        sprintf(buf, "%.9lg", g[gno].t[1].tminor);
        xv_setstr(define_world_yminor, buf);
    }
}

static void define_world_proc(Widget, XtPointer, XtPointer)
{
    char val[80];
    int i, which, ming, maxg, errpos;
    double x, y, a, b, c, d;
    double tx1, tx2, ty1, ty2;
    double tm1, tm2;
    extern double result;
    /* result passed from the expression
    * interpreter */
    which = (int)GetChoice(world_applyto_item);
    if (which)
    {
        ming = 0;
        maxg = maxgraph - 1;
    }
    else
    {
        ming = cg;
        maxg = cg;
    }
    if (ming == cg && maxg == cg)
    {
        if (!isactive_graph(cg))
        {
            errwin("Current graph is not active!");
            return;
        }
    }
    for (i = ming; i <= maxg; i++)
    {
        if (isactive_graph(i))
        {
            /* TODO, I doubt the scanner recognizes these anymore */
            x = (g[i].w.xg2 - g[i].w.xg1); /* DX */
            y = (g[i].w.yg2 - g[i].w.yg1); /* DY */
            a = g[i].w.xg1; /* XMIN */
            b = g[i].w.yg1; /* XMAX */
            c = g[i].w.xg2; /* YMIN */
            d = g[i].w.yg2; /* YMAX */
            if (which <= 1 || which == 2)
            {
                strcpy(val, (char *)xv_getstr(define_world_xg1));
                fixupstr(val);
                scanner(val, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &errpos);
                if (errpos)
                {
                    return;
                }
                tx1 = result;

                strcpy(val, (char *)xv_getstr(define_world_xg2));
                fixupstr(val);
                scanner(val, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &errpos);
                if (errpos)
                {
                    return;
                }
                tx2 = result;

                strcpy(val, (char *)xv_getstr(define_world_xmajor));
                fixupstr(val);
                scanner(val, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &errpos);
                if (errpos)
                {
                    return;
                }
                tm1 = result;

                strcpy(val, (char *)xv_getstr(define_world_xminor));
                fixupstr(val);
                scanner(val, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &errpos);
                if (errpos)
                {
                    return;
                }
                tm2 = result;
                if (tx1 < tx2)
                {
                    g[i].w.xg1 = tx1;
                    g[i].w.xg2 = tx2;
                }
                else
                {
                    errwin("World scaling along X improperly set, scale not changed");
                    return;
                }
                if (tm1 >= 0.0)
                {
                    if (check_nticks(i, X_AXIS, g[i].w.xg1, g[i].w.xg2,
                                     tm1, maxmajorticks))
                    {
                        g[i].t[X_AXIS].tmajor = tm1;
                    }
                    else
                    {
                        sprintf(buf, "Too many major ticks along X-axis, %d",
                                maxmajorticks);
                        errwin(buf);
                        return;
                    }
                    g[i].t[0].tmajor = tm1;
                }
                else
                {
                    errwin("Major ticks must be > 0.0");
                    return;
                }
                if (tm2 >= 0.0)
                {
                    if (check_nticks(i, X_AXIS, g[i].w.xg1, g[i].w.xg2,
                                     tm2, maxminorticks))
                    {
                        g[i].t[X_AXIS].tminor = tm2;
                    }
                    else
                    {
                        sprintf(buf, "Too many minor ticks along X-axis, %d",
                                maxminorticks);
                        errwin(buf);
                        return;
                    }
                    g[i].t[0].tminor = tm2;
                }
                else
                {
                    errwin("Minor ticks must be > 0.0");
                    return;
                }
            }
            if (which <= 1 || which == 3)
            {

                strcpy(val, (char *)xv_getstr(define_world_yg1));
                fixupstr(val);
                scanner(val, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &errpos);
                if (errpos)
                {
                    return;
                }
                ty1 = result;
                strcpy(val, (char *)xv_getstr(define_world_yg2));
                fixupstr(val);
                scanner(val, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &errpos);
                if (errpos)
                {
                    return;
                }
                ty2 = result;

                strcpy(val, (char *)xv_getstr(define_world_ymajor));
                fixupstr(val);
                scanner(val, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &errpos);
                if (errpos)
                {
                    return;
                }
                tm1 = result;

                strcpy(val, (char *)xv_getstr(define_world_yminor));
                fixupstr(val);
                scanner(val, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &errpos);
                if (errpos)
                {
                    return;
                }
                tm2 = result;

                if (ty1 < ty2)
                {
                    g[i].w.yg1 = ty1;
                    g[i].w.yg2 = ty2;
                }
                else
                {
                    errwin("World scaling along Y improperly set, scale not changed");
                    return;
                }
                if (tm1 >= 0.0)
                {
                    if (check_nticks(i, Y_AXIS, g[i].w.yg1, g[i].w.yg2,
                                     tm1, maxmajorticks))
                    {
                        g[i].t[Y_AXIS].tmajor = tm1;
                    }
                    else
                    {
                        sprintf(buf, "Too many minor ticks along Y-axis, %d",
                                maxmajorticks);
                        errwin(buf);
                        return;
                    }
                }
                else
                {
                    errwin("Major ticks must be > 0.0");
                    return;
                }
                if (tm2 >= 0.0)
                {
                    if (check_nticks(i, Y_AXIS, g[i].w.yg1, g[i].w.yg2,
                                     tm2, maxminorticks))
                    {
                        g[i].t[Y_AXIS].tminor = tm2;
                    }
                    else
                    {
                        sprintf(buf, "Too many minor ticks along Y-axis, %d",
                                maxminorticks);
                        errwin(buf);
                        return;
                    }
                }
                else
                {
                    errwin("Minor ticks must be > 0.0");
                    return;
                }
            }
        }
    }
    drawgraph();
}

void update_world_proc(void)
{
    update_world(cg);
}

/*
 * Create the world Frame and the world Panel
 */
void create_world_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget rc, rc2, fr;
    Widget wlabel;
    Widget buts[3];
    set_wait_cursor();
    if (world_frame == NULL)
    {
        char *blabels[3];
        blabels[0] = (char *)"Accept";
        blabels[1] = (char *)"Update";
        blabels[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        world_frame = XmCreateDialogShell(app_shell, (char *)"World", NULL, 0);
        handle_close(world_frame);
        XtVaSetValues(world_frame, XmNx, x, XmNy, y, NULL);
        world_panel = XmCreateRowColumn(world_frame, (char *)"world_rc", NULL, 0);

        rc2 = XmCreateRowColumn(world_panel, (char *)"rc", NULL, 0);
        XtVaSetValues(rc2, XmNorientation, XmHORIZONTAL, NULL);
        fr = XmCreateFrame(rc2, (char *)"fr", NULL, 0);
        rc = XmCreateRowColumn(fr, (char *)"rc", NULL, 0);
        wlabel = XtVaCreateManagedWidget("World (axis scaling)", xmLabelWidgetClass, rc, NULL);
        define_world_xg1 = CreateTextItem2(rc, 10, "Xmin:");
        define_world_xg2 = CreateTextItem2(rc, 10, "Xmax:");
        define_world_yg1 = CreateTextItem2(rc, 10, "Ymin:");
        define_world_yg2 = CreateTextItem2(rc, 10, "Ymax:");
        XtManageChild(rc);
        XtManageChild(fr);

        fr = XmCreateFrame(rc2, (char *)"fr", NULL, 0);
        rc = XmCreateRowColumn(fr, (char *)"rc", NULL, 0);
        wlabel = XtVaCreateManagedWidget("Tick spacing", xmLabelWidgetClass, rc, NULL);
        define_world_xmajor = CreateTextItem2(rc, 10, "X-major:");
        define_world_xminor = CreateTextItem2(rc, 10, "X-minor:");
        define_world_ymajor = CreateTextItem2(rc, 10, "Y-major:");
        define_world_yminor = CreateTextItem2(rc, 10, "Y-minor:");
        XtManageChild(rc);
        XtManageChild(fr);
        XtManageChild(rc2);

        world_applyto_item = (Widget *)CreatePanelChoice(world_panel, "Apply to:",
                                                         5,
                                                         "Current graph",
                                                         "All active graphs",
                                                         "X, all active graphs",
                                                         "Y, all active graphs",
                                                         NULL,
                                                         NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, world_panel, NULL);

        CreateCommandButtons(world_panel, 3, buts, blabels);
        XtAddCallback(buts[0], XmNactivateCallback, (XtCallbackProc)define_world_proc, (XtPointer)NULL);
        XtAddCallback(buts[1], XmNactivateCallback, (XtCallbackProc)update_world_proc, (XtPointer)NULL);
        XtAddCallback(buts[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)world_frame);

        XtManageChild(world_panel);
    }
    XtRaise(world_frame);
    update_world(cg);
    unset_wait_cursor();
} /* end create_world_panel */

/*
 * Viewport popup routines
 */
void update_view(int gno)
{
    if (view_frame)
    {
        sprintf(buf, "%.9lg", g[gno].v.xv1);
        xv_setstr(define_view_xv1, buf);
        sprintf(buf, "%.9lg", g[gno].v.xv2);
        xv_setstr(define_view_xv2, buf);
        sprintf(buf, "%.9lg", g[gno].v.yv1);
        xv_setstr(define_view_yv1, buf);
        sprintf(buf, "%.9lg", g[gno].v.yv2);
        xv_setstr(define_view_yv2, buf);
    }
}

/*
 * For snaps (not used)
 */
double fround(double x, double r)
{
    x = x + r * 0.5;
    x = (long)(x / r);
    return x * r;
}

static void define_view_proc(Widget, XtPointer, XtPointer)
{
    char val[80];
    double tmpx1, tmpx2;
    double tmpy1, tmpy2;
    double r;
    int i, snap = 0, ming, maxg, which;
    which = (int)GetChoice(view_applyto_item);
    if (which)
    {
        ming = 0;
        maxg = maxgraph - 1;
    }
    else
    {
        ming = cg;
        maxg = cg;
    }
    if (ming == cg && maxg == cg)
    {
        if (!isactive_graph(cg))
        {
            errwin("Current graph is not active!");
            return;
        }
    }
    strcpy(val, xv_getstr(define_view_xv1));
    tmpx1 = atof(val);
    strcpy(val, xv_getstr(define_view_xv2));
    tmpx2 = atof(val);
    strcpy(val, xv_getstr(define_view_yv1));
    tmpy1 = atof(val);
    strcpy(val, xv_getstr(define_view_yv2));
    tmpy2 = atof(val);
    snap = 0; /* snap off for now TODO */
    switch (snap)
    {
    case 0: /* no snap */
        break;
    case 1: /* .1 */
        r = 0.1;
        break;
    case 2: /* .05 */
        r = 0.05;
        break;
    case 3: /* .01 */
        r = 0.01;
        break;
    case 4: /* .005 */
        r = 0.005;
        break;
    case 5: /* .001 */
        r = 0.001;
        break;
    }
    if (snap)
    {
        tmpx1 = fround(tmpx1, r);
        tmpx2 = fround(tmpx2, r);
        tmpy1 = fround(tmpy1, r);
        tmpy2 = fround(tmpy2, r);
        update_view(cg);
    }
    for (i = ming; i <= maxg; i++)
    {
        if (isactive_graph(i))
        {
            if (which <= 1 || which == 2)
            {
                g[i].v.xv1 = tmpx1;
                g[i].v.xv2 = tmpx2;
            }
            if (which <= 1 || which == 3)
            {
                g[i].v.yv1 = tmpy1;
                g[i].v.yv2 = tmpy2;
            }
        }
    }
    drawgraph();
}

/*
static void define_viewm_proc(Widget , XtPointer , XtPointer )
{
    set_action(0);
    set_action(VIEW_1ST);
}

static void snap_proc(Widget , XtPointer , XtPointer )
{
}
*/
void create_view_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget buts[3];
    set_wait_cursor();
    if (view_frame == NULL)
    {
        char *blabels[3];
        blabels[0] = (char *)"Accept";
        blabels[1] = (char *)"Pick";
        blabels[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        view_frame = XmCreateDialogShell(app_shell, (char *)"Viewports", NULL, 0);
        handle_close(view_frame);
        XtVaSetValues(view_frame, XmNx, x, XmNy, y, NULL);
        view_panel = XmCreateRowColumn(view_frame, (char *)"view_rc", NULL, 0);

        define_view_xv1 = CreateTextItem2(view_panel, 10, "Xmin:");
        define_view_xv2 = CreateTextItem2(view_panel, 10, "Xmax:");
        define_view_yv1 = CreateTextItem2(view_panel, 10, "Ymin:");
        define_view_yv2 = CreateTextItem2(view_panel, 10, "Ymax:");

        view_applyto_item = CreatePanelChoice(view_panel, "Apply to:",
                                              5,
                                              "Current graph",
                                              "All active graphs",
                                              "X, all active graphs",
                                              "Y, all active graphs",
                                              NULL,
                                              NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, view_panel, NULL);

        CreateCommandButtons(view_panel, 3, buts, blabels);
        XtAddCallback(buts[0], XmNactivateCallback, (XtCallbackProc)define_view_proc, (XtPointer)NULL);
        XtAddCallback(buts[1], XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)VIEW_1ST);
        XtAddCallback(buts[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)view_frame);

        XtManageChild(view_panel);
    }
    XtRaise(view_frame);
    update_view(cg);
    unset_wait_cursor();
}

/*
 * Arrange graphs popup routines
 */
static void define_arrange_proc(Widget, XtPointer, XtPointer)
{
    int nrows, ncols, pack;
    double vgap, hgap, sx, sy, wx, wy;

    nrows = GetChoice(arrange_rows_item) + 1;
    ncols = GetChoice(arrange_cols_item) + 1;
    if (nrows == 1 && ncols == 1)
    {
        errwin("For a single graph use View/Viewport...");
        return;
    }
    pack = GetChoice(arrange_packed_item);
    vgap = atof((char *)xv_getstr(arrange_vgap_item));
    hgap = atof((char *)xv_getstr(arrange_hgap_item));
    sx = atof((char *)xv_getstr(arrange_startx_item));
    sy = atof((char *)xv_getstr(arrange_starty_item));
    wx = atof((char *)xv_getstr(arrange_widthx_item));
    wy = atof((char *)xv_getstr(arrange_widthy_item));
    if (wx <= 0.0)
    {
        errwin("Graph width must be > 0.0");
        return;
    }
    if (wy <= 0.0)
    {
        errwin("Graph height must be > 0.0");
        return;
    }
    define_arrange(nrows, ncols, pack, vgap, hgap, sx, sy, wx, wy);
}

void update_arrange(void)
{
    xv_setstr(arrange_vgap_item, "0.05");
    xv_setstr(arrange_hgap_item, "0.05");
    xv_setstr(arrange_startx_item, "0.1");
    xv_setstr(arrange_starty_item, "0.1");
    xv_setstr(arrange_widthx_item, "0.8");
    xv_setstr(arrange_widthy_item, "0.8");
}

void create_arrange_frame(Widget, XtPointer client_ata, XtPointer)
{
    (void)client_ata;
    int x, y;
    Widget rc;
    set_wait_cursor();
    if (arrange_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        arrange_frame = XmCreateDialogShell(app_shell, (char *)"Arrange graphs", NULL, 0);
        handle_close(arrange_frame);
        XtVaSetValues(arrange_frame, XmNx, x, XmNy, y, NULL);
        arrange_panel = XmCreateRowColumn(arrange_frame, (char *)"arrange_rc", NULL, 0);

        rc = XtVaCreateWidget("rc", xmRowColumnWidgetClass, arrange_panel,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 9, /* nitems / 2 */
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        XtVaCreateManagedWidget("Rows: ", xmLabelWidgetClass, rc, NULL);
        arrange_rows_item = CreatePanelChoice(rc, " ",
                                              11,
                                              "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                                              NULL, NULL);
        XtVaCreateManagedWidget("Columns: ", xmLabelWidgetClass, rc, NULL);
        arrange_cols_item = CreatePanelChoice(rc, " ",
                                              11,
                                              "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                                              NULL, NULL);
        XtVaCreateManagedWidget("Packing: ", xmLabelWidgetClass, rc, NULL);
        arrange_packed_item = CreatePanelChoice(rc, " ",
                                                5,
                                                "None", "Horizontal", "Vertical", "Both",
                                                NULL, NULL);

        arrange_vgap_item = CreateTextItem4(rc, 10, "Vertical gap:");
        arrange_hgap_item = CreateTextItem4(rc, 10, "Horizontal gap:");
        arrange_startx_item = CreateTextItem4(rc, 10, "Start at X =");
        arrange_starty_item = CreateTextItem4(rc, 10, "Start at Y =");
        arrange_widthx_item = CreateTextItem4(rc, 10, "Graph width:");
        arrange_widthy_item = CreateTextItem4(rc, 10, "Graph height:");
        XtManageChild(rc);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, arrange_panel, NULL);

        CreateCommandButtons(arrange_panel, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)define_arrange_proc, (XtPointer)NULL);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)arrange_frame);

        XtManageChild(arrange_panel);
        update_arrange();
    }
    /*	update_arrange(); */
    XtRaise(arrange_frame);
    unset_wait_cursor();
}

/*
 * Overlay graphs popup routines
 */
static void define_overlay_proc(Widget, XtPointer, XtPointer)
{
    int i;
    int g1 = GetChoice(graph_overlay1_choice_item);
    int g2 = GetChoice(graph_overlay2_choice_item);
    int type = GetChoice(graph_overlaytype_item);
    if (g1 == g2)
    {
        errwin("Can't overlay a graph onto itself");
        return;
    }
    if (!isactive_graph(g1))
    {
        set_graph_active(g1);
    }
    if (!isactive_graph(g2))
    {
        set_graph_active(g2);
    }
    /* set identical viewports g2 is the controlling graph */
    g[g1].v = g[g2].v;
    switch (type)
    {
    case 0:
        g[g1].w = g[g2].w;
        for (i = 0; i < MAXAXES; i++)
        {
            /*
                   g[g1].t[i] = g[g2].t[i];
            */
            g[g1].t[i].active = OFF;
            g[g2].t[i].active = ON;
        }
        g[g1].t[1].tl_op = LEFT;
        g[g2].t[1].tl_op = LEFT;
        g[g1].t[1].t_op = BOTH;
        g[g2].t[1].t_op = BOTH;

        g[g1].t[0].tl_op = BOTTOM;
        g[g2].t[0].tl_op = BOTTOM;
        g[g1].t[0].t_op = BOTH;
        g[g2].t[0].t_op = BOTH;
        break;
    case 1:
        g[g1].w.xg1 = g[g2].w.xg1;
        g[g1].w.xg2 = g[g2].w.xg2;
        for (i = 0; i < MAXAXES; i++)
        {
            if (i % 2 == 0)
            {
                g[g1].t[i].active = OFF;
            }
            else
            {
                g[g1].t[i].active = ON;
            }
        }
        g[g2].t[1].tl_op = LEFT;
        g[g1].t[1].tl_op = RIGHT;
        g[g2].t[1].t_op = LEFT;
        g[g1].t[1].t_op = RIGHT;

        g[g2].t[0].tl_op = BOTTOM;
        g[g1].t[0].tl_op = BOTTOM;
        g[g2].t[0].t_op = BOTH;
        g[g1].t[0].t_op = BOTH;

        break;
    case 2:
        g[g1].w.yg1 = g[g2].w.yg1;
        g[g1].w.yg2 = g[g2].w.yg2;
        for (i = 0; i < MAXAXES; i++)
        {
            if (i % 2 == 1)
            {
                g[g1].t[i].active = OFF;
            }
            else
            {
                g[g1].t[i].active = ON;
            }
            /*
                   g[g1].t[i] = g[g2].t[i];
            */
        }
        g[g2].t[0].tl_op = BOTTOM;
        g[g1].t[0].tl_op = TOP;
        g[g2].t[0].t_op = BOTTOM;
        g[g1].t[0].t_op = TOP;

        g[g2].t[1].tl_op = LEFT;
        g[g1].t[1].tl_op = LEFT;
        g[g2].t[1].t_op = BOTH;
        g[g1].t[1].t_op = BOTH;
        break;
    case 3:
        for (i = 0; i < MAXAXES; i++)
        {
            g[g1].t[i].active = ON;
            g[g2].t[i].active = ON;
        }
        g[g2].t[1].tl_op = LEFT;
        g[g1].t[1].tl_op = RIGHT;
        g[g2].t[0].tl_op = BOTTOM;
        g[g1].t[0].tl_op = TOP;
        g[g2].t[1].t_op = LEFT;
        g[g1].t[1].t_op = RIGHT;
        g[g2].t[0].t_op = BOTTOM;
        g[g1].t[0].t_op = TOP;
        break;
    }
    update_all(cg);
    drawgraph();
}

void create_overlay_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    set_wait_cursor();
    if (overlay_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        overlay_frame = XmCreateDialogShell(app_shell, (char *)"Overlay graphs", NULL, 0);
        handle_close(overlay_frame);
        XtVaSetValues(overlay_frame, XmNx, x, XmNy, y, NULL);
        overlay_panel = XmCreateRowColumn(overlay_frame, (char *)"overlay_rc", NULL, 0);
        graph_overlay1_choice_item = CreateGraphChoice(overlay_panel, "Overlay graph: ", maxgraph, 0);
        graph_overlay2_choice_item = CreateGraphChoice(overlay_panel, "Onto graph: ", maxgraph, 0);
        graph_overlaytype_item = CreatePanelChoice(overlay_panel, "Overlay type:",
                                                   5,
                                                   "Same axes scaling along X and Y",
                                                   "X-axes same, Y-axes different:",
                                                   "Y-axes same, X-axes different:",
                                                   "X and Y axes different:",
                                                   NULL, NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, overlay_panel, NULL);

        CreateCommandButtons(overlay_panel, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)define_overlay_proc, (XtPointer)NULL);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)overlay_frame);

        XtManageChild(overlay_panel);
    }
    /*	update_overlay(); */
    XtRaise(overlay_frame);
    unset_wait_cursor();
}

/*
 * autoscale popup
 */
typedef struct _Auto_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *on_item;
    Widget *method_item;
    Widget *nticksx_item;
    Widget *nticksy_item;
    Widget *applyto_item;
} Auto_ui;

static Auto_ui aui;

static void define_autos_proc(Widget, XtPointer client_data, XtPointer)
{
    int aon, ameth, antx, anty, au, ap;
    Auto_ui *ui = (Auto_ui *)client_data;
    aon = GetChoice(ui->on_item) - 4;
    au = GetSelectedSet(ui->sel);
    if (au == SET_SELECT_ALL)
    {
        au = -1;
    }
    ap = GetChoice(ui->applyto_item);
    ameth = GetChoice(ui->method_item);
    antx = GetChoice(ui->nticksx_item);
    anty = GetChoice(ui->nticksy_item);
    define_autos(aon, au, ap, ameth, antx, anty);
}

void update_autos(int gno)
{
    if (aui.top)
    {
        SetChoice(aui.method_item, g[gno].auto_type == SPEC);
        SetChoice(aui.nticksx_item, g[gno].t[0].t_num - 2);
        SetChoice(aui.nticksy_item, g[gno].t[1].t_num - 2);
    }
}

void create_autos_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget panel, rc;

    set_wait_cursor();
    if (aui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        aui.top = XmCreateDialogShell(app_shell, (char *)"Autoscale graphs", NULL, 0);
        handle_close(aui.top);
        XtVaSetValues(aui.top, XmNx, x, XmNy, y, NULL);
        panel = XmCreateRowColumn(aui.top, (char *)"autos_rc", NULL, 0);

        rc = XtVaCreateWidget("rc", xmRowColumnWidgetClass, panel,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 5,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);
        XtVaCreateManagedWidget("Autoscale axis:", xmLabelWidgetClass, rc, NULL);
        aui.on_item = CreatePanelChoice(rc, " ",
                                        11,
                                        "None", "All", "All X-axes", "All Y-axes",
                                        "X-axis", "Y-axis", "Zero X-axis", "Zero Y-axis",
                                        "Alternate X-axis", "Alternate Y-axis",
                                        NULL,
                                        NULL);

        XtVaCreateManagedWidget("Autoscale type:", xmLabelWidgetClass, rc, NULL);
        aui.method_item = CreatePanelChoice(rc, " ",
                                            3,
                                            "Heckbert",
                                            "Fixed",
                                            NULL,
                                            NULL);
        XtVaCreateManagedWidget("Number of ticks in X:", xmLabelWidgetClass, rc, NULL);
        aui.nticksx_item = CreatePanelChoice(rc, " ",
                                             12,
                                             "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                                             NULL,
                                             NULL);
        XtVaCreateManagedWidget("Number of ticks in Y:", xmLabelWidgetClass, rc, NULL);
        aui.nticksy_item = CreatePanelChoice(rc, " ",
                                             12,
                                             "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
                                             NULL,
                                             NULL);
        XtVaCreateManagedWidget("Apply to:", xmLabelWidgetClass, rc, NULL);
        aui.applyto_item = CreatePanelChoice(rc, " ",
                                             3,
                                             "Current graph",
                                             "All active graphs",
                                             NULL,
                                             NULL);
        XtManageChild(rc);

        aui.sel = CreateSetSelector(panel, "Use set:",
                                    SET_SELECT_ALL,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, panel, NULL);

        CreateCommandButtons(panel, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)define_autos_proc, (XtPointer)&aui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)aui.top);

        XtManageChild(panel);
    }
    XtRaise(aui.top);
    update_autos(cg);
    unset_wait_cursor();
}
