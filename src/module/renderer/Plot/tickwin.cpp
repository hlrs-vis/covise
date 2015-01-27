/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: tickwin.c,v 1.4 1994/09/29 03:37:37 pturner Exp pturner $
 *
 * ticks / tick labels / axes labels
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "extern.h"
#include <math.h>

#include <Xm/Xm.h>
#include <Xm/BulletinB.h>
#include <Xm/DialogS.h>
#include <Xm/Frame.h>
#include <Xm/Form.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/ToggleB.h>
#include <Xm/RowColumn.h>
#include <Xm/Text.h>
#include <Xm/TextF.h>
#include <Xm/ScrolledW.h>
#include <Xm/Scale.h>
#include <Xm/Separator.h>

#include "globals.h"
#include "motifinc.h"
#include "extern2.h"

static Widget ticks_frame;
static Widget ticks_panel;

static Widget *editaxis; /* which axis to edit */
static Widget *axis_applyto; /* ovverride */
static Widget offx; /* x offset of axis in viewport coords */
static Widget offy; /* y offset of axis in viewport coords */
// static Widget altmap;		/* DEFUNCT alternate mapping for axis */
// static Widget altmin;		/* DEFUNCT alternate mapping for axis */
// static Widget altmax;		/* DEFUNCT alternate mapping for axis */
static Widget tonoff; /* toggle display of axis ticks */
static Widget tlonoff; /* toggle display of tick labels */
static Widget axislabel; /* axis label */
// static Widget axislabelop;	/* axis label on opposite side */
static Widget *axislabellayout; /* axis label layout (perp or parallel) */
static Widget *axislabelfont; /* axis label font */
static Widget axislabelcharsize; /* axis label charsize */
static Widget *axislabelcolor; /* axis label color */
static Widget *axislabellinew; /* axis label linew */
static Widget tmajor; /* major tick spacing */
static Widget tminor; /* minor tick spacing */
static Widget *tickop; /* ticks opposite */
static Widget *ticklop; /* tick labels opposite */
static Widget *ticklabel_applyto; /* ovverride */
static Widget *tlform; /* format for labels */
static Widget *tlprec; /* precision for labels */
static Widget *tlfont; /* tick label font */
static Widget tlcharsize; /* tick label charsize */
static Widget *tlcolor; /* tick label color */
static Widget *tllinew; /* tick label color */
static Widget tlappstr; /* tick label append string */
static Widget tlprestr; /* tick label prepend string */
// static Widget tlvgap;		/* */
// static Widget tlhgap;		/* */
static Widget *tlskip; /* tick marks to skip */
// static Widget tltype;		/* tick label type (auto or specified) */
// static Widget ttype;		/* tick mark type (auto or specified) */
static Widget *tlstarttype; /* use graph min or starting value */
static Widget tlstart; /* value to start tick labels */
static Widget *tlstoptype; /* use graph max or stop value */
static Widget tlstop; /* value to stop tick labels */
static Widget *tllayout;
/* tick labels perp or horizontal or use the *
 * angle */
static Widget tlangle; /* angle */
static Widget *tlstagger; /* stagger */
static Widget *tlsign;
/* sign of tick label (normal, negate, *
 * absolute) */
// static Widget tlspec;		/* tick labels specified */
static Widget *tick_applyto; /* override */
// static Widget tnum;		/* number of ticks for autoscaling */
static Widget tgrid; /* major ticks grid */
static Widget *tgridcol;
static Widget *tgridlinew;
static Widget *tgridlines;
static Widget tmgrid; /* minor ticks grid */
static Widget *tmgridcol;
static Widget *tmgridlinew;
static Widget *tmgridlines;
static Widget tlen; /* tick length */
static Widget tmlen;
static Widget *tinout; /* ticks in out or both */
// static Widget tspec;		/* tick marks specified */
static Widget baronoff; /* axis bar */
static Widget *barcolor;
static Widget *barlinew;
static Widget *barlines;

static Widget specticks; /* special ticks and tick labels */
static Widget specticklabels;
static Widget nspec;
//static Widget specnum[MAX_TICK_LABELS];	/* label denoting which tick/label */
static Widget specloc[MAX_TICK_LABELS];
static Widget speclabel[MAX_TICK_LABELS];

void set_axis_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void ticks_define_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void accept_axis_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void update_axis_items(int gno);
static void do_axis_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void accept_axislabel_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void update_axislabel_items(int gno);
static void do_axislabel_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void accept_ticklabel_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void update_ticklabel_items(int gno);
static void do_ticklabels_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void update_tickmark_items(int gno);
static void accept_tickmark_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_tickmarks_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void accept_axisbar_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void update_axisbar_items(int gno);
static void do_axisbar_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void accept_special_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void update_special_items(int gno);
static void load_special(int gno, int a);
void page_special_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_special_proc(Widget w, XtPointer client_data, XtPointer call_data);

static Widget but1[2];

void update_ticks(int gno)
{
    update_ticks_items(gno);
    update_axis_items(gno);
    update_axislabel_items(gno);
    update_ticklabel_items(gno);
    update_tickmark_items(gno);
    update_axisbar_items(gno);
    load_special(gno, curaxis);
    update_special_items(gno);
}

void update_ticks_items(int gno)
{
    tickmarks t;

    if (ticks_frame)
    {
        SetChoice(editaxis, curaxis);
        get_graph_tickmarks(gno, &t, curaxis);
        XmToggleButtonSetState(tlonoff, t.tl_flag == ON, False);
        XmToggleButtonSetState(tonoff, t.t_flag == ON, False);
        XmToggleButtonSetState(baronoff, t.t_drawbar == ON, False);
        XmTextSetString(axislabel, t.label.s);

        if (islogx(gno) && (curaxis % 2 == 0))
        {
            t.tmajor = (int)t.tmajor;
            if (t.tmajor == 0)
            {
                t.tmajor = 1;
            }
            sprintf(buf, "%.0lf", t.tmajor);
        }
        else if (islogy(gno) && (curaxis % 2 == 1))
        {
            t.tmajor = (int)t.tmajor;
            if (t.tmajor == 0)
            {
                t.tmajor = 1;
            }
            sprintf(buf, "%.0lf", t.tmajor);
        }
        else if (t.tmajor > 0)
        {
            sprintf(buf, "%.5lg", t.tmajor);
        }
        else
        {
            strcpy(buf, "UNDEFINED");
        }
        XmTextSetString(tmajor, buf);
        if (islogx(gno) && (curaxis % 2 == 0))
        {
            t.tminor = (int)t.tminor;
            if (t.tminor < 0 || t.tminor > 5)
            {
                t.tminor = 0;
            }
            sprintf(buf, "%.0lf", t.tminor);
        }
        else if (islogy(gno) && (curaxis % 2 == 1))
        {
            t.tminor = (int)t.tminor;
            if (t.tminor < 0 || t.tminor > 5)
            {
                t.tminor = 0;
            }
            sprintf(buf, "%.0lf", t.tminor);
        }
        else if (t.tminor > 0)
        {
            sprintf(buf, "%.5lg", t.tminor);
        }
        else
        {
            strcpy(buf, "UNDEFINED");
        }
        XmTextSetString(tminor, buf);
    }
}

void set_axis_proc(Widget, XtPointer client_data, XtPointer)
{
    int cd = (long)client_data;
    if (ismaster)
    {
        cm->sendCommandMessage(SET_AXIS_PROC, cd, 0);
    }
    curaxis = cd;
    update_ticks(cg);
}

/*
 * Create the ticks popup
 */
void create_ticks_frame(Widget, XtPointer, XtPointer)
{
    Widget wbut, rc;
    int x, y;
    long i;
    set_wait_cursor();
    if (ticks_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        ticks_frame = XmCreateDialogShell(app_shell, (char *)"Axes", NULL, 0);
        handle_close(ticks_frame);
        XtVaSetValues(ticks_frame, XmNx, x, XmNy, y, NULL);
        ticks_panel = XmCreateRowColumn(ticks_frame, (char *)"ticks_rc", NULL, 0);

        editaxis = (Widget *)CreatePanelChoice(ticks_panel, "Edit:",
                                               5,
                                               "X axis",
                                               "Y axis",
                                               "Zero X axis",
                                               "Zero Y axis",
                                               /* DEFUNCT
                        "Alternate X axis",
                        "Alternate Y axis",
      */
                                               NULL,
                                               NULL);
        for (i = 0; i < 4; i++)
        {
            XtAddCallback(editaxis[2 + i], XmNactivateCallback, (XtCallbackProc)set_axis_proc, (XtPointer)i);
        }

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, ticks_panel, NULL);
        axislabel = CreateTextItem2(ticks_panel, 30, "Axis label:");
        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, ticks_panel, NULL);
        tmajor = CreateTextItem2(ticks_panel, 10, "Major tick spacing:");
        tminor = CreateTextItem2(ticks_panel, 10, "Minor tick spacing:");

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, ticks_panel, NULL);
        tlonoff = XtVaCreateManagedWidget("Display tick labels",
                                          xmToggleButtonWidgetClass, ticks_panel,
                                          NULL);
        tonoff = XtVaCreateManagedWidget("Display tick marks",
                                         xmToggleButtonWidgetClass, ticks_panel,
                                         NULL);
        baronoff = XtVaCreateManagedWidget("Display axis bar",
                                           xmToggleButtonWidgetClass, ticks_panel,
                                           NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, ticks_panel, NULL);
        rc = XmCreateRowColumn(ticks_panel, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
        wbut = XtVaCreateManagedWidget("Axis props...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_axis_proc, (XtPointer)0);
        wbut = XtVaCreateManagedWidget("Axis label...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_axislabel_proc, (XtPointer)0);
        wbut = XtVaCreateManagedWidget("Tick labels...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_ticklabels_proc, (XtPointer)0);
        XtManageChild(rc);

        rc = XmCreateRowColumn(ticks_panel, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
        wbut = XtVaCreateManagedWidget("Tick marks...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_tickmarks_proc, (XtPointer)0);
        wbut = XtVaCreateManagedWidget("Axis bar...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_axisbar_proc, (XtPointer)0);
        wbut = XtVaCreateManagedWidget("User ticks/tick labels...",
                                       xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_special_proc, (XtPointer)0);
        XtManageChild(rc);
        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, ticks_panel, NULL);
        axis_applyto = (Widget *)CreatePanelChoice(ticks_panel,
                                                   "Apply to:",
                                                   5,
                                                   "Current axis",
                                                   "All axes, current graph",
                                                   "Current axis, all graphs",
                                                   "All axes, all graphs",
                                                   NULL,
                                                   NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, ticks_panel, NULL);

        CreateCommandButtons(ticks_panel, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)ticks_define_notify_proc, (XtPointer)0);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)ticks_frame);

        XtManageChild(ticks_panel);
    }
    XtRaise(ticks_frame);
    update_ticks(cg);
    unset_wait_cursor();
}

/*
 * define tick marks
 */
static void ticks_define_notify_proc(Widget, XtPointer, XtPointer)
{
    char val[80];
    int i, j;
    int applyto;
    extern double result;
    double x = (g[cg].w.xg2 - g[cg].w.xg1), y = (g[cg].w.yg2 - g[cg].w.yg1), a = g[cg].w.xg1, b = g[cg].w.yg1, c = g[cg].w.xg2, d = g[cg].w.yg2;
    int errpos;
    tickmarks t;

    get_graph_tickmarks(cg, &t, curaxis);

    applyto = GetChoice(axis_applyto);
    strcpy(val, (char *)xv_getstr(tmajor));
    fixupstr(val);
    scanner(val, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &errpos);
    if (errpos)
    {
        return;
    }
    t.tmajor = result;
    if (islogx(cg) && (curaxis % 2 == 0))
    {
        t.tmajor = (int)t.tmajor;
    }
    else if (islogy(cg) && (curaxis % 2 == 1))
    {
        t.tmajor = (int)t.tmajor;
    }
    strcpy(val, (char *)xv_getstr(tminor));
    fixupstr(val);
    scanner(val, &x, &y, 1, &a, &b, &c, &d, 1, 0, 0, &errpos);
    if (errpos)
    {
        return;
    }
    t.tminor = result;
    if (islogx(cg) && (curaxis % 2 == 0))
    {
        t.tminor = (int)t.tminor;
        if (t.tminor < 0 || t.tminor > 5)
        {
            t.tminor = 0;
        }
    }
    else if (islogy(cg) && (curaxis % 2 == 1))
    {
        t.tminor = (int)t.tminor;
        if (t.tminor < 0 || t.tminor > 5)
        {
            t.tminor = 0;
        }
    }
    t.tl_flag = XmToggleButtonGetState(tlonoff) ? ON : OFF;
    t.t_flag = XmToggleButtonGetState(tonoff) ? ON : OFF;
    t.t_drawbar = XmToggleButtonGetState(baronoff) ? ON : OFF;
    set_plotstr_string(&t.label, (char *)xv_getstr(axislabel));
    if (ismaster)
    {
        cm->sendCommand_StringMessage(TICKS_DEFINE_NOTIFY_PROC, t.label.s);
        cm->sendCommand_FloatMessage(TICKS_DEFINE_NOTIFY_PROC, t.tmajor, t.tminor, 0, 0, 0, 0, 0, 0, 0, 0);
        cm->sendCommand_ValuesMessage(TICKS_DEFINE_NOTIFY_PROC, applyto, t.tl_flag, t.t_flag, t.t_drawbar, 0, 0, 0, 0, 0, 0);
    }
    switch (applyto)
    {
    case 0: /* current axis */
        set_graph_tickmarks(cg, &t, curaxis);
        break;
    case 1: /* all axes, current graph */
        for (i = 0; i < MAXAXES; i++)
        {
            g[cg].t[i].tl_flag = t.tl_flag;
            g[cg].t[i].t_flag = t.t_flag;
            g[cg].t[i].t_drawbar = t.t_drawbar;
            set_plotstr_string(&g[cg].t[i].label, t.label.s);
            g[cg].t[i].tmajor = t.tmajor;
            g[cg].t[i].tminor = t.tminor;
        }
        break;
    case 2: /* current axis, all graphs */
        for (i = 0; i < maxgraph; i++)
        {
            g[i].t[curaxis].tl_flag = t.tl_flag;
            g[i].t[curaxis].t_flag = t.t_flag;
            g[i].t[curaxis].t_drawbar = t.t_drawbar;
            set_plotstr_string(&g[i].t[curaxis].label, t.label.s);
            g[i].t[curaxis].tmajor = t.tmajor;
            g[i].t[curaxis].tminor = t.tminor;
        }
        break;
    case 3: /* all axes, all graphs */
        for (i = 0; i < maxgraph; i++)
        {
            for (j = 0; j < 6; j++)
            {
                g[i].t[j].tl_flag = t.tl_flag;
                g[i].t[j].t_flag = t.t_flag;
                g[i].t[j].t_drawbar = t.t_drawbar;
                set_plotstr_string(&g[i].t[j].label, t.label.s);
                g[i].t[j].tmajor = t.tmajor;
                g[i].t[j].tminor = t.tminor;
            }
        }
        break;
    }
    drawgraph();
}

static Widget axis_frame;
static Widget axis_panel;

static void accept_axis_proc(Widget, XtPointer, XtPointer)
{
    tickmarks t;

    get_graph_tickmarks(cg, &t, curaxis);
    /*
   Alternate ticks are being removed

       t.alt = XmToggleButtonGetState(altmap) ? ON : OFF;
       t.tmin = atof((char *) xv_getstr(altmin));
       t.tmax = atof((char *) xv_getstr(altmax));
   */
    t.alt = OFF;
    t.tmin = 0.0;
    t.tmax = 1.0;
    t.offsx = atof((char *)xv_getstr(offx));
    t.offsy = atof((char *)xv_getstr(offy));
    if (ismaster)
    {
        cm->sendCommand_FloatMessage(ACCEPT_AXIS_PROC, t.offsx, t.offsy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    set_graph_tickmarks(cg, &t, curaxis);
    drawgraph();
}

static void update_axis_items(int gno)
{
    tickmarks t;

    if (axis_frame)
    {
        get_graph_tickmarks(gno, &t, curaxis);
        /* removing alternate ticks
         XmToggleButtonSetState(altmap, t.alt == ON, False);
         sprintf(buf, "%.5g", t.tmin);
         XmTextSetString(altmin, buf);
         sprintf(buf, "%.5g", t.tmax);
         XmTextSetString(altmax, buf);
      */
        sprintf(buf, "%.5g", t.offsx);
        XmTextSetString(offx, buf);
        sprintf(buf, "%.5g", t.offsy);
        XmTextSetString(offy, buf);
    }
}

static void do_axis_proc(Widget, XtPointer, XtPointer)
{
    //Widget wlabel;
    int x, y;
    set_wait_cursor();
    if (axis_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        axis_frame = XmCreateDialogShell(app_shell, (char *)"Axis props", NULL, 0);
        handle_close(axis_frame);
        XtVaSetValues(axis_frame, XmNx, x, XmNy, y, NULL);
        axis_panel = XmCreateRowColumn(axis_frame, (char *)"axis_rc", NULL, 0);

        /*
         altmap = XtVaCreateManagedWidget("Use alternate map", xmToggleButtonWidgetClass, axis_panel,
                      NULL);
         altmin = CreateTextItem2(axis_panel, 10, "Alternate min:");
         altmax = CreateTextItem2(axis_panel, 10, "Alternate max:");

         XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, axis_panel, NULL);
      */

        //wlabel =
        XtVaCreateManagedWidget("Axis offset (viewport coordinates):",
                                xmLabelWidgetClass, axis_panel,
                                NULL);
        offx = CreateTextItem2(axis_panel, 10, "Left or bottom:");
        offy = CreateTextItem2(axis_panel, 10, "Right or top:");

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, axis_panel, NULL);

        CreateCommandButtons(axis_panel, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)accept_axis_proc, (XtPointer)0);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)axis_frame);

        XtManageChild(axis_panel);
    }
    XtRaise(axis_frame);
    update_axis_items(cg);
    unset_wait_cursor();
}

static Widget axislabel_frame;
static Widget axislabel_panel;

static void accept_axislabel_proc(Widget, XtPointer, XtPointer)
{
    Arg a;
    tickmarks t;
    int iv;

    get_graph_tickmarks(cg, &t, curaxis);
    t.label_layout = GetChoice(axislabellayout) ? PERP : PARA;
    t.label.font = GetChoice(axislabelfont);
    t.label.color = GetChoice(axislabelcolor);
    t.label.linew = GetChoice(axislabellinew) + 1;
    XtSetArg(a, XmNvalue, &iv);
    XtGetValues(axislabelcharsize, &a, 1);
    t.label.charsize = iv / 100.0;
    if (ismaster)
    {
        cm->sendCommand_ValuesMessage(ACCEPT_AXISLABEL_PROC, t.label_layout, t.label.font, t.label.color, t.label.linew, iv, 0, 0, 0, 0, 0);
    }
    set_graph_tickmarks(cg, &t, curaxis);
    drawgraph();
}

static void update_axislabel_items(int gno)
{
    Arg a;
    tickmarks t;
    int iv;

    if (axislabel_frame)
    {
        get_graph_tickmarks(gno, &t, curaxis);
        SetChoice(axislabellayout, t.label_layout == PERP ? 1 : 0);
        SetChoice(axislabelfont, t.label.font);
        SetChoice(axislabelcolor, t.label.color);
        SetChoice(axislabellinew, t.label.linew - 1);
        iv = (int)(100 * t.label.charsize);
        XtSetArg(a, XmNvalue, iv);
        XtSetValues(axislabelcharsize, &a, 1);
    }
}

static void do_axislabel_proc(Widget, XtPointer, XtPointer)
{
    Widget rc;
    int x, y;
    set_wait_cursor();
    if (axislabel_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        axislabel_frame = XmCreateDialogShell(app_shell, (char *)"Axis label", NULL, 0);
        handle_close(axislabel_frame);
        XtVaSetValues(axislabel_frame, XmNx, x, XmNy, y, NULL);
        axislabel_panel = XmCreateRowColumn(axislabel_frame, (char *)"axislabel_rc", NULL, 0);

        axislabellayout = (Widget *)CreatePanelChoice(axislabel_panel, "Axis layout:",
                                                      3,
                                                      "Parallel to axis",
                                                      "Perpendicular to axis",
                                                      NULL,
                                                      NULL);

        axislabelfont = CreatePanelChoice(axislabel_panel, "Font:",
                                          11,
                                          "Times-Roman", "Times-Bold", "Times-Italic",
                                          "Times-BoldItalic", "Helvetica",
                                          "Helvetica-Bold", "Helvetica-Oblique",
                                          "Helvetica-BoldOblique", "Greek", "Symbol",
                                          0,
                                          0);

        rc = XmCreateRowColumn(axislabel_panel, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
        axislabelcolor = CreateColorChoice(rc, "Color:", 0);
        axislabellinew = CreatePanelChoice(rc, "Line width:",
                                           10,
                                           "1", "2", "3", "4", "5", "6", "7", "8", "9", 0,
                                           0);
        XtManageChild(rc);

        //wlabel =
        XtVaCreateManagedWidget("Size:", xmLabelWidgetClass, axislabel_panel,
                                NULL);
        axislabelcharsize = XtVaCreateManagedWidget("stringsize", xmScaleWidgetClass, axislabel_panel,
                                                    XmNminimum, 0,
                                                    XmNmaximum, 400,
                                                    XmNvalue, 100,
                                                    XmNshowValue, True,
                                                    XmNprocessingDirection, XmMAX_ON_RIGHT,
                                                    XmNorientation, XmHORIZONTAL,
                                                    NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, axislabel_panel, NULL);

        CreateCommandButtons(axislabel_panel, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)accept_axislabel_proc, (XtPointer)0);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)axislabel_frame);

        XtManageChild(axislabel_panel);
    }
    XtRaise(axislabel_frame);
    update_axislabel_items(cg);
    unset_wait_cursor();
}

static Widget ticklabel_frame;
static Widget ticklabel_panel;

static void accept_ticklabel_proc(Widget, XtPointer, XtPointer)
{
    Arg a;
    tickmarks t;
    int iv;
    int i, j, applyto, gstart = 0, gstop = 0, astart = 0, astop = 0;
    applyto = GetChoice(ticklabel_applyto);
    switch (applyto)
    {
    case 0:
        gstart = gstop = cg;
        astart = astop = curaxis;
        break;
    case 1:
        gstart = gstop = cg;
        astart = 0;
        astop = 5;
        break;
    case 2:
        gstart = 0;
        gstop = maxgraph - 1;
        astart = astop = curaxis;
        break;
    case 3:
        gstart = 0;
        gstop = maxgraph - 1;
        astart = 0;
        astop = 5;
        break;
    default:
        fprintf(stderr, "tickwin.cpp: accept_ticklabel_proc(): gstart, gstop, ... uninitialized\n");
        break;
    }
    for (i = gstart; i <= gstop; i++)
    {
        for (j = astart; j <= astop; j++)
        {
            get_graph_tickmarks(i, &t, j);
            t.tl_font = GetChoice(tlfont);
            t.tl_color = GetChoice(tlcolor);
            t.tl_linew = GetChoice(tllinew) + 1;
            t.tl_skip = GetChoice(tlskip);
            t.tl_prec = GetChoice(tlprec);
            t.tl_staggered = (int)GetChoice(tlstagger);
            strcpy(t.tl_appstr, xv_getstr(tlappstr));
            strcpy(t.tl_prestr, xv_getstr(tlprestr));
            t.tl_starttype = (int)GetChoice(tlstarttype) == 0 ? AUTO : SPEC;
            if (t.tl_starttype == SPEC)
            {
                t.tl_start = atof((char *)xv_getstr(tlstart));
            }
            t.tl_stoptype = (int)GetChoice(tlstoptype) == 0 ? AUTO : SPEC;
            if (t.tl_stoptype == SPEC)
            {
                t.tl_stop = atof((char *)xv_getstr(tlstop));
            }
            t.tl_format = format_types[(int)GetChoice(tlform)];
            switch (GetChoice(ticklop))
            {
            case 0:
                if (j % 2)
                {
                    t.tl_op = LEFT;
                }
                else
                {
                    t.tl_op = BOTTOM;
                }
                break;
            case 1:
                if (j % 2)
                {
                    t.tl_op = RIGHT;
                }
                else
                {
                    t.tl_op = TOP;
                }
                break;
            case 2:
                t.tl_op = BOTH;
                break;
            }
            switch ((int)GetChoice(tlsign))
            {
            case 0:
                t.tl_sign = NORMAL;
                break;
            case 1:
                t.tl_sign = ABSOLUTE;
                break;
            case 2:
                t.tl_sign = NEGATE;
                break;
            }
            switch ((int)GetChoice(tllayout))
            {
            case 0:
                t.tl_layout = HORIZONTAL;
                break;
            case 1:
                t.tl_layout = VERTICAL;
                break;
            case 2:
                t.tl_layout = SPEC;
                XtSetArg(a, XmNvalue, &iv);
                XtGetValues(tlangle, &a, 1);
                t.tl_angle = iv;
                break;
            }
            XtSetArg(a, XmNvalue, &iv);
            XtGetValues(tlcharsize, &a, 1);
            t.tl_charsize = iv / 100.0;
            if (ismaster)
            {
                cm->sendCommand_ValuesMessage(ACCEPT_TICKLABEL_PROC, i, j, t.tl_angle, t.tl_layout, t.tl_sign, t.tl_op, t.tl_font, t.tl_color, t.tl_linew, t.tl_skip);
                cm->sendCommand_StringMessage(ACCEPT_TICKLABEL_PROC, t.tl_appstr);
                cm->sendCommand_StringMessage(ACCEPT_TICKLABEL_PROC2, t.tl_prestr);
                cm->sendCommand_FloatMessage(ACCEPT_TICKLABEL_PROC, t.tl_start, t.tl_stop, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                cm->sendCommand_ValuesMessage(ACCEPT_TICKLABEL_PROC2, i, j, t.tl_prec, t.tl_staggered, t.tl_starttype, t.tl_stoptype, t.tl_format, 0, 0, iv);
            }
            set_graph_tickmarks(i, &t, j);
        }
    }
    if (ismaster)
    {
        cm->sendCommandMessage(DRAWGRAPH, 0, 0);
    }
    drawgraph();
}

static void update_ticklabel_items(int gno)
{
    Arg a;
    tickmarks t;
    int iv;

    if (ticklabel_frame)
    {
        get_graph_tickmarks(gno, &t, curaxis);
        SetChoice(tlfont, t.tl_font);
        SetChoice(tlcolor, t.tl_color);
        SetChoice(tllinew, t.tl_linew - 1);
        SetChoice(tlskip, t.tl_skip);
        SetChoice(tlstagger, t.tl_staggered);
        xv_setstr(tlappstr, t.tl_appstr);
        xv_setstr(tlprestr, t.tl_prestr);
        SetChoice(tlstarttype, t.tl_starttype == SPEC);
        if (t.tl_starttype == SPEC)
        {
            sprintf(buf, "%lf", t.tl_start);
            xv_setstr(tlstart, buf);
            sprintf(buf, "%lf", t.tl_stop);
            xv_setstr(tlstop, buf);
        }
        SetChoice(tlstoptype, t.tl_stoptype == SPEC);
        if (t.tl_stoptype == SPEC)
        {
            sprintf(buf, "%lf", t.tl_stop);
            xv_setstr(tlstop, buf);
        }
        iv = getFormat_index(t.tl_format);
        SetChoice(tlform, iv);
        switch (t.tl_op)
        {
        case LEFT:
            SetChoice(ticklop, 0);
            break;
        case RIGHT:
            SetChoice(ticklop, 1);
            break;
        case BOTTOM:
            SetChoice(ticklop, 0);
            break;
        case TOP:
            SetChoice(ticklop, 1);
            break;
        case BOTH:
            SetChoice(ticklop, 2);
            break;
        }
        switch (t.tl_sign)
        {
        case NORMAL:
            SetChoice(tlsign, 0);
            break;
        case ABSOLUTE:
            SetChoice(tlsign, 1);
            break;
        case NEGATE:
            SetChoice(tlsign, 2);
            break;
        }
        SetChoice(tlprec, t.tl_prec);
        iv = (int)(100 * t.tl_charsize);
        XtSetArg(a, XmNvalue, iv);
        XtSetValues(tlcharsize, &a, 1);
        switch (t.tl_layout)
        {
        case HORIZONTAL:
            SetChoice(tllayout, 0);
            break;
        case VERTICAL:
            SetChoice(tllayout, 1);
            break;
        case SPEC:
            SetChoice(tllayout, 2);
            break;
        }
        iv = (int)t.tl_angle % 360;
        XtSetArg(a, XmNvalue, iv);
        XtSetValues(tlangle, &a, 1);
    }
}

static void do_ticklabels_proc(Widget, XtPointer, XtPointer)
{
    Widget rc;
    int x, y;
    set_wait_cursor();
    if (ticklabel_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        ticklabel_frame = XmCreateDialogShell(app_shell, (char *)"Tick labels", NULL, 0);
        handle_close(ticklabel_frame);
        XtVaSetValues(ticklabel_frame, XmNx, x, XmNy, y, NULL);
        ticklabel_panel = XmCreateRowColumn(ticklabel_frame, (char *)"ticklabel_rc", NULL, 0);

        tlfont = CreatePanelChoice(ticklabel_panel, "Font:",
                                   11,
                                   "Times-Roman", "Times-Bold", "Times-Italic",
                                   "Times-BoldItalic", "Helvetica",
                                   "Helvetica-Bold", "Helvetica-Oblique",
                                   "Helvetica-BoldOblique", "Greek", "Symbol",
                                   0,
                                   0);

        rc = XmCreateRowColumn(ticklabel_panel, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
        tlcolor = CreateColorChoice(rc, "Color:", 0);

        tllinew = CreatePanelChoice(rc, "Line width:",
                                    10,
                                    "1", "2", "3", "4", "5", "6", "7", "8", "9", 0,
                                    0);
        XtManageChild(rc);

        rc = XmCreateRowColumn(ticklabel_panel, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
        //wlabel =
        XtVaCreateManagedWidget("Char size:", xmLabelWidgetClass, rc,
                                NULL);
        tlcharsize = XtVaCreateManagedWidget("stringsize", xmScaleWidgetClass, rc,
                                             XmNminimum, 0,
                                             XmNmaximum, 400,
                                             XmNvalue, 0,
                                             XmNshowValue, True,
                                             XmNprocessingDirection, XmMAX_ON_RIGHT,
                                             XmNorientation, XmHORIZONTAL,
                                             NULL);
        XtManageChild(rc);
        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, ticklabel_panel, NULL);

        tlform = CreatePanelChoice0(ticklabel_panel,
                                    "Format:", 4,
                                    29,
                                    "Decimal",
                                    "Exponential",
                                    "Power",
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
                                    "MM' SS.s\" (lat)", 0,
                                    0);

        tlprec = CreatePanelChoice(ticklabel_panel, "Precision:",
                                   11,
                                   "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 0,
                                   0);
        tlappstr = CreateTextItem2(ticklabel_panel, 10, "Append to labels:");
        tlprestr = CreateTextItem2(ticklabel_panel, 10, "Prepend to labels:");

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, ticklabel_panel, NULL);

        tlstagger = CreatePanelChoice(ticklabel_panel, "Stagger labels:",
                                      11,
                                      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 0,
                                      0);

        tlskip = CreatePanelChoice(ticklabel_panel, "Skip every:",
                                   11,
                                   "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 0,
                                   0);
        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, ticklabel_panel, NULL);

        rc = XmCreateRowColumn(ticklabel_panel, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
        tlstarttype = CreatePanelChoice(rc, "Start labels at:",
                                        3,
                                        "Graph min", "Specified:", 0,
                                        0);
        tlstart = XtVaCreateManagedWidget("tlstart", xmTextWidgetClass, rc,
                                          XmNtraversalOn, True,
                                          XmNcolumns, 10,
                                          NULL);
        XtManageChild(rc);

        rc = XmCreateRowColumn(ticklabel_panel, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
        tlstoptype = CreatePanelChoice(rc, "Stop labels at:",
                                       3,
                                       "Graph max", "Specified:", 0,
                                       0);
        tlstop = XtVaCreateManagedWidget("tlstop", xmTextWidgetClass, rc,
                                         XmNtraversalOn, True,
                                         XmNcolumns, 10,
                                         NULL);
        XtManageChild(rc);

        rc = XmCreateRowColumn(ticklabel_panel, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
        tllayout = (Widget *)CreatePanelChoice(rc, "Layout:",
                                               4,
                                               "Horizontal",
                                               "Vertical",
                                               "Specified (degrees):",
                                               NULL,
                                               NULL);
        tlangle = XtVaCreateManagedWidget("ticklangle", xmScaleWidgetClass, rc,
                                          XmNminimum, 0,
                                          XmNmaximum, 360,
                                          XmNvalue, 100,
                                          XmNshowValue, True,
                                          XmNprocessingDirection, XmMAX_ON_RIGHT,
                                          XmNorientation, XmHORIZONTAL,
                                          NULL);
        XtManageChild(rc);
        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, ticklabel_panel, NULL);

        ticklop = CreatePanelChoice(ticklabel_panel, "Draw tick labels on:",
                                    4,
                                    "Normal side", "Opposite side", "Both", 0,
                                    0);

        tlsign = CreatePanelChoice(ticklabel_panel, "Sign of label:",
                                   4,
                                   "As is", "Absolute value", "Negate",
                                   NULL,
                                   0);

        ticklabel_applyto = CreatePanelChoice(ticklabel_panel, "Apply to:",
                                              4,
                                              "Current axis",
                                              "All axes, current graph",
                                              "Current axis, all graphs",
                                              "All axes, all graphs",
                                              NULL,
                                              0);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, ticklabel_panel, NULL);

        CreateCommandButtons(ticklabel_panel, 2, but1, label1);

        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)accept_ticklabel_proc, (XtPointer)0);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)ticklabel_frame);

        XtManageChild(ticklabel_panel);
    }
    XtRaise(ticklabel_frame);
    update_ticklabel_items(cg);
    unset_wait_cursor();
}

static Widget tickmark_frame;
static Widget tickmark_panel;

static void update_tickmark_items(int gno)
{
    Arg a;
    tickmarks t;
    int iv;

    if (tickmark_frame)
    {
        get_graph_tickmarks(gno, &t, curaxis);
        switch (t.t_inout)
        {
        case IN:
            SetChoice(tinout, 0);
            break;
        case OUT:
            SetChoice(tinout, 1);
            break;
        case BOTH:
            SetChoice(tinout, 2);
            break;
        }
        switch (t.t_op)
        {
        case LEFT:
            SetChoice(tickop, 0);
            break;
        case RIGHT:
            SetChoice(tickop, 1);
            break;
        case BOTTOM:
            SetChoice(tickop, 0);
            break;
        case TOP:
            SetChoice(tickop, 1);
            break;
        case BOTH:
            SetChoice(tickop, 2);
            break;
        }
        SetChoice(tgridcol, t.t_color);
        SetChoice(tgridlinew, t.t_linew - 1);
        SetChoice(tgridlines, t.t_lines - 1);
        SetChoice(tmgridcol, t.t_mcolor);
        SetChoice(tmgridlinew, t.t_mlinew - 1);
        SetChoice(tmgridlines, t.t_mlines - 1);
        iv = (int)(100 * t.t_size);
        XtSetArg(a, XmNvalue, iv);
        XtSetValues(tlen, &a, 1);
        iv = (int)(100 * t.t_msize);
        XtSetArg(a, XmNvalue, iv);
        XtSetValues(tmlen, &a, 1);
        XmToggleButtonSetState(tgrid, t.t_gridflag == ON, False);
        XmToggleButtonSetState(tmgrid, t.t_mgridflag == ON, False);
    }
}

static void accept_tickmark_proc(Widget, XtPointer, XtPointer)
{
    Arg a;
    tickmarks t;
    int iv, iv2;
    int i, j, applyto, gstart = 0, gstop = 0, astart = 0, astop = 0;
    applyto = GetChoice(tick_applyto);
    switch (applyto)
    {
    case 0:
        gstart = gstop = cg;
        astart = astop = curaxis;
        break;
    case 1:
        gstart = gstop = cg;
        astart = 0;
        astop = 5;
        break;
    case 2:
        gstart = 0;
        gstop = maxgraph - 1;
        astart = astop = curaxis;
        break;
    case 3:
        gstart = 0;
        gstop = maxgraph - 1;
        astart = 0;
        astop = 5;
        break;
    default:
        fprintf(stderr, "tickwin.cpp: accept_tickmark_proc(): gstart, gstop, ... uninitialized\n");
        break;
    }
    for (i = gstart; i <= gstop; i++)
    {
        for (j = astart; j <= astop; j++)
        {
            get_graph_tickmarks(i, &t, j);
            switch ((int)GetChoice(tinout))
            {
            case 0:
                t.t_inout = IN;
                break;
            case 1:
                t.t_inout = OUT;
                break;
            case 2:
                t.t_inout = BOTH;
                break;
            }
            switch (GetChoice(tickop))
            {
            case 0:
                if (j % 2)
                {
                    t.t_op = LEFT;
                }
                else
                {
                    t.t_op = BOTTOM;
                }
                break;
            case 1:
                if (j % 2)
                {
                    t.t_op = RIGHT;
                }
                else
                {
                    t.t_op = TOP;
                }
                break;
            case 2:
                t.t_op = BOTH;
                break;
            }
            t.t_color = GetChoice(tgridcol);
            t.t_linew = GetChoice(tgridlinew) + 1;
            t.t_lines = GetChoice(tgridlines) + 1;
            t.t_mcolor = GetChoice(tmgridcol);
            t.t_mlinew = GetChoice(tmgridlinew) + 1;
            t.t_mlines = GetChoice(tmgridlines) + 1;
            XtSetArg(a, XmNvalue, &iv);
            XtGetValues(tlen, &a, 1);
            t.t_size = iv / 100.0;
            XtSetArg(a, XmNvalue, &iv2);
            XtGetValues(tmlen, &a, 1);
            t.t_msize = iv2 / 100.0;
            t.t_gridflag = XmToggleButtonGetState(tgrid) ? ON : OFF;
            t.t_mgridflag = XmToggleButtonGetState(tmgrid) ? ON : OFF;
            if (ismaster)
            {
                cm->sendCommand_ValuesMessage(ACCEPT_TICKMARK_PROC, i, j, t.t_inout, t.t_op, t.t_color, t.t_linew, t.t_lines, t.t_mcolor, t.t_mlinew, t.t_mlines);
                cm->sendCommand_ValuesMessage(ACCEPT_TICKMARK_PROC2, i, j, iv, iv2, t.t_gridflag, t.t_mgridflag, 0, 0, 0, 0);
            }
            set_graph_tickmarks(i, &t, j);
        }
    }
    if (ismaster)
    {
        cm->sendCommandMessage(DRAWGRAPH, 0, 0);
    }
    drawgraph();
}

static void do_tickmarks_proc(Widget, XtPointer, XtPointer)
{
    Widget rc, rc2, rc3, fr;
    int x, y;
    set_wait_cursor();
    if (tickmark_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        tickmark_frame = XmCreateDialogShell(app_shell, (char *)"Tick marks", NULL, 0);
        handle_close(tickmark_frame);
        XtVaSetValues(tickmark_frame, XmNx, x, XmNy, y, NULL);
        tickmark_panel = XmCreateRowColumn(tickmark_frame, (char *)"tickmark_rc", NULL, 0);

        tinout = CreatePanelChoice(tickmark_panel, "Tick marks pointing:",
                                   4,
                                   "In", "Out", "Both", 0,
                                   0);

        tickop = CreatePanelChoice(tickmark_panel, "Draw tick marks on:",
                                   4,
                                   "Normal side", "Opposite side", "Both sides", 0,
                                   0);

        rc2 = XmCreateRowColumn(tickmark_panel, (char *)"rc2", NULL, 0);
        XtVaSetValues(rc2, XmNorientation, XmHORIZONTAL, NULL);

        /* major tick marks */
        fr = XmCreateFrame(rc2, (char *)"fr", NULL, 0);
        rc = XmCreateRowColumn(fr, (char *)"rc", NULL, 0);

        tgrid = XtVaCreateManagedWidget("Major ticks grid lines",
                                        xmToggleButtonWidgetClass, rc,
                                        NULL);

        rc3 = XmCreateRowColumn(rc, (char *)"rc3", NULL, 0);
        //wlabel =
        XtVaCreateManagedWidget("Major tick length:", xmLabelWidgetClass, rc3,
                                NULL);
        tlen = XtVaCreateManagedWidget("ticklength", xmScaleWidgetClass, rc3,
                                       XmNminimum, 0,
                                       XmNmaximum, 400,
                                       XmNvalue, 100,
                                       XmNshowValue, True,
                                       XmNprocessingDirection, XmMAX_ON_RIGHT,
                                       XmNorientation, XmHORIZONTAL,
                                       NULL);
        XtManageChild(rc3);

        tgridcol = CreateColorChoice(rc, "Color:", 0);

        tgridlinew = CreatePanelChoice(rc, "Line width:",
                                       10,
                                       "1", "2", "3", "4", "5", "6", "7", "8", "9", 0,
                                       0);
        tgridlines = (Widget *)CreatePanelChoice(rc, "Line style:",
                                                 6,
                                                 "Solid line",
                                                 "Dotted line",
                                                 "Dashed line",
                                                 "Long Dashed",
                                                 "Dot-dashed",
                                                 NULL,
                                                 NULL);
        XtManageChild(rc);
        XtManageChild(fr);

        fr = XmCreateFrame(rc2, (char *)"fr", NULL, 0);
        rc = XmCreateRowColumn(fr, (char *)"rc", NULL, 0);

        tmgrid = XtVaCreateManagedWidget("Minor ticks grid lines", xmToggleButtonWidgetClass, rc,
                                         NULL);
        rc3 = XmCreateRowColumn(rc, (char *)"rc", NULL, 0);
        //wlabel =
        XtVaCreateManagedWidget("Minor tick length:", xmLabelWidgetClass, rc3,
                                NULL);
        tmlen = XtVaCreateManagedWidget("mticklength", xmScaleWidgetClass, rc3,
                                        XmNminimum, 0,
                                        XmNmaximum, 400,
                                        XmNvalue, 100,
                                        XmNshowValue, True,
                                        XmNprocessingDirection, XmMAX_ON_RIGHT,
                                        XmNorientation, XmHORIZONTAL,
                                        NULL);
        XtManageChild(rc3);

        tmgridcol = CreateColorChoice(rc, "Color:", 0);
        tmgridlinew = CreatePanelChoice(rc, "Line width:",
                                        10,
                                        "1", "2", "3", "4", "5", "6", "7", "8", "9", 0,
                                        0);
        tmgridlines = (Widget *)CreatePanelChoice(rc, "Line style:",
                                                  6,
                                                  "Solid line",
                                                  "Dotted line",
                                                  "Dashed line",
                                                  "Long Dashed",
                                                  "Dot-dashed",
                                                  NULL,
                                                  NULL);
        XtManageChild(rc);
        XtManageChild(fr);
        XtManageChild(rc2);

        tick_applyto = CreatePanelChoice(tickmark_panel, "Apply to:",
                                         4,
                                         "Current axis",
                                         "All axes, current graph",
                                         "Current axis, all graphs",
                                         "All axes, all graphs",
                                         NULL,
                                         0);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, tickmark_panel, NULL);

        CreateCommandButtons(tickmark_panel, 2, but1, label1);

        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)accept_tickmark_proc, (XtPointer)0);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)tickmark_frame);

        XtManageChild(tickmark_panel);
    }
    XtRaise(tickmark_frame);
    update_tickmark_items(cg);
    unset_wait_cursor();
}

static Widget axisbar_frame;
static Widget axisbar_panel;

static void accept_axisbar_proc(Widget, XtPointer, XtPointer)
{
    tickmarks t;

    get_graph_tickmarks(cg, &t, curaxis);
    t.t_drawbarcolor = GetChoice(barcolor);
    t.t_drawbarlinew = GetChoice(barlinew) + 1;
    t.t_drawbarlines = GetChoice(barlines) + 1;
    if (ismaster)
    {
        cm->sendCommand_ValuesMessage(ACCEPT_AXISBAR_PROC, t.t_drawbarcolor, t.t_drawbarlinew, t.t_drawbarlines, 0, 0, 0, 0, 0, 0, 0);
    }
    set_graph_tickmarks(cg, &t, curaxis);
    drawgraph();
}

static void update_axisbar_items(int gno)
{
    tickmarks t;

    if (axisbar_frame)
    {
        get_graph_tickmarks(gno, &t, curaxis);
        SetChoice(barcolor, t.t_drawbarcolor);
        SetChoice(barlinew, t.t_drawbarlinew - 1);
        SetChoice(barlines, t.t_drawbarlines - 1);
    }
}

static void do_axisbar_proc(Widget, XtPointer, XtPointer)
{
    int x, y;
    set_wait_cursor();
    if (axisbar_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        axisbar_frame = XmCreateDialogShell(app_shell, (char *)"Axis bar", NULL, 0);
        handle_close(axisbar_frame);
        XtVaSetValues(axisbar_frame, XmNx, x, XmNy, y, NULL);
        axisbar_panel = XmCreateRowColumn(axisbar_frame, (char *)"axisbar_rc", NULL, 0);

        barcolor = CreateColorChoice(axisbar_panel, "Color:", 0);

        barlinew = CreatePanelChoice(axisbar_panel, "Line width:",
                                     10,
                                     "1", "2", "3", "4", "5", "6", "7", "8", "9", 0,
                                     0);

        barlines = (Widget *)CreatePanelChoice(axisbar_panel, "Line style:",
                                               6,
                                               "Solid line",
                                               "Dotted line",
                                               "Dashed line",
                                               "Long Dashed",
                                               "Dot-dashed",
                                               NULL,
                                               NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, axisbar_panel, NULL);

        CreateCommandButtons(axisbar_panel, 2, but1, label1);

        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)accept_axisbar_proc, (XtPointer)0);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)axisbar_frame);

        XtManageChild(axisbar_panel);
    }
    XtRaise(axisbar_frame);
    update_axisbar_items(cg);
    unset_wait_cursor();
}

static Widget special_frame;
static Widget special_panel;

#define TPAGESIZE 5
#define NPAGES (MAX_TICK_LABELS / TPAGESIZE)
// static int tcurpage = 0;

static void accept_special_proc(Widget, XtPointer, XtPointer)
{
    tickmarks t;
    int iv, i;

    get_graph_tickmarks(cg, &t, curaxis);
    t.t_type = XmToggleButtonGetState(specticks) ? SPEC : AUTO;
    t.tl_type = XmToggleButtonGetState(specticklabels) ? SPEC : AUTO;
    iv = atoi((char *)xv_getstr(nspec));
    if (iv > MAX_TICK_LABELS)
    {
        sprintf(buf, "Number of ticks/tick labels exceeds %d", MAX_TICK_LABELS);
        errwin(buf);
        return;
    }
    t.t_spec = iv;
    for (i = 0; i < MAX_TICK_LABELS; i++)
    {
        t.t_specloc[i] = atof((char *)xv_getstr(specloc[i]));
        set_plotstr_string(&t.t_speclab[i], (char *)xv_getstr(speclabel[i]));
        if (ismaster)
        {
            cm->sendCommand_FloatMessage(ACCEPT_SPECIAL_PROC, (double)i, t.t_specloc[i], 0, 0, 0, 0, 0, 0, 0, 0);
            cm->sendCommand_StringMessage(ACCEPT_SPECIAL_PROC, (t.t_speclab[i]).s);
        }
    }
    if (ismaster)
    {
        cm->sendCommand_ValuesMessage(ACCEPT_SPECIAL_PROC, t.t_type, t.tl_type, t.t_spec, 0, 0, 0, 0, 0, 0, 0);
    }
    set_graph_tickmarks(cg, &t, curaxis);
    drawgraph();
}

static void update_special_items(int gno)
{
    tickmarks t;

    if (special_frame)
    {
        get_graph_tickmarks(gno, &t, curaxis);
        XmToggleButtonSetState(specticks, t.t_type == SPEC, False);
        XmToggleButtonSetState(specticklabels, t.tl_type == SPEC, False);
    }
}

static void load_special(int gno, int a)
{
    int i;
    char buf[128];
    tickmarks t;

    if (special_frame)
    {
        get_graph_tickmarks(gno, &t, a);
        sprintf(buf, "%d", t.t_spec);
        xv_setstr(nspec, buf);
        for (i = 0; i < t.t_spec; i++)
        {
            sprintf(buf, "%lf", t.t_specloc[i]);
            xv_setstr(specloc[i], buf);
            if (t.t_speclab[i].s != NULL)
            {
                xv_setstr(speclabel[i], t.t_speclab[i].s);
            }
        }
    }
}

void page_special_notify_proc(Widget, XtPointer, XtPointer)
{
    if (ismaster)
    {
        cm->sendCommandMessage(PAGE_SPECIAL_NOTIFY_PROC, 0, 0);
    }
    update_special_items(cg);
}

static void do_special_proc(Widget, XtPointer, XtPointer)
{
    Widget rc, rc3, sw;
    int i, x, y;
    char buf[10];
    set_wait_cursor();
    if (special_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        special_frame = XmCreateDialogShell(app_shell, (char *)"Specified ticks/ticklabels", NULL, 0);
        handle_close(special_frame);
        XtVaSetValues(special_frame, XmNx, x, XmNy, y, NULL);
        special_panel = XmCreateForm(special_frame, (char *)"special_form", NULL, 0);

        rc = XmCreateRowColumn(special_panel, (char *)"rc", NULL, 0);
        specticks = XtVaCreateManagedWidget("Use special tick locations",
                                            xmToggleButtonWidgetClass, rc,
                                            NULL);
        specticklabels = XtVaCreateManagedWidget("Use special tick labels",
                                                 xmToggleButtonWidgetClass, rc,
                                                 NULL);

        nspec = CreateTextItem2(rc, 10, "# of user ticks/labels to use:");
        //wlabel =
        XtVaCreateManagedWidget("Tick location - Label:", xmLabelWidgetClass,
                                rc, NULL);
        XtManageChild(rc);
        XtVaSetValues(rc,
                      XmNleftAttachment, XmATTACH_FORM,
                      XmNrightAttachment, XmATTACH_FORM,
                      XmNtopAttachment, XmATTACH_FORM,
                      NULL);

        sw = XtVaCreateManagedWidget("sw",
                                     xmScrolledWindowWidgetClass, special_panel,
                                     XmNscrollingPolicy, XmAUTOMATIC,
                                     XmNtopAttachment, XmATTACH_WIDGET,
                                     XmNtopWidget, rc,
                                     XmNleftAttachment, XmATTACH_FORM,
                                     XmNrightAttachment, XmATTACH_FORM,
                                     NULL);
        rc = XmCreateRowColumn(sw, (char *)"rc", NULL, 0);
        XtVaSetValues(sw,
                      XmNworkWindow, rc,
                      NULL);

        for (i = 0; i < MAX_TICK_LABELS; i++)
        {
            rc3 = XmCreateRowColumn(rc, (char *)"rc3", NULL, 0);
            XtVaSetValues(rc3, XmNorientation, XmHORIZONTAL, NULL);
            sprintf(buf, "%2d", i + 1);
            //specnum[i] =
            XtVaCreateManagedWidget(buf, xmLabelWidgetClass, rc3,
                                    NULL);
            specloc[i] = XtVaCreateManagedWidget("tickmark", xmTextFieldWidgetClass, rc3,
                                                 XmNcolumns, 10,
                                                 NULL);
            speclabel[i] = XtVaCreateManagedWidget("ticklabel", xmTextFieldWidgetClass, rc3,
                                                   XmNcolumns, 35,
                                                   NULL);
            XtManageChild(rc3);
        }
        XtManageChild(rc);
        XtManageChild(sw);

        rc = XmCreateRowColumn(special_panel, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);

        CreateCommandButtons(rc, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)accept_special_proc, (XtPointer)0);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)special_frame);
        XtManageChild(rc);
        XtVaSetValues(rc,
                      XmNleftAttachment, XmATTACH_FORM,
                      XmNrightAttachment, XmATTACH_FORM,
                      XmNbottomAttachment, XmATTACH_FORM,
                      NULL);
        XtVaSetValues(sw,
                      XmNbottomAttachment, XmATTACH_WIDGET,
                      XmNbottomWidget, rc,
                      NULL);

        load_special(cg, curaxis);
        XtManageChild(special_panel);
    }
    XtRaise(special_frame);
    update_special_items(cg);
    unset_wait_cursor();
}
