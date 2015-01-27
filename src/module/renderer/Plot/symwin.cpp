/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: symwin.c,v 1.7 1994/11/04 07:04:49 pturner Exp pturner $
 *
 * symbols, legends, and error bars
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "extern.h"
#include <Xm/Xm.h>
#include <Xm/BulletinB.h>
#include <Xm/DialogS.h>
#include <Xm/Frame.h>
#include <Xm/Form.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/Scale.h>
#include <Xm/ScrolledW.h>
#include <Xm/Separator.h>
#include <Xm/ToggleB.h>
#include <Xm/Text.h>

#include "globals.h"
#include "motifinc.h"

extern Widget app_shell;

int cset = 0; /* the current set from the symbols panel */

static Widget define_symbols_frame;

static Widget define_legend_frame;
static Widget define_legend_panel;

// static Widget *toggle_set_item;
static Widget *toggle_symbols_item;
static Widget symchar_item;
static Widget symsize_item;
static Widget *symfill_item;
static Widget symskip_item;
static Widget *symcolor_item;
static Widget *symlinew_item;
static Widget *symlines_item;
static Widget *symbols_apply_item;
static Widget *toggle_color_item;
static Widget *toggle_width_item;
static Widget *toggle_lines_item;
static Widget *toggle_fill_item;
static Widget *toggle_fillusing_item;
static Widget *toggle_fillpat_item;
static Widget *toggle_fillcol_item;
static SetChoiceItem toggle_symset_item;
//static Widget symmsg;

static Widget define_errbar_frame;
static Widget define_errbar_panel;
static Widget errbar_size_item;
static Widget *errbar_width_item;
static Widget *errbar_lines_item;
static Widget *errbar_type_item;
static Widget *errbar_riser_item;
static Widget *errbar_riserlinew_item;
static Widget *errbar_riserlines_item;
static Widget *errbar_apply_item;

static Widget define_boxplot_frame;
//static Widget define_boxplot_panel;
//static Widget boxplot_size_item;
//static Widget boxplot_outliers_item;
static Widget *boxplot_type_item;
static Widget *boxplot_apply_item;

Widget legend_x_panel; /* needed in the canvas event proc */
Widget legend_y_panel;
static Widget toggle_legends_item;
static Widget *toggle_legendloc_item;
static Widget legend_str_panel;
static Widget *legends_gap_item;
static Widget *legends_len_item;
//static Widget leglocbut;
static Widget *legend_font_item;
static Widget legend_charsize_item;
static Widget *legend_color_item;
//static Widget *legend_linew_item;
static Widget legend_box_item;
static Widget legend_boxfill_item;
static Widget *legend_boxfillusing_item;
static Widget *legend_boxfillcolor_item;
static Widget *legend_boxfillpat_item;
static Widget *legend_boxlinew_item;
static Widget *legend_boxlines_item;
static Widget *legend_boxcolor_item;

static void define_symbols(int set_mode);
// static void create_symmisc_frame(Widget w, XtPointer client_data, XtPointer call_data);
static void define_symbols_proc(Widget w, XtPointer client_data, XtPointer call_data);
void setall_colors_proc(Widget w, XtPointer client_data, XtPointer call_data);
void setall_sym_proc(Widget w, XtPointer client_data, XtPointer call_data);
void setall_linew_proc(Widget w, XtPointer client_data, XtPointer call_data);
void set_cset_proc(Widget w, XtPointer client_data, XtPointer call_data);
//static void reset_symleg_proc(void);
static void define_errbar_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void define_errbar_popup(Widget w, XtPointer client_data, XtPointer call_data);
void accept_ledit_proc(Widget w, XtPointer client_data, XtPointer call_data);
void update_ledit_items(int gno);
static void create_ledit_frame(Widget w, XtPointer client_data, XtPointer call_data);

extern void set_plotstr_string(plotstr *pstr, char *buf);

/*
 * define symbols for the current set
 */
static void define_symbols(int set_mode)
{
    int sym, symchar, symskip, symfill, symcolor, symlinew, symlines;
    int line, pen, wid, fill, fillusing, fillpat, fillcol, i;
    double symsize;
    char s[30];
    int value;
    Arg a;

    XtSetArg(a, XmNvalue, &value);
    XtGetValues(symsize_item, &a, 1);
    symsize = value / 100.0;
    sym = (int)GetChoice(toggle_symbols_item);
    pen = (int)GetChoice(toggle_color_item);
    wid = (int)GetChoice(toggle_width_item);
    line = (int)GetChoice(toggle_lines_item);
    fill = (int)GetChoice(toggle_fill_item);
    fillusing = (int)GetChoice(toggle_fillusing_item) ? PATTERN : COLOR;
    fillpat = (int)GetChoice(toggle_fillpat_item);
    fillcol = (int)GetChoice(toggle_fillcol_item);
    symskip = atoi((char *)xv_getstr(symskip_item));
    symfill = (int)GetChoice(symfill_item);
    symcolor = (int)GetChoice(symcolor_item);
    symlinew = (int)GetChoice(symlinew_item);
    symlines = (int)GetChoice(symlines_item);
    strcpy(s, (char *)xv_getstr(symchar_item));
    symchar = s[0];
    if (ismaster)
    {
        if (set_mode == 0)
            cm->sendCommand_StringMessage(DEFINE_SYMBOLS3, (char *)xv_getstr(legend_str_panel));
        cm->sendCommand_ValuesMessage(DEFINE_SYMBOLS, symskip, symfill, symcolor, symlinew, symlines, symchar, cset, 0, 0, 0);
        cm->sendCommand_ValuesMessage(DEFINE_SYMBOLS2, value, sym, pen, wid, line, fill, fillusing, fillpat, fillcol, set_mode);
    }
    if (set_mode == 0)
    {
        g[cg].p[cset].symskip = symskip;
        g[cg].p[cset].symsize = symsize;
        g[cg].p[cset].symchar = symchar;
        g[cg].p[cset].symfill = symfill;
        g[cg].p[cset].symlinew = symlinew;
        g[cg].p[cset].symlines = symlines;
        g[cg].p[cset].fill = fill;
        g[cg].p[cset].fillusing = fillusing;
        g[cg].p[cset].fillpattern = fillpat;
        g[cg].p[cset].fillcolor = fillcol;
        set_plotstr_string(&g[cg].l.str[cset], (char *)xv_getstr(legend_str_panel));
        setplotsym(cg, cset, sym);
        setplotlines(cg, cset, line);
        setplotlinew(cg, cset, wid);
        setplotcolor(cg, cset, pen);
        setplotsymcolor(cg, cset, symcolor);
    }
    else
    {
        for (i = 0; i < g[cg].maxplot; i++)
        {
            if (isactive(cg, i))
            {
                g[cg].p[i].symskip = symskip;
                g[cg].p[i].symsize = symsize;
                g[cg].p[i].symchar = symchar;
                g[cg].p[i].symfill = symfill;
                g[cg].p[i].symlinew = symlinew;
                g[cg].p[i].symlines = symlines;
                g[cg].p[i].fill = fill;
                g[cg].p[i].fillusing = fillusing;
                g[cg].p[i].fillpattern = fillpat;
                g[cg].p[i].fillcolor = fillcol;
                setplotsym(cg, i, sym);
                setplotlines(cg, i, line);
                setplotlinew(cg, i, wid);
                setplotcolor(cg, i, pen);
                setplotsymcolor(cg, i, symcolor);
            }
        }
    }
    updatesymbols(cg, cset);
    drawgraph();
}

static void define_symbols_proc(Widget, XtPointer, XtPointer)
{
    define_symbols((int)GetChoice(symbols_apply_item));
}

/*
 * define colors incrementally
 */
void setall_colors_proc(Widget, XtPointer, XtPointer)
{

    printf("DANIELA TEST\n");
    int i;
    if (ismaster)
    {
        cm->sendCommandMessage(SETALL_COLORS_PROC, 0, 0);
    }
    for (i = 0; i < g[cg].maxplot; i++)
    {
        if (isactive(cg, i))
        {
            setplotcolor(cg, i, (i % 14) + 1);
        }
    }
    updatesymbols(cg, cset);
    drawgraph();
}

/*
 * define symbols incrementally mod 10
 */
void setall_sym_proc(Widget, XtPointer, XtPointer)
{
    int i;
    if (ismaster)
    {
        cm->sendCommandMessage(SETALL_SYM_PROC, 0, 0);
    }

    for (i = 0; i < g[cg].maxplot; i++)
    {
        if (isactive(cg, i))
        {
            setplotsym(cg, i, (i % 10) + 2);
        }
    }
    updatesymbols(cg, cset);
    drawgraph();
}

/*
 * define linewidths incrementally mod 7
 */
void setall_linew_proc(Widget, XtPointer, XtPointer)
{
    int i;
    if (ismaster)
    {
        cm->sendCommandMessage(SETALL_LINEW_PROC, 0, 0);
    }

    for (i = 0; i < g[cg].maxplot; i++)
    {
        if (isactive(cg, i))
        {
            setplotlinew(cg, i, (i % 7) + 1);
        }
    }
    updatesymbols(cg, cset);
    drawgraph();
}

/*
 * freshen up symbol items, generally after a parameter
 * file has been read
 */
void updatesymbols(int gno, int value)
{
    Arg a;
    int iv;
    char s[2], val[24];

    if (define_symbols_frame && cset == value)
    {
        iv = (int)(100.0 * g[gno].p[value].symsize);
        XtSetArg(a, XmNvalue, iv);
        XtSetValues(symsize_item, &a, 1);
        if (value < maxplot)
        {
            /*
                SetChoice(toggle_symset_item, value);
         */
        }
        sprintf(val, "%d", g[gno].p[value].symskip);
        xv_setstr(symskip_item, val);
        SetChoice(symfill_item, g[gno].p[value].symfill);
        if (g[gno].p[value].symchar > ' ' && g[gno].p[value].symchar < 127)
        {
            s[0] = g[gno].p[value].symchar;
            s[1] = 0;
        }
        else
        {
            s[0] = 0;
        }
        xv_setstr(symchar_item, s);
        SetChoice(toggle_symbols_item, getsetplotsym(gno, value));
        SetChoice(symcolor_item, getsetplotsymcolor(gno, value));
        SetChoice(symlinew_item, g[gno].p[value].symlinew);
        SetChoice(symlines_item, g[gno].p[value].symlines);
        SetChoice(toggle_color_item, getsetcolor(gno, value));
        SetChoice(toggle_width_item, getsetlinew(gno, value));
        SetChoice(toggle_lines_item, getsetlines(gno, value));
        SetChoice(toggle_fill_item, g[gno].p[value].fill);
        SetChoice(toggle_fillusing_item, g[gno].p[value].fillusing == COLOR ? 0 : 1);
        SetChoice(toggle_fillcol_item, g[gno].p[value].fillcolor);
        SetChoice(toggle_fillpat_item, g[gno].p[value].fillpattern);
        updatelegendstr(gno);
        update_ledit_items(gno);
        updateerrbar(gno, value);
    }
}

void set_cset_proc(Widget, XtPointer, XtPointer)
{
    int cd = GetSelectedSet(toggle_symset_item);
    if (ismaster)
    {
        cm->sendCommandMessage(SET_CSET_PROC, 0, 0);
    }
    if (cd != SET_SELECT_ERROR)
    {
        cset = cd;
        updatesymbols(cg, cd);
    }
    else
    {
    }
}

/*
 * legends
 */

// static int firstrun = TRUE;

/*
 * freshen up legend items, generally after a parameter
 * file has been read
 */

void updatelegendstr(int gno)
{
    if (define_symbols_frame)
    {
        xv_setstr(legend_str_panel, g[gno].l.str[cset].s);
    }
}

/*
 * Cancel proc
 */
/*
static void reset_symleg_proc(void)
{
} */

/*
 * create the symbols popup
 */
void define_symbols_popup(Widget, XtPointer, XtPointer)
{
    Widget dialog, wbut, fr, rc, rc2, rc3;
    //Widget wlabel;
    int i, x, y;
    Widget buts[5];

    set_wait_cursor();
    if (define_symbols_frame == NULL)
    {
        char *label1[5];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Error bars...";
        label1[2] = (char *)"Legends...";
        label1[3] = (char *)"Close";
        /*
         label1[3] = "Misc...";
      */
        XmGetPos(app_shell, 0, &x, &y);
        define_symbols_frame = XmCreateDialogShell(app_shell, (char *)"Symbols/legends", NULL, 0);
        handle_close(define_symbols_frame);
        XtVaSetValues(define_symbols_frame, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(define_symbols_frame, (char *)"symbols_rc", NULL, 0);

        /*
         toggle_symset_item = CreateSetChoice(dialog, "Select set:", maxplot, 0);
      */

        toggle_symset_item = CreateSetSelector(dialog, "Select set:",
                                               SET_SELECT_ACTIVE,
                                               FILTER_SELECT_NONE,
                                               GRAPH_SELECT_CURRENT,
                                               SELECTION_TYPE_SINGLE);
        XtVaSetValues(toggle_symset_item.list,
                      XmNselectionPolicy, XmSINGLE_SELECT,
                      NULL);
        XtAddCallback(toggle_symset_item.list, XmNsingleSelectionCallback,
                      (XtCallbackProc)set_cset_proc, (XtPointer)0);

        /*
         for (i = 0; i < maxplot; i++) {
             XtAddCallback(toggle_symset_item[2 + i], XmNactivateCallback, (XtCallbackProc) set_cset_proc, (XtPointer) i);
         }
      */

        /*
          wlabel = XtVaCreateManagedWidget("Symbol:", xmLabelWidgetClass, dialog,
                 NULL);
      */

        rc2 = XmCreateRowColumn(dialog, (char *)"rc", NULL, 0);
        XtVaSetValues(rc2, XmNorientation, XmHORIZONTAL, NULL);

        fr = XtVaCreateManagedWidget("symframe", xmFrameWidgetClass, rc2,
                                     NULL);
        rc = XtVaCreateManagedWidget("symbolsbb", xmRowColumnWidgetClass, fr,
                                     NULL);
        toggle_symbols_item = CreatePanelChoice0(rc,
                                                 " ", 4,
                                                 44,
                                                 "No symbol", /* 0 */
                                                 "Dot", /* 1 */
                                                 "Circle", /* 2 */
                                                 "Square", /* 3 */
                                                 "Diamond", /* 4 */
                                                 "Triangle up", /* 5 */
                                                 "Triangle left", /* 6 */
                                                 "Triangle down", /* 7 */
                                                 "Triangle right", /* 8 */
                                                 "Plus", /* 9 */
                                                 "X", /* 10 */
                                                 "Star", /* 11 */
                                                 "Impulse at X", /* 12 */
                                                 "Impulse at Y", /* 13 */
                                                 "Vert line at X", /* 14 */
                                                 "Horiz line at Y", /* 15 */
                                                 "Histogram X", /* 16 */
                                                 "Histogram Y", /* 17 */
                                                 "Stair step X", /* 18 */
                                                 "Stair step Y", /* 19 */
                                                 "Bar X", /* 20 */
                                                 "Bar Y", /* 21 */
                                                 "Range", /* 22 */
                                                 "Loc", /* 23 */
                                                 "Set #", /* 24 */
                                                 "Set #, loc", /* 25 */
                                                 "*Bar and whisker", /* 26 */
                                                 "Segments", /* 27 */
                                                 "Character", /* 28 */
                                                 "Tag first point", /* 29 */
                                                 "Tag last point", /* 30 */
                                                 "Tag center point", /* 31 */
                                                 "*String (n/a)", /* 32 */
                                                 "Hi low X", /* 33 */
                                                 "Hi low Y", /* 34 */
                                                 "Open/close X", /* 35 */
                                                 "Open/close Y", /* 36 */
                                                 "Box plot X", /* 37 */
                                                 "Box plot Y", /* 38 */
                                                 "Average Y", /* 39 */
                                                 "Average Y+-1*std", /* 40 */
                                                 "Average Y+-2*std", /* 41 */
                                                 "Average Y+-3*std", /* 42 */
                                                 /*
                         "Average Y+-value", 43
                         "Median Y",	 44
                         "Geom. mean Y", 45
                         "Harm. mean Y", 46
      */
                                                 0,
                                                 0);

        symfill_item = CreatePanelChoice(rc,
                                         "Sym fill:",
                                         4,
                                         "None", "Filled", "Opaque",
                                         NULL,
                                         NULL);

        //wlabel =
        XtVaCreateManagedWidget("Sym size:", xmLabelWidgetClass, rc, NULL);
        symsize_item = XtVaCreateManagedWidget("SymSize", xmScaleWidgetClass, rc,
                                               XmNminimum, 0,
                                               XmNmaximum, 800,
                                               XmNvalue, 100,
                                               XmNshowValue, True,
                                               XmNprocessingDirection, XmMAX_ON_RIGHT,
                                               XmNorientation, XmHORIZONTAL,
                                               NULL);
        symcolor_item = CreateColorChoice(rc, "Color:", 0);

        symchar_item = CreateTextItem2(rc, 2, "Sym char:");
        symskip_item = CreateTextItem2(rc, 5, "Sym skip:");

        /*
         symskip_item = CreatePanelChoice0(rc,
                       "Sym skip:", 3, 19,
                       "None",
                 "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                 "20", "50", "100", "500", "1000", "5000", "10000",
                       NULL,
                       0);
      */
        symlines_item = CreatePanelChoice(rc, "Style:",
                                          7,
                                          "None",
                                          "Solid line",
                                          "Dotted line",
                                          "Dashed line",
                                          "Long Dashed",
                                          "Dot-dashed",
                                          NULL,
                                          0);
        symlinew_item = CreatePanelChoice(rc, "Width:",
                                          11,
                                          "None",
                                          "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                          NULL,
                                          0);
        XtManageChild(rc);

        rc3 = XmCreateRowColumn(rc2, (char *)"rc", NULL, 0);
        fr = XtVaCreateManagedWidget("lineframe", xmFrameWidgetClass, rc3,
                                     NULL);
        rc = XtVaCreateManagedWidget("linesrc", xmRowColumnWidgetClass, fr,
                                     NULL);
        //wlabel =
        XtVaCreateManagedWidget("Line properties:", xmLabelWidgetClass, rc,
                                NULL);
        toggle_lines_item = CreatePanelChoice(rc, "Style:",
                                              7,
                                              "None",
                                              "Solid line",
                                              "Dotted line",
                                              "Dashed line",
                                              "Long Dashed",
                                              "Dot-dashed",
                                              NULL,
                                              0);
        toggle_width_item = CreatePanelChoice(rc, "Width:",
                                              11,
                                              "None",
                                              "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                              NULL,
                                              0);
        toggle_color_item = CreateColorChoice(rc, "Color:", 0);
        XtManageChild(rc);

        fr = XtVaCreateManagedWidget("fillframe", xmFrameWidgetClass, rc3,
                                     NULL);
        rc = XtVaCreateManagedWidget("fillsbb", xmRowColumnWidgetClass, fr,
                                     NULL);
        //wlabel =
        XtVaCreateManagedWidget("Fill properties:", xmLabelWidgetClass, rc,
                                NULL);
        toggle_fill_item = CreatePanelChoice(rc, "Fill: ",
                                             9,
                                             "None",
                                             "As polygon",
                                             "To Y=0.0",
                                             "To X=0.0",
                                             "To X min",
                                             "To X max",
                                             "To Y min",
                                             "To Y max",
                                             0,
                                             0);
        toggle_fillusing_item = CreatePanelChoice(rc, "Using:",
                                                  3,
                                                  "Colors",
                                                  "Patterns",
                                                  0,
                                                  0);
        toggle_fillpat_item = CreatePanelChoice0(rc,
                                                 "Pattern:", 4,
                                                 17,
                                                 "0",
                                                 "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                                                 "11", "12", "13", "14", "15",
                                                 NULL,
                                                 0);
        toggle_fillcol_item = CreateColorChoice(rc, "Color:", 0);
        XtManageChild(rc);
        XtManageChild(rc3);
        XtManageChild(rc2);

        if (!g[cg].l.active)
        {
            for (i = 0; i < MAXPLOT; i++)
            {
                g[cg].l.str[i].s[0] = '\0';
            }
        }
        legend_str_panel = CreateTextItem2(dialog, 30, "Legend:");
        symbols_apply_item = CreatePanelChoice(dialog,
                                               "Apply to:",
                                               3,
                                               "selected set",
                                               "all sets",
                                               NULL,
                                               NULL);

        rc2 = XmCreateRowColumn(dialog, (char *)"rc", NULL, 0);
        XtVaSetValues(rc2, XmNorientation, XmHORIZONTAL, NULL);
        wbut = XtVaCreateManagedWidget("All colors", xmPushButtonWidgetClass, rc2, NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)setall_colors_proc, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("All symbols", xmPushButtonWidgetClass, rc2, NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)setall_sym_proc, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("All linewidths", xmPushButtonWidgetClass, rc2, NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)setall_linew_proc, (XtPointer)0);

        XtManageChild(rc2);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 4, buts, label1);

        XtAddCallback(buts[0], XmNactivateCallback,
                      (XtCallbackProc)define_symbols_proc, (XtPointer)0);
        XtAddCallback(buts[1], XmNactivateCallback,
                      (XtCallbackProc)define_errbar_popup, (XtPointer)0);
        XtAddCallback(buts[2], XmNactivateCallback,
                      (XtCallbackProc)define_legend_popup, (XtPointer)0);
        XtAddCallback(buts[3], XmNactivateCallback,
                      (XtCallbackProc)destroy_dialog, (XtPointer)define_symbols_frame);
        /*
         XtAddCallback(buts[4], XmNactivateCallback,
                  (XtCallbackProc) create_symmisc_frame, (XtPointer) 0);
      */

        XtManageChild(dialog);
    }
    XtRaise(define_symbols_frame);
    updatesymbols(cg, cset);
    unset_wait_cursor();
}

/*
 * legend popup
 */
void updatelegends(int gno)
{
    Arg a;
    int iv;

    if (define_legend_frame)
    {
        iv = (int)(100.0 * g[gno].l.charsize);
        XtSetArg(a, XmNvalue, iv);
        XtSetValues(legend_charsize_item, &a, 1);
        XmToggleButtonSetState(toggle_legends_item, g[gno].l.active == ON, False);
        sprintf(buf, "%.9lg", g[gno].l.legx);
        xv_setstr(legend_x_panel, buf);
        sprintf(buf, "%.9lg", g[gno].l.legy);
        xv_setstr(legend_y_panel, buf);
        SetChoice(legends_gap_item, g[gno].l.vgap - 1);
        SetChoice(legends_len_item, g[gno].l.len - 1);
        SetChoice(toggle_legendloc_item, g[gno].l.loctype == VIEW);
        SetChoice(legend_font_item, g[gno].l.font);
        SetChoice(legend_color_item, g[gno].l.color);
        XmToggleButtonSetState(legend_box_item, g[gno].l.box == ON, False);
        XmToggleButtonSetState(legend_boxfill_item, g[gno].l.boxfill == ON, False);
        SetChoice(legend_boxfillusing_item, g[gno].l.boxfillusing == PATTERN);
        SetChoice(legend_boxfillcolor_item, g[gno].l.boxfillcolor);
        SetChoice(legend_boxfillpat_item, g[gno].l.boxfillpat);
        SetChoice(legend_boxcolor_item, g[gno].l.boxlcolor);
        SetChoice(legend_boxlinew_item, g[gno].l.boxlinew - 1);
        SetChoice(legend_boxlines_item, g[gno].l.boxlines - 1);
    }
}

/*
 * define legends for the current graph
 */
void define_legends_proc(Widget, XtPointer, XtPointer)
{
    Arg a;
    char val[80];
    int value;

    if (define_legend_frame)
    {
        XtSetArg(a, XmNvalue, &value);
        XtGetValues(legend_charsize_item, &a, 1);
        g[cg].l.charsize = value / 100.0;
        g[cg].l.active = XmToggleButtonGetState(toggle_legends_item) ? ON : OFF;
        g[cg].l.vgap = GetChoice(legends_gap_item) + 1;
        g[cg].l.len = (int)GetChoice(legends_len_item) + 1;
        g[cg].l.loctype = (int)GetChoice(toggle_legendloc_item) ? VIEW : WORLD;
        strcpy(val, (char *)xv_getstr(legend_x_panel));
        g[cg].l.legx = atof(val);
        strcpy(val, (char *)xv_getstr(legend_y_panel));
        g[cg].l.legy = atof(val);
        g[cg].l.font = (int)GetChoice(legend_font_item);
        g[cg].l.color = (int)GetChoice(legend_color_item);
        g[cg].l.box = XmToggleButtonGetState(legend_box_item) ? ON : OFF;
        g[cg].l.boxfill = XmToggleButtonGetState(legend_boxfill_item) ? ON : OFF;
        g[cg].l.boxfillusing = (int)GetChoice(legend_boxfillusing_item) ? PATTERN : COLOR;
        g[cg].l.boxfillcolor = (int)GetChoice(legend_boxfillcolor_item);
        g[cg].l.boxfillpat = (int)GetChoice(legend_boxfillpat_item);
        g[cg].l.boxlcolor = (int)GetChoice(legend_boxcolor_item);
        g[cg].l.boxlinew = (int)GetChoice(legend_boxlinew_item) + 1;
        g[cg].l.boxlines = (int)GetChoice(legend_boxlines_item) + 1;
        update_ledit_items(cg);
        if (ismaster)
        {
            cm->sendCommand_FloatMessage(DEFINE_LEGENDS_PROC, g[cg].l.legx, g[cg].l.legy, g[cg].l.charsize, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            cm->sendCommand_ValuesMessage(DEFINE_LEGENDS_PROC, 0, g[cg].l.active, g[cg].l.vgap, g[cg].l.len, g[cg].l.loctype, 0, 0, g[cg].l.font, g[cg].l.color, 0);
            cm->sendCommand_ValuesMessage(DEFINE_LEGENDS_PROC2, g[cg].l.box, g[cg].l.boxfill, g[cg].l.boxfillusing, g[cg].l.boxfillcolor, g[cg].l.boxfillpat, g[cg].l.boxlcolor, g[cg].l.boxlinew, g[cg].l.boxlines, 0, 0);
        }
    }
    drawgraph();
}

/*
 * activate the legend location flag
 */
void legend_loc_proc(Widget, XtPointer, XtPointer)
{
    if (define_legend_frame)
    {
        g[cg].l.loctype = (int)GetChoice(toggle_legendloc_item) ? VIEW : WORLD;
    }
    set_action(0);
    set_action(LEG_LOC);
}

/*
 * load legend strings from set comments
 */
void legend_load_proc(Widget, XtPointer, XtPointer)
{
    int i;
    if (ismaster)
    {
        cm->sendCommandMessage(LEGEND_LOAD_PROC, 0, 0);
    }

    for (i = 0; i < MAXPLOT; i++)
    {
        if (isactive(cg, i))
        {
            set_plotstr_string(&g[cg].l.str[i], g[cg].p[i].comments);
        }
    }
    update_ledit_items(cg);
}

/*
 * create the legend popup
 */
void define_legend_popup(Widget, XtPointer, XtPointer)
{
    Widget fr, rc, rc0, rc1;
    int x, y;
    Widget buts[5];
    set_wait_cursor();
    if (define_legend_frame == NULL)
    {
        char *label1[5];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Place";
        label1[2] = (char *)"Comments";
        label1[3] = (char *)"Edit...";
        label1[4] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        define_legend_frame = XmCreateDialogShell(app_shell, (char *)"Legends", NULL, 0);
        handle_close(define_legend_frame);
        XtVaSetValues(define_legend_frame, XmNx, x, XmNy, y, NULL);
        define_legend_panel = XmCreateRowColumn(define_legend_frame, (char *)"legend_rc", NULL, 0);

        rc0 = XmCreateRowColumn(define_legend_panel, (char *)"rc0", NULL, 0);
        XtVaSetValues(rc0, XmNorientation, XmHORIZONTAL, NULL);

        fr = XtVaCreateManagedWidget("frame", xmFrameWidgetClass, rc0, NULL);
        rc = XtVaCreateManagedWidget("rc", xmRowColumnWidgetClass, fr, NULL);
        toggle_legends_item = XtVaCreateManagedWidget("Display legend",
                                                      xmToggleButtonWidgetClass, rc,
                                                      NULL);

        toggle_legendloc_item = CreatePanelChoice(rc, "Locate in:",
                                                  3,
                                                  "World coords",
                                                  "Viewport coords",
                                                  0, 0);
        legend_font_item = CreatePanelChoice(rc, "Font:",
                                             11,
                                             "Times-Roman", "Times-Bold", "Times-Italic",
                                             "Times-BoldItalic", "Helvetica",
                                             "Helvetica-Bold", "Helvetica-Oblique",
                                             "Helvetica-BoldOblique", "Greek", "Symbol",
                                             0,
                                             0);

        XtVaCreateManagedWidget("Char size:", xmLabelWidgetClass, rc,
                                NULL);

        legend_charsize_item = XtVaCreateManagedWidget("charsize", xmScaleWidgetClass, rc,
                                                       XmNminimum, 0,
                                                       XmNmaximum, 400,
                                                       XmNvalue, 100,
                                                       XmNshowValue, True,
                                                       XmNprocessingDirection, XmMAX_ON_RIGHT,
                                                       XmNorientation, XmHORIZONTAL,
                                                       NULL);
        legend_color_item = CreateColorChoice(rc, "Color:", 0);

        legends_gap_item = CreatePanelChoice(rc, "Legend gap:",
                                             5,
                                             "1", "2", "3", "4",
                                             0, 0);
        legends_len_item = CreatePanelChoice(rc, "Legend length:",
                                             9,
                                             "1", "2", "3", "4", "5", "6", "7", "8",
                                             0, 0);
        legend_x_panel = CreateTextItem2(rc, 10, "X location:");
        legend_y_panel = CreateTextItem2(rc, 10, "Y location:");
        XtManageChild(fr);
        XtManageChild(rc);

        rc1 = XtVaCreateManagedWidget("rc", xmRowColumnWidgetClass,
                                      rc0, NULL);
        fr = XtVaCreateManagedWidget("frame", xmFrameWidgetClass, rc1,
                                     NULL);
        rc = XtVaCreateManagedWidget("rc", xmRowColumnWidgetClass, fr,
                                     NULL);
        legend_box_item = XtVaCreateManagedWidget("Legend frame",
                                                  xmToggleButtonWidgetClass, rc,
                                                  NULL);
        legend_boxcolor_item = CreateColorChoice(rc, "Line color:", 0);
        legend_boxlinew_item = CreatePanelChoice(rc, "Line width:",
                                                 10,
                                                 "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                                 NULL,
                                                 NULL);
        legend_boxlines_item = CreatePanelChoice(rc,
                                                 "Line style:",
                                                 6,
                                                 "Solid",
                                                 "Dotted",
                                                 "Dashed",
                                                 "Long Dashed",
                                                 "Dot-dashed",
                                                 NULL,
                                                 NULL);
        XtManageChild(fr);
        XtManageChild(rc);

        fr = XtVaCreateManagedWidget("frame", xmFrameWidgetClass, rc1,
                                     NULL);
        rc = XtVaCreateManagedWidget("rc", xmRowColumnWidgetClass, fr,
                                     NULL);
        legend_boxfill_item = XtVaCreateManagedWidget("Fill frame", xmToggleButtonWidgetClass, rc,
                                                      NULL);
        legend_boxfillusing_item = CreatePanelChoice(rc, "Fill with:",
                                                     3,
                                                     "Color", "Pattern",
                                                     NULL,
                                                     NULL);
        legend_boxfillcolor_item = CreateColorChoice(rc, "Color:", 0);

        legend_boxfillpat_item = CreatePanelChoice0(rc,
                                                    "Pattern:", 4, 17,
                                                    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                                                    "11", "12", "13", "14", "15",
                                                    NULL, 0);
        XtManageChild(fr);
        XtManageChild(rc);

        XtManageChild(rc0);

        CreateCommandButtons(define_legend_panel, 5, buts, label1);
        XtAddCallback(buts[0], XmNactivateCallback,
                      (XtCallbackProc)define_legends_proc, (XtPointer)0);
        XtAddCallback(buts[1], XmNactivateCallback,
                      (XtCallbackProc)legend_loc_proc, (XtPointer)0);
        XtAddCallback(buts[2], XmNactivateCallback,
                      (XtCallbackProc)legend_load_proc, (XtPointer)0);
        XtAddCallback(buts[3], XmNactivateCallback,
                      (XtCallbackProc)create_ledit_frame, (XtPointer)0);
        XtAddCallback(buts[4], XmNactivateCallback,
                      (XtCallbackProc)destroy_dialog, (XtPointer)define_legend_frame);

        XtManageChild(define_legend_panel);
    }
    XtRaise(define_legend_frame);
    updatelegends(cg);
    unset_wait_cursor();
}

/*
 * define errbars for the current set
 */
static void define_errbar_proc(Widget, XtPointer, XtPointer)
{
    int i, itmp, applyto, nstart, nstop;
    Arg a;
    int value;

    applyto = GetChoice(errbar_apply_item);
    if (applyto)
    {
        nstart = 0;
        nstop = g[cg].maxplot - 1;
    }
    else
    {
        nstart = nstop = cset;
    }
    for (i = nstart; i <= nstop; i++)
    {

        XtSetArg(a, XmNvalue, &value);
        XtGetValues(errbar_size_item, &a, 1);
        g[cg].p[i].errbarper = value / 100.0;

        itmp = (int)GetChoice(errbar_type_item);
        switch (dataset_type(cg, i))
        {
        case XYDX:
        case XYDXDX:
            if (itmp == 0)
            {
                itmp = BOTH;
                ;
            }
            else if (itmp == 1)
            {
                itmp = LEFT;
            }
            else
            {
                itmp = RIGHT;
            }
            break;
        case XYDY:
        case XYDYDY:
            if (itmp == 0)
            {
                itmp = BOTH;
                ;
            }
            else if (itmp == 1)
            {
                itmp = TOP;
            }
            else
            {
                itmp = BOTTOM;
            }
            break;
        default:
            itmp = BOTH;
            break;
        }

        g[cg].p[i].errbarxy = itmp;
        g[cg].p[i].errbar_linew = (int)GetChoice(errbar_width_item) + 1;
        g[cg].p[i].errbar_lines = (int)GetChoice(errbar_lines_item) + 1;
        g[cg].p[i].errbar_riser = (int)GetChoice(errbar_riser_item) ? ON : OFF;
        g[cg].p[i].errbar_riser_linew = (int)GetChoice(errbar_riserlinew_item) + 1;
        g[cg].p[i].errbar_riser_lines = (int)GetChoice(errbar_riserlines_item) + 1;
        if (ismaster)
        {
            cm->sendCommand_FloatMessage(DEFINE_ERRBAR_PROC, g[cg].p[i].errbarper, (double)i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            cm->sendCommand_ValuesMessage(DEFINE_ERRBAR_PROC, 0, g[cg].p[i].errbarxy, g[cg].p[i].errbar_linew, g[cg].p[i].errbar_lines, g[cg].p[i].errbar_riser, g[cg].p[i].errbar_riser_linew, g[cg].p[i].errbar_riser_lines, i, i == nstop, 0);
        }
    }
    drawgraph();
}

/*
 */
void updateerrbar(int gno, int value)
{
    int itmp = 0;
    Arg a;
    int iv;

    if (value == -1)
    {
        value = cset;
    }
    if (define_errbar_frame && cset == value)
    {

        iv = (int)(100.0 * g[gno].p[value].errbarper);
        XtSetArg(a, XmNvalue, iv);
        XtSetValues(errbar_size_item, &a, 1);

        switch (g[gno].p[value].errbarxy)
        {
        case BOTH:
            itmp = 0;
            break;
        case TOP:
        case LEFT:
            itmp = 1;
            break;
        case BOTTOM:
        case RIGHT:
            itmp = 2;
            break;
        }
        SetChoice(errbar_type_item, itmp);
        SetChoice(errbar_width_item, g[gno].p[value].errbar_linew - 1);
        SetChoice(errbar_lines_item, g[gno].p[value].errbar_lines - 1);
        SetChoice(errbar_riser_item, g[gno].p[value].errbar_riser == ON ? 1 : 0);
        SetChoice(errbar_riserlinew_item, g[gno].p[value].errbar_riser_linew - 1);
        SetChoice(errbar_riserlines_item, g[gno].p[value].errbar_riser_lines - 1);
    }
}

/*
 * create the errbar popup
 */
static void define_errbar_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget buts[2];
    set_wait_cursor();
    if (define_errbar_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        define_errbar_frame = XmCreateDialogShell(app_shell, (char *)"Error bars", NULL, 0);
        handle_close(define_errbar_frame);
        XtVaSetValues(define_errbar_frame, XmNx, x, XmNy, y, NULL);
        define_errbar_panel = XmCreateRowColumn(define_errbar_frame, (char *)"errbar_rc", NULL, 0);

        XtVaCreateManagedWidget("Size:", xmLabelWidgetClass, define_errbar_panel,
                                NULL);
        errbar_size_item = XtVaCreateManagedWidget("Size", xmScaleWidgetClass, define_errbar_panel,
                                                   XmNminimum, 0,
                                                   XmNmaximum, 400,
                                                   XmNvalue, 100,
                                                   XmNshowValue, True,
                                                   XmNprocessingDirection, XmMAX_ON_RIGHT,
                                                   XmNorientation, XmHORIZONTAL,
                                                   NULL);

        errbar_width_item = CreatePanelChoice(define_errbar_panel, "Line width:",
                                              10,
                                              "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                              NULL,
                                              NULL);
        errbar_lines_item = CreatePanelChoice(define_errbar_panel,
                                              "Line style:",
                                              6,
                                              "Solid",
                                              "Dotted",
                                              "Dashed",
                                              "Long Dashed",
                                              "Dot-dashed",
                                              NULL,
                                              NULL);
        errbar_riser_item = CreatePanelChoice(define_errbar_panel,
                                              "Riser:",
                                              3, "OFF", "ON", NULL,
                                              NULL);
        errbar_riserlinew_item = CreatePanelChoice(define_errbar_panel,
                                                   "Riser line width:",
                                                   10,
                                                   "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                                   NULL,
                                                   NULL);
        errbar_riserlines_item = CreatePanelChoice(define_errbar_panel,
                                                   "Riser line style:",
                                                   6,
                                                   "Solid",
                                                   "Dotted",
                                                   "Dashed",
                                                   "Long Dashed",
                                                   "Dot-dashed",
                                                   NULL,
                                                   NULL);
        errbar_type_item = CreatePanelChoice(define_errbar_panel,
                                             "Display:",
                                             4,
                                             "Both", "Top/left", "Bottom/right",
                                             NULL,
                                             NULL);
        errbar_apply_item = CreatePanelChoice(define_errbar_panel,
                                              "Apply to:",
                                              3,
                                              "selected set",
                                              "all sets",
                                              NULL,
                                              NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, define_errbar_panel, NULL);

        CreateCommandButtons(define_errbar_panel, 2, buts, label1);
        XtAddCallback(buts[0], XmNactivateCallback,
                      (XtCallbackProc)define_errbar_proc, (XtPointer)0);
        XtAddCallback(buts[1], XmNactivateCallback,
                      (XtCallbackProc)destroy_dialog, (XtPointer)define_errbar_frame);

        XtManageChild(define_errbar_panel);
    }
    updateerrbar(cg, cset);
    XtRaise(define_errbar_frame);
    unset_wait_cursor();
}

static Widget ledit_frame;
static Widget ledit_panel;
static Widget leglabel[MAXPLOT];

void accept_ledit_proc(Widget, XtPointer, XtPointer)
{
    int i;

    for (i = 0; i < maxplot; i++)
    {
        if (ismaster)
        {
            set_plotstr_string(&g[cg].l.str[i], (char *)xv_getstr(leglabel[i]));
            cm->sendCommand_StringMessage(ACCEPT_LEDIT_PROC, (char *)xv_getstr(leglabel[i]));
            cm->sendCommandMessage(ACCEPT_LEDIT_PROC, cg, i);
        }
    }
    updatelegendstr(cg);
    drawgraph();
}

void update_ledit_items(int)
{
    int i;

    if (ledit_frame)
    {
        for (i = 0; i < maxplot; i++)
        {
            xv_setstr(leglabel[i], g[cg].l.str[i].s);
        }
    }
}

static void create_ledit_frame(Widget, XtPointer, XtPointer)
{
    Widget wbut, rc, sw;
    int i, x, y;
    char buf[10];
    set_wait_cursor();
    if (ledit_frame == NULL)
    {
        //char *label1[4];
        //label1[0] = "Accept";
        //label1[1] = "Place";
        //label1[2] = "Load comments";
        //label1[3] = "Close";
        XmGetPos(app_shell, 0, &x, &y);
        ledit_frame = XmCreateDialogShell(app_shell, (char *)"Edit legend labels", NULL, 0);
        handle_close(ledit_frame);
        XtVaSetValues(ledit_frame, XmNx, x, XmNy, y, NULL);
        ledit_panel = XmCreateForm(ledit_frame, (char *)"ledit_rc", NULL, 0);

        sw = XtVaCreateManagedWidget("ledit_sw",
                                     xmScrolledWindowWidgetClass, ledit_panel,
                                     XmNscrollingPolicy, XmAUTOMATIC,
                                     NULL);
        rc = XmCreateRowColumn(sw, (char *)"rc", NULL, 0);
        XtVaSetValues(sw,
                      XmNworkWindow, rc,
                      NULL);
        for (i = 0; i < maxplot; i++)
        {
            sprintf(buf, "%2d:", i);
            leglabel[i] = CreateTextItem2(rc, 20, buf);
        }
        XtManageChild(rc);
        XtManageChild(sw);

        rc = XmCreateRowColumn(ledit_panel, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);

        wbut = XtVaCreateManagedWidget("Accept", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)accept_ledit_proc, (XtPointer)0);
        wbut = XtVaCreateManagedWidget("Place", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)legend_loc_proc, (XtPointer)0);
        wbut = XtVaCreateManagedWidget("Load comments", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)legend_load_proc, (XtPointer)0);
        wbut = XtVaCreateManagedWidget("Close", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)ledit_frame);
        XtManageChild(rc);

        XtVaSetValues(rc,
                      XmNleftAttachment, XmATTACH_FORM,
                      XmNrightAttachment, XmATTACH_FORM,
                      XmNbottomAttachment, XmATTACH_FORM,
                      NULL);
        XtVaSetValues(sw,
                      XmNtopAttachment, XmATTACH_FORM,
                      XmNleftAttachment, XmATTACH_FORM,
                      XmNrightAttachment, XmATTACH_FORM,
                      XmNbottomAttachment, XmATTACH_WIDGET,
                      XmNbottomWidget, rc,
                      NULL);

        XtManageChild(ledit_panel);
    }
    XtRaise(ledit_frame);
    update_ledit_items(cg);
    unset_wait_cursor();
}

/*
If you decide to incorporate these into xmgr (I and others here
would be eternally grateful if you did), then I would suggest that
the following should be user-defined quantities:
1) The inner limits (the box, 25th and 75th percentiles in the example),
2) The outer limits (the whiskers, 10th and 90th percentiles in the example),
3) The number of points below which the box plot is unacceptable, and
therefore the points are plotted instead (e.g. 10 in the examples), and
4) Whether to plot outlying points or not.
*/

/*
 * define boxplot for the current set
 */
void define_boxplot_proc(Widget, XtPointer, XtPointer)
{
    int i, applyto, nstart, nstop;
    if (ismaster)
    {
        cm->sendCommandMessage(DEFINE_BOXPLOT_PROC, 0, 0);
    }

    applyto = GetChoice(boxplot_apply_item);
    if (applyto)
    {
        nstart = 0;
        nstop = g[cg].maxplot - 1;
    }
    else
    {
        nstart = nstop = cset;
    }
    for (i = nstart; i <= nstop; i++)
    {
    }
    drawgraph();
}

/*
 */
void updateboxplot(int, int value)
{
    int itmp = 0;

    if (value == -1)
    {
        value = cset;
    }
    if (define_boxplot_frame && cset == value)
    {
        SetChoice(boxplot_type_item, itmp);
    }
}

/*
 * create the boxplot popup
 */
/*
static void define_boxplot_popup(Widget , XtPointer , XtPointer )
{
Widget wbut, rc;
int x, y;
set_wait_cursor();
if (define_boxplot_frame == NULL) {
char *label1[2];
label1[0] = "Accept";
label1[1] = "Close";
XmGetPos(app_shell, 0, &x, &y);
define_boxplot_frame = XmCreateDialogShell(app_shell, "Error bars", NULL, 0);
handle_close(define_boxplot_frame);
XtVaSetValues(define_boxplot_frame, XmNx, x, XmNy, y, NULL);
define_boxplot_panel = XmCreateRowColumn(define_boxplot_frame, "boxplot_rc", NULL, 0);

boxplot_type_item = CreatePanelChoice(define_boxplot_panel,
"Box width is:",
3,
"Symbol size",
"Standard deviation",
NULL,
NULL);

boxplot_outliers_item = XtVaCreateManagedWidget("Display outliers",xmToggleButtonWidgetClass, define_boxplot_panel,NULL);

boxplot_apply_item = CreatePanelChoice(define_boxplot_panel,
"Apply to:",
3,
"selected set",
"all sets",
NULL,
NULL);

XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, define_boxplot_panel,
NULL);

rc = XmCreateRowColumn(define_boxplot_panel, "rc", NULL, 0);
XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
wbut = XtVaCreateManagedWidget("Accept", xmPushButtonWidgetClass, rc,
NULL);
XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc) define_boxplot_proc, (XtPointer) 0);
wbut = XtVaCreateManagedWidget("Close", xmPushButtonWidgetClass, rc,
NULL);
XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc) destroy_dialog, (XtPointer) define_boxplot_frame);
XtManageChild(rc);

XtManageChild(define_boxplot_panel);
}
XtRaise(define_boxplot_frame);
updateboxplot(cg, cset);
unset_wait_cursor();
} */

static Widget define_symmisc_frame;
static Widget *symmisc_apply_item;
// static Widget symmisc_avg_item;
// static Widget symmisc_avgstd_item;
// static Widget symmisc_med_item;
// static Widget symmisc_geommean_item;
// static Widget symmisc_harmmean_item;

void accept_symmisc(Widget, XtPointer, XtPointer)
{
    int i, applyto, nstart, nstop;
    if (ismaster)
    {
        cm->sendCommandMessage(ACCEPT_SYMMISC, 0, 0);
    }

    applyto = GetChoice(symmisc_apply_item);
    if (applyto)
    {
        nstart = 0;
        nstop = g[cg].maxplot - 1;
    }
    else
    {
        nstart = nstop = cset;
    }
    for (i = nstart; i <= nstop; i++)
    {
    }
    drawgraph();
}

void updatesymmisc(int)
{
    if (define_symmisc_frame)
    {
    }
}

/*
static void create_symmisc_frame(Widget , XtPointer, XtPointer )
{
    Widget define_symmisc_panel;
    Widget buts[2];
    int x, y;
    set_wait_cursor();
    if (define_symmisc_frame == NULL) {
   char *label1[2];
   label1[0] = "Accept";
   label1[1] = "Close";
XmGetPos(app_shell, 0, &x, &y);
define_symmisc_frame = XmCreateDialogShell(app_shell, "Misc", NULL, 0);
handle_close(define_symmisc_frame);
XtVaSetValues(define_symmisc_frame, XmNx, x, XmNy, y, NULL);
define_symmisc_panel = XmCreateRowColumn(define_symmisc_frame, "symmisc_rc", NULL, 0);

symmisc_avg_item = XtVaCreateManagedWidget("Display average",
xmToggleButtonWidgetClass, define_symmisc_panel,
NULL);
symmisc_avgstd_item = XtVaCreateManagedWidget("Display average+-standard deviation",
xmToggleButtonWidgetClass, define_symmisc_panel,
NULL);
symmisc_med_item = XtVaCreateManagedWidget("Display median",
xmToggleButtonWidgetClass, define_symmisc_panel,
NULL);
symmisc_geommean_item = XtVaCreateManagedWidget("Display geometric mean",
xmToggleButtonWidgetClass, define_symmisc_panel,
NULL);
symmisc_harmmean_item = XtVaCreateManagedWidget("Display harmonic mean",
xmToggleButtonWidgetClass, define_symmisc_panel,
NULL);

XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, define_symmisc_panel,
NULL);

CreateCommandButtons(define_symmisc_panel, 2, buts, label1);
XtAddCallback(buts[0], XmNactivateCallback, (XtCallbackProc) accept_symmisc, (XtPointer) 0);
XtAddCallback(buts[1], XmNactivateCallback, (XtCallbackProc) destroy_dialog, (XtPointer) define_symmisc_frame);

XtManageChild(define_symmisc_panel);
}
XtRaise(define_symmisc_frame);
updatesymmisc(cg);
unset_wait_cursor();
} */
