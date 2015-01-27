/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: labelwin.c,v 1.3 1994/09/29 03:37:37 pturner Exp pturner $
 *
 * label Panel
 *
 */

#include <stdio.h>

#include <Xm/Xm.h>
#include <Xm/BulletinB.h>
#include <Xm/DialogS.h>
#include <Xm/Frame.h>
#include <Xm/Label.h>
#include <Xm/RowColumn.h>
#include <Xm/PushB.h>
#include <Xm/Scale.h>
#include <Xm/Separator.h>
#include <Xm/Text.h>

#include "globals.h"
#include "motifinc.h"

static Widget label_frame;
static Widget label_panel;

static Widget labelprops_frame;
static Widget labelprops_panel;

/*
 * Panel item declarations
 */
static Widget label_title_text_item;
static Widget label_subtitle_text_item;
static Widget *title_color_item;
static Widget *title_linew_item;
static Widget *title_font_item;
static Widget title_size_item;
static Widget *stitle_color_item;
static Widget *stitle_linew_item;
static Widget *stitle_font_item;
static Widget stitle_size_item;

/*
 * Event and Notify proc declarations
 */
static void label_props_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void label_define_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void labelprops_define_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
extern void set_plotstr_string(plotstr *pstr, char *buf);

void update_label_proc(void)
{
    if (label_frame)
    {
        if (g[cg].labs.title.s != NULL)
        {
            xv_setstr(label_title_text_item, g[cg].labs.title.s);
        }
        if (g[cg].labs.stitle.s != NULL)
        {
            xv_setstr(label_subtitle_text_item, g[cg].labs.stitle.s);
        }
    }
}

/*
 * Create the label Frame and the label Panel
 */
void create_label_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    set_wait_cursor();
    if (label_frame == NULL)
    {
        Widget buts[3];
        char *label1[3];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Props...";
        label1[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        label_frame = XmCreateDialogShell(app_shell, (char *)"Title/Subtitle", NULL, 0);
        handle_close(label_frame);
        XtVaSetValues(label_frame, XmNx, x, XmNy, y, NULL);
        label_panel = XtVaCreateWidget("label panel", xmRowColumnWidgetClass, label_frame,
                                       NULL);

        label_title_text_item = CreateTextItem2(label_panel, 30, "Title:");
        label_subtitle_text_item = CreateTextItem2(label_panel, 30, "Subtitle:");

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, label_panel,
                                NULL);

        CreateCommandButtons(label_panel, 3, buts, label1);
        XtAddCallback(buts[0], XmNactivateCallback,
                      (XtCallbackProc)label_define_notify_proc, (XtPointer)NULL);
        XtAddCallback(buts[1], XmNactivateCallback,
                      (XtCallbackProc)label_props_notify_proc, (XtPointer)NULL);
        XtAddCallback(buts[2], XmNactivateCallback,
                      (XtCallbackProc)destroy_dialog, (XtPointer)label_frame);

        XtManageChild(label_panel);
    }
    XtRaise(label_frame);
    update_label_proc();
    unset_wait_cursor();
}

static void label_define_notify_proc(Widget, XtPointer, XtPointer)
{
    set_plotstr_string(&g[cg].labs.title, (char *)xv_getstr(label_title_text_item));
    set_plotstr_string(&g[cg].labs.stitle, (char *)xv_getstr(label_subtitle_text_item));
    drawgraph();
}

static void labelprops_define_notify_proc(Widget, XtPointer, XtPointer)
{
    Arg a;
    int value;

    g[cg].labs.title.font = (int)GetChoice(title_font_item);
    g[cg].labs.title.color = (int)GetChoice(title_color_item);
    g[cg].labs.title.linew = (int)GetChoice(title_linew_item) + 1;
    XtSetArg(a, XmNvalue, &value);
    XtGetValues(title_size_item, &a, 1);
    g[cg].labs.title.charsize = value / 100.0;

    g[cg].labs.stitle.font = (int)GetChoice(stitle_font_item);
    g[cg].labs.stitle.color = (int)GetChoice(stitle_color_item);
    g[cg].labs.stitle.linew = (int)GetChoice(stitle_linew_item) + 1;
    XtSetArg(a, XmNvalue, &value);
    XtGetValues(stitle_size_item, &a, 1);
    g[cg].labs.stitle.charsize = value / 100.0;
    drawgraph();
}

void update_labelprops_proc(void)
{
    Arg a;
    int iv;

    if (labelprops_frame)
    {
        SetChoice(title_font_item, g[cg].labs.title.font);
        SetChoice(title_color_item, g[cg].labs.title.color);
        SetChoice(title_linew_item, g[cg].labs.title.linew - 1);
        iv = (int)(100 * g[cg].labs.title.charsize);
        XtSetArg(a, XmNvalue, iv);
        XtSetValues(title_size_item, &a, 1);

        SetChoice(stitle_font_item, g[cg].labs.stitle.font);
        SetChoice(stitle_color_item, g[cg].labs.stitle.color);
        SetChoice(stitle_linew_item, g[cg].labs.stitle.linew - 1);
        iv = (int)(100 * g[cg].labs.stitle.charsize);
        XtSetArg(a, XmNvalue, iv);
        XtSetValues(stitle_size_item, &a, 1);
    }
}

static void label_props_notify_proc(Widget, XtPointer, XtPointer)
{
    Widget wlabel, rc, rc2, rc3, fr;
    int x, y;
    set_wait_cursor();
    if (labelprops_frame == NULL)
    {
        Widget buts[2];
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        labelprops_frame = XmCreateDialogShell(app_shell, (char *)"Title/Subtitle props", NULL, 0);
        handle_close(labelprops_frame);
        XtVaSetValues(labelprops_frame, XmNx, x, XmNy, y, NULL);
        labelprops_panel = XtVaCreateWidget("labelprops panel",
                                            xmRowColumnWidgetClass, labelprops_frame,
                                            NULL);

        rc = XmCreateRowColumn(labelprops_panel, (char *)"rc", NULL, 0);
        XtVaSetValues(rc,
                      XmNorientation, XmHORIZONTAL,
                      NULL);

        fr = XtVaCreateManagedWidget("frame", xmFrameWidgetClass, rc,
                                     NULL);
        rc2 = XtVaCreateManagedWidget("rc2", xmRowColumnWidgetClass, fr,
                                      NULL);
        XtVaCreateManagedWidget("Title:", xmLabelWidgetClass, rc2,
                                NULL);
        title_font_item = CreatePanelChoice(rc2, "Font:",
                                            11,
                                            "Times-Roman", "Times-Bold", "Times-Italic",
                                            "Times-BoldItalic", "Helvetica",
                                            "Helvetica-Bold", "Helvetica-Oblique",
                                            "Helvetica-BoldOblique", "Greek", "Symbol",
                                            0,
                                            0);
        wlabel = XtVaCreateManagedWidget("Character size:", xmLabelWidgetClass, rc2,
                                         NULL);
        title_size_item = XtVaCreateManagedWidget("stringsize", xmScaleWidgetClass, rc2,
                                                  XmNminimum, 0,
                                                  XmNmaximum, 400,
                                                  XmNvalue, 100,
                                                  XmNshowValue, True,
                                                  XmNprocessingDirection, XmMAX_ON_RIGHT,
                                                  XmNorientation, XmHORIZONTAL,
                                                  NULL);

        rc3 = XmCreateRowColumn(rc2, (char *)"rc3", NULL, 0);
        XtVaSetValues(rc3, XmNorientation, XmHORIZONTAL, NULL);
        title_color_item = CreateColorChoice(rc3, "Color:", 0);
        title_linew_item = CreatePanelChoice(rc3, "Width:",
                                             10,
                                             "1", "2", "3", "4", "5", "6", "7", "8", "9", 0,
                                             0);
        XtManageChild(rc3);
        XtManageChild(rc2);
        XtManageChild(fr);

        fr = XtVaCreateManagedWidget("frame", xmFrameWidgetClass, rc,
                                     NULL);
        rc2 = XtVaCreateManagedWidget("rc2", xmRowColumnWidgetClass, fr,
                                      NULL);
        wlabel = XtVaCreateManagedWidget("Subtitle:", xmLabelWidgetClass, rc2,
                                         NULL);
        stitle_font_item = CreatePanelChoice(rc2, "Font:",
                                             11,
                                             "Times-Roman", "Times-Bold", "Times-Italic",
                                             "Times-BoldItalic", "Helvetica",
                                             "Helvetica-Bold", "Helvetica-Oblique",
                                             "Helvetica-BoldOblique", "Greek", "Symbol",
                                             0,
                                             0);
        wlabel = XtVaCreateManagedWidget("Character size:", xmLabelWidgetClass, rc2,
                                         NULL);
        stitle_size_item = XtVaCreateManagedWidget("stringsize", xmScaleWidgetClass, rc2,
                                                   XmNminimum, 0,
                                                   XmNmaximum, 400,
                                                   XmNvalue, 100,
                                                   XmNshowValue, True,
                                                   XmNprocessingDirection, XmMAX_ON_RIGHT,
                                                   XmNorientation, XmHORIZONTAL,
                                                   NULL);
        rc3 = XmCreateRowColumn(rc2, (char *)"rc3", NULL, 0);
        XtVaSetValues(rc3, XmNorientation, XmHORIZONTAL, NULL);
        stitle_color_item = CreateColorChoice(rc3, "Color:", 0);
        stitle_linew_item = CreatePanelChoice(rc3, "Width:",
                                              10,
                                              "1", "2", "3", "4", "5", "6", "7", "8", "9", 0,
                                              0);

        XtManageChild(rc3);
        XtManageChild(rc2);
        XtManageChild(fr);

        XtManageChild(rc);

        CreateCommandButtons(labelprops_panel, 2, buts, label1);
        XtAddCallback(buts[0], XmNactivateCallback,
                      (XtCallbackProc)labelprops_define_notify_proc, (XtPointer)NULL);
        XtAddCallback(buts[1], XmNactivateCallback,
                      (XtCallbackProc)destroy_dialog, (XtPointer)labelprops_frame);
        XtManageChild(labelprops_panel);
    }
    update_labelprops_proc();
    XtRaise(labelprops_frame);
    unset_wait_cursor();
}
