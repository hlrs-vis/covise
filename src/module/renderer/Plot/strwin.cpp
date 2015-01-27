/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: strwin.c,v 1.3 1994/11/02 02:49:22 pturner Exp pturner $
 *
 * strings, lines, and boxes
 *
 */

#include <stdio.h>
#include <math.h>
#include <sys/types.h>

#include <Xm/Xm.h>
#include <Xm/BulletinB.h>
#include <Xm/DialogS.h>
#include <Xm/Frame.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/ToggleB.h>
#include <Xm/Text.h>
#include <Xm/Scale.h>
#include <Xm/Separator.h>

#include "extern.h"
#include "globals.h"
#include "motifinc.h"

static Widget objects_frame;
static Widget strings_frame;
static Widget lines_frame;
static Widget boxes_frame;

// static Widget string_item;
// static Widget strings_item;
static Widget *strings_font_item;
static Widget strings_rot_item;
static Widget strings_size_item;
static Widget *strings_loc_item;
static Widget *strings_pen_item;
static Widget *strings_just_item;
Widget strings_x_item;
Widget strings_y_item;

static Widget *lines_arrow_item;
static Widget lines_asize_item;
static Widget *lines_atype_item;
static Widget *lines_pen_item;
static Widget *lines_style_item;
static Widget *lines_width_item;
static Widget *lines_loc_item;
static Widget *boxes_pen_item;
static Widget *boxes_lines_item;
static Widget *boxes_linew_item;
static Widget *boxes_fill_item;
static Widget *boxes_fillpat_item;
static Widget *boxes_fillcol_item;
static Widget *boxes_loc_item;

void boxes_def_proc(Widget, XtPointer, XtPointer)
{
    box_color = (int)GetChoice(boxes_pen_item);
    box_loctype = (int)GetChoice(boxes_loc_item) ? VIEW : WORLD;
    box_lines = (int)GetChoice(boxes_lines_item) + 1;
    box_linew = (int)GetChoice(boxes_linew_item) + 1;
    switch (GetChoice(boxes_fill_item))
    {
    case 0:
        box_fill = covise::NONE;
        break;
    case 1:
        box_fill = COLOR;
        break;
    case 2:
        box_fill = PATTERN;
        break;
    }
    box_fillcolor = (int)GetChoice(boxes_fillcol_item);
    box_fillpat = (int)GetChoice(boxes_fillpat_item);
    if (ismaster)
        cm->sendCommand_ValuesMessage(BOXES_DEF_PROC, box_color, box_loctype, box_lines, box_linew, box_fill, box_fillcolor, box_fillpat, 0, 0, 0);
}

void lines_def_proc(Widget, XtPointer, XtPointer)
{
    Arg a;
    int value;
    XtSetArg(a, XmNvalue, &value);
    XtGetValues(lines_asize_item, &a, 1);
    line_asize = value / 50.0;
    line_color = (int)GetChoice(lines_pen_item);
    line_arrow = (int)GetChoice(lines_arrow_item);
    line_atype = (int)GetChoice(lines_atype_item);
    line_lines = (int)GetChoice(lines_style_item) + 1;
    line_linew = (int)GetChoice(lines_width_item) + 1;
    line_loctype = (int)GetChoice(lines_loc_item) ? VIEW : WORLD;
    if (ismaster)
        cm->sendCommand_ValuesMessage(LINES_DEF_PROC, value, line_color, line_arrow, line_atype, line_lines, line_linew, line_loctype, 0, 0, 0);
}

void updatestrings(void)
{
    Arg a;
    int iv;
    if (strings_frame)
    {
        SetChoice(strings_font_item, string_font);
        SetChoice(strings_pen_item, string_color);
        iv = (int)(100 * string_size);
        XtSetArg(a, XmNvalue, iv);
        XtSetValues(strings_size_item, &a, 1);
        XtSetArg(a, XmNvalue, string_rot);
        XtSetValues(strings_rot_item, &a, 1);
        SetChoice(strings_loc_item, string_loctype == VIEW ? 1 : 0);
        SetChoice(strings_just_item, string_just);
    }
}

void update_lines(void)
{
    Arg a;
    int iv;
    if (lines_frame)
    {
        SetChoice(lines_pen_item, line_color);
        SetChoice(lines_style_item, line_lines - 1);
        SetChoice(lines_width_item, line_linew - 1);
        SetChoice(lines_arrow_item, line_arrow);
        SetChoice(lines_atype_item, line_atype);
        iv = (int)(50 * line_asize);
        XtSetArg(a, XmNvalue, iv);
        XtSetValues(lines_asize_item, &a, 1);
        SetChoice(lines_loc_item, line_loctype == VIEW ? 1 : 0);
    }
}

void update_boxes(void)
{
    if (boxes_frame)
    {
        SetChoice(boxes_pen_item, box_color);
        SetChoice(boxes_lines_item, box_lines - 1);
        SetChoice(boxes_linew_item, box_linew - 1);
        switch (box_fill)
        {
        case covise::NONE:
            SetChoice(boxes_fill_item, 0);
            break;
        case COLOR:
            SetChoice(boxes_fill_item, 1);
            break;
        case PATTERN:
            SetChoice(boxes_fill_item, 2);
            break;
        }
        SetChoice(boxes_fillpat_item, box_fillpat);
        SetChoice(boxes_fillcol_item, box_fillcolor);
        SetChoice(boxes_loc_item, box_loctype == VIEW ? 1 : 0);
    }
}

void define_string_defaults(Widget, XtPointer, XtPointer)
{
    Arg a;
    int value, s_size;

    if (strings_frame)
    {
        string_font = (int)GetChoice(strings_font_item);
        string_color = (int)GetChoice(strings_pen_item);
        XtSetArg(a, XmNvalue, &s_size);
        XtGetValues(strings_size_item, &a, 1);
        string_size = s_size / 100.0;
        XtSetArg(a, XmNvalue, &value);
        XtGetValues(strings_rot_item, &a, 1);
        string_rot = value;
        string_loctype = (int)GetChoice(strings_loc_item) ? VIEW : WORLD;
        string_just = (int)GetChoice(strings_just_item);
        if (ismaster)
            cm->sendCommand_ValuesMessage(DEFINE_STRING_DEFAULTS, string_font, string_color, s_size, string_rot, string_loctype, string_just, 0, 0, 0, 0);
    }
}

void define_objects_popup(Widget, XtPointer, XtPointer)
{
    Widget wbut;
    Widget panel;
    int x, y;
    set_wait_cursor();
    if (objects_frame == NULL)
    {
        XmGetPos(app_shell, 0, &x, &y);
        objects_frame = XmCreateDialogShell(app_shell, (char *)"Objects", NULL, 0);
        handle_close(objects_frame);
        XtVaSetValues(objects_frame, XmNx, x, XmNy, y, NULL);
        panel = XmCreateRowColumn(objects_frame, (char *)"ticks_rc", NULL, 0);

        wbut = XtVaCreateManagedWidget("Text", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)STR_LOC);

        wbut = XtVaCreateManagedWidget("Text at angle", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)STR_LOC1ST);

        wbut = XtVaCreateManagedWidget("Edit Text", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)STR_EDIT);

        wbut = XtVaCreateManagedWidget("Text props...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)define_strings_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Line", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)MAKE_LINE_1ST);

        wbut = XtVaCreateManagedWidget("Line props...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)define_lines_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Box", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)MAKE_BOX_1ST);

        wbut = XtVaCreateManagedWidget("Box props...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)define_boxes_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Move object", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)MOVE_OBJECT_1ST);

        wbut = XtVaCreateManagedWidget("Copy object", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)COPY_OBJECT1ST);

        wbut = XtVaCreateManagedWidget("Delete object", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)DEL_OBJECT);

        wbut = XtVaCreateManagedWidget("Clear all text", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_clear_text, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Clear all lines", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_clear_lines, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Clear all boxes", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_clear_boxes, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Close", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)objects_frame);
        XtManageChild(panel);
    }
    XtRaise(objects_frame);
    unset_wait_cursor();
}

void define_strings_popup(Widget, XtPointer, XtPointer)
{
    Widget wlabel;
    int x, y;
    Widget buts[2];
    Widget panel;
    Widget rc;

    set_wait_cursor();
    if (strings_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        strings_frame = XmCreateDialogShell(app_shell, (char *)"Strings", NULL, 0);
        handle_close(strings_frame);
        XtVaSetValues(strings_frame, XmNx, x, XmNy, y, NULL);
        panel = XmCreateRowColumn(strings_frame, (char *)"strings_rc", NULL, 0);

        rc = XtVaCreateWidget("rc", xmRowColumnWidgetClass, panel,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 4,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        XtVaCreateManagedWidget("Font:", xmLabelWidgetClass, rc, NULL);
        strings_font_item = CreatePanelChoice(rc, " ",
                                              11,
                                              "Times-Roman", "Times-Bold", "Times-Italic",
                                              "Times-BoldItalic", "Helvetica",
                                              "Helvetica-Bold", "Helvetica-Oblique",
                                              "Helvetica-BoldOblique", "Greek", "Symbol",
                                              0,
                                              0);

        XtVaCreateManagedWidget("Color:", xmLabelWidgetClass, rc, NULL);
        strings_pen_item = CreateColorChoice(rc, " ", 0);

        XtVaCreateManagedWidget("Justification:", xmLabelWidgetClass, rc, NULL);
        strings_just_item = CreatePanelChoice(rc, " ",
                                              4,
                                              "Left",
                                              "Right",
                                              "Centered",
                                              0,
                                              0);

        XtVaCreateManagedWidget("Position in:", xmLabelWidgetClass, rc, NULL);
        strings_loc_item = CreatePanelChoice(rc, " ",
                                             3,
                                             "World coordinates",
                                             "Viewport coordinates",
                                             0,
                                             0);
        XtManageChild(rc);

        wlabel = XtVaCreateManagedWidget("Rotation:", xmLabelWidgetClass, panel, NULL);
        strings_rot_item = XtVaCreateManagedWidget("rotation", xmScaleWidgetClass, panel,
                                                   XmNminimum, 0,
                                                   XmNmaximum, 360,
                                                   XmNvalue, 0,
                                                   XmNshowValue, True,
                                                   XmNprocessingDirection, XmMAX_ON_RIGHT,
                                                   XmNorientation, XmHORIZONTAL,
                                                   NULL);

        wlabel = XtVaCreateManagedWidget("Size:", xmLabelWidgetClass, panel, NULL);
        strings_size_item = XtVaCreateManagedWidget("stringsize", xmScaleWidgetClass, panel,
                                                    XmNminimum, 0,
                                                    XmNmaximum, 400,
                                                    XmNvalue, 100,
                                                    XmNshowValue, True,
                                                    XmNprocessingDirection, XmMAX_ON_RIGHT,
                                                    XmNorientation, XmHORIZONTAL,
                                                    NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, panel, NULL);

        CreateCommandButtons(panel, 2, buts, label1);
        XtAddCallback(buts[0], XmNactivateCallback,
                      (XtCallbackProc)define_string_defaults, (XtPointer)0);
        XtAddCallback(buts[1], XmNactivateCallback,
                      (XtCallbackProc)destroy_dialog, (XtPointer)strings_frame);

        XtManageChild(panel);
    }
    XtRaise(strings_frame);
    updatestrings();
    unset_wait_cursor();
}

void define_lines_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget buts[2];
    Widget panel;
    Widget rc;

    set_wait_cursor();
    if (lines_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        lines_frame = XmCreateDialogShell(app_shell, (char *)"Lines", NULL, 0);
        handle_close(lines_frame);
        XtVaSetValues(lines_frame, XmNx, x, XmNy, y, NULL);
        panel = XmCreateRowColumn(lines_frame, (char *)"lines_rc", NULL, 0);

        rc = XtVaCreateWidget("rc", xmRowColumnWidgetClass, panel,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 7,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        XtVaCreateManagedWidget("Color:", xmLabelWidgetClass, rc, NULL);
        lines_pen_item = CreateColorChoice(rc, " ", 0);

        XtVaCreateManagedWidget("Width:", xmLabelWidgetClass, rc, NULL);
        lines_width_item = (Widget *)CreatePanelChoice(rc, " ",
                                                       10,
                                                       "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                                       NULL,
                                                       NULL);

        XtVaCreateManagedWidget("Style:", xmLabelWidgetClass, rc, NULL);
        lines_style_item = (Widget *)CreatePanelChoice(rc, " ",
                                                       6,
                                                       "Solid line",
                                                       "Dotted line",
                                                       "Dashed line",
                                                       "Long Dashed",
                                                       "Dot-dashed",
                                                       NULL,
                                                       NULL);

        XtVaCreateManagedWidget("Arrow:", xmLabelWidgetClass, rc, NULL);
        lines_arrow_item = CreatePanelChoice(rc, " ",
                                             5,
                                             "None",
                                             "At start",
                                             "At end",
                                             "Both ends",
                                             0,
                                             0);

        XtVaCreateManagedWidget("Arrow type:", xmLabelWidgetClass, rc, NULL);
        lines_atype_item = CreatePanelChoice(rc, " ",
                                             4,
                                             "Line",
                                             "Filled",
                                             "Hollow",
                                             0,
                                             0);

        XtVaCreateManagedWidget("Arrow head size:", xmLabelWidgetClass, rc, NULL);
        lines_asize_item = XtVaCreateManagedWidget("arrowsize", xmScaleWidgetClass, rc,
                                                   XmNminimum, 0,
                                                   XmNmaximum, 400,
                                                   XmNvalue, 100,
                                                   XmNshowValue, True,
                                                   XmNprocessingDirection, XmMAX_ON_RIGHT,
                                                   XmNorientation, XmHORIZONTAL,
                                                   NULL);

        XtVaCreateManagedWidget("Position in:", xmLabelWidgetClass, rc, NULL);
        lines_loc_item = CreatePanelChoice(rc, " ",
                                           3,
                                           "World coordinates",
                                           "Viewport coordinates",
                                           0,
                                           0);
        XtManageChild(rc);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, panel, NULL);

        CreateCommandButtons(panel, 2, buts, label1);
        XtAddCallback(buts[0], XmNactivateCallback,
                      (XtCallbackProc)lines_def_proc, (XtPointer)0);
        XtAddCallback(buts[1], XmNactivateCallback,
                      (XtCallbackProc)destroy_dialog, (XtPointer)lines_frame);

        XtManageChild(panel);
    }
    update_lines();
    XtRaise(lines_frame);
    unset_wait_cursor();
}

void define_boxes_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget buts[2];
    Widget panel;
    Widget rc;

    set_wait_cursor();
    if (boxes_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        boxes_frame = XmCreateDialogShell(app_shell, (char *)"Boxes", NULL, 0);
        handle_close(boxes_frame);
        XtVaSetValues(boxes_frame, XmNx, x, XmNy, y, NULL);
        panel = XmCreateRowColumn(boxes_frame, (char *)"boxes_rc", NULL, 0);

        rc = XtVaCreateWidget("rc", xmRowColumnWidgetClass, panel,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 7,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        XtVaCreateManagedWidget("Color:", xmLabelWidgetClass, rc, NULL);
        boxes_pen_item = CreateColorChoice(rc, " ", 0);

        XtVaCreateManagedWidget("Line width:", xmLabelWidgetClass, rc, NULL);
        boxes_linew_item = CreatePanelChoice(rc, " ",
                                             10,
                                             "1", "2", "3", "4", "5", "6", "7", "8", "9", 0,
                                             NULL,
                                             0);

        XtVaCreateManagedWidget("Line style:", xmLabelWidgetClass, rc, NULL);
        boxes_lines_item = (Widget *)CreatePanelChoice(rc, " ",
                                                       6,
                                                       "Solid line",
                                                       "Dotted line",
                                                       "Dashed line",
                                                       "Long Dashed",
                                                       "Dot-dashed",
                                                       NULL,
                                                       NULL);

        XtVaCreateManagedWidget("Fill:", xmLabelWidgetClass, rc, NULL);
        boxes_fill_item = CreatePanelChoice(rc, " ",
                                            4,
                                            "None",
                                            "Color",
                                            "Pattern",
                                            NULL,
                                            0);

        XtVaCreateManagedWidget("Fill pattern:", xmLabelWidgetClass, rc, NULL);
        boxes_fillpat_item = CreatePanelChoice0(rc,
                                                "Pattern:", 4, 17,
                                                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                                "10", "11", "12", "13", "14", "15", 0, 0);
        XtVaCreateManagedWidget("Fill color:", xmLabelWidgetClass, rc, NULL);
        boxes_fillcol_item = CreateColorChoice(rc, " ", 0);
        XtVaCreateManagedWidget("Position in:", xmLabelWidgetClass, rc, NULL);
        boxes_loc_item = CreatePanelChoice(rc, " ",
                                           3,
                                           "World coordinates",
                                           "Viewport coordinates",
                                           0,
                                           0);
        XtManageChild(rc);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, panel, NULL);

        CreateCommandButtons(panel, 2, buts, label1);
        XtAddCallback(buts[0], XmNactivateCallback,
                      (XtCallbackProc)boxes_def_proc, (XtPointer)0);
        XtAddCallback(buts[1], XmNactivateCallback,
                      (XtCallbackProc)destroy_dialog, (XtPointer)boxes_frame);

        XtManageChild(panel);
    }
    XtRaise(boxes_frame);
    update_boxes();
    unset_wait_cursor();
}
