/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: setwin.c,v 1.12 1994/11/04 06:02:10 pturner Exp pturner $
 *
 * setops - operations on sets
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "extern.h"

#include <Xm/Xm.h>
#include <Xm/BulletinB.h>
#include <Xm/DialogS.h>
#include <Xm/FileSB.h>
#include <Xm/Frame.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/ToggleB.h>
#include <Xm/RowColumn.h>
#include <Xm/Text.h>
#include <Xm/List.h>
#include <Xm/Separator.h>
#include <Xm/Protocols.h>

#include "globals.h"
#include "motifinc.h"

static Widget but1[2];
// static Widget but2[3];

char format[128] = "%16lg %16lg";
char sformat[128] = "%16lg %16lg";

static Widget saveall_sets_format_item;
//static Widget saveall_sets_file_item;
static Widget *swap_from_item;
static Widget *swap_fgraph_item;
static Widget *swap_to_item;
static Widget *swap_tgraph_item;

static void do_activate_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_deactivate_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_reactivate_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_setlength_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_changetype_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_copy_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_setmove_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_drop_points_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_join_sets_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_split_sets_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_kill_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_sort_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_write_sets_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_saveall_sets_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_reverse_sets_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_coalesce_sets_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_swap_proc(Widget w, XtPointer client_data, XtPointer call_data);
// static void do_break_sets_proc(Widget w, XtPointer client_data, XtPointer call_data);

/* for pick ops */
static void define_pickops_popup(Widget w, XtPointer client_data, XtPointer call_data);

static void create_activate_popup(Widget w, XtPointer client_data, XtPointer call_data);
static void create_deactivate_popup(Widget w, XtPointer client_data, XtPointer call_data);
static void create_reactivate_popup(Widget w, XtPointer client_data, XtPointer call_data);
static void create_change_popup(Widget w, XtPointer client_data, XtPointer call_data);
static void create_copy_popup(Widget w, XtPointer client_data, XtPointer call_data);
static void create_setlength_popup(Widget w, XtPointer client_data, XtPointer call_data);
static void create_move_popup(Widget w, XtPointer client_data, XtPointer call_data);
static void create_drop_popup(Widget w, XtPointer client_data, XtPointer call_data);
static void create_join_popup(Widget w, XtPointer client_data, XtPointer call_data);
static void create_split_popup(Widget w, XtPointer client_data, XtPointer call_data);
// static void create_break_popup(Widget w, XtPointer client_data, XtPointer call_data);
static void create_kill_popup(Widget w, XtPointer client_data, XtPointer call_data);
static void create_sort_popup(Widget w, XtPointer client_data, XtPointer call_data);
void create_write_popup(Widget w, XtPointer client_data, XtPointer call_data);
static void create_reverse_popup(Widget w, XtPointer client_data, XtPointer call_data);
static void create_coalesce_popup(Widget w, XtPointer client_data, XtPointer call_data);
static void create_swap_popup(Widget w, XtPointer client_data, XtPointer call_data);

// static char errbuf[256];

extern int index_set_types[]; /* declared in setutils.c */
extern int index_set_ncols[];

extern void DefineSetSelectorFilter(SetChoiceItem *s);
extern int GetSelectedSets(SetChoiceItem l, int **sets);

void define_setops_popup(Widget, XtPointer, XtPointer)
{
    static Widget top;
    Widget panel, wbut, rc;
    int x, y;
    set_wait_cursor();
    if (top == NULL)
    {
        XmGetPos(app_shell, 0, &x, &y);
        top = XmCreateDialogShell(app_shell, (char *)"Set ops", NULL, 0);
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        panel = XmCreateRowColumn(top, (char *)"setops_rc", NULL, 0);
        XtVaSetValues(panel, XmNorientation, XmHORIZONTAL, NULL);

        rc = XmCreateRowColumn(panel, (char *)"rc", NULL, 0);
        wbut = XtVaCreateManagedWidget("Pick ops...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)define_pickops_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Activate...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_activate_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("De-activate...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_deactivate_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Re-activate...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_reactivate_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Set length...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_setlength_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Change set type...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_change_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Copy...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_copy_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Move...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_move_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Drop points...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_drop_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Join sets...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_join_popup, (XtPointer)NULL);
        XtManageChild(rc);

        rc = XmCreateRowColumn(panel, (char *)"rc", NULL, 0);
        wbut = XtVaCreateManagedWidget("Split...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_split_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Kill...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_kill_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Kill all", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_flush, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Sort...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_sort_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Write sets...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_write_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Reverse sets...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_reverse_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Coalesce sets...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_coalesce_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Swap sets...", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_swap_popup, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Pack sets", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)do_packsets, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Close", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)top);
        XtManageChild(rc);

        XtManageChild(panel);
    }
    XtRaise(top);
    unset_wait_cursor();
}

static void define_pickops_popup(Widget, XtPointer, XtPointer)
{
    static Widget top;
    Widget panel, wbut;
    int x, y;

    set_wait_cursor();
    if (top == NULL)
    {
        XmGetPos(app_shell, 0, &x, &y);
        top = XmCreateDialogShell(app_shell, (char *)"Pick ops", NULL, 0);
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        panel = XmCreateRowColumn(top, (char *)"pickops_rc", NULL, 0);

        wbut = XtVaCreateManagedWidget("Kill nearest set",
                                       xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)KILL_NEAREST);

        wbut = XtVaCreateManagedWidget("Copy nearest set",
                                       xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)COPY_NEAREST1ST);

        wbut = XtVaCreateManagedWidget("Move nearest set",
                                       xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)MOVE_NEAREST1ST);

        wbut = XtVaCreateManagedWidget("Reverse nearest set",
                                       xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)REVERSE_NEAREST);

        wbut = XtVaCreateManagedWidget("De-activate nearest set",
                                       xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)DEACTIVATE_NEAREST);

        wbut = XtVaCreateManagedWidget("Join nearest sets",
                                       xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)JOIN_NEAREST1ST);

        wbut = XtVaCreateManagedWidget("Delete range in nearest set",
                                       xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)DELETE_NEAREST1ST);

        wbut = XtVaCreateManagedWidget("Break set",
                                       xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)PICK_BREAK);

        wbut = XtVaCreateManagedWidget("Cancel operation",
                                       xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)set_actioncb, (XtPointer)0);

        wbut = XtVaCreateManagedWidget("Close", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)destroy_dialog,
                      (XtPointer)top);

        XtManageChild(panel);
    }
    XtRaise(top);
    unset_wait_cursor();
}

typedef struct _Act_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget len_item;
    Widget name_item;
    Widget comment_item;
    Widget *type_item;
    Widget *graph_item;
} Act_ui;

static Act_ui aui;

static void create_activate_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;
    set_wait_cursor();
    if (aui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        aui.top = XmCreateDialogShell(app_shell, (char *)"Activate set", NULL, 0);
        handle_close(aui.top);
        XtVaSetValues(aui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(aui.top, (char *)"dialog_rc", NULL, 0);

        aui.sel = CreateSetSelector(dialog, (char *)"Activate set:",
                                    SET_SELECT_ACTIVE,
                                    FILTER_SELECT_INACT,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);
        DefineSetSelectorFilter(&aui.sel);

        aui.name_item = CreateTextItem2(dialog, 10, (char *)"Set name:");
        aui.comment_item = CreateTextItem2(dialog, 25, (char *)"Comment:");
        aui.len_item = CreateTextItem2(dialog, 10, (char *)"Length:");
        aui.type_item = CreatePanelChoice(dialog,
                                          "Set type:",
                                          10,
                                          "XY",
                                          "XY DX",
                                          "XY DY",
                                          "XY DX1 DX2",
                                          "XY DY1 DY2",
                                          "XY DX DY",
                                          "XY Z",
                                          "XY HILO",
                                          "XY R",
                                          NULL, 0);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_activate_proc, (XtPointer)&aui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)aui.top);

        XtManageChild(dialog);
    }
    XtRaise(aui.top);
    unset_wait_cursor();
}

typedef struct _Deact_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *graph_item;
} Deact_ui;

static Deact_ui deactui;

static void create_deactivate_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (deactui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        deactui.top = XmCreateDialogShell(app_shell, (char *)"De-activate set", NULL, 0);
        handle_close(deactui.top);
        XtVaSetValues(deactui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(deactui.top, (char *)"dialog_rc", NULL, 0);

        deactui.sel = CreateSetSelector(dialog, "De-activate set:",
                                        SET_SELECT_ACTIVE,
                                        FILTER_SELECT_NONE,
                                        GRAPH_SELECT_CURRENT,
                                        SELECTION_TYPE_MULTIPLE);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_deactivate_proc, (XtPointer)&deactui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)deactui.top);

        XtManageChild(dialog);
    }
    XtRaise(deactui.top);
    unset_wait_cursor();
}

typedef struct _React_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *graph_item;
} React_ui;

static React_ui reactui;

static void create_reactivate_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (reactui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        reactui.top = XmCreateDialogShell(app_shell, (char *)"Re-activate set", NULL, 0);
        handle_close(reactui.top);
        XtVaSetValues(reactui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(reactui.top, (char *)"dialog_rc", NULL, 0);

        reactui.sel = CreateSetSelector(dialog, "Re-activate set:",
                                        SET_SELECT_ACTIVE,
                                        FILTER_SELECT_DEACT,
                                        GRAPH_SELECT_CURRENT,
                                        SELECTION_TYPE_MULTIPLE);
        DefineSetSelectorFilter(&reactui.sel);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_reactivate_proc, (XtPointer)&reactui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)reactui.top);

        XtManageChild(dialog);
    }
    XtRaise(reactui.top);
    unset_wait_cursor();
}

typedef struct _Type_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget name_item;
    Widget comment_item;
    Widget *type_item;
    Widget *graph_item;
} Type_ui;

static Type_ui tui;

static void changetypeCB(Widget, XtPointer clientd, XtPointer calld)
{
    Type_ui *ui = (Type_ui *)clientd;
    XmListCallbackStruct *cbs = (XmListCallbackStruct *)calld;
    char *s;
    XmStringGetLtoR(cbs->item, charset, &s);
    if (cbs->reason == XmCR_SINGLE_SELECT)
    {
        int setno = GetSelectedSet(ui->sel);
        /*
         int setno = GetSetFromString(s);
      */
        if (setno >= 0)
        {
            xv_setstr(ui->comment_item, getcomment(cg, setno));
            xv_setstr(ui->name_item, getsetname(cg, setno));
        }
    }
    XtFree(s);
}

static void create_change_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (tui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        tui.top = XmCreateDialogShell(app_shell, (char *)"Change set type", NULL, 0);
        handle_close(tui.top);
        XtVaSetValues(tui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(tui.top, (char *)"dialog_rc", NULL, 0);

        tui.sel = CreateSetSelector(dialog, "Apply to set:",
                                    SET_SELECT_ACTIVE,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);

        XtVaSetValues(tui.sel.list,
                      XmNselectionPolicy, XmSINGLE_SELECT,
                      NULL);
        XtAddCallback(tui.sel.list, XmNdefaultActionCallback, changetypeCB, &tui);
        XtAddCallback(tui.sel.list, XmNsingleSelectionCallback, changetypeCB, &tui);
        tui.name_item = CreateTextItem2(dialog, 10, "Set name:");
        tui.comment_item = CreateTextItem2(dialog, 20, "Comment:");
        tui.type_item = CreatePanelChoice(dialog,
                                          "Type:",
                                          10,
                                          "XY",
                                          "XY DX",
                                          "XY DY",
                                          "XY DX1 DX2",
                                          "XY DY1 DY2",
                                          "XY DX DY",
                                          "XY Z",
                                          "XY HILO",
                                          "XY R",
                                          NULL, 0);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_changetype_proc, (XtPointer)&tui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)tui.top);

        XtManageChild(dialog);
    }
    XtRaise(tui.top);
    unset_wait_cursor();
}

typedef struct _Length_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget length_item;
    Widget *graph_item;
} Length_ui;

static Length_ui lui;

static void create_setlength_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (lui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        lui.top = XmCreateDialogShell(app_shell, (char *)"Set length", NULL, 0);
        handle_close(lui.top);
        XtVaSetValues(lui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(lui.top, (char *)"dialog_rc", NULL, 0);

        lui.sel = CreateSetSelector(dialog, "Set length of set:",
                                    SET_SELECT_ACTIVE,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);
        lui.length_item = CreateTextItem2(dialog, 10, "Length:");

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_setlength_proc, (XtPointer)&lui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)lui.top);

        XtManageChild(dialog);
    }
    XtRaise(lui.top);
    unset_wait_cursor();
}

typedef struct _Copy_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *graph_item;
} Copy_ui;

static Copy_ui cui;

static void create_copy_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (cui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        cui.top = XmCreateDialogShell(app_shell, (char *)"Copy set", NULL, 0);
        handle_close(cui.top);
        XtVaSetValues(cui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(cui.top, (char *)"dialog_rc", NULL, 0);

        cui.sel = CreateSetSelector(dialog, "Copy set:",
                                    SET_SELECT_ALL,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);
        cui.graph_item = CreateGraphChoice(dialog, "To graph:", maxgraph, 1);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_copy_proc, (XtPointer)&cui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)cui.top);

        XtManageChild(dialog);
    }
    XtRaise(cui.top);
    unset_wait_cursor();
}

typedef struct _Move_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *graph_item;
} Move_ui;

static Move_ui mui;

static void create_move_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (mui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        mui.top = XmCreateDialogShell(app_shell, (char *)"Move set", NULL, 0);
        handle_close(mui.top);
        XtVaSetValues(mui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(mui.top, (char *)"dialog_rc", NULL, 0);

        mui.sel = CreateSetSelector(dialog, "Move set:",
                                    SET_SELECT_ACTIVE,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);
        mui.graph_item = CreateGraphChoice(dialog, "To graph:", maxgraph, 1);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_setmove_proc, (XtPointer)&mui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)mui.top);

        XtManageChild(dialog);
    }
    XtRaise(mui.top);
    unset_wait_cursor();
}

typedef struct _Drop_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget start_item;
    Widget stop_item;
} Drop_ui;

static Drop_ui dui;

static void create_drop_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (dui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        dui.top = XmCreateDialogShell(app_shell, (char *)"Drop points", NULL, 0);
        handle_close(dui.top);
        XtVaSetValues(dui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(dui.top, (char *)"dialog_rc", NULL, 0);

        dui.sel = CreateSetSelector(dialog, "Drop points from set:",
                                    SET_SELECT_ALL,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);
        dui.start_item = CreateTextItem2(dialog, 6, "Start drop at:");
        dui.stop_item = CreateTextItem2(dialog, 6, "End drop at:");

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_drop_points_proc, (XtPointer)&dui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)dui.top);

        XtManageChild(dialog);
    }
    XtRaise(dui.top);
    unset_wait_cursor();
}

typedef struct _Join_ui
{
    Widget top;
    SetChoiceItem sel1;
    SetChoiceItem sel2;
    Widget *graph_item;
} Join_ui;

static Join_ui jui;

static void create_join_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (jui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        jui.top = XmCreateDialogShell(app_shell, (char *)"Join sets", NULL, 0);
        handle_close(jui.top);
        XtVaSetValues(jui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(jui.top, (char *)"dialog_rc", NULL, 0);

        jui.sel1 = CreateSetSelector(dialog, "Join set:",
                                     SET_SELECT_ACTIVE,
                                     FILTER_SELECT_NONE,
                                     GRAPH_SELECT_CURRENT,
                                     SELECTION_TYPE_SINGLE);
        jui.sel2 = CreateSetSelector(dialog, "To the end of set:",
                                     SET_SELECT_ACTIVE,
                                     FILTER_SELECT_NONE,
                                     GRAPH_SELECT_CURRENT,
                                     SELECTION_TYPE_SINGLE);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_join_sets_proc, (XtPointer)&jui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)jui.top);

        XtManageChild(dialog);
    }
    XtRaise(jui.top);
    unset_wait_cursor();
}

typedef struct _Split_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget len_item;
    Widget *graph_item;
} Split_ui;

static Split_ui sui;

static void create_split_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (sui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        sui.top = XmCreateDialogShell(app_shell, (char *)"Split sets", NULL, 0);
        handle_close(sui.top);
        XtVaSetValues(sui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(sui.top, (char *)"dialog_rc", NULL, 0);

        sui.sel = CreateSetSelector(dialog, "Split set:",
                                    SET_SELECT_ALL,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);
        sui.len_item = CreateTextItem2(dialog, 10, "Length:");

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_split_sets_proc, (XtPointer)&sui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)sui.top);

        XtManageChild(dialog);
    }
    XtRaise(sui.top);
    unset_wait_cursor();
}

typedef struct _Kill_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget soft_toggle;
    Widget *graph_item;
} Kill_ui;

static Kill_ui kui;

static void create_kill_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (kui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        kui.top = XmCreateDialogShell(app_shell, (char *)"Kill set", NULL, 0);
        handle_close(kui.top);
        XtVaSetValues(kui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(kui.top, (char *)"dialog_rc", NULL, 0);

        kui.sel = CreateSetSelector(dialog, "Kill set:",
                                    SET_SELECT_ALL,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);
        kui.soft_toggle = XtVaCreateManagedWidget("Preserve parameters",
                                                  xmToggleButtonWidgetClass, dialog,
                                                  NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_kill_proc, (XtPointer)&kui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)kui.top);

        XtManageChild(dialog);
    }
    XtRaise(kui.top);
    unset_wait_cursor();
}

typedef struct _Sort_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *xy_item;
    Widget *up_down_item;
    Widget *graph_item;
} Sort_ui;

static Sort_ui sortui;

static void create_sort_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (sortui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        sortui.top = XmCreateDialogShell(app_shell, (char *)"Sort sets", NULL, 0);
        handle_close(sortui.top);
        XtVaSetValues(sortui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(sortui.top, (char *)"dialog_rc", NULL, 0);

        sortui.sel = CreateSetSelector(dialog, "Sort set:",
                                       SET_SELECT_ACTIVE,
                                       FILTER_SELECT_NONE,
                                       GRAPH_SELECT_CURRENT,
                                       SELECTION_TYPE_MULTIPLE);
        sortui.xy_item = CreatePanelChoice(dialog,
                                           "Sort on:",
                                           9,
                                           "X",
                                           "Y",
                                           "Y1",
                                           "Y2",
                                           "Y3",
                                           "Y4",
                                           "Y5",
                                           "Y6",
                                           0, 0);
        sortui.up_down_item = CreatePanelChoice(dialog,
                                                "Order:",
                                                3,
                                                "Ascending",
                                                "Descending", 0,
                                                0);
        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_sort_proc, (XtPointer)&sortui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)sortui.top);

        XtManageChild(dialog);
    }
    XtRaise(sortui.top);
    unset_wait_cursor();
}

typedef struct _Write_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *graph_item;
    Widget imbed_item;
    Widget binary_item;
    Widget format_item;
} Write_ui;

Write_ui wui;

void create_write_popup(Widget, XtPointer, XtPointer)
{
    Widget dialog;
    Widget fr;

    set_wait_cursor();
    if (wui.top == NULL)
    {
        wui.top = XmCreateFileSelectionDialog(app_shell, (char *)"write_sets", NULL, 0);
        XtVaSetValues(XtParent(wui.top), XmNtitle, "Write sets", NULL);

        XtAddCallback(wui.top, XmNokCallback, (XtCallbackProc)do_write_sets_proc, (XtPointer)&wui);
        XtAddCallback(wui.top, XmNcancelCallback, (XtCallbackProc)destroy_dialog, (XtPointer)wui.top);

        fr = XmCreateFrame(wui.top, (char *)"fr", NULL, 0);
        dialog = XmCreateRowColumn(fr, (char *)"dialog_rc", NULL, 0);

        wui.sel = CreateSetSelector(dialog, "Write set:",
                                    SET_SELECT_ALL,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);
        wui.graph_item = CreateGraphChoice(dialog, "From graph:", maxgraph, 2);
        wui.imbed_item = XtVaCreateManagedWidget("Imbed parameters",
                                                 xmToggleButtonWidgetClass, dialog,
                                                 NULL);
        wui.binary_item = XtVaCreateManagedWidget("Write binary data",
                                                  xmToggleButtonWidgetClass, dialog,
                                                  NULL);
        wui.format_item = CreateTextItem2(dialog, 15, "Format: ");

        XtManageChild(dialog);
        XtManageChild(fr);
        xv_setstr(wui.format_item, format);
    }
    XtRaise(wui.top);
    unset_wait_cursor();
}

typedef struct _Reverse_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *graph_item;
} Reverse_ui;

static Reverse_ui rui;

static void create_reverse_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (rui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        rui.top = XmCreateDialogShell(app_shell, (char *)"Reverse sets", NULL, 0);
        handle_close(rui.top);
        XtVaSetValues(rui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(rui.top, (char *)"dialog_rc", NULL, 0);

        rui.sel = CreateSetSelector(dialog, "Reverse set:",
                                    SET_SELECT_ALL,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_reverse_sets_proc, (XtPointer)&rui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)rui.top);

        XtManageChild(dialog);
    }
    XtRaise(rui.top);
    unset_wait_cursor();
}

typedef struct _Coal_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *graph_item;
} Coal_ui;

static Coal_ui coalui;

static void create_coalesce_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (coalui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        coalui.top = XmCreateDialogShell(app_shell, (char *)"Coalesce sets", NULL, 0);
        handle_close(coalui.top);
        XtVaSetValues(coalui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(coalui.top, (char *)"dialog_rc", NULL, 0);

        coalui.sel = CreateSetSelector(dialog, "Coalesce active sets to set:",
                                       SET_SELECT_ALL,
                                       FILTER_SELECT_NONE,
                                       GRAPH_SELECT_CURRENT,
                                       SELECTION_TYPE_MULTIPLE);
        DefineSetSelectorFilter(&coalui.sel);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_coalesce_sets_proc, (XtPointer)&coalui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)coalui.top);

        XtManageChild(dialog);
    }
    XtRaise(coalui.top);
    unset_wait_cursor();
}

typedef struct _Swap_ui
{
    Widget top;
    SetChoiceItem sel1;
    SetChoiceItem sel2;
    Widget *graph1_item;
    Widget *graph2_item;
} Swap_ui;

static Swap_ui swapui;

static void create_swap_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (swapui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        swapui.top = XmCreateDialogShell(app_shell, (char *)"Swap sets", NULL, 0);
        handle_close(swapui.top);
        XtVaSetValues(swapui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(swapui.top, (char *)"dialog_rc", NULL, 0);

        swapui.sel1 = CreateSetSelector(dialog, "Swap set:",
                                        SET_SELECT_ACTIVE,
                                        FILTER_SELECT_NONE,
                                        GRAPH_SELECT_CURRENT,
                                        SELECTION_TYPE_SINGLE);
        swapui.graph1_item = CreateGraphChoice(dialog, "In graph:", maxgraph, 1);

        swapui.sel2 = CreateSetSelector(dialog, "With set:",
                                        SET_SELECT_ACTIVE,
                                        FILTER_SELECT_ALL,
                                        GRAPH_SELECT_CURRENT,
                                        SELECTION_TYPE_SINGLE);
        DefineSetSelectorFilter(&swapui.sel2);
        swapui.graph2_item = CreateGraphChoice(dialog, "In graph:", maxgraph, 1);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_swap_proc, (XtPointer)&swapui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)swapui.top);

        XtManageChild(dialog);
    }
    XtRaise(swapui.top);
    unset_wait_cursor();
}

/*
 * setops - combine, copy sets - callbacks
 */

/*
 * activate a set and set its length
 */
static void do_activate_proc(Widget, XtPointer client_data, XtPointer)
{
    int setno, len, type;
    Act_ui *ui = (Act_ui *)client_data;
    setno = GetSelectedSet(ui->sel);
    type = GetChoice(ui->type_item);
    len = atoi((char *)xv_getstr(ui->len_item));
    setcomment(cg, setno, (char *)xv_getstr(ui->comment_item));
    setname(cg, setno, (char *)xv_getstr(ui->name_item));
    set_wait_cursor();
    do_activate(setno, type, len);
    if (ismaster)
    {
        cm->sendCommand_StringMessage(DO_ACTIVATE, (char *)xv_getstr(ui->comment_item));
        cm->sendCommand_StringMessage(DO_ACTIVATE2, (char *)xv_getstr(ui->name_item));
        cm->sendCommand_ValuesMessage(DO_ACTIVATE, setno, type, len, 0, 0, 0, 0, 0, 0, 0);
    }
    unset_wait_cursor();
}

/*
 * de-activate a set
 */
static void do_deactivate_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno;
    Deact_ui *ui = (Deact_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    set_wait_cursor();
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        if (ismaster)
            cm->sendCommandMessage(DO_DEACTIVATE, setno, 0);
        do_deactivate(cg, setno);
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

/*
 * re-activate a set
 */
static void do_reactivate_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno;
    React_ui *ui = (React_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    set_wait_cursor();
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        if (ismaster)
            cm->sendCommandMessage(DO_REACTIVATE, setno, 0);
        do_reactivate(cg, setno);
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

/*
 * change the type of a set
 */
static void do_changetype_proc(Widget, XtPointer client_data, XtPointer)
{
    int setno, type;
    Type_ui *ui = (Type_ui *)client_data;
    setno = GetSelectedSet(ui->sel);
    type = GetChoice(ui->type_item);
    setcomment(cg, setno, (char *)xv_getstr(ui->comment_item));
    setname(cg, setno, (char *)xv_getstr(ui->name_item));
    set_wait_cursor();
    do_changetype(setno, type);
    if (ismaster)
    {
        cm->sendCommand_StringMessage(DO_CHANGETYPE, (char *)xv_getstr(ui->comment_item));
        cm->sendCommand_StringMessage(DO_CHANGETYPE2, (char *)xv_getstr(ui->name_item));
        cm->sendCommandMessage(DO_CHANGETYPE, setno, type);
    }
    unset_wait_cursor();
}

/*
 * set the length of an active set - contents are destroyed
 */
static void do_setlength_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno, len;
    Length_ui *ui = (Length_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    len = atoi((char *)xv_getstr(ui->length_item));
    set_wait_cursor();
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        if (ismaster)
            cm->sendCommandMessage(DO_SETLENGTH, setno, len);
        do_setlength(setno, len);
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

/*
 * copy a set to another set, if the to set doesn't exist
 * get a new one, if it does, ask if it is okay to overwrite
 */
static void do_copy_proc(Widget, XtPointer client_data, XtPointer)
{
    int j1, j2, gto, i, *selsets;
    Copy_ui *ui = (Copy_ui *)client_data;
    int cnt;
    set_wait_cursor();
    cnt = GetSelectedSets(ui->sel, &selsets);
    for (i = 0; i < cnt; i++)
    {
        j1 = selsets[i];
        j2 = SET_SELECT_NEXT;
        gto = GetChoice(ui->graph_item);
        if (ismaster)
            cm->sendCommand_ValuesMessage(DO_COPY, j1, cg, j2, gto, 0, 0, 0, 0, 0, 0);
        do_copy(j1, cg, j2, gto);
    }
    free(selsets);
    unset_wait_cursor();
    drawgraph();
}

/*
 * move a set to another set, if the to set doesn't exist
 * get a new one, if it does, ask if it is okay to overwrite
 */
static void do_setmove_proc(Widget, XtPointer client_data, XtPointer)
{
    int j1, j2, gto, i, *selsets;
    Move_ui *ui = (Move_ui *)client_data;
    int cnt;
    set_wait_cursor();
    cnt = GetSelectedSets(ui->sel, &selsets);
    for (i = 0; i < cnt; i++)
    {
        j1 = selsets[i];
        j2 = SET_SELECT_NEXT;
        gto = GetChoice(ui->graph_item);
        if (ismaster)
            cm->sendCommand_ValuesMessage(DO_MOVE, j1, cg, j2, gto, 0, 0, 0, 0, 0, 0);
        do_move(j1, cg, j2, gto);
    }
    free(selsets);
    unset_wait_cursor();
    drawgraph();
}

/*
 * swap a set with another set
 */
static void do_swap_proc(Widget, XtPointer, XtPointer)
{
    int j1, j2, gto, gfrom;

    j1 = (int)GetChoice(swap_from_item);
    gfrom = (int)GetChoice(swap_fgraph_item);
    j2 = (int)GetChoice(swap_to_item);
    gto = (int)GetChoice(swap_tgraph_item);
    set_wait_cursor();
    if (ismaster)
        cm->sendCommand_ValuesMessage(DO_SWAP, j1, gfrom, j2, gto, 0, 0, 0, 0, 0, 0);
    do_swap(j1, gfrom, j2, gto);
    unset_wait_cursor();
    drawgraph();
}

/*
 * drop points from an active set
 */
static void do_drop_points_proc(Widget, XtPointer client_data, XtPointer)
{
    int i, *selsets;
    int cnt;
    int startno, endno, setno;
    Drop_ui *ui = (Drop_ui *)client_data;
    setno = GetSelectedSet(ui->sel);
    startno = atoi((char *)xv_getstr(ui->start_item)) - 1;
    endno = atoi((char *)xv_getstr(ui->stop_item)) - 1;
    set_wait_cursor();
    cnt = GetSelectedSets(ui->sel, &selsets);
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        if (ismaster)
            cm->sendCommand_ValuesMessage(DO_DROP_POINTS, setno, startno, endno, 0, 0, 0, 0, 0, 0, 0);
        do_drop_points(setno, startno, endno);
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

/*
 * append one set to another
 */
static void do_join_sets_proc(Widget, XtPointer client_data, XtPointer)
{
    int j1, j2;
    Join_ui *ui = (Join_ui *)client_data;
    j1 = GetSelectedSet(ui->sel1);
    j2 = GetSelectedSet(ui->sel2);
    set_wait_cursor();
    if (ismaster)
        cm->sendCommand_ValuesMessage(DO_JOIN_SETS, cg, j1, cg, j2, 0, 0, 0, 0, 0, 0);
    do_join_sets(cg, j1, cg, j2);
    unset_wait_cursor();
    drawgraph();
}

/*
 * reverse the order of a set
 */
static void do_reverse_sets_proc(Widget, XtPointer client_data, XtPointer)
{
    int setno;
    int cnt, i, *selsets;
    Reverse_ui *ui = (Reverse_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    set_wait_cursor();
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        if (ismaster)
            cm->sendCommandMessage(DO_REVERSE_SETS, setno, 0);
        do_reverse_sets(setno);
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

/*
 * coalesce sets
 */
static void do_coalesce_sets_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno;
    Coal_ui *ui = (Coal_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    set_wait_cursor();
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        if (ismaster)
            cm->sendCommandMessage(DO_COALESCE_SETS, setno, 0);
        do_coalesce_sets(setno);
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

/*
 * kill a set
 */
static void do_kill_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno, soft;
    Kill_ui *ui = (Kill_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    soft = (int)XmToggleButtonGetState(ui->soft_toggle);
    set_wait_cursor();
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        if (ismaster)
            cm->sendCommandMessage(DO_KILL, setno, soft);
        do_kill(cg, setno, soft);
    }
    free(selsets);
    unset_wait_cursor();
    drawgraph();
}

/*
 sort sets
*/
static void do_sort_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno, sorton, stype;
    int son[] = { PLOT_X, PLOT_Y, Y1, Y2, Y3, Y4, Y5 };
    Sort_ui *ui = (Sort_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    sorton = son[(int)GetChoice(ui->xy_item)];
    stype = (int)GetChoice(ui->up_down_item);

    set_wait_cursor();
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        if (ismaster)
            cm->sendCommand_ValuesMessage(DO_SORT, setno, sorton, stype, 0, 0, 0, 0, 0, 0, 0);
        do_sort(setno, sorton, stype);
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

/*
 *  write a set or sets to a file
 */
static void do_write_sets_proc(Widget, XtPointer client_data, XtPointer)
{
    int which_graph;
    int setno;
    int imbed, bin;
    char fn[256], *s;
    Write_ui *ui = (Write_ui *)client_data;
    Arg arg;
    XmString list_item;
    Widget p = ui->top;

    XtSetArg(arg, XmNtextString, &list_item);
    XtGetValues(p, &arg, 1);
    XmStringGetLtoR(list_item, charset, &s);

    strcpy(fn, s);
    XtFree(s);

    imbed = (int)XmToggleButtonGetState(ui->imbed_item);
    bin = (int)XmToggleButtonGetState(ui->binary_item);
    setno = GetSelectedSet(ui->sel);
    if (setno == SET_SELECT_ALL)
    {
        setno = -1;
    }
    which_graph = (int)GetChoice(ui->graph_item) - 1;
    strcpy(format, (char *)xv_getstr(ui->format_item));
    set_wait_cursor();
    if (ismaster)
        cm->sendCommand_StringMessage(DO_WRITESETS, fn);
    if (bin)
    {
        if (ismaster)
            cm->sendCommandMessage(DO_WRITESETS_BINARY, setno, 0);
        do_writesets_binary(cg, setno, fn);
    }
    else
    {
        if (ismaster)
        {
            cm->sendCommand_StringMessage(DO_WRITESETS2, format);
            cm->sendCommand_ValuesMessage(DO_WRITESETS, which_graph, setno, imbed, 0, 0, 0, 0, 0, 0, 0);
        }
        do_writesets(which_graph, setno, imbed, fn, format);
    }
    unset_wait_cursor();
}

/*
 * split sets split by itmp, remainder in last set.
 */
static void do_split_sets_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno, lpart;
    Split_ui *ui = (Split_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    lpart = atoi((char *)xv_getstr(ui->len_item));
    set_wait_cursor();
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        if (ismaster)
            cm->sendCommandMessage(DO_SPLITSETS, setno, lpart);
        do_splitsets(cg, setno, lpart);
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

void create_saveall_popup(Widget, XtPointer, XtPointer)
{
    static Widget top;
    Widget dialog;
    Widget fr;

    set_wait_cursor();
    if (top == NULL)
    {
        top = XmCreateFileSelectionDialog(app_shell, (char *)"save_all_sets", NULL, 0);
        XtVaSetValues(XtParent(top), XmNtitle, "Save all sets", NULL);

        XtAddCallback(top, XmNokCallback, (XtCallbackProc)do_saveall_sets_proc, (XtPointer)top);
        XtAddCallback(top, XmNcancelCallback, (XtCallbackProc)destroy_dialog, (XtPointer)top);

        fr = XmCreateFrame(top, (char *)"fr", NULL, 0);
        dialog = XmCreateRowColumn(fr, (char *)"dialog_rc", NULL, 0);

        saveall_sets_format_item = CreateTextItem2(dialog, 15, "Format: ");

        XtManageChild(dialog);
        XtManageChild(fr);

        xv_setstr(saveall_sets_format_item, sformat);
    }
    XtRaise(top);
    unset_wait_cursor();
}

/*
 *  write a set or sets to a file
 */
static void do_saveall_sets_proc(Widget, XtPointer client_data, XtPointer)
{
    char fn[256], *s;

    Arg arg;
    XmString list_item;
    Widget p = (Widget)client_data;

    XtSetArg(arg, XmNtextString, &list_item);
    XtGetValues(p, &arg, 1);
    XmStringGetLtoR(list_item, charset, &s);

    strcpy(fn, s);
    XtFree(s);

    strcpy(sformat, (char *)xv_getstr(saveall_sets_format_item));
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommand_StringMessage(DO_WRITESETS, fn);
        cm->sendCommand_StringMessage(DO_WRITESETS2, sformat);
        cm->sendCommand_ValuesMessage(DO_WRITESETS, maxgraph, -1, 1, 0, 0, 0, 0, 0, 0, 0);
    }
    do_writesets(maxgraph, -1, 1, fn, sformat);
    unset_wait_cursor();
}
