/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: graphwin.c,v 1.5 1994/09/29 03:37:37 pturner Exp pturner $
 *
 * graph Panel
 *
 */

#include <stdio.h>
#include <stdlib.h>

#include <Xm/Xm.h>
#include <Xm/BulletinB.h>
#include <Xm/DialogS.h>
#include <Xm/Frame.h>
#include <Xm/FileSB.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/Separator.h>
#include <Xm/ToggleB.h>

#include "globals.h"
#include "motifinc.h"

static Widget but1[2];

static Widget graphs_frame;
static Widget graph_show_frame;
static Widget graph_focus_frame;
static Widget graph_type_frame;
static Widget graph_invert_frame;

// static XmString gstring;

/*
 * Panel item declarations
 */
// static Widget graph_curgraph_message_item;
static Widget *graph_setcur_choice_item;
static Widget *graph_rendsets_choice_item;
static Widget *graph_focus_choice_item;
static Widget graph_drawfocus_choice_item;
static Widget *graph_activate_choice_item;
static Widget *graph_kill_choice_item;
static Widget *graph_copyfrom_choice_item;
static Widget *graph_copyto_choice_item;
static Widget *graph_swapfrom_choice_item;
static Widget *graph_swapto_choice_item;
static Widget *graph_show_choice_item;
static Widget *graph_invert_choice_item;
static Widget graph_invertx_item;
static Widget graph_inverty_item;
static Widget graph_flipxy_item;

/*
 * Event and Notify proc declarations
 */
static void graph_rendsets_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void graph_focus_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void graph_activate_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void graph_kill_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void graph_copy_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void graph_show_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void graph_swap_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void graph_invert_notify_proc(Widget w, XtPointer cdata, XtPointer call_data);
static void update_invert_items(int gno);
static void update_type_items(int gno);
static void update_focus_items(int gno);
static void update_show_items(void);
static void create_gactive_frame(Widget w, XtPointer client_data, XtPointer call_data);
static void create_gcopy_frame(Widget w, XtPointer client_data, XtPointer call_data);
static void create_gswap_frame(Widget w, XtPointer client_data, XtPointer call_data);
static void create_gkill_frame(Widget w, XtPointer client_data, XtPointer call_data);
static void create_gfocus_frame(Widget w, XtPointer client_data, XtPointer call_data);
static void create_gshow_frame(Widget w, XtPointer client_data, XtPointer call_data);
static void create_ginvert_frame(Widget w, XtPointer client_data, XtPointer call_data);

static int gtypes[] = {
    XY, LOGX, LOGY, LOGXY, BAR,
    STACKEDBAR, HBAR, STACKEDHBAR, BOXPLOT, HBOXPLOT, XY, XY, XY, XY
};

void update_graph_items(void)
{
    update_focus_items(cg);
    update_show_items();
    update_type_items(cg);
    update_invert_items(cg);
}

void create_graph_frame(Widget, XtPointer, XtPointer)
{
    Widget wbut, graphs_panel;
    int x, y;
    set_wait_cursor();
    if (graphs_frame == NULL)
    {
        XmGetPos(app_shell, 0, &x, &y);
        graphs_frame = XmCreateDialogShell(app_shell, (char *)"Graph ops", NULL, 0);
        handle_close(graphs_frame);
        XtVaSetValues(graphs_frame, XmNx, x, XmNy, y, NULL);
        graphs_panel = XmCreateRowColumn(graphs_frame, (char *)"graphs_rc", NULL, 0);

        wbut = XtVaCreateManagedWidget("Activate...", xmPushButtonWidgetClass, graphs_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_gactive_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Copy...", xmPushButtonWidgetClass, graphs_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_gcopy_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Swap...", xmPushButtonWidgetClass, graphs_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_gswap_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Kill...", xmPushButtonWidgetClass, graphs_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_gkill_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Focus...", xmPushButtonWidgetClass, graphs_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_gfocus_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Show...", xmPushButtonWidgetClass, graphs_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_gshow_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Set graph type...", xmPushButtonWidgetClass, graphs_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_gtype_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Arrange...", xmPushButtonWidgetClass, graphs_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_arrange_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Overlay...", xmPushButtonWidgetClass, graphs_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_overlay_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Invert/flip axes...", xmPushButtonWidgetClass, graphs_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_ginvert_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Image...", xmPushButtonWidgetClass, graphs_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_image_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Close", xmPushButtonWidgetClass, graphs_panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)graphs_frame);

        XtManageChild(graphs_panel);
    }
    update_graph_items();
    XtRaise(graphs_frame);
    unset_wait_cursor();
}

/*
 * Notify and event procs
 */
static void graph_rendsets_notify_proc(Widget, XtPointer, XtPointer)
{
    int graphtype = (int)GetChoice(graph_rendsets_choice_item);

    if (g[cg].type != gtypes[graphtype])
    {
        g[cg].type = gtypes[graphtype];
        autoscale_graph(cg, -3);
        update_all(cg);
        drawgraph();
    }
}

static void graph_focus_notify_proc(Widget, XtPointer, XtPointer)
{
    int newcg;

    switch ((int)GetChoice(graph_focus_choice_item))
    {
    case 0:
        focus_policy = CLICK;
        break;
    case 1:
        focus_policy = SET;
        break;
    case 2:
        focus_policy = FOLLOWS;
        break;
    }
    draw_focus_flag = (int)XmToggleButtonGetState(graph_drawfocus_choice_item) ? ON : OFF;
    newcg = (int)GetChoice(graph_setcur_choice_item);
    if (newcg != cg)
    {
        switch_current_graph(cg, newcg);
    }
}

static void graph_activate_notify_proc(Widget, XtPointer, XtPointer)
{
    int gno = (int)GetChoice(graph_activate_choice_item);

    set_graph_active(gno);
    update_all(cg);
    drawgraph();
}

static void graph_kill_notify_proc(Widget, XtPointer, XtPointer)
{
    kill_graph((int)GetChoice(graph_kill_choice_item));
    update_all(cg);
    drawgraph();
}

static void graph_copy_notify_proc(Widget, XtPointer, XtPointer)
{
    int from = (int)GetChoice(graph_copyfrom_choice_item);
    int to = (int)GetChoice(graph_copyto_choice_item);

    if (from == to)
    {
        errwin("Graph from and graph to are the same");
        return;
    }
    if (!isactive_graph(from))
    {
        errwin("Graph from isn't active");
        return;
    }
    if (isactive_graph(to))
    {
        if (!yesno("Graph to copy to is active, kill it?", NULL, NULL, NULL))
        {
            return;
        }
    }
    copy_graph(from, to);
    update_all(cg);
    drawgraph();
}

static void graph_show_notify_proc(Widget, XtPointer, XtPointer)
{
    int i;

    for (i = 0; i < maxgraph; i++)
    {
        if (XmToggleButtonGetState(graph_show_choice_item[i]) == True)
        {
            g[i].hidden = FALSE;
        }
        else
        {
            g[i].hidden = TRUE;
        }
    }
    update_all(cg);
    drawgraph();
}

static void graph_swap_notify_proc(Widget, XtPointer, XtPointer)
{
    int from = (int)GetChoice(graph_swapfrom_choice_item);
    int to = (int)GetChoice(graph_swapto_choice_item);

    if (from == to)
    {
        errwin("Graph from and graph to are the same");
        return;
    }
    swap_graph(from, to);
    update_all(cg);
    drawgraph();
}

/*
 */
static void update_type_items(int gno)
{
    int i;

    if (graph_type_frame)
    {
        i = 0;
        while (g[gno].type != gtypes[i])
            i++;
        if (i > 8)
        {
            errwin("Graph type not found");
        }
        else
        {
            SetChoice(graph_rendsets_choice_item, i);
        }
    }
}

static void update_focus_items(int gno)
{
    int itest = 0;

    if (graph_focus_frame)
    {
        SetChoice(graph_setcur_choice_item, gno);
        if (focus_policy == SET)
        {
            itest = 1;
        }
        else if (focus_policy == CLICK)
        {
            itest = 0;
        }
        else if (focus_policy == FOLLOWS)
        {
            itest = 2;
        }
        SetChoice(graph_focus_choice_item, itest);
        XmToggleButtonSetState(graph_drawfocus_choice_item,
                               draw_focus_flag == ON ? True : False, False);
    }
}

static void update_show_items(void)
{
    int i;

    if (graph_show_frame)
    {
        for (i = 0; i < maxgraph; i++)
        {
            if (g[i].hidden)
            {
                XmToggleButtonSetState(graph_show_choice_item[i], False, False);
            }
            else
            {
                XmToggleButtonSetState(graph_show_choice_item[i], True, False);
            }
        }
    }
}

static void update_invert_items(int /*gno*/)
{
    if (graph_invert_frame)
    {
    }
}

static void graph_invert_notify_proc(Widget, XtPointer, XtPointer)
{
    int i, invx, invy, fxy, gstart, gstop;
    invx = XmToggleButtonGetState(graph_invertx_item);
    invy = XmToggleButtonGetState(graph_inverty_item);
    fxy = XmToggleButtonGetState(graph_flipxy_item);
    if (GetChoice(graph_invert_choice_item) == 1)
    {
        gstart = 0;
        gstop = maxgraph - 1;
    }
    else
    {
        gstart = gstop = cg;
    }
    for (i = gstart; i <= gstop; i++)
    {
        if (isactive_graph(i))
        {
            if (invx)
            {
                invertx(i);
            }
            if (invy)
            {
                inverty(i);
            }
            if (fxy)
            {
                flipxy(i);
            }
        }
    }
    update_all(cg);
    drawgraph();
}

static void create_gactive_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    static Widget top, dialog;

    set_wait_cursor();
    if (top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        top = XmCreateDialogShell(app_shell, (char *)"Activate graphs", NULL, 0);
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(top, (char *)"rc", NULL, 0);

        graph_activate_choice_item = CreateGraphChoice(dialog, "Activate graph: ", maxgraph, 0);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)graph_activate_notify_proc, (XtPointer)NULL);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)top);

        XtManageChild(dialog);
    }
    XtRaise(top);
    unset_wait_cursor();
}

static void create_gcopy_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    static Widget top, dialog;

    set_wait_cursor();
    if (top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        top = XmCreateDialogShell(app_shell, (char *)"Copy graphs", NULL, 0);
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(top, (char *)"rc", NULL, 0);

        graph_copyfrom_choice_item = CreateGraphChoice(dialog, "Copy graph: ", maxgraph, 0);
        graph_copyto_choice_item = CreateGraphChoice(dialog, "To graph: ", maxgraph, 0);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)graph_copy_notify_proc, (XtPointer)NULL);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)top);

        XtManageChild(dialog);
    }
    XtRaise(top);
    unset_wait_cursor();
}

static void create_gswap_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    static Widget top, dialog;
    // Widget rc;

    set_wait_cursor();
    if (top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        top = XmCreateDialogShell(app_shell, (char *)"Swap graphs", NULL, 0);
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(top, (char *)"rc", NULL, 0);

        graph_swapfrom_choice_item = CreateGraphChoice(dialog, "Swap graph: ", maxgraph, 0);

        graph_swapto_choice_item = CreateGraphChoice(dialog, "With graph: ", maxgraph, 0);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)graph_swap_notify_proc, (XtPointer)NULL);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)top);

        XtManageChild(dialog);
    }
    XtRaise(top);
    unset_wait_cursor();
}

static void create_gkill_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    static Widget top, dialog;

    set_wait_cursor();
    if (top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        top = XmCreateDialogShell(app_shell, (char *)"Kill graphs", NULL, 0);
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(top, (char *)"rc", NULL, 0);

        graph_kill_choice_item = CreateGraphChoice(dialog, "Kill graph: ", maxgraph, 0);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)graph_kill_notify_proc, (XtPointer)NULL);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)top);

        XtManageChild(dialog);
    }
    XtRaise(top);
    unset_wait_cursor();
}

static void create_gfocus_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    static Widget dialog;

    set_wait_cursor();
    if (graph_focus_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        graph_focus_frame = XmCreateDialogShell(app_shell, (char *)"Set focus", NULL, 0);
        handle_close(graph_focus_frame);
        XtVaSetValues(graph_focus_frame, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(graph_focus_frame, (char *)"rc", NULL, 0);
        graph_setcur_choice_item = CreateGraphChoice(dialog, "Set current graph to", maxgraph, 0);

        graph_focus_choice_item = (Widget *)CreatePanelChoice(dialog, "Graph focus",
                                                              4,
                                                              "Button press",
                                                              "As set",
                                                              "Follows mouse",
                                                              NULL,
                                                              NULL);
        graph_drawfocus_choice_item = XtVaCreateManagedWidget("Display focus markers",
                                                              xmToggleButtonWidgetClass, dialog,
                                                              NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)graph_focus_notify_proc, (XtPointer)NULL);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)graph_focus_frame);

        XtManageChild(dialog);
    }
    XtRaise(graph_focus_frame);
    update_focus_items(cg);
    unset_wait_cursor();
}

static void create_gshow_frame(Widget, XtPointer, XtPointer)
{
    int i, x, y;
    static Widget dialog;
    Widget lab, rc;

    set_wait_cursor();
    if (graph_show_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        graph_show_frame = XmCreateDialogShell(app_shell, (char *)"Show graphs", NULL, 0);
        handle_close(graph_show_frame);
        XtVaSetValues(graph_show_frame, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(graph_show_frame, (char *)"rc", NULL, 0);

        lab = XtVaCreateManagedWidget("Select graphs for display (a graph must also be active to be seen):", xmLabelWidgetClass, dialog,
                                      NULL);
        rc = XtVaCreateManagedWidget("rc", xmRowColumnWidgetClass, dialog,
                                     XmNorientation, XmHORIZONTAL,
                                     XmNnumColumns, maxgraph / 10,
                                     XmNpacking, XmPACK_COLUMN,
                                     NULL);
        graph_show_choice_item = (Widget *)XtMalloc(maxgraph * sizeof(Widget));
        for (i = 0; i < maxgraph; i++)
        {
            sprintf(buf, "%d", i);
            graph_show_choice_item[i] = XtVaCreateManagedWidget(buf,
                                                                xmToggleButtonWidgetClass, rc,
                                                                NULL);
        }

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)graph_show_notify_proc, (XtPointer)NULL);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)graph_show_frame);

        XtManageChild(dialog);
    }
    XtRaise(graph_show_frame);
    update_show_items();
    unset_wait_cursor();
}

void create_gtype_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    static Widget dialog;

    set_wait_cursor();
    if (graph_type_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        graph_type_frame = XmCreateDialogShell(app_shell, (char *)"Set graph type", NULL, 0);
        handle_close(graph_type_frame);
        XtVaSetValues(graph_type_frame, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(graph_type_frame, (char *)"rc", NULL, 0);

        graph_rendsets_choice_item = CreatePanelChoice(dialog, "Set current graph type to",
                                                       9,
                                                       "XY graph",
                                                       "Log-linear",
                                                       "Linear-log",
                                                       "Log-log",
                                                       "Bar chart",
                                                       "Stacked bar",
                                                       "Horizontal bar chart",
                                                       "Horizontal stacked bar",
                                                       NULL,
                                                       NULL);
        /*
                               "Boxplot",
                               "Horizontal box plot",
      */

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)graph_rendsets_notify_proc, (XtPointer)NULL);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)graph_type_frame);

        XtManageChild(dialog);
    }
    XtRaise(graph_type_frame);
    update_type_items(cg);
    unset_wait_cursor();
}

static Widget image_frame;
// static Widget image_name_item;
static Widget image_x_item;
static Widget image_y_item;
static Widget image_display_item;
XImage *img = NULL;
int imagew, imageh;
int drawimage_flag = 1;
static int open_image_dialog = 0;
void do_rimage_proc(Widget w, XtPointer client_data, XtPointer call_data);

void create_rimage_popup(Widget w, XtPointer client_data, XtPointer call_data);

void drawimage()
{
    extern Display *disp;
    extern Window xwin;
    extern GC gc;
    if (img != NULL && drawimage_flag)
    {
        XPutImage(disp, xwin, gc, img, 0, 0, imagex, imagey, imagew, imageh);
    }
}

static void update_image()
{
    char buf[256];
    XmToggleButtonSetState(image_display_item, drawimage_flag, False);
    sprintf(buf, "%d", imagex);
    xv_setstr(image_x_item, buf);
    sprintf(buf, "%d", imagey);
    xv_setstr(image_y_item, buf);
}

void do_accept_image_proc(Widget, XtPointer, XtPointer)
{
    drawimage_flag = XmToggleButtonGetState(image_display_item);
    imagex = atoi((char *)xv_getstr(image_x_item));
    imagey = atoi((char *)xv_getstr(image_y_item));
    drawgraph();
}

void create_image_frame(Widget w, XtPointer client_data, XtPointer call_data)
{
    int x, y;
    Widget dialog;
    Widget but3[3];

    if (img == NULL)
    {
        open_image_dialog = 1;
        create_rimage_popup(w, client_data, call_data);
        return;
    }
    set_wait_cursor();
    if (image_frame == NULL)
    {
        char *label3[3];
        label3[0] = (char *)"Accept";
        label3[1] = (char *)"Read image...";
        label3[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        image_frame = XmCreateDialogShell(app_shell, (char *)"Image", NULL, 0);
        handle_close(image_frame);
        XtVaSetValues(image_frame, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(image_frame, (char *)"dialog_rc", NULL, 0);

        /*    image_name_item = CreateTextItem2(dialog, 15, "Image file name: ");*/
        image_x_item = CreateTextItem2(dialog, 20, "Anchor top left to X (in pixels) = : ");
        image_y_item = CreateTextItem2(dialog, 20, "Anchor top left to Y (in pixels) = : ");
        image_display_item = XtVaCreateManagedWidget("Display image",
                                                     xmToggleButtonWidgetClass, dialog,
                                                     NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 3, but3, label3);
        XtAddCallback(but3[0], XmNactivateCallback, (XtCallbackProc)do_accept_image_proc, (XtPointer)NULL);
        XtAddCallback(but3[1], XmNactivateCallback, (XtCallbackProc)create_rimage_popup, (XtPointer)NULL);
        XtAddCallback(but3[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)image_frame);

        XtManageChild(dialog);
    }
    XtRaise(image_frame);
    update_image();
    unset_wait_cursor();
}

static Widget rimage_dialog;

void close_rimage_popup(Widget, XtPointer, XtPointer)
{
    XtUnmanageChild(rimage_dialog);
}

void read_image(char *fname)
{
    extern Display *disp;
    extern Window xwin;
    int width, height, depth;
    if (img != NULL)
    {
        XDestroyImage(img);
    }
    set_wait_cursor();
    img = read_image_from_disk(disp,
                               xwin,
                               fname,
                               &width,
                               &height,
                               &depth);
    unset_wait_cursor();
    if (img != NULL)
    {
        imagew = width;
        imageh = height;
    }
    else
    {
        errwin("Unable to load image");
    }
}

void do_rimage_proc(Widget, XtPointer, XtPointer)
{
    extern Display *disp;
    extern Window xwin;
    extern GC gc;
    int width, height, depth;

    Arg args;
    XmString list_item;
    char *s;

    XtSetArg(args, XmNtextString, &list_item);
    XtGetValues(rimage_dialog, &args, 1);
    XmStringGetLtoR(list_item, charset, &s);
    strcpy(image_filename, s);
    if (img != NULL)
    {
        XDestroyImage(img);
    }
    set_wait_cursor();
    img = read_image_from_disk(disp,
                               xwin,
                               image_filename,
                               &width,
                               &height,
                               &depth);
    if (img != NULL)
    {
        imagew = width;
        imageh = height;
        XPutImage(disp, xwin, gc, img, 0, 0, 0, 0, width, height);
    }
    else
    {
        errwin("Unable to load image");
    }
    unset_wait_cursor();

    XtFree(s);

    XtUnmanageChild(rimage_dialog);
    if (open_image_dialog)
    {
        open_image_dialog = 0;
        create_image_frame((Widget)NULL, (XtPointer)NULL, (XtPointer)NULL);
    }
}

void create_rimage_popup(Widget, XtPointer, XtPointer)
{
    if (rimage_dialog == NULL)
    {
        rimage_dialog = XmCreateFileSelectionDialog(app_shell, (char *)"rimage_dialog", NULL, 0);
        XtVaSetValues(XtParent(rimage_dialog),
                      XmNtitle, "Read image (.xwd format)",
                      NULL);
        XtVaSetValues(rimage_dialog,
                      XmNdirMask, XmStringCreate((char *)"*.xwd", charset),
                      NULL);
        XtAddCallback(rimage_dialog, XmNcancelCallback, (XtCallbackProc)close_rimage_popup, (XtPointer)NULL);
        XtAddCallback(rimage_dialog, XmNokCallback, (XtCallbackProc)do_rimage_proc, (XtPointer)NULL);
    }
    XtRaise(rimage_dialog);
}

static void create_ginvert_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;
    //  Widget lab, wbut, rc;

    set_wait_cursor();
    if (graph_invert_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        graph_invert_frame = XmCreateDialogShell(app_shell, (char *)"Invert/flip axes", NULL, 0);
        handle_close(graph_invert_frame);
        XtVaSetValues(graph_invert_frame, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(graph_invert_frame, (char *)"rc", NULL, 0);
        graph_invertx_item = XtVaCreateManagedWidget("Invert X",
                                                     xmToggleButtonWidgetClass, dialog,
                                                     NULL);
        graph_inverty_item = XtVaCreateManagedWidget("Invert Y",
                                                     xmToggleButtonWidgetClass, dialog,
                                                     NULL);
        graph_flipxy_item = XtVaCreateManagedWidget("Flip XY",
                                                    xmToggleButtonWidgetClass, dialog,
                                                    NULL);
        graph_invert_choice_item = CreatePanelChoice(dialog, "Apply to graph",
                                                     3,
                                                     "Current",
                                                     "All",
                                                     NULL,
                                                     NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)graph_invert_notify_proc, (XtPointer)NULL);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)graph_invert_frame);

        XtManageChild(dialog);
    }
    XtRaise(graph_invert_frame);
    update_invert_items(cg);
    unset_wait_cursor();
}
