/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: hotwin.c,v 1.6 1994/11/02 04:40:52 pturner Exp pturner $
 *
 * hot links
 *
 */

#include <stdio.h>
#include <math.h>
#include "extern.h"
#include "globals.h"

#include <Xm/Xm.h>
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

#include "motifinc.h"

#include "extern2.h"

static Widget hotlink_frame = (Widget)NULL;
static SetChoiceItem hotlink_set_item;
static Widget hotlink_list_item;
static Widget hotlink_file_item;
static Widget *hotlink_source_item;

void create_hotfiles_popup(Widget w, XtPointer client_data, XtPointer call_data);

static void do_hotlink_proc(Widget, XtPointer, XtPointer)
{
    int setno, src;
    char fname[256];
    char buf[256];
    XmString xms;

    set_wait_cursor();

    setno = GetSelectedSet(hotlink_set_item);
    src = GetChoice(hotlink_source_item);
    strcpy(fname, xv_getstr(hotlink_file_item));

    sprintf(buf, "S%02d -> %s -> %s", setno, src == 0 ? "DISK" : "PIPE", fname);

    xms = XmStringCreateLtoR(buf, charset);
    XmListAddItemUnselected(hotlink_list_item, xms, 0);

    set_hotlink(cg, setno, TRUE, fname, src == 0 ? DISK : PIPE);

    XmStringFree(xms);

    unset_wait_cursor();
}

static void do_hotunlink_proc(Widget, XtPointer, XtPointer)
{
    XmString *s, cs;
    int *pos_list;
    int pos_cnt, cnt;
    char *cstr;
    int setno;

    set_wait_cursor();

    if (XmListGetSelectedPos(hotlink_list_item, &pos_list, &pos_cnt))
    {
        //j = pos_list[0];
        XtVaGetValues(hotlink_list_item,
                      XmNselectedItemCount, &cnt,
                      XmNselectedItems, &s,
                      NULL);
        cs = XmStringCopy(*s);
        if (XmStringGetLtoR(cs, charset, &cstr))
        {
            sscanf(cstr, "S%d", &setno);
            if (setno >= 0 && setno < g[cg].maxplot)
            {
                set_hotlink(cg, setno, FALSE, NULL, 0);
            }
            XtFree(cstr);
        }
        XmStringFree(cs);
        update_hotlinks();
    }

    unset_wait_cursor();
}

void update_hotlinks(void)
{
    int i, j;
    char buf[256];
    XmString xms;

    if (hotlink_frame != NULL)
    {
        set_wait_cursor();
        XmListDeleteAllItems(hotlink_list_item);
        for (i = 0; i < maxgraph; i++)
        {
            for (j = 0; j < g[i].maxplot; j++)
            {
                if (is_hotlinked(i, j))
                {
                    sprintf(buf, "S%02d -> %s -> %s", j,
                            get_hotlink_src(i, j) == DISK ? "DISK" : "PIPE",
                            get_hotlink_file(i, j));
                    xms = XmStringCreateLtoR(buf, charset);
                    XmListAddItemUnselected(hotlink_list_item, xms, 0);
                    XmStringFree(xms);
                }
            }
        }
        unset_wait_cursor();
    }
}

static void do_hotupdate_proc(Widget, XtPointer, XtPointer)
{
    int i;

    set_wait_cursor();

    for (i = 0; i < g[cg].maxplot; i++)
    {
        if (is_hotlinked(cg, i))
        {
            do_update_hotlink(cg, i);
        }
    }

    unset_wait_cursor();
    drawgraph();
}

void create_hotlinks_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    static Widget top, dialog;
    Arg args[3];
    set_wait_cursor();
    if (top == NULL)
    {
        char *label1[5];
        Widget but1[5];
        label1[0] = (char *)"Link";
        label1[1] = (char *)"Files...";
        label1[2] = (char *)"Unlink";
        label1[3] = (char *)"Update";
        label1[4] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        top = XmCreateDialogShell(app_shell, (char *)"Hot links", NULL, 0);
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(top, (char *)"dialog_rc", NULL, 0);

        XtSetArg(args[0], XmNlistSizePolicy, XmRESIZE_IF_POSSIBLE);
        XtSetArg(args[1], XmNvisibleItemCount, 5);
        hotlink_list_item = XmCreateScrolledList(dialog, (char *)"list", args, 2);
        XtManageChild(hotlink_list_item);

        hotlink_set_item = CreateSetSelector(dialog, "Link set:",
                                             SET_SELECT_ACTIVE,
                                             FILTER_SELECT_ALL,
                                             GRAPH_SELECT_CURRENT,
                                             SELECTION_TYPE_MULTIPLE);
        DefineSetSelectorFilter(&hotlink_set_item);

        hotlink_file_item = CreateTextItem2(dialog, 30, "To file or pipe:");
        hotlink_source_item = CreatePanelChoice(dialog, "Source: ", 3,
                                                "Disk file",
                                                "Pipe",
                                                NULL,
                                                NULL);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 5, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_hotlink_proc,
                      (XtPointer)NULL);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)create_hotfiles_popup,
                      (XtPointer)NULL);
        XtAddCallback(but1[2], XmNactivateCallback, (XtCallbackProc)do_hotunlink_proc,
                      (XtPointer)NULL);
        XtAddCallback(but1[3], XmNactivateCallback, (XtCallbackProc)do_hotupdate_proc,
                      (XtPointer)NULL);
        XtAddCallback(but1[4], XmNactivateCallback, (XtCallbackProc)destroy_dialog,
                      (XtPointer)top);

        XtManageChild(dialog);
        hotlink_frame = top;
    }
    XtRaise(top);
    update_hotlinks();
    unset_wait_cursor();
}

static void do_hotlinkfile_proc(Widget, XtPointer client_data, XtPointer)
{
    Widget dialog = (Widget)client_data;
    Arg args;
    XmString list_item;
    char *s;

    set_wait_cursor();

    XtSetArg(args, XmNtextString, &list_item);
    XtGetValues(dialog, &args, 1);
    XmStringGetLtoR(list_item, charset, &s);

    xv_setstr(hotlink_file_item, s);

    XtFree(s);

    unset_wait_cursor();

    XtUnmanageChild(dialog);
}

void create_hotfiles_popup(Widget, XtPointer, XtPointer)
{
    static Widget top;

    set_wait_cursor();
    if (top == NULL)
    {
        top = XmCreateFileSelectionDialog(app_shell, (char *)"hotlinks", NULL, 0);
        XtVaSetValues(XtParent(top), XmNtitle, "Select hot link file", NULL);

        XtAddCallback(top, XmNokCallback, (XtCallbackProc)do_hotlinkfile_proc, (XtPointer)top);
        XtAddCallback(top, XmNcancelCallback, (XtCallbackProc)destroy_dialog, (XtPointer)top);
    }
    XtRaise(top);
    unset_wait_cursor();
}
