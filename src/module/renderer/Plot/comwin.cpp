/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: comwin.c,v 1.4 1994/09/29 03:37:37 pturner Exp pturner $
 *
 * Command Panel
 *
 */

#include <stdio.h>

#include <Xm/Xm.h>
#include <Xm/DialogS.h>
#include <Xm/BulletinB.h>
#include <Xm/Frame.h>
#include <Xm/FileSB.h>
#include <Xm/Command.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/Text.h>
#include <Xm/List.h>
#include <Xm/Separator.h>

#include "globals.h"
#include "motifinc.h"

/* all declared in pars.yacc */
extern int gotbatch;
extern int gotparams;
extern int gotread;
extern int readsrc, readtype;
extern char batchfile[];
extern char paramfile[];
extern char readfile[];
extern double result;

/*
 * Widget item declarations
 */

static Widget command;
static Widget comshell;

void excovise_cmd(char *s);
static void comcall(Widget w, XtPointer cd, XtPointer calld);
static void clear_history(Widget w, XtPointer client_data, XtPointer call_data);
static void replay_history(Widget w, XtPointer client_data, XtPointer call_data);
static void whist_apply_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);

void excovise_cmd(char *s)
{
    static int errpos, errorcount;
    // static char val[256];
    errorcount = errpos = 0;
    if (s == NULL)
    {
        return;
    }
    if (s[strlen(s) - 1] == '\n')
    {
        s[strlen(s) - 1] = 0;
    }
    fixupstr(s);
    scanner(s, getx(cg, curset), gety(cg, curset),
            getsetlength(cg, curset), ax, bx, cx, dx, MAXARR, 0, curset, &errpos);
    if (errpos)
    {
        errorcount++;
    }
    /*
    * the yacc grammar is not re-entrant so run these separately
    */
    if (gotbatch && batchfile[0])
    {
        runbatch(batchfile);
        gotbatch = 0;
    }
    else if (gotparams && paramfile[0])
    {
        if (!getparms(cg, paramfile))
        {
        }
        gotparams = 0;
    }
    else if (gotread && readfile[0])
    {
        if (getdata(cg, readfile, readsrc, readtype))
        {
            drawgraph();
        }
        gotread = 0;
    }
}

static void comcall(Widget, XtPointer, XtPointer calld)
{
    static int errpos, errorcount;
    static char val[256];
    char *ts;
    XmCommandCallbackStruct *s = (XmCommandCallbackStruct *)calld;
    errorcount = errpos = 0;
    XmStringGetLtoR(s->value, charset, &ts);
    strcpy(val, ts);
    XtFree(ts);
    fixupstr(val);
    scanner(val, getx(cg, curset), gety(cg, curset),
            getsetlength(cg, curset), ax, bx, cx, dx,
            MAXARR, 0, curset, &errpos);
    if (errpos)
    {
        errorcount++;
    }
    /*
    * the yacc grammar is not re-entrant so run these separately
    */
    if (gotbatch && batchfile[0])
    {
        runbatch(batchfile);
        gotbatch = 0;
    }
    else if (gotparams && paramfile[0])
    {
        if (!getparms(cg, paramfile))
        {
        }
        gotparams = 0;
    }
    else if (gotread && readfile[0])
    {
        if (getdata(cg, readfile, readsrc, readtype))
        {
            drawgraph();
        }
        gotread = 0;
    }
}

static void clear_history(Widget, XtPointer, XtPointer)
{
    int i;
    int ac = 0, hc;
    Arg al[5];
    Widget h = XmCommandGetChild(command, XmDIALOG_HISTORY_LIST);
    ac = 0;
    XtSetArg(al[ac], XmNhistoryItemCount, &hc);
    ac++;
    XtGetValues(command, al, ac);
    for (i = 0; i < hc; i++)
    {
        XmListDeletePos(h, 0);
    }
}

static const char *helptext = "Several errors have been found in the history list, check the history list for errors, delete them and try again.";

static void replay_history(Widget, XtPointer, XtPointer)
{
    static int errpos, errcount;
    extern int gotbatch;
    extern char batchfile[];
    extern int gotparams;
    extern char paramfile[];
    char buf[256], *ts;
    int i;
    int ac = 0, hc;
    XmStringTable xmstrs;
    Arg al[5];
    ac = 0;
    XtSetArg(al[ac], XmNhistoryItems, &xmstrs);
    ac++;
    XtSetArg(al[ac], XmNhistoryItemCount, &hc);
    ac++;
    XtGetValues(command, al, ac);
    errcount = 0;
    for (i = 0; i < hc; i++)
    {
        errpos = 0;
        XmStringGetLtoR(xmstrs[i], charset, &ts);
        strcpy(buf, ts);
        XtFree(ts);
        fixupstr(buf);
        scanner(buf, getx(cg, curset), gety(cg, curset), getsetlength(cg, curset), ax, bx, cx, dx, MAXARR, 0, curset, &errpos);
        if (errpos)
        {
            errcount++;
        }
        if (errcount > 3)
        {
            if (yesno("Lots of errors, cancel?", "Press YES or NO", helptext, "NO"))
            {
                break;
            }
            else
            {
                errcount = 0;
            }
        }
        /*
       * the yacc grammar is not re-entrant so run these separately
       */
        if (gotbatch && batchfile[0])
        {
            runbatch(batchfile);
            gotbatch = 0;
        }
        else if (gotparams && paramfile[0])
        {
            if (!getparms(cg, paramfile))
            {
            }
            gotparams = 0;
        }
        else if (gotread && readfile[0])
        {
            if (getdata(cg, readfile, readsrc, readtype))
            {
                drawgraph();
            }
            gotread = 0;
        }
    }
}

void open_command(Widget, XtPointer, XtPointer)
{
    Widget bt, fr, rc;
    int x, y;
    set_wait_cursor();
    if (command == NULL)
    {
        XmGetPos(app_shell, 0, &x, &y);
        comshell = XmCreateDialogShell(app_shell, (char *)"Commands", NULL, 0);
        handle_close(comshell);
        XtVaSetValues(comshell, XmNx, x, XmNy, y, NULL);
        command = XmCreateCommand(comshell, (char *)"command", NULL, 0);
        XtVaSetValues(command,
                      XmNpromptString,
                      XmStringCreateLtoR((char *)"Command", charset),
                      NULL);

        fr = XmCreateFrame(command, (char *)"commandframe", NULL, 0);

        rc = XmCreateRowColumn(fr, (char *)"commandrc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);

        bt = XtVaCreateManagedWidget((char *)"Save...", xmPushButtonWidgetClass, rc,
                                     NULL);
        XtAddCallback(bt, XmNactivateCallback, (XtCallbackProc)create_whist_frame, (XtPointer)NULL);

        bt = XtVaCreateManagedWidget((char *)"Read...", xmPushButtonWidgetClass, rc,
                                     NULL);
        XtAddCallback(bt, XmNactivateCallback, (XtCallbackProc)create_rhist_popup, (XtPointer)NULL);

        bt = XtVaCreateManagedWidget((char *)"Clear", xmPushButtonWidgetClass, rc,
                                     NULL);
        XtAddCallback(bt, XmNactivateCallback, (XtCallbackProc)clear_history, (XtPointer)NULL);

        bt = XtVaCreateManagedWidget((char *)"Replay", xmPushButtonWidgetClass, rc,
                                     NULL);
        XtAddCallback(bt, XmNactivateCallback, (XtCallbackProc)replay_history, (XtPointer)NULL);

        bt = XtVaCreateManagedWidget((char *)"Close", xmPushButtonWidgetClass, rc,
                                     NULL);
        XtAddCallback(bt, XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)comshell);

        XtAddCallback(command, XmNcommandEnteredCallback, (XtCallbackProc)comcall, (XtPointer)NULL);
        XtManageChild(rc);
        XtManageChild(fr);

        XtManageChild(command);
        XtManageChild(comshell);
    }
    XtRaise(comshell);
    unset_wait_cursor();
}

static Widget rhist_dialog;

void close_rhist_popup(Widget, XtPointer, XtPointer)
{
    XtUnmanageChild(rhist_dialog);
}

void do_rhist_proc(Widget, XtPointer, XtPointer)
{
    Arg args;
    XmString list_item;
    char *s, buf[256];
    FILE *fp;
    int sl;
    Widget h = XmCommandGetChild(command, XmDIALOG_HISTORY_LIST);
    XtSetArg(args, XmNtextString, &list_item);
    XtGetValues(rhist_dialog, &args, 1);
    XmStringGetLtoR(list_item, charset, &s);
    strcpy(buf, s);
    XtFree(s);
    if ((fp = fopen(buf, "r")) != NULL)
    {
        while (fgets(buf, 255, fp) != NULL)
        {
            sl = strlen(buf);
            buf[sl - 1] = 0;
            list_item = XmStringCreateLtoR(buf, charset);
            XmListAddItemUnselected(h, list_item, 0);
            XmStringFree(list_item);
        }
        fclose(fp);
    }
    else
    {
        errwin("Unable to open file");
    }
    XtUnmanageChild(rhist_dialog);
}

void create_rhist_popup(Widget, XtPointer, XtPointer)
{
    set_wait_cursor();
    if (rhist_dialog == NULL)
    {
        rhist_dialog = XmCreateFileSelectionDialog(app_shell, (char *)"Read history", NULL, 0);
        XtAddCallback(rhist_dialog, XmNcancelCallback, (XtCallbackProc)close_rhist_popup, (XtPointer)NULL);
        XtAddCallback(rhist_dialog, XmNokCallback, (XtCallbackProc)do_rhist_proc, (XtPointer)NULL);
        XtManageChild(rhist_dialog);
    }
    XtRaise(rhist_dialog);
    unset_wait_cursor();
}

/*
 * Panel item declarations
 */
static Widget whist_frame;
static Widget whist_panel;
static Widget whist_text_item;
// static Widget *whist_choice_item;

/*
 * Create the whist Frame and the whist Panel
 */
void create_whist_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    set_wait_cursor();
    if (whist_frame == NULL)
    {
        Widget buts[2];
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Cancel";
        XmGetPos(app_shell, 0, &x, &y);
        whist_frame = XmCreateDialogShell(app_shell, (char *)"Write history", NULL, 0);
        handle_close(whist_frame);
        XtVaSetValues(whist_frame, XmNx, x, XmNy, y, NULL);
        whist_panel = XmCreateRowColumn(whist_frame, (char *)"whist_rc", NULL, 0);

        whist_text_item = CreateTextItem2(whist_panel, 30, (char *)"Write history to:");

        XtVaCreateManagedWidget((char *)"sep", xmSeparatorWidgetClass, whist_panel, NULL);

        CreateCommandButtons(whist_panel, 2, buts, label1);
        XtAddCallback(buts[0], XmNactivateCallback,
                      (XtCallbackProc)whist_apply_notify_proc, (XtPointer)NULL);
        XtAddCallback(buts[1], XmNactivateCallback,
                      (XtCallbackProc)destroy_dialog, (XtPointer)whist_frame);
        XtManageChild(whist_panel);
    }
    XtRaise(whist_frame);
    unset_wait_cursor();
}

static void whist_apply_notify_proc(Widget, XtPointer, XtPointer)
{
    int i, ac = 0, hc;
    char s[256], *ts;
    XmStringTable xmstrs;
    Arg al[5];

    strcpy(s, (char *)xv_getstr(whist_text_item));
    if (!fexists(s))
    {
        FILE *pp = fopen(s, "w");
        if (pp != NULL)
        {
            ac = 0;
            XtSetArg(al[ac], XmNhistoryItems, &xmstrs);
            ac++;
            XtSetArg(al[ac], XmNhistoryItemCount, &hc);
            ac++;
            XtGetValues(command, al, ac);
            for (i = 0; i < hc; i++)
            {
                XmStringGetLtoR(xmstrs[i], charset, &ts);
                fprintf(pp, "%s\n", ts);
                XtFree(ts);
            }
            fclose(pp);
        }
        else
        {
            errwin("Unable to open file");
        }
    }
    XtUnmanageChild(whist_frame);
}
