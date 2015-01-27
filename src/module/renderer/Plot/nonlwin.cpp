/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: nonlwin.c,v 1.6 1994/09/29 03:37:37 pturner Exp pturner $
 *
 * non linear curve fitting
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
#include <Xm/Separator.h>
#include <Xm/ScrolledW.h>

#include "globals.h"
#include "motifinc.h"

/* info strings
info = 0  improper input parameters.
info = 1  algorithm estimates that the relative error in the sum of squares is at most tol.
info = 2  algorithm estimates that the relative error between x and the solution is at most tol.
info = 3  conditions for info = 1 and info = 2 both hold.
info = 4  fvec is orthogonal to the columns of the jacobian to machine precision.
info = 5  number of calls to fcn has reached or exceeded 200*(n+1).
info = 6  tol is too small. no further reduction in the sum of squares is possible.
info = 7  tol is too small. no further improvement in the approximate solution x is possible.
*/

extern double nonl_parms[];

static void do_nonl_proc(Widget w, XtPointer client_data, XtPointer call_data);
//static void create_nonleval_frame(Widget w, XtPointer client_data, XtPointer call_data);

#define MAXPARM 10

static Widget nonl_frame;
static Widget nonl_panel;
static Widget nonl_formula_item;
static Widget *nonl_set_item;
static Widget *nonl_load_item;
static Widget *nonl_loadgraph_item;
static Widget nonl_initial_item[MAXPARM];
static Widget nonl_computed_item[MAXPARM];
static Widget nonl_tol_item;
static Widget nonl_nparm_item;
static void do_nonl_proc(Widget w, XtPointer client_data, XtPointer call_data);

// static Widget nonleval_frame;

/* ARGSUSED */
void create_nonl_frame(Widget, XtPointer, XtPointer)
{
    int i;
    int x, y;
    Widget sw, fr, rc, rc1, rc2, lab, but1[3];
    set_wait_cursor();
    if (nonl_frame == NULL)
    {
        char *label1[3];
        /*
         label1[0] = "Accept";
         label1[1] = "Eval...";
         label1[2] = "Close";
      */
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        nonl_frame = XmCreateDialogShell(app_shell, (char *)"Non-linear curve fitting", NULL, 0);
        handle_close(nonl_frame);
        XtVaSetValues(nonl_frame, XmNx, x, XmNy, y, NULL);
        nonl_panel = XmCreateForm(nonl_frame, (char *)"nonl_frame_rc", NULL, 0);
        fr = XmCreateFrame(nonl_panel, (char *)"nonl_frame", NULL, 0);
        rc = XmCreateRowColumn(fr, (char *)"nonl_rc", NULL, 0);

        nonl_set_item = CreateSetChoice(rc, (char *)"Use set:", maxplot, 0);

        nonl_load_item = CreatePanelChoice(rc,
                                           "Load:",
                                           4,
                                           "Fitted values",
                                           "Residuals",
                                           "None",
                                           NULL, NULL);

        nonl_loadgraph_item = CreateGraphChoice(rc, (char *)"To graph:", maxgraph, 0);

        nonl_formula_item = (Widget)CreateTextItem2(rc, 35, (char *)"Function:");
        xv_setstr(nonl_formula_item, (char *)"y = ");

        nonl_nparm_item = CreateTextItem2(rc, 5, (char *)"# of parameters:");
        nonl_tol_item = CreateTextItem2(rc, 10, (char *)"Tolerance:");
        XtManageChild(rc);
        XtManageChild(fr);
        sw = XtVaCreateManagedWidget("sw",
                                     xmScrolledWindowWidgetClass, nonl_panel,
                                     XmNscrollingPolicy, XmAUTOMATIC,
                                     NULL);

        rc = XmCreateRowColumn(sw, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);

        rc1 = XmCreateRowColumn(rc, (char *)"rc1", NULL, 0);
        lab = XmCreateLabel(rc1, (char *)"Initial guess:", NULL, 0);
        XtManageChild(lab);
        rc2 = XmCreateRowColumn(rc, (char *)"rc2", NULL, 0);
        lab = XmCreateLabel(rc2, (char *)"Computed values:", NULL, 0);
        XtManageChild(lab);

        for (i = 0; i < MAXPARM; i++)
        {
            sprintf(buf, "A%1d: ", i);
            nonl_initial_item[i] = CreateTextItem2(rc1, 15, buf);
        }
        for (i = 0; i < MAXPARM; i++)
        {
            nonl_computed_item[i] = CreateTextItem2(rc2, 15, (char *)"");
        }
        XtManageChild(rc1);
        XtManageChild(rc2);
        XtManageChild(rc);
        XtVaSetValues(sw,
                      XmNworkWindow, rc,
                      NULL);

        rc = XmCreateRowColumn(nonl_panel, (char *)"rc", NULL, 0);
        CreateCommandButtons(rc, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback,
                      (XtCallbackProc)do_nonl_proc, (XtPointer)NULL);
        /*
         XtAddCallback(but1[1], XmNactivateCallback,
                (XtCallbackProc) create_nonleval_frame, (XtPointer) NULL);
      */
        XtAddCallback(but1[1], XmNactivateCallback,
                      (XtCallbackProc)destroy_dialog, (XtPointer)nonl_frame);
        XtManageChild(rc);

        XtVaSetValues(fr,
                      XmNtopAttachment, XmATTACH_FORM,
                      XmNleftAttachment, XmATTACH_FORM,
                      XmNrightAttachment, XmATTACH_FORM,
                      NULL);
        XtVaSetValues(sw,
                      XmNtopAttachment, XmATTACH_WIDGET,
                      XmNtopWidget, fr,
                      XmNleftAttachment, XmATTACH_FORM,
                      XmNrightAttachment, XmATTACH_FORM,
                      XmNbottomAttachment, XmATTACH_WIDGET,
                      XmNbottomWidget, rc,
                      NULL);
        XtVaSetValues(rc,
                      XmNleftAttachment, XmATTACH_FORM,
                      XmNrightAttachment, XmATTACH_FORM,
                      XmNbottomAttachment, XmATTACH_FORM,
                      NULL);

        XtManageChild(nonl_panel);
    }
    XtRaise(nonl_frame);
    unset_wait_cursor();
}

void update_nonl(void)
{
}

/* ARGSUSED */
static void do_nonl_proc(Widget, XtPointer, XtPointer)
{
    int i, setno, loadset, loadto, graphto, npar, info;
    double tol, a[MAXPARM];
    char fstr[256];
    double *y, *yp;

    set_wait_cursor();
    curset = setno = (int)GetChoice(nonl_set_item);
    loadto = (int)GetChoice(nonl_load_item);
    graphto = (int)GetChoice(nonl_loadgraph_item) - 1;
    tol = atof((char *)xv_getstr(nonl_tol_item));
    if (graphto < 0)
    {
        graphto = cg;
    }
    npar = atoi((char *)xv_getstr(nonl_nparm_item));
    strcpy(fstr, (char *)xv_getstr(nonl_formula_item));
    for (i = 0; i < MAXPARM; i++)
    {
        a[i] = 0.0;
        strcpy(buf, (char *)xv_getstr(nonl_initial_item[i]));
        sscanf(buf, "%lf", &a[i]);
    }
    yp = (double *)calloc(getsetlength(cg, setno), sizeof(double));
    if (yp == NULL)
    {
        errwin("Memory allocation error, operation cancelled");
        unset_wait_cursor();
        return;
    }
    y = gety(cg, setno);
    for (i = 0; i < getsetlength(cg, setno); i++)
    {
        yp[i] = y[i];
    }
    sprintf(buf, "Fitting: %s\n", fstr);
    stufftext(buf, 0);
    sprintf(buf, "Initial guess:\n");
    stufftext(buf, 0);
    for (i = 0; i < npar; i++)
    {
        sprintf(buf, "\ta%1d = %.9lf\n", i, a[i]);
        stufftext(buf, 0);
    }
    sprintf(buf, "Tolerance = %.9lf\n", tol);
    stufftext(buf, 0);
    lmfit(fstr, getsetlength(cg, setno), getx(cg, setno),
          yp, y, npar, a, tol, &info);
    for (i = 0; i < getsetlength(cg, setno); i++)
    {
        y[i] = yp[i];
    }
    free(yp);
    for (i = 0; i < MAXPARM; i++)
    {
        sprintf(buf, "%.9lf", a[i]);
        xv_setstr(nonl_computed_item[i], buf);
        nonl_parms[i] = a[i];
    }
    if (info > 0 && info < 4)
    {
        sprintf(buf, "Computed values:\n");
        stufftext(buf, 0);
        for (i = 0; i < npar; i++)
        {
            sprintf(buf, "\ta%1d = %.9lf\n", i, a[i]);
            stufftext(buf, 0);
        }
        loadset = nextset(cg);
        if (loadset != -1)
        {
            do_copyset(cg, setno, cg, loadset);
        }
        else
        {
            unset_wait_cursor();
            return;
        }
        switch (loadto)
        {
        case 0:
            sprintf(buf, "Evaluating function and loading result to set %d:\n", loadset);
            stufftext(buf, 0);
            do_compute(loadset, 0, graphto, fstr);
            break;
        case 1:
            sprintf(buf, "Evaluating function and loading residuals to set %d:\n", loadset);
            stufftext(buf, 0);
            do_compute(loadset, 0, graphto, fstr);
            break;
        case 2:
            sprintf(buf, "Computed function not evaluated\n");
            stufftext(buf, 0);
            break;
        }
    }
    /*
       if (info >= 4) {
      do_compute(setno, 1, graphto, fstr);
       }
   */
    if (info >= 0 && info <= 7)
    {
        char *s;
        switch (info)
        {
        case 0:
            s = (char *)"Improper input parameters.\n";
            break;
        case 1:
            s = (char *)"Relative error in the sum of squares is at most tol.\n";
            break;
        case 2:
            s = (char *)"Relative error between A and the solution is at most tol.\n";
            break;
        case 3:
            s = (char *)"Relative error in the sum of squares and A and the solution is at most tol.\n";
            break;
        case 4:
            s = (char *)"Fvec is orthogonal to the columns of the jacobian to machine precision.\n";
            break;
        case 5:
            s = (char *)"Number of calls to fcn has reached or exceeded 200*(n+1).\n";
            break;
        case 6:
            s = (char *)"Tol is too small. No further reduction in the sum of squares is possible.\n";
            break;
        case 7:
            s = (char *)"Tol is too small. No further improvement in the approximate solution A is possible.\n";
            break;
        }
        stufftext(s, 0);
        stufftext((char *)"\n", 0);
    }
    unset_wait_cursor();
}

/* ARGSUSED */ /*
static void create_nonleval_frame(Widget , XtPointer , XtPointer )
{
    int x, y;
    Widget panel, wbut, rc;
    set_wait_cursor();
    if (nonleval_frame == NULL) {
   XmGetPos(app_shell, 0, &x, &y);
   nonleval_frame = XmCreateDialogShell(app_shell, "Evaluate fitted curve", NULL, 0);
   handle_close(nonleval_frame);
   XtVaSetValues(nonleval_frame, XmNx, x, XmNy, y, NULL);
panel = XmCreateRowColumn(nonleval_frame, "nonleval_rc", NULL, 0);

rc = XmCreateRowColumn(panel, "rc", NULL, 0);
XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
wbut = XtVaCreateManagedWidget("Accept", xmPushButtonWidgetClass, rc,
NULL);
XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc) do_nonl_proc, (XtPointer) NULL);
wbut = XtVaCreateManagedWidget("Close", xmPushButtonWidgetClass, rc,
NULL);
XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc) destroy_dialog, (XtPointer) nonleval_frame);
XtManageChild(rc);

XtManageChild(panel);
}
XtRaise(nonleval_frame);
unset_wait_cursor();
}*/
