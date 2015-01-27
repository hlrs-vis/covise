/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: eblockwin.c,v 1.6 1994/11/01 05:35:57 pturner Exp pturner $
 *
 * Edit block data Panel
 *
 *
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <Xm/Xm.h>
#include <Xm/DialogS.h>
#include <Xm/Frame.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/ToggleB.h>
#include <Xm/RowColumn.h>
#include <Xm/Separator.h>
#include <Xm/Text.h>

#include "globals.h"
#include "motifinc.h"

Widget *CreateBlockChoice(Widget parent, char *labelstr, int ncols, int type);

static char ncolsbuf[128];

static int block_curtype = XY;

static Widget eblock_frame;
static Widget eblock_panel;

/*
 * Panel item declarations
 */
static Widget eblock_ncols_item;
static Widget *eblock_type_choice_item;
static Widget *eblock_x_choice_item;
static Widget *eblock_y_choice_item;
static Widget *eblock_e1_choice_item;
static Widget *eblock_e2_choice_item;
static Widget *eblock_e3_choice_item;
static Widget *eblock_e4_choice_item;
// static Widget *eblock_set_choice_item;
static Widget *eblock_graph_choice_item;

/*
 * Event and Notify proc declarations
 */
static void eblock_type_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void eblock_accept_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void update_eblock(void);

extern "C" {
extern void cfree(void *);
}

/*
 * Create the files Frame and the files Panel
 */
void create_eblock_frame(Widget, XtPointer, XtPointer)
{
    long i;
    int x, y;
    Widget rc, buts[2];

    if (blockncols == 0)
    {
        errwin("Need to read block data first");
        return;
    }
    set_wait_cursor();
    if (eblock_frame == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        eblock_frame = XmCreateDialogShell(app_shell, (char *)"Edit block data", NULL, 0);
        handle_close(eblock_frame);
        XtVaSetValues(eblock_frame, XmNx, x, XmNy, y, NULL);
        eblock_panel = XmCreateRowColumn(eblock_frame, (char *)"eblock_rc", NULL, 0);

        eblock_ncols_item = XtVaCreateManagedWidget((char *)"tmp", xmLabelWidgetClass, eblock_panel,
                                                    NULL);

        rc = XtVaCreateWidget((char *)"rc", xmRowColumnWidgetClass, eblock_panel,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 9,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        XtVaCreateManagedWidget((char *)"Set type: ", xmLabelWidgetClass, rc, NULL);
        eblock_type_choice_item = CreatePanelChoice(rc,
                                                    " ",
                                                    13,
                                                    "XY",
                                                    "XY DX",
                                                    "XY DY",
                                                    "XY DX1 DX2",
                                                    "XY DY1 DY2",
                                                    "XY DX DY",
                                                    "XY Z",
                                                    "XY HILO",
                                                    "XY R",
                                                    "XY BOX",
                                                    "Rawspice",
                                                    "XY BOXPLOT",
                                                    NULL, 0);
        for (i = 0; i < 12; i++)
        {
            XtAddCallback(eblock_type_choice_item[2 + i],
                          XmNactivateCallback, (XtCallbackProc)eblock_type_notify_proc, (XtPointer)i);
        }

        XtVaCreateManagedWidget("X from column:", xmLabelWidgetClass, rc, NULL);
        eblock_x_choice_item = CreateBlockChoice(rc, (char *)" ", maxblock, 1);
        XtVaCreateManagedWidget("Y from column:", xmLabelWidgetClass, rc, NULL);
        eblock_y_choice_item = CreateBlockChoice(rc, (char *)" ", maxblock, 0);
        XtVaCreateManagedWidget("E1 from column:", xmLabelWidgetClass, rc, NULL);
        eblock_e1_choice_item = CreateBlockChoice(rc, (char *)" ", maxblock, 0);
        XtVaCreateManagedWidget("E2 from column:", xmLabelWidgetClass, rc, NULL);
        eblock_e2_choice_item = CreateBlockChoice(rc, (char *)" ", maxblock, 0);
        XtVaCreateManagedWidget("E3 from column:", xmLabelWidgetClass, rc, NULL);
        eblock_e3_choice_item = CreateBlockChoice(rc, (char *)" ", maxblock, 0);
        XtVaCreateManagedWidget("E4 from column:", xmLabelWidgetClass, rc, NULL);
        eblock_e4_choice_item = CreateBlockChoice(rc, (char *)" ", maxblock, 0);
        /*
         XtVaCreateManagedWidget("Load to set:", xmLabelWidgetClass, rc, NULL);
         eblock_set_choice_item = CreateSetChoice(rc, " ", maxplot, 4);
      */

        XtVaCreateManagedWidget((char *)"Load to set in graph:", xmLabelWidgetClass, rc, NULL);
        eblock_graph_choice_item = CreateGraphChoice(rc, (char *)" ", maxgraph, 1);

        XtManageChild(rc);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, eblock_panel, NULL);

        CreateCommandButtons(eblock_panel, 2, buts, label1);
        XtAddCallback(buts[0], XmNactivateCallback,
                      (XtCallbackProc)eblock_accept_notify_proc, (XtPointer)0);
        XtAddCallback(buts[1], XmNactivateCallback,
                      (XtCallbackProc)destroy_dialog, (XtPointer)eblock_frame);

        XtManageChild(eblock_panel);
    }
    XtRaise(eblock_frame);
    update_eblock();
    unset_wait_cursor();
} /* end create_eblock_panel */

/*
 * Notify and event procs
 */

static void update_eblock(void)
{
    XmString string;
    Arg al;
    if (!eblock_frame)
    {
        return;
    }
    if (blockncols == 0)
    {
        errwin("Need to read block data first");
        return;
    }
    sprintf(ncolsbuf, "%d columns of length %d", blockncols, blocklen);
    string = XmStringCreateLtoR(ncolsbuf, charset);
    XtSetArg(al, XmNlabelString, string);
    XtSetValues(eblock_ncols_item, &al, 1);
    XmStringFree(string);
    switch (block_curtype)
    {
    case XY:
        XtSetSensitive(eblock_e1_choice_item[0], False);
        XtSetSensitive(eblock_e2_choice_item[0], False);
        XtSetSensitive(eblock_e3_choice_item[0], False);
        XtSetSensitive(eblock_e4_choice_item[0], False);
        break;
    case XYRT:
    case XYDX:
    case XYDY:
    case XYZ:
        XtSetSensitive(eblock_e1_choice_item[0], True);
        XtSetSensitive(eblock_e2_choice_item[0], False);
        XtSetSensitive(eblock_e3_choice_item[0], False);
        XtSetSensitive(eblock_e4_choice_item[0], False);
        break;
    case XYDXDX:
    case XYDYDY:
    case XYDXDY:
        XtSetSensitive(eblock_e1_choice_item[0], True);
        XtSetSensitive(eblock_e2_choice_item[0], True);
        XtSetSensitive(eblock_e3_choice_item[0], False);
        XtSetSensitive(eblock_e4_choice_item[0], False);
        break;
    case XYHILO:
    case XYBOX:
        XtSetSensitive(eblock_e1_choice_item[0], True);
        XtSetSensitive(eblock_e2_choice_item[0], True);
        XtSetSensitive(eblock_e3_choice_item[0], True);
        XtSetSensitive(eblock_e4_choice_item[0], False);
        break;
    case XYBOXPLOT:
        XtSetSensitive(eblock_e1_choice_item[0], True);
        XtSetSensitive(eblock_e2_choice_item[0], True);
        XtSetSensitive(eblock_e3_choice_item[0], True);
        XtSetSensitive(eblock_e4_choice_item[0], True);
        break;
    }
}

static void eblock_type_notify_proc(Widget, XtPointer client_data, XtPointer)
{
    int cd = (long)client_data;
    switch (cd)
    {
    case 0:
        block_curtype = XY;
        break;
    case 1:
        block_curtype = XYDX;
        break;
    case 2:
        block_curtype = XYDY;
        break;
    case 3:
        block_curtype = XYDXDX;
        break;
    case 4:
        block_curtype = XYDYDY;
        break;
    case 5:
        block_curtype = XYDXDY;
        break;
    case 6:
        block_curtype = XYZ;
        break;
    case 7:
        block_curtype = XYHILO;
        break;
    case 8:
        block_curtype = XYRT;
        break;
    case 9:
        block_curtype = XYBOX;
        break;
    case 10:
        block_curtype = RAWSPICE;
        break;
    case 11:
        block_curtype = XYBOXPLOT;
        break;
    }
    update_eblock();
}

static void eblock_accept_notify_proc(Widget, XtPointer, XtPointer)
{
    int i = 0;
    char buf[256];
    int setno, graphno;
    int cx, cy, c1 = 0, c2 = 0, c3 = 0, c4 = 0; //d1,
    double *tx, *ty, *t2, *t3, *t4, *t5;

    //d1 = (int) GetChoice(eblock_type_choice_item);
    cx = (int)GetChoice(eblock_x_choice_item) - 1;
    cy = (int)GetChoice(eblock_y_choice_item);
    if (cx >= 0 && cx >= blockncols)
    {
        errwin("Column for X exceeds the number of columns in block data");
        return;
    }
    if (cy >= blockncols)
    {
        errwin("Column for Y exceeds the number of columns in block data");
        return;
    }
    switch (block_curtype)
    {
    case XY:
        break;
    case XYRT:
    case XYDX:
    case XYDY:
    case XYZ:
        c1 = (int)GetChoice(eblock_e1_choice_item);
        if (c1 >= blockncols)
        {
            errwin("Column for E1 exceeds the number of columns in block data");
            return;
        }
        break;
    case XYDXDX:
    case XYDYDY:
    case XYDXDY:
        c1 = (int)GetChoice(eblock_e1_choice_item);
        c2 = (int)GetChoice(eblock_e2_choice_item);
        if (c1 >= blockncols)
        {
            errwin("Column for E1 exceeds the number of columns in block data");
            return;
        }
        if (c2 >= blockncols)
        {
            errwin("Column for E2 exceeds the number of columns in block data");
            return;
        }
        break;
    case XYHILO:
    case XYBOX:
        c1 = (int)GetChoice(eblock_e1_choice_item);
        c2 = (int)GetChoice(eblock_e2_choice_item);
        c3 = (int)GetChoice(eblock_e3_choice_item);
        if (c1 >= blockncols)
        {
            errwin("Column for E1 exceeds the number of columns in block data");
            return;
        }
        if (c2 >= blockncols)
        {
            errwin("Column for E2 exceeds the number of columns in block data");
            return;
        }
        if (c3 >= blockncols)
        {
            errwin("Column for E3 exceeds the number of columns in block data");
            return;
        }
        break;
    case XYBOXPLOT:
        c1 = (int)GetChoice(eblock_e1_choice_item);
        c2 = (int)GetChoice(eblock_e2_choice_item);
        c3 = (int)GetChoice(eblock_e3_choice_item);
        c4 = (int)GetChoice(eblock_e4_choice_item);
        if (c1 >= blockncols)
        {
            errwin("Column for E1 exceeds the number of columns in block data");
            return;
        }
        if (c2 >= blockncols)
        {
            errwin("Column for E2 exceeds the number of columns in block data");
            return;
        }
        if (c3 >= blockncols)
        {
            errwin("Column for E3 exceeds the number of columns in block data");
            return;
        }
        if (c4 >= blockncols)
        {
            errwin("Column for E4 exceeds the number of columns in block data");
            return;
        }
    }
    setno = -1;
    graphno = (int)GetChoice(eblock_graph_choice_item) - 1;

    if (graphno == -1)
    {
        graphno = cg;
    }
    if (setno == -1)
    {
        setno = nextset(graphno);
    }
    if (setno == -1)
    {
        return;
    }
    if (g[graphno].active == OFF)
    {
        set_graph_active(graphno);
    }
    activateset(graphno, setno);
    settype(graphno, setno, block_curtype);

    tx = (double *)calloc(blocklen, sizeof(double));
    if (tx == NULL)
    {
        errwin("Can't allocate memory for X");
        return;
    }
    ty = (double *)calloc(blocklen, sizeof(double));
    if (ty == NULL)
    {
        cfree(tx);
        errwin("Can't allocate memory for Y");
        return;
    }
    for (i = 0; i < blocklen; i++)
    {
        if (cx == -1)
        {
            tx[i] = i + 1;
        }
        else
        {
            tx[i] = blockdata[cx][i];
        }
        ty[i] = blockdata[cy][i];
    }
    setcol(graphno, tx, setno, blocklen, 0);
    setcol(graphno, ty, setno, blocklen, 1);

    switch (block_curtype)
    {
    case XY:
        sprintf(buf, "Cols %d %d", cx + 1, cy + 1);
        break;
    case XYRT:
    case XYDX:
    case XYDY:
    case XYZ:
        sprintf(buf, "Cols %d %d %d", cx + 1, cy + 1, c1 + 1);
        t2 = (double *)calloc(blocklen, sizeof(double));
        for (i = 0; i < blocklen; i++)
        {
            t2[i] = blockdata[c1][i];
        }
        setcol(graphno, t2, setno, blocklen, 2);
        break;
    case XYDXDX:
    case XYDYDY:
    case XYDXDY:
        sprintf(buf, "Cols %d %d %d %d", cx + 1, cy + 1, c1 + 1, c2 + 1);
        t2 = (double *)calloc(blocklen, sizeof(double));
        t3 = (double *)calloc(blocklen, sizeof(double));
        for (i = 0; i < blocklen; i++)
        {
            t2[i] = blockdata[c1][i];
            t3[i] = blockdata[c2][i];
        }
        setcol(graphno, t2, setno, blocklen, 2);
        setcol(graphno, t3, setno, blocklen, 3);
        break;
    case XYHILO:
    case XYBOX:
        sprintf(buf, "Cols %d %d %d %d %d", cx + 1, cy + 1, c1 + 1, c2 + 1, c3 + 1);
        t2 = (double *)calloc(blocklen, sizeof(double));
        t3 = (double *)calloc(blocklen, sizeof(double));
        t4 = (double *)calloc(blocklen, sizeof(double));
        for (i = 0; i < blocklen; i++)
        {
            t2[i] = blockdata[c1][i];
            t3[i] = blockdata[c2][i];
            t4[i] = blockdata[c3][i];
        }
        setcol(graphno, t2, setno, blocklen, 2);
        setcol(graphno, t3, setno, blocklen, 3);
        setcol(graphno, t4, setno, blocklen, 4);
        break;
    case XYBOXPLOT:
        sprintf(buf, "Cols %d %d %d %d %d %d", cx + 1, cy + 1, c1 + 1, c2 + 1, c3 + 1, c4 + 1);
        t2 = (double *)calloc(blocklen, sizeof(double));
        t3 = (double *)calloc(blocklen, sizeof(double));
        t4 = (double *)calloc(blocklen, sizeof(double));
        t5 = (double *)calloc(blocklen, sizeof(double));
        for (i = 0; i < blocklen; i++)
        {
            t2[i] = blockdata[c1][i];
            t3[i] = blockdata[c2][i];
            t4[i] = blockdata[c3][i];
            t5[i] = blockdata[c4][i];
        }
        setcol(graphno, t2, setno, blocklen, 2);
        setcol(graphno, t3, setno, blocklen, 3);
        setcol(graphno, t4, setno, blocklen, 4);
        setcol(graphno, t5, setno, blocklen, 5);
        break;
    }

    setcomment(graphno, setno, buf);
    updatesetminmax(graphno, setno);
    update_status_popup(NULL, NULL, NULL);
    drawgraph();
}

Widget *CreateBlockChoice(Widget parent, char *labelstr, int nsets, int type)
{
    int nmal, i = 0;
    XmString str;
    char *name = (char *)"setchoice";
    char buf[10];
    Widget *retval = NULL;

    switch (type)
    {
    case 0:
        nmal = nsets + 2;
        retval = (Widget *)XtMalloc(nmal * sizeof(Widget));
        retval[1] = XmCreatePulldownMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[1],
                      XmNorientation, XmVERTICAL,
                      XmNpacking, XmPACK_COLUMN,
                      XmNnumColumns, nsets / 10,
                      NULL);
        i = 0;
        for (i = 0; i < nsets; i++)
        {
            sprintf(buf, "%d", i + 1);
            retval[i + 2] = XmCreatePushButton(retval[1], buf, NULL, 0);
        }
        XtManageChildren(retval + 2, nsets);

        str = XmStringCreate(labelstr, charset);

        retval[0] = XmCreateOptionMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[0],
                      XmNlabelString, str,
                      XmNsubMenuId, retval[1],
                      XmNentryBorder, 2,
                      XmNwhichButton, 1,
                      NULL);
        XtManageChild(retval[0]);
        break;
    case 1:
        nmal = nsets + 3;
        retval = (Widget *)XtMalloc(nmal * sizeof(Widget));
        retval[1] = XmCreatePulldownMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[1],
                      XmNorientation, XmVERTICAL,
                      XmNpacking, XmPACK_COLUMN,
                      XmNnumColumns, nsets / 10,
                      NULL);
        i = 0;
        retval[2] = XmCreatePushButton(retval[1], (char *)"Index", NULL, 0);
        for (i = 1; i < nsets + 1; i++)
        {
            sprintf(buf, "%d", i);
            retval[i + 2] = XmCreatePushButton(retval[1], buf, NULL, 0);
        }
        XtManageChildren(retval + 2, nsets + 1);

        str = XmStringCreate(labelstr, charset);

        retval[0] = XmCreateOptionMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[0],
                      XmNlabelString, str,
                      XmNsubMenuId, retval[1],
                      XmNentryBorder, 2,
                      XmNwhichButton, 1,
                      NULL);
        XtManageChild(retval[0]);
        break;
    default:
        fprintf(stderr, "eblockwin.cpp:CreateBlockChoice(): retval not initialized\n");
        break;
    }
    return retval;
}
