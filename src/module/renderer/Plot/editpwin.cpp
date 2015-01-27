/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: editpwin.c,v 1.13 1994/10/28 23:46:04 pturner Exp pturner $
 *
 * spreadsheet-like editing of data points
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
#include "noxprotos.h"

// static Widget but1[2];

Widget *editp_set_item;
Widget *editp_format_item;
Widget *editp_precision_item;
Widget *editp_width_item;

#ifdef HAS_XBAE

#include <Xbae/Matrix.h>

String **cells;

static short widths[6] = { 10, 10, 10, 10, 10, 10 };
static int precision[6] = { 3, 3, 3, 3, 3, 3 };
static int format[6] = { DECIMAL, DECIMAL, DECIMAL, DECIMAL, DECIMAL, DECIMAL };

String labels1[2] = { "X", "Y" };
String labels2[3] = { "X", "Y", "DX" };
String labels3[3] = { "X", "Y", "DY" };
String labels4[4] = { "X", "Y", "DX1", "DX2" };
String labels5[4] = { "X", "Y", "DY1", "DY2" };
String labels6[4] = { "X", "Y", "DX", "DY" };
String labels7[3] = { "X", "Y", "DZ" };
String labels8[5] = { "X", "HI", "LO", "OPEN", "CLOSE" };
String labels9[4] = { "X", "Y", "Radius", "Theta" };

String *rowlabels;
String *collabels = labels1;

/* */
typedef struct _EditPoints
{
    int gno;
    int setno;
    int nrows;
    int ncols;
    String **cells;
    String *data;
    String label;
    String *rowlabels;
    String *collabels;
    short *widths;
    short width;
    int *lengths;
    int length;
    Widget top;
    Widget mw;
    Widget *editp_format_item;
    Widget *editp_precision_item;
    Widget *editp_width_item;
    int cformat[10];
    int cprec[10];
    int cwidth[10];
    double **vals;
} EditPoints;

typedef enum
{
    NoSelection,
    CellSelection,
    RowSelection,
    ColumnSelection
} SelectionType;

typedef enum
{
    AddMode,
    ExclusiveMode
} SelectionMode;

typedef struct _SelectionStruct
{
    int row, column;
    SelectionType type;
    SelectionMode mode;
    Boolean selected;
    Widget matrix;
} *SelectionPtr, SelectionStruct;

EditPoints *newep(int gno, int setno)
{
    EditPoints *ep;
    ep = (EditPoints *)malloc(sizeof(EditPoints));
    ep->gno = gno;
    ep->setno = setno;
    ep->ncols = getncols(gno, setno);
    ep->nrows = getsetlength(gno, setno);
    g[gno].p[setno].ep = ep;
    switch (dataset_type(gno, setno))
    {
    case XY:
        ep->collabels = labels1;
        break;
    case XYDX:
        ep->collabels = labels2;
        break;
    case XYDY:
        ep->collabels = labels3;
        break;
    case XYDXDX:
        ep->collabels = labels4;
        break;
    case XYDYDY:
        ep->collabels = labels5;
        break;
    case XYDXDY:
        ep->collabels = labels6;
        break;
    case XYZ:
        ep->collabels = labels7;
        break;
    case XYHILO:
        ep->collabels = labels8;
        break;
    case XYRT:
        ep->collabels = labels9;
        break;
    }
    return ep;
}

void epdtor(EditPoints *ep)
{ /* This could be a source of leakage TODO */
    /* need to free the ep->cells stuff
       int i, j, len = getsetlength(ep->gno, ep->setno);
       for (i = 0; i < len; i++) {
      for (j = 0; j < ep->ncols; j++) {
          XtFree((XtPointer) cells[i][j]);
      }
      XtFree((XtPointer) cells[i]);
       }
       XtFree((XtPointer) cells);
   */
    XtUnmanageChild(ep->top);
    XtDestroyWidget(ep->top);
}

void create_ss_frame(EditPoints *ep);

void do_ss_frame(Widget w, XtPointer client_data, XtPointer call_data)
{
    EditPoints *ep;
    int setno = GetChoice(editp_set_item);
    int gno = cg;
    if (isactive(gno, setno))
    {
        if ((ep = (EditPoints *)geteditpoints(gno, setno)) != NULL)
        {
            XtRaise(ep->top);
        }
        else
        {
            ep = newep(gno, setno);
            create_ss_frame(ep);
        }
    }
    else
    {
        errwin("Set not active");
    }
}

void do_ss_proc(Widget w, XtPointer client_data, XtPointer call_data)
{
    int rows;
    String **cells;
    String *data;
    String label, *row_labels, *column_labels;
    short *widths, width;
    int *lengths, length;
    int i, j;
    double *datap;
    EditPoints *ep = (EditPoints *)client_data;
    Widget matrix = ep->mw;
    XtUnmanageChild(ep->top);

    XtVaGetValues(matrix,
                  XmNcells, &cells,
                  XmNrows, &rows,
                  XmNcolumnWidths, &widths,
                  XmNcolumnMaxLengths, &lengths,
                  XmNrowLabels, &row_labels,
                  XmNcolumnLabels, &column_labels,
                  NULL);
    for (j = 0; j < ep->ncols; j++)
    {
        datap = getcol(ep->gno, ep->setno, j);
        for (i = 0; i < rows; i++)
        {
            if (strcmp(ep->cells[i][j], cells[i][j]) != 0)
            {
                printf("Cell contents changed %s to %s\n", ep->cells[i][j], cells[i][j]);
                datap[i] = atof(cells[i][j]);
            }
        }
    }
    updatesetminmax(ep->gno, ep->setno);
    update_set_status(ep->gno, ep->setno);
    drawgraph();
}

void verifyCB(Widget w, XtPointer client_data, XtPointer calld)
{
    XbaeMatrixModifyVerifyCallbackStruct *cs = (XbaeMatrixModifyVerifyCallbackStruct *)calld;
    printf("Called verify %d %d\n", cs->row, cs->column);
}

void enterCB(Widget w, XtPointer client_data, XtPointer calld)
{
    XbaeMatrixEnterCellCallbackStruct *cs = (XbaeMatrixEnterCellCallbackStruct *)calld;
    printf("Called enter %d %d\n", cs->row, cs->column);
}

void leaveCB(Widget w, XtPointer client_data, XtPointer calld)
{
    XbaeMatrixLeaveCellCallbackStruct *cs = (XbaeMatrixLeaveCellCallbackStruct *)calld;
    printf("Called leave %d %d, %s\n", cs->row, cs->column, cs->value);
}

void drawcellCB(Widget w, XtPointer client_data, XtPointer calld)
{
    double *datap;
    String cell;
    static char buf[128];
    int i, j;
    XbaeMatrixDrawCellCallbackStruct *cs = (XbaeMatrixDrawCellCallbackStruct *)calld;
    printf("Called draw cell %d %d\n", i = cs->row, j = cs->column);
    datap = getcol(cg, 0, j);
    sprintf(buf, "%8.1lf", datap[i]);
    cell = (String)XtMalloc(sizeof(char) * (strlen(buf) + 1));
    strcpy(cell, buf);
    cs->type = XbaeString;
    cs->string = cell;
}

void selectCB(Widget w, XtPointer client_data, XtPointer *call_data)
{
    XbaeMatrixSelectCellCallbackStruct *sc = (XbaeMatrixSelectCellCallbackStruct *)call_data;
    /*
       SelectionPtr selection = GetSelectionFromWidget(w);

       if (selection->mode == ExclusiveMode && selection->selected)
           switch (selection->type) {
           case CellSelection:
               XbaeMatrixDeselectCell(w, selection->row, selection->column);
               break;
           case RowSelection:
               XbaeMatrixDeselectRow(w, selection->row);
               break;
   case ColumnSelection:
   XbaeMatrixDeselectColumn(w, selection->column);
   break;
   }

   selection->row = call_data->row;
   selection->column = call_data->column;

   switch (selection->type) {
   case CellSelection:
   XbaeMatrixSelectCell(w, selection->row, selection->column);
   break;
   case RowSelection:
   XbaeMatrixSelectRow(w, selection->row);
   break;
   case ColumnSelection:
   XbaeMatrixSelectColumn(w, selection->column);
   break;
   }

   selection->selected = True;
   */
    XbaeMatrixSelectRow(w, sc->row);
    XbaeMatrixSelectColumn(w, sc->column);
    printf("Selected %d %d\n", sc->row, sc->column);
}

void create_editp_frame(Widget w, XtPointer client_data, XtPointer call_data)
{
    int x, y;
    static Widget top;
    Widget dialog;
    Widget wbut, rc;

    set_wait_cursor();

    if (top == NULL)
    {
        char buf[32];
        double *xp, *yp;
        char *label1[2];
        label1[0] = "Accept";
        label1[1] = "Close";
        XmGetPos(app_shell, 0, &x, &y);
        top = XmCreateDialogShell(app_shell, "Edit set", NULL, 0);
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(top, "dialog_rc", NULL, 0);
        editp_set_item = CreateSetChoice(dialog, "Edit set:", maxplot, 0);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_ss_frame, NULL);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)top);

        XtManageChild(dialog);
    }
    XtRaise(top);
    unset_wait_cursor();
}

void create_ss_frame(EditPoints *ep)
{
    int x, y;
    Widget dialog;
    Widget wbut, rc;
    char buf[32];
    int i, j, len;
    double *datap;
    char *label1[2];
    label1[0] = "Accept";
    label1[1] = "Close";

    set_wait_cursor();
    XmGetPos(app_shell, 0, &x, &y);
    ep->top = XmCreateDialogShell(app_shell, "Edit set", NULL, 0);
    handle_close(ep->top);
    XtVaSetValues(ep->top, XmNx, x, XmNy, y, NULL);
    dialog = XmCreateRowColumn(ep->top, "dialog_rc", NULL, 0);

    len = getsetlength(ep->gno, ep->setno);
    cells = (String **)XtMalloc(sizeof(String *) * len);
    for (i = 0; i < len; i++)
    {
        cells[i] = (String *)XtMalloc(sizeof(String) * ep->ncols);
        for (j = 0; j < ep->ncols; j++)
        {
            datap = getcol(ep->gno, ep->setno, j);
            sprintf(buf, "%8.1lf", datap[i]);
            cells[i][j] = XtNewString(buf);
        }
    }

    rc = XtVaCreateWidget("rc", xmRowColumnWidgetClass, dialog,
                          XmNorientation, XmHORIZONTAL,
                          NULL);

    ep->editp_format_item = CreatePanelChoice(rc, "Format:",
                                              4,
                                              "Decimal",
                                              "General",
                                              "Exponential",
                                              NULL, 0);

    ep->editp_precision_item = CreatePanelChoice(rc, "Precision:",
                                                 11,
                                                 "0", "1", "2", "3", "4",
                                                 "5", "6", "7", "8", "9",
                                                 NULL, 0);

    ep->editp_width_item = CreatePanelChoice(rc, "Width:",
                                             11,
                                             "1", "2", "3", "4", "5",
                                             "6", "7", "8", "9", "10",
                                             NULL, 0);

    XtManageChild(rc);

    XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

    rowlabels = (String *)malloc(len * sizeof(String));
    for (i = 0; i < len; i++)
    {
        sprintf(buf, "%d", i + 1);
        rowlabels[i] = (String)malloc((sizeof(buf) + 1) * sizeof(char));
        strcpy(rowlabels[i], buf);
    }
    ep->mw = XtVaCreateManagedWidget("mw",
                                     xbaeMatrixWidgetClass, dialog,
                                     XmNrows, ep->nrows,
                                     XmNcolumns, ep->ncols,
                                     XmNcolumnWidths, widths,
                                     XmNvisibleRows, 10,
                                     XmNvisibleColumns, 2,
                                     XmNrowLabels, rowlabels,
                                     XmNcolumnLabels, ep->collabels,
                                     XmNcells, cells,
                                     NULL);
    XtAddCallback(ep->mw, XmNdrawCellCallback, drawcellCB, NULL);
    XtAddCallback(ep->mw, XmNmodifyVerifyCallback, verifyCB, NULL);
    XtAddCallback(ep->mw, XmNleaveCellCallback, leaveCB, NULL);
    XtAddCallback(ep->mw, XmNenterCellCallback, enterCB, NULL);
    XtAddCallback(ep->mw, XmNselectCellCallback, selectCB, NULL);
    /*
       XtVaSetValues(w,
                     XmNuserData, selection,
                     NULL);
   */

    ep->cells = cells;
    /*
       for (i = 0; i < len; i++) {
      for (j = 0; j < ep->ncols; j++) {
          XtFree((XtPointer) cells[i][j]);
      }
      XtFree((XtPointer) cells[i]);
       }
       XtFree((XtPointer) cells);
   */

    XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

    CreateCommandButtons(dialog, 2, but1, label1);
    XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_ss_proc, (XtPointer)ep);
    XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)ep->top);

    XtManageChild(dialog);
    XtRaise(ep->top);
    unset_wait_cursor();
}

#else

/*
 * Create the editor using xterm & GR_EDITOR
 */
void create_editp_frame(Widget, XtPointer, XtPointer)
{
    char tbuf[256], buf[256], *fname;
    char ebuf[256], *s;
    if ((s = getenv("GR_EDITOR")) != NULL)
    {
        strcpy(ebuf, s);
    }
    else
    {
        strcpy(ebuf, "vi");
    }
    strcpy(tbuf, "/tmp/ACEgrXXXXXX");
    mkstemp(tbuf);
    sprintf(buf, "xterm -e %s %s", ebuf, tbuf);
    system(buf);
    getdata(cg, fname, DISK, XY);
}
#endif
