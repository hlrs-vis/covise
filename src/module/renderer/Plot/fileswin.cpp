/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: fileswin.c,v 1.7 1994/09/29 03:37:37 pturner Exp pturner $
 *
 * read/write data/parameter files
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <sys/param.h>

#include <Xm/Xm.h>
#include <Xm/DialogS.h>
#include <Xm/BulletinB.h>
#include <Xm/FileSB.h>
#include <Xm/Frame.h>
#include <Xm/Form.h>
#include <Xm/Label.h>
#include <Xm/List.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/SelectioB.h>
#include <Xm/Separator.h>
#include <Xm/ToggleB.h>
#define REGIONSINC

#include "globals.h"
#include "motifinc.h"
#include "noxprotos.h"

static Widget rdata_dialog; /* read data popup */
static Widget *read_graph_item; /* graph choice item */
static Widget *read_ftype_item; /* set type choice item */
static Widget read_auto_item; /* autoscale on read button */
static Widget wparam_frame; /* write params popup */
static Widget wparam_panel;
// static Widget wparam_text_item;
static Widget *wparam_choice_item;
static void set_type_proc(int data);
static void set_src_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void rdata_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_rparams_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void wparam_apply_notify_proc(Widget w, XtPointer client_data, XtPointer call_data);

static Widget rparams_dialog; /* read params popup */

extern void expand_tilde(char *buf);
extern void set_title(char *ts);

static void set_type_proc(int data)
{
    switch (data)
    {
    case 0:
        curtype = XY;
        break;
    case 1:
        curtype = NXY;
        break;
    case 2:
        curtype = IHL;
        break;
    case 3:
        curtype = BIN;
        break;
    case 4:
        curtype = XYDX;
        break;
    case 5:
        curtype = XYDY;
        break;
    case 6:
        curtype = XYDXDX;
        break;
    case 7:
        curtype = XYDYDY;
        break;
    case 8:
        curtype = XYDXDY;
        break;
    case 9:
        curtype = XYZ;
        break;
    case 10:
        curtype = XYHILO;
        break;
    case 11:
        curtype = XYRT;
        break;
    case 12:
        curtype = XYBOX;
        break;
    case 13:
        curtype = RAWSPICE;
        break;
    case 14:
        curtype = XYBOXPLOT;
        break;
    }
}

static void set_src_proc(Widget, XtPointer client_data, XtPointer)
{
    int data = (long)client_data;

    switch (data)
    {
    case 0:
        cursource = DISK;
        break;
    case 1:
        cursource = PIPE;
        break;
    }
}

static void rdata_proc(Widget, XtPointer, XtPointer)
{
    int graphno, autoflag;
    Arg args;
    XmString list_item;
    char *s;

    XtSetArg(args, XmNtextString, &list_item);
    XtGetValues(rdata_dialog, &args, 1);
    XmStringGetLtoR(list_item, charset, &s);
    graphno = GetChoice(read_graph_item) - 1;
    autoflag = XmToggleButtonGetState(read_auto_item);
    if (graphno == -1)
    {
        graphno = cg;
    }
    if (g[graphno].active == OFF)
    {
        set_graph_active(graphno);
    }
    set_type_proc(GetChoice(read_ftype_item));
    set_wait_cursor();
    if (getdata(graphno, s, cursource, curtype))
    {
        if (autoscale_onread || autoflag)
        {
            autoscale_proc((Widget)NULL, (XtPointer)0, (XtPointer)NULL);
        }
        else
        {
            drawgraph();
        }
    }
    XtFree(s);
    unset_wait_cursor();
}

void create_file_popup(Widget, XtPointer, XtPointer)
{
    long i;
    Widget lab, rc, rc2, fr, rb, w[3];

    set_wait_cursor();

    if (rdata_dialog == NULL)
    {
        rdata_dialog = XmCreateFileSelectionDialog(app_shell, (char *)"rdata_dialog", NULL, 0);
        XtVaSetValues(XtParent(rdata_dialog), XmNtitle, "Read sets", NULL);
        XtAddCallback(rdata_dialog, XmNcancelCallback, (XtCallbackProc)destroy_dialog, rdata_dialog);
        XtAddCallback(rdata_dialog, XmNokCallback, rdata_proc, 0);

        curtype = XY;

        rc = XmCreateRowColumn(rdata_dialog, (char *)"Read data main RC", NULL, 0);

        fr = XmCreateFrame(rc, (char *)"frame_1", NULL, 0);
        rc2 = XmCreateRowColumn(fr, (char *)"Read data main RC", NULL, 0);
        XtVaSetValues(rc2, XmNorientation, XmHORIZONTAL, NULL);
        read_ftype_item = CreatePanelChoice(rc2, "File format: ", 16,
                                            "X Y",
                                            "X Y1 Y2 ... ",
                                            "IHL",
                                            "Binary",
                                            "X Y DX",
                                            "X Y DY",
                                            "X Y DX1 DX2",
                                            "X Y DY1 DY2",
                                            "X Y DX DY",
                                            "X Y Z",
                                            "X HI LO OPEN CLOSE",
                                            "X Y RADIUS",
                                            "X Y BOX",
                                            "Rawspice",
                                            "X Y BOXPLOT",
                                            NULL, NULL);

        XtManageChild(rc2);
        XtManageChild(fr);

        fr = XmCreateFrame(rc, (char *)"frame_2", NULL, 0);
        rc2 = XmCreateRowColumn(fr, (char *)"Read data main RC", NULL, 0);
        XtVaSetValues(rc2, XmNorientation, XmHORIZONTAL, NULL);
        lab = XmCreateLabel(rc2, (char *)"File Source:", NULL, 0);
        rb = XmCreateRadioBox(rc2, (char *)"radio_box_2", NULL, 0);
        XtVaSetValues(rb, XmNorientation, XmHORIZONTAL, NULL);
        w[0] = XmCreateToggleButton(rb, (char *)"Disk", NULL, 0);
        w[1] = XmCreateToggleButton(rb, (char *)"Pipe", NULL, 0);
        for (i = 0; i < 2; i++)
        {
            XtAddCallback(w[i], XmNvalueChangedCallback, set_src_proc, (XtPointer)i);
        }
        XtManageChild(lab);
        XtManageChild(rb);
        XtManageChildren(w, 2);
        XtManageChild(rc2);
        XtManageChild(fr);
        XmToggleButtonSetState(w[0], True, False);

        fr = XmCreateFrame(rc, (char *)"frame_3", NULL, 0);
        rc2 = XmCreateRowColumn(fr, (char *)"Read data main RC", NULL, 0);
        read_graph_item = CreateGraphChoice(rc2, "Read to graph: ", maxgraph, 1);
        read_auto_item = XmCreateToggleButton(rc2, (char *)"Autoscale on read", NULL, 0);
        XtManageChild(read_auto_item);
        XtManageChild(rc2);
        XtManageChild(fr);
        XtManageChild(rc);

        XtManageChild(rc);
    }
    XtRaise(rdata_dialog);
    unset_wait_cursor();
}

static void do_rparams_proc(Widget, XtPointer, XtPointer)
{
    Arg args;
    XmString list_item;
    char *s;

    XtSetArg(args, XmNtextString, &list_item);
    XtGetValues(rparams_dialog, &args, 1);
    XmStringGetLtoR(list_item, charset, &s);
    set_wait_cursor();
    getparms(cg, s);
    unset_wait_cursor();
    XtFree(s);
}

void create_rparams_popup(Widget, XtPointer, XtPointer)
{
    set_wait_cursor();
    if (rparams_dialog == NULL)
    {
        rparams_dialog = XmCreateFileSelectionDialog(app_shell, (char *)"rparams_dialog", NULL, 0);
        XtVaSetValues(XtParent(rparams_dialog), XmNtitle, "Read parameters", NULL);
        XtAddCallback(rparams_dialog, XmNcancelCallback, (XtCallbackProc)destroy_dialog, rparams_dialog);
        XtAddCallback(rparams_dialog, XmNokCallback, (XtCallbackProc)do_rparams_proc, 0);
    }
    XtRaise(rparams_dialog);
    unset_wait_cursor();
}

/*
 * Create the wparam Frame and the wparam Panel
 */
void create_wparam_frame(Widget, XtPointer, XtPointer)
{
    Widget fr;

    set_wait_cursor();
    if (wparam_frame == NULL)
    {
        wparam_frame = XmCreateFileSelectionDialog(app_shell, (char *)"wparam_frame", NULL, 0);
        XtVaSetValues(XtParent(wparam_frame), XmNtitle, "Write plot parameters", NULL);
        XtAddCallback(wparam_frame, XmNcancelCallback, (XtCallbackProc)destroy_dialog, wparam_frame);
        XtAddCallback(wparam_frame, XmNokCallback, (XtCallbackProc)wparam_apply_notify_proc, 0);

        /* may not be needed
         handle_close(wparam_frame);
      */

        fr = XmCreateFrame(wparam_frame, (char *)"fr", NULL, 0);
        wparam_panel = XmCreateRowColumn(fr, (char *)"wparam_rc", NULL, 0);
        wparam_choice_item = CreateGraphChoice(wparam_panel, "Write parameters from graph: ", maxgraph, 2);

        XtManageChild(fr);
        XtManageChild(wparam_panel);
    }
    XtRaise(wparam_frame);
    unset_wait_cursor();
}

static void wparam_apply_notify_proc(Widget, XtPointer, XtPointer)
{
    char fname[256], *s;
    Arg args;
    XmString list_item;
    int wparamno = (int)GetChoice(wparam_choice_item);

    XtSetArg(args, XmNtextString, &list_item);
    XtGetValues(wparam_frame, &args, 1);
    XmStringGetLtoR(list_item, charset, &s);

    wparamno--;

    strcpy(fname, s);

    if (!fexists(fname))
    {
        FILE *pp = fopen(fname, "w");

        if (pp != NULL)
        {
            set_wait_cursor();
            if (wparamno == -1)
            {
                wparamno = cg;
                putparms(wparamno, pp, 0);
                fclose(pp);
            }
            else if (wparamno == maxgraph)
            {
                putparms(-1, pp, 0);
                fclose(pp);
            }
            else
            {
                putparms(wparamno, pp, 0);
                fclose(pp);
            }
            unset_wait_cursor();
        }
        else
        {
            errwin("Unable to open file");
        }
    }
}

static Widget workingd_dialog;

static Widget dir_item;

static void workingdir_apply_notify_proc(Widget, XtPointer, XtPointer)
{
    Arg args;
    XmString list_item;
    char *s;
    char buf[MAXPATHLEN];

    XtSetArg(args, XmNtextString, &list_item);
    XtGetValues(workingd_dialog, &args, 1);
    XmStringGetLtoR(list_item, charset, &s);
    strcpy(buf, s);
    XtFree(s);

    if (buf[0] == '~')
    {
        expand_tilde(buf);
    }
    if (chdir(buf) >= 0)
    {
        strcpy(workingdir, buf);
        set_title(workingdir);
        XmFileSelectionDoSearch(workingd_dialog, NULL);
    }
    else
    {
        errwin("Can't change to directory");
    }
    XtUnmanageChild(workingd_dialog);
}

static void select_dir(Widget, XtPointer, XmListCallbackStruct *cbs)
{
    char buf[MAXPATHLEN], *str;

    XmStringGetLtoR(cbs->item, charset, &str);
    strcpy(buf, str);
    XtFree(str);

    xv_setstr(dir_item, buf);
    XmFileSelectionDoSearch(workingd_dialog, NULL);
}

void create_workingdir_popup(Widget w, XtPointer, XtPointer)
{
    XmString str;

    set_wait_cursor();
    if (workingd_dialog == NULL)
    {
        workingd_dialog = XmCreateFileSelectionDialog(app_shell, (char *)"workingd_dialog", NULL, 0);
        XtVaSetValues(XtParent(workingd_dialog), XmNtitle, "Set working directory", NULL);
        XtAddCallback(workingd_dialog, XmNcancelCallback, (XtCallbackProc)destroy_dialog, (XtPointer)workingd_dialog);
        XtAddCallback(workingd_dialog, XmNokCallback, (XtCallbackProc)workingdir_apply_notify_proc, (XtPointer)0);

        /* unmanage unneeded items */
        w = XmFileSelectionBoxGetChild(workingd_dialog, XmDIALOG_LIST);
        XtUnmanageChild(XtParent(w));
        w = XmFileSelectionBoxGetChild(workingd_dialog, XmDIALOG_LIST_LABEL);
        XtUnmanageChild(w);
        w = XmFileSelectionBoxGetChild(workingd_dialog, XmDIALOG_FILTER_LABEL);
        XtUnmanageChild(w);
        w = XmFileSelectionBoxGetChild(workingd_dialog, XmDIALOG_FILTER_TEXT);
        XtUnmanageChild(w);
        w = XmFileSelectionBoxGetChild(workingd_dialog, XmDIALOG_APPLY_BUTTON);
        XtUnmanageChild(w);

        /* save the name of the text item used for definition */
        dir_item = XmFileSelectionBoxGetChild(workingd_dialog, XmDIALOG_TEXT);

        /* Add a callback to the dir list */
        w = XmFileSelectionBoxGetChild(workingd_dialog, XmDIALOG_DIR_LIST);
        XtAddCallback(w, XmNsingleSelectionCallback, (XtCallbackProc)select_dir, (XtPointer)0);
        XtVaSetValues(w, XmNselectionPolicy, XmSINGLE_SELECT, NULL);
    }
    xv_setstr(dir_item, workingdir);
    XtVaSetValues(workingd_dialog, XmNdirectory,
                  str = XmStringCreateLtoR(workingdir, charset), NULL);
    XmFileSelectionDoSearch(workingd_dialog, NULL);
    XmStringFree(str);
    XtRaise(workingd_dialog);
    unset_wait_cursor();
}

#if defined(HAVE_NETCDF) || defined(HAVE_MFHDF)

#include <netcdf.h>

/*
 *
 * netcdf reader
 *
 */

extern int readcdf; /* declared in main.c */

extern char netcdf_name[], xvar_name[], yvar_name[];

static Widget netcdf_frame = (Widget)NULL;

static Widget *netcdf_set_item;
static Widget netcdf_listx_item;
static Widget netcdf_listy_item;
static Widget netcdf_file_item;
static Widget netcdf_basex_item; /* base for X if index */
static Widget netcdf_incrx_item; /* increment for X if index */
static Widget netcdf_auto_item;

void create_netcdffiles_popup(Widget w, XtPointer client_data, XtPointer call_data);

static void do_netcdfquery_proc(Widget w, XtPointer client_data, XtPointer call_data);

void update_netcdfs(void);

int getnetcdfvars(void);

static void do_netcdf_proc(Widget, XtPointer, XtPointer)
{
    int setno, src;
    char fname[256];
    char buf[256], xvar[256], yvar[256];
    XmString xms;
    XmString *s, cs;
    int *pos_list;
    int j, pos_cnt, cnt, autoflag, retval;
    char *cstr;

    set_wait_cursor();
    autoflag = XmToggleButtonGetState(netcdf_auto_item);

    /*
    * setno == -1, then next set
    */
    setno = GetChoice(netcdf_set_item) - 1;
    strcpy(fname, xv_getstr(netcdf_file_item));
    if (XmListGetSelectedPos(netcdf_listx_item, &pos_list, &pos_cnt))
    {
        XtVaGetValues(netcdf_listx_item,
                      XmNselectedItemCount, &cnt,
                      XmNselectedItems, &s,
                      NULL);
        cs = XmStringCopy(*s);
        if (XmStringGetLtoR(cs, charset, &cstr))
        {
            strcpy(xvar, cstr);
            XtFree(cstr);
        }
        XmStringFree(cs);
    }
    else
    {
        errwin("Need to select X, either variable name or INDEX");
        unset_wait_cursor();
        return;
    }
    if (XmListGetSelectedPos(netcdf_listy_item, &pos_list, &pos_cnt))
    {
        j = pos_list[0];
        XtVaGetValues(netcdf_listy_item,
                      XmNselectedItemCount, &cnt,
                      XmNselectedItems, &s,
                      NULL);
        cs = XmStringCopy(*s);
        if (XmStringGetLtoR(cs, charset, &cstr))
        {
            strcpy(yvar, cstr);
            XtFree(cstr);
        }
        XmStringFree(cs);
    }
    else
    {
        errwin("Need to select Y");
        unset_wait_cursor();
        return;
    }
    if (strcmp(xvar, "INDEX") == 0)
    {
        retval = readnetcdf(cg, setno, fname, NULL, yvar, -1, -1, 1);
    }
    else
    {
        retval = readnetcdf(cg, setno, fname, xvar, yvar, -1, -1, 1);
    }
    if (retval)
    {
        if (autoflag)
        {
            autoscale_proc((Widget)NULL, (XtPointer)0, (XtPointer)NULL);
        }
        else
        {
            drawgraph();
        }
    } /* error from readnetcdf() */
    else
    {
    }

    unset_wait_cursor();
}

void update_netcdfs(void)
{
    int i, j;
    char buf[256], fname[512];
    XmString xms;
    int cdfid; /* netCDF id */
    int ndims, nvars, ngatts, recdim;
    int var_id;
    long start[2];
    long count[2];
    char varname[256];
    nc_type datatype = 0;
    int dim[100], natts;
    long dimlen[100];
    long len;
    extern int ncopts;

    ncopts = 0; /* no crash on error */

    if (netcdf_frame != NULL)
    {
        strcpy(fname, xv_getstr(netcdf_file_item));
        set_wait_cursor();
        XmListDeleteAllItems(netcdf_listx_item);
        XmListDeleteAllItems(netcdf_listy_item);
        xms = XmStringCreateLtoR("INDEX", charset);
        XmListAddItemUnselected(netcdf_listx_item, xms, 0);
        XmStringFree(xms);

        if (strlen(fname) < 2)
        {
            unset_wait_cursor();
            return;
        }
        if ((cdfid = ncopen(fname, NC_NOWRITE)) == -1)
        {
            errwin("Can't open file.");
            unset_wait_cursor();
            return;
        }
        ncinquire(cdfid, &ndims, &nvars, &ngatts, &recdim);
        /*
          printf("%d %d %d %d\n", ndims, nvars, ngatts, recdim);
      */
        for (i = 0; i < ndims; i++)
        {
            ncdiminq(cdfid, i, NULL, &dimlen[i]);
        }
        for (i = 0; i < nvars; i++)
        {
            ncvarinq(cdfid, i, varname, &datatype, &ndims, dim, &natts);
            if ((var_id = ncvarid(cdfid, varname)) == -1)
            {
                char ebuf[256];
                sprintf(ebuf, "update_netcdfs(): No such variable %s", varname);
                errwin(ebuf);
                continue;
            }
            if (ndims != 1)
            {
                continue;
            }
            ncdiminq(cdfid, dim[0], (char *)NULL, &len);
            sprintf(buf, "%s", varname);
            xms = XmStringCreateLtoR(buf, charset);
            XmListAddItemUnselected(netcdf_listx_item, xms, 0);
            XmListAddItemUnselected(netcdf_listy_item, xms, 0);
            XmStringFree(xms);
        }
        ncclose(cdfid);
        unset_wait_cursor();
    }
}

static void do_netcdfupdate_proc(Widget, XtPointer, XtPointer)
{
    int i;
    char buf[256];

    set_wait_cursor();
    update_netcdfs();
    unset_wait_cursor();
}

void create_netcdfs_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    static Widget top, dialog;
    Widget wbut, lab, rc, rcl, rc1, rc2, form;
    Arg args[3];

    set_wait_cursor();
    if (top == NULL)
    {
        char *label1[5];
        Widget but1[5];

        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Files...";
        label1[2] = (char *)"Update";
        label1[3] = (char *)"Query";
        label1[4] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
#ifdef HAVE_MFHDF
        top = XmCreateDialogShell(app_shell, (char *)"netCDF/HDF", NULL, 0);
#else
#ifdef HAVE_NETCDF
        top = XmCreateDialogShell(app_shell, (char *)"netCDF", NULL, 0);
#endif
#endif
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(top, (char *)"dialog_rc", NULL, 0);

        /*
         form = XmCreateForm(dialog, "form", NULL, 0);
      */
        form = XmCreateRowColumn(dialog, (char *)"form", NULL, 0);
        XtVaSetValues(form,
                      XmNpacking, XmPACK_COLUMN,
                      XmNnumColumns, 1,
                      XmNorientation, XmHORIZONTAL,
                      XmNisAligned, True,
                      XmNadjustLast, False,
                      XmNentryAlignment, XmALIGNMENT_END,
                      NULL);

        XtSetArg(args[0], XmNlistSizePolicy, XmRESIZE_IF_POSSIBLE);
        XtSetArg(args[1], XmNvisibleItemCount, 5);

        rc1 = XmCreateRowColumn(form, (char *)"rc1", NULL, 0);
        lab = XmCreateLabel(rc1, (char *)"Select set X:", NULL, 0);
        XtManageChild(lab);
        netcdf_listx_item = XmCreateScrolledList(rc1, (char *)"list", args, 2);
        XtManageChild(netcdf_listx_item);
        XtManageChild(rc1);

        rc2 = XmCreateRowColumn(form, (char *)"rc2", NULL, 0);
        lab = XmCreateLabel(rc2, (char *)"Select set Y:", NULL, 0);
        XtManageChild(lab);
        netcdf_listy_item = XmCreateScrolledList(rc2, (char *)"list", args, 2);
        XtManageChild(netcdf_listy_item);
        XtManageChild(rc2);

        XtManageChild(form);

        netcdf_file_item = CreateTextItem2(dialog, 30, "netCDF file:");
        netcdf_set_item = CreateSetChoice(dialog, "Read to set:", maxplot, 4);
        netcdf_auto_item = XmCreateToggleButton(dialog, "Autoscale on read", NULL, 0);
        XtManageChild(netcdf_auto_item);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 5, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_netcdf_proc,
                      (XtPointer)NULL);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)create_netcdffiles_popup,
                      (XtPointer)NULL);
        XtAddCallback(but1[2], XmNactivateCallback, (XtCallbackProc)do_netcdfupdate_proc,
                      (XtPointer)NULL);
        XtAddCallback(but1[3], XmNactivateCallback, (XtCallbackProc)do_netcdfquery_proc,
                      (XtPointer)NULL);
        XtAddCallback(but1[4], XmNactivateCallback, (XtCallbackProc)destroy_dialog,
                      (XtPointer)top);

        XtManageChild(dialog);
        netcdf_frame = top;
        if (strlen(netcdf_name))
        {
            xv_setstr(netcdf_file_item, netcdf_name);
        }
    }
    update_netcdfs();
    XtRaise(top);
    unset_wait_cursor();
}

static void do_netcdffile_proc(Widget, XtPointer, XtPointer)
{
    Widget dialog = (Widget)client_data;
    Arg args;
    XmString list_item;
    char *s;
    char fname[256];

    set_wait_cursor();

    XtSetArg(args, XmNtextString, &list_item);
    XtGetValues(dialog, &args, 1);
    XmStringGetLtoR(list_item, charset, &s);

    xv_setstr(netcdf_file_item, s);

    XtFree(s);

    unset_wait_cursor();

    XtUnmanageChild(dialog);
    update_netcdfs();
}

void create_netcdffiles_popup(Widget, XtPointer, XtPointer)
{
    int x, y;
    static Widget top;
    Widget dialog;
    Widget wbut, rc, fr;
    Arg args[2];

    set_wait_cursor();
    if (top == NULL)
    {
        top = XmCreateFileSelectionDialog(app_shell, "netcdfs", NULL, 0);
        XtVaSetValues(XtParent(top), XmNtitle, "Select netCDF file", NULL);

        XtAddCallback(top, XmNokCallback, (XtCallbackProc)do_netcdffile_proc, (XtPointer)top);
        XtAddCallback(top, XmNcancelCallback, (XtCallbackProc)destroy_dialog, (XtPointer)top);
    }
    XtRaise(top);
    unset_wait_cursor();
}

char *getcdf_type(nc_type datatype)
{
    switch (datatype)
    {
    case NC_SHORT:
        return "NC_SHORT";
        break;
    case NC_LONG:
        return "NC_LONG";
        break;
    case NC_FLOAT:
        return "NC_FLOAT";
        break;
    case NC_DOUBLE:
        return "NC_DOUBLE";
        break;
    default:
        return "UNKNOWN (can't read this)";
        break;
    }
}

/*
 * TODO, lots of declared, but unused variables here
 */
static void do_netcdfquery_proc(Widget, XtPointer, XtPointer)
{
    int setno, src;
    char xvar[256], yvar[256];
    char buf[256], fname[512];
    XmString xms;
    XmString *s, cs;
    int *pos_list;
    int i, j, pos_cnt, cnt;
    char *cstr;

    int cdfid; /* netCDF id */
    int ndims, nvars, ngatts, recdim;
    int var_id;
    long start[2];
    long count[2];
    char varname[256];
    nc_type datatype = 0;
    int dim[100], natts;
    long dimlen[100];
    long len;

    int x_id, y_id;
    nc_type xdatatype = 0;
    nc_type ydatatype = 0;
    int xndims, xdim[10], xnatts;
    int yndims, ydim[10], ynatts;
    long nx, ny;

    int atlen;
    char attname[256];
    char atcharval[256];

    extern int ncopts;

    ncopts = 0; /* no crash on error */

    set_wait_cursor();

    strcpy(fname, xv_getstr(netcdf_file_item));

    if ((cdfid = ncopen(fname, NC_NOWRITE)) == -1)
    {
        errwin("Can't open file.");
        goto out2;
    }
    if (XmListGetSelectedPos(netcdf_listx_item, &pos_list, &pos_cnt))
    {
        XtVaGetValues(netcdf_listx_item,
                      XmNselectedItemCount, &cnt,
                      XmNselectedItems, &s,
                      NULL);
        cs = XmStringCopy(*s);
        if (XmStringGetLtoR(cs, charset, &cstr))
        {
            strcpy(xvar, cstr);
            XtFree(cstr);
        }
        XmStringFree(cs);
    }
    else
    {
        errwin("Need to select X, either variable name or INDEX");
        goto out1;
    }
    if (XmListGetSelectedPos(netcdf_listy_item, &pos_list, &pos_cnt))
    {
        XtVaGetValues(netcdf_listy_item,
                      XmNselectedItemCount, &cnt,
                      XmNselectedItems, &s,
                      NULL);
        cs = XmStringCopy(*s);
        if (XmStringGetLtoR(cs, charset, &cstr))
        {
            strcpy(yvar, cstr);
            XtFree(cstr);
        }
        XmStringFree(cs);
    }
    else
    {
        errwin("Need to select Y");
        goto out1;
    }
    if (strcmp(xvar, "INDEX") == 0)
    {
        stufftext("X is the index of the Y variable\n", STUFF_START);
    }
    else
    {
        if ((x_id = ncvarid(cdfid, xvar)) == -1)
        {
            char ebuf[256];
            sprintf(ebuf, "do_query(): No such variable %s for X", xvar);
            errwin(ebuf);
            goto out1;
        }
        ncvarinq(cdfid, x_id, NULL, &xdatatype, &xndims, xdim, &xnatts);
        ncdiminq(cdfid, xdim[0], NULL, &nx);
        sprintf(buf, "X is %s, data type %s \t length [%d]\n", xvar, getcdf_type(xdatatype), nx);
        stufftext(buf, STUFF_TEXT);
        sprintf(buf, "\t%d Attributes:\n", xnatts);
        stufftext(buf, STUFF_TEXT);
        for (i = 0; i < xnatts; i++)
        {
            atcharval[0] = 0;
            ncattname(cdfid, x_id, i, attname);
            ncattinq(cdfid, x_id, attname, &datatype, &atlen);
            switch (datatype)
            {
            case NC_CHAR:
                ncattget(cdfid, x_id, attname, (void *)atcharval);
                break;
            }
            sprintf(buf, "\t\t%s: %s\n", attname, atcharval);
            stufftext(buf, STUFF_TEXT);
        }
    }
    if ((y_id = ncvarid(cdfid, yvar)) == -1)
    {
        char ebuf[256];
        sprintf(ebuf, "do_query(): No such variable %s for Y", yvar);
        errwin(ebuf);
        goto out1;
    }
    ncvarinq(cdfid, y_id, NULL, &ydatatype, &yndims, ydim, &ynatts);
    ncdiminq(cdfid, ydim[0], NULL, &ny);
    sprintf(buf, "Y is %s, data type %s \t length [%d]\n", yvar, getcdf_type(ydatatype), ny);
    stufftext(buf, STUFF_TEXT);
    sprintf(buf, "\t%d Attributes:\n", ynatts);
    stufftext(buf, STUFF_TEXT);
    for (i = 0; i < ynatts; i++)
    {
        atcharval[0] = 0;
        ncattname(cdfid, y_id, i, attname);
        ncattinq(cdfid, y_id, attname, &datatype, &atlen);
        switch (datatype)
        {
        case NC_CHAR:
            ncattget(cdfid, y_id, attname, (void *)atcharval);
            break;
        }
        sprintf(buf, "\t\t%s: %s\n", attname, atcharval);
        stufftext(buf, STUFF_TEXT);
    }

out1:
    ;
    ncclose(cdfid);

out2:
    ;
    stufftext("\n", STUFF_STOP);
    unset_wait_cursor();
}
#endif
