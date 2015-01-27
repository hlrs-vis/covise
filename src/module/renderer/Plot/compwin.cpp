/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: compwin.c,v 1.21 1994/11/04 07:04:49 pturner Exp pturner $
 *
 * transformations, curve fitting, etc.
 *
 * formerly, this was all one big popup, now it is several.
 * All are created as needed
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "extern.h"

#include <Xm/Xm.h>
#include <Xm/BulletinB.h>
#include <Xm/DialogS.h>
#include <Xm/Form.h>
#include <Xm/Label.h>
#include <Xm/LabelG.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/Separator.h>
#include <Xm/ToggleB.h>
#include <Xm/Text.h>
#include <Xm/List.h>

#include "globals.h"
#include "motifinc.h"

extern int nonlflag;
/* true if nonlinear curve fitting module is
 * to be included */

int lsmethod = 0; /* 1 == AS274, 0 = old method */

static Widget but1[2];
static Widget but2[3];
extern int savedretval; // globale Variable aus PlotCommunication

void create_ntiles_frame(Widget w, XtPointer client_data, XtPointer call_data);
void create_geom_frame(Widget w, XtPointer client_data, XtPointer call_data);

static void do_compute_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_load_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_compute_proc2(Widget w, XtPointer client_data, XtPointer call_data);
static void do_digfilter_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_linearc_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_xcor_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_spline_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_int_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_differ_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_seasonal_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_interp_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_regress_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_runavg_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_fourier_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_fft_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_window_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_histo_proc(Widget w, XtPointer client_data, XtPointer call_data);
static void do_sample_proc(Widget w, XtPointer client_data, XtPointer call_data);

extern int nactive(int gno);
extern void do_seasonal_diff(int setno, int period);
extern void do_ntiles(int gno, int setno, int nt);

void do_pick_compose(Widget, XtPointer client_data, XtPointer)
{
    set_action(0);
    set_action((long)client_data);
}

static SetChoiceItem *plist = NULL;
static int nplist = 0;

int GetSetFromString(char *buf)
{
    int retval = SET_SELECT_ERROR;
    if (strcmp(buf, "New set") == 0)
    {
        retval = SET_SELECT_NEXT;
    }
    else if (strcmp(buf, "All sets") == 0)
    {
        retval = SET_SELECT_ALL;
    }
    else
    {
        sscanf(buf, "S%d", &retval);
    }
    return retval;
}

int GetSelectedSet(SetChoiceItem l)
{
    int retval = SET_SELECT_ERROR;
    int *pos_list;
    int pos_cnt, cnt;
    char buf[256];
    if (!ismaster)
        return savedretval; // globale Variable aus PlotCommunication
    if (XmListGetSelectedPos(l.list, &pos_list, &pos_cnt))
    {
        XmString *s, cs;
        char *cstr;
        XtVaGetValues(l.list,
                      XmNselectedItemCount, &cnt,
                      XmNselectedItems, &s,
                      NULL);
        cs = XmStringCopy(*s);
        if (XmStringGetLtoR(cs, charset, &cstr))
        {
            strcpy(buf, cstr);
            if (strcmp(buf, "New set") == 0)
            {
                retval = SET_SELECT_NEXT;
            }
            else if (strcmp(buf, "All sets") == 0)
            {
                retval = SET_SELECT_ALL;
            }
            else
            {
                sscanf(buf, "S%d", &retval);
            }
            XtFree(cstr);
        }
        XmStringFree(cs);
    }
    if (ismaster)
    {
        cm->sendCommandMessage(GETSELECTEDSET, retval, 0);
    }
    return retval;
}

/*
 * if the set selection type is multiple, then get a
 * list of sets, returns the number of selected sets.
 */
int GetSelectedSets(SetChoiceItem l, int **sets)
{
    int i;
    int cnt = 0, retval = SET_SELECT_ERROR;
    int *ptr = NULL;
    int *pos_list;
    int pos_cnt, gno;
    if (XmListGetSelectedPos(l.list, &pos_list, &pos_cnt))
    {
        char buf[256];
        char *cstr;
        XmString *s, cs;

        XtVaGetValues(l.list,
                      XmNselectedItemCount, &cnt,
                      XmNselectedItems, &s,
                      NULL);
        *sets = (int *)malloc((cnt + 10) * sizeof(int));
        ptr = *sets;
        for (i = 0; i < cnt; i++)
        {
            cs = XmStringCopy(s[i]);
            if (XmStringGetLtoR(cs, charset, &cstr))
            {
                strcpy(buf, cstr);
                if (strcmp(buf, "New set") == 0)
                {
                    retval = SET_SELECT_NEXT;
                    if (ismaster)
                        cm->sendCommand_ValuesMessage(GETSELECTEDSETS0, retval, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                    return retval;
                }
                else if (strcmp(buf, "All sets") == 0)
                {
                    int j, nsets = 0;
                    retval = SET_SELECT_ALL;
                    if (l.gno == GRAPH_SELECT_CURRENT)
                    {
                        gno = cg;
                    }
                    else
                    {
                        gno = l.gno;
                    }
                    retval = nactive(gno);
                    *sets = (int *)realloc(*sets, retval * sizeof(int));
                    ptr = *sets;
                    for (j = 0; j < g[gno].maxplot; j++)
                    {
                        if (isactive(gno, j))
                        {
                            ptr[nsets] = j;
                            nsets++;
                        }
                    }
                    if (nsets != retval)
                    {
                        errwin("Nsets != reval, can't happen!");
                    }
                    if ((ismaster) && (ptr != NULL))
                    {
                        if (nsets > 29)
                        {
                            cm->sendCommand_ValuesMessage(GETSELECTEDSETS0, nsets, ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8]);
                            cm->sendCommand_ValuesMessage(GETSELECTEDSETS1, ptr[9], ptr[10], ptr[11], ptr[12], ptr[13], ptr[14], ptr[15], ptr[16], ptr[17], ptr[18]);
                            cm->sendCommand_ValuesMessage(GETSELECTEDSETS2, ptr[19], ptr[20], ptr[21], ptr[22], ptr[23], ptr[24], ptr[25], ptr[26], ptr[27], ptr[28]);
                            cm->sendCommand_ValuesMessage(GETSELECTEDSETS3, ptr[29], ptr[30], ptr[31], ptr[32], ptr[33], ptr[34], ptr[35], ptr[36], ptr[37], ptr[38]);
                        }
                        if (nsets > 19)
                        {
                            cm->sendCommand_ValuesMessage(GETSELECTEDSETS0, nsets, ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8]);
                            cm->sendCommand_ValuesMessage(GETSELECTEDSETS1, ptr[9], ptr[10], ptr[11], ptr[12], ptr[13], ptr[14], ptr[15], ptr[16], ptr[17], ptr[18]);
                            cm->sendCommand_ValuesMessage(GETSELECTEDSETS2, ptr[19], ptr[20], ptr[21], ptr[22], ptr[23], ptr[24], ptr[25], ptr[26], ptr[27], ptr[28]);
                        }
                        if (nsets > 9)
                        {
                            cm->sendCommand_ValuesMessage(GETSELECTEDSETS0, nsets, ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8]);
                            cm->sendCommand_ValuesMessage(GETSELECTEDSETS1, ptr[9], ptr[10], ptr[11], ptr[12], ptr[13], ptr[14], ptr[15], ptr[16], ptr[17], ptr[18]);
                        }
                        if (nsets > 0)
                            cm->sendCommand_ValuesMessage(GETSELECTEDSETS0, nsets, ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8]);
                    }
                    return retval;
                }
                else
                {
                    sscanf(buf, "S%d", &retval);
                }
                ptr[i] = retval;
                /*
                  printf("S%d %d\n", retval, ptr[i]);
            */
                XtFree(cstr);
            }
            XmStringFree(cs);
        }
    }
    /*
       printf("Selected sets:");
       for (i = 0; i < cnt; i++) {
      printf(" %d", ptr[i]);
       }
       printf("\n");
   */
    if ((ismaster) && (ptr != NULL))
    {
        if (cnt > 29)
        {
            cm->sendCommand_ValuesMessage(GETSELECTEDSETS0, cnt, ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8]);
            cm->sendCommand_ValuesMessage(GETSELECTEDSETS1, ptr[9], ptr[10], ptr[11], ptr[12], ptr[13], ptr[14], ptr[15], ptr[16], ptr[17], ptr[18]);
            cm->sendCommand_ValuesMessage(GETSELECTEDSETS2, ptr[19], ptr[20], ptr[21], ptr[22], ptr[23], ptr[24], ptr[25], ptr[26], ptr[27], ptr[28]);
            cm->sendCommand_ValuesMessage(GETSELECTEDSETS3, ptr[29], ptr[30], ptr[31], ptr[32], ptr[33], ptr[34], ptr[35], ptr[36], ptr[37], ptr[38]);
        }
        if (cnt > 19)
        {
            cm->sendCommand_ValuesMessage(GETSELECTEDSETS0, cnt, ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8]);
            cm->sendCommand_ValuesMessage(GETSELECTEDSETS1, ptr[9], ptr[10], ptr[11], ptr[12], ptr[13], ptr[14], ptr[15], ptr[16], ptr[17], ptr[18]);
            cm->sendCommand_ValuesMessage(GETSELECTEDSETS2, ptr[19], ptr[20], ptr[21], ptr[22], ptr[23], ptr[24], ptr[25], ptr[26], ptr[27], ptr[28]);
        }
        if (cnt > 9)
        {
            cm->sendCommand_ValuesMessage(GETSELECTEDSETS0, cnt, ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8]);
            cm->sendCommand_ValuesMessage(GETSELECTEDSETS1, ptr[9], ptr[10], ptr[11], ptr[12], ptr[13], ptr[14], ptr[15], ptr[16], ptr[17], ptr[18]);
        }
        if (cnt > 0)
            cm->sendCommand_ValuesMessage(GETSELECTEDSETS0, cnt, ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8]);
    }
    return cnt;
}

void SetSelectorFilterCB(Widget w, XtPointer clientd, XtPointer)
{
    SetChoiceItem *s = (SetChoiceItem *)clientd;
    if (XmToggleButtonGetState(w))
    {
        int i;
        for (i = 0; i < 4; i++)
        {
            if (w == s->but[i])
            {
                break;
            }
        }
        /* update the list in s->list */
        s->fflag = i + 1;
        update_set_list(cg, *s);
    }
}

void DefineSetSelectorFilter(SetChoiceItem *s)
{
    XtAddCallback(s->but[0], XmNvalueChangedCallback, (XtCallbackProc)SetSelectorFilterCB, (XtPointer)s);
    XtAddCallback(s->but[1], XmNvalueChangedCallback, (XtCallbackProc)SetSelectorFilterCB, (XtPointer)s);
    XtAddCallback(s->but[2], XmNvalueChangedCallback, (XtCallbackProc)SetSelectorFilterCB, (XtPointer)s);
    XtAddCallback(s->but[3], XmNvalueChangedCallback, (XtCallbackProc)SetSelectorFilterCB, (XtPointer)s);
}

SetChoiceItem CreateSetSelector(Widget parent,
                                const char *label,
                                int type,
                                int ff,
                                int gtype,
                                int stype)
{
    Arg args[3];
    Widget rc, lab;
    SetChoiceItem sel;
    lab = XmCreateLabel(parent, (char *)label, NULL, 0);
    XtManageChild(lab);
    XtSetArg(args[0], XmNlistSizePolicy, XmRESIZE_IF_POSSIBLE);
    XtSetArg(args[1], XmNvisibleItemCount, 5);
    sel.list = XmCreateScrolledList(parent, (char *)"list", args, 2);
    if (stype == SELECTION_TYPE_MULTIPLE) /* multiple select */
    {
        XtVaSetValues(sel.list,
                      XmNselectionPolicy, XmMULTIPLE_SELECT,
                      NULL);
    } /* single select */
    else
    {
    }
    sel.type = type;
    sel.fflag = ff;
    sel.gno = gtype;
    XtManageChild(sel.list);
    save_set_list(sel);
    update_set_list(cg, sel);
    if (ff) /* need a filter gadget */
    {
        rc = XmCreateRowColumn(parent, (char *)"rc", NULL, 0);
        XtVaSetValues(rc,
                      XmNorientation, XmHORIZONTAL,
                      NULL);
        lab = XmCreateLabel(rc, (char *)"Display:", NULL, 0);
        XtManageChild(lab);
        sel.rb = XmCreateRadioBox(rc, (char *)"rb", NULL, 0);
        XtVaSetValues(sel.rb,
                      XmNorientation, XmHORIZONTAL,
                      XmNpacking, XmPACK_TIGHT,
                      NULL);
        sel.but[0] = XtVaCreateManagedWidget((char *)"Active", xmToggleButtonWidgetClass, sel.rb,
                                             XmNalignment, XmALIGNMENT_CENTER,
                                             XmNindicatorOn, False, XmNshadowThickness, 2,
                                             NULL);
        sel.but[1] = XtVaCreateManagedWidget((char *)"All", xmToggleButtonWidgetClass, sel.rb,
                                             XmNalignment, XmALIGNMENT_CENTER,
                                             XmNindicatorOn, False, XmNshadowThickness, 2,
                                             NULL);
        sel.but[2] = XtVaCreateManagedWidget((char *)"Inactive", xmToggleButtonWidgetClass, sel.rb,
                                             XmNalignment, XmALIGNMENT_CENTER,
                                             XmNindicatorOn, False, XmNshadowThickness, 2,
                                             NULL);
        sel.but[3] = XtVaCreateManagedWidget((char *)"Deact", xmToggleButtonWidgetClass, sel.rb,
                                             XmNalignment, XmALIGNMENT_CENTER,
                                             XmNindicatorOn, False, XmNshadowThickness, 2,
                                             NULL);
        XmToggleButtonSetState(sel.but[ff - 1], True, False);
        XtManageChild(sel.rb);
        XtManageChild(rc);
    }
    return sel;
}

void save_set_list(SetChoiceItem l)
{
    nplist++;
    if (plist == NULL)
    {
        plist = (SetChoiceItem *)malloc(nplist * sizeof(SetChoiceItem));
    }
    else
    {
        plist = (SetChoiceItem *)realloc(plist, nplist * sizeof(SetChoiceItem));
    }
    plist[nplist - 1] = l;
}

void update_set_list(int gno, SetChoiceItem l)
{
    int i, cnt = 0;
    char buf[256];
    XmString *xms;
    XmListDeleteAllItems(l.list);
    for (i = 0; i < g[gno].maxplot; i++)
    {
        switch (l.fflag)
        {
        case FILTER_SELECT_NONE: /* Active sets */
            if (isactive(gno, i))
            {
                cnt++;
            }
            break;
        case FILTER_SELECT_ALL: /* All sets */
            cnt++;
            break;
        case FILTER_SELECT_ACTIVE: /* Active sets */
            if (isactive(gno, i))
            {
                cnt++;
            }
            break;
        case FILTER_SELECT_INACT: /* Inactive sets */
            if (!isactive(gno, i))
            {
                cnt++;
            }
            break;
        case FILTER_SELECT_DEACT: /* Deactivated sets */
            if (!isactive(gno, i) && g[gno].p[i].deact)
            {
                cnt++;
            }
            break;
        }
    }
    switch (l.type) /* TODO */
    {
    case SET_SELECT_ACTIVE:
        xms = (XmString *)malloc(sizeof(XmString) * cnt);
        cnt = 0;
        break;
    case SET_SELECT_ALL:
        xms = (XmString *)malloc(sizeof(XmString) * (cnt + 1));
        xms[0] = XmStringCreateLtoR((char *)"All sets", charset);
        cnt = 1;
        break;
    case SET_SELECT_NEXT:
        xms = (XmString *)malloc(sizeof(XmString) * (cnt + 1));
        xms[0] = XmStringCreateLtoR((char *)"New set", charset);
        cnt = 1;
        break;
    default:
        xms = (XmString *)malloc(sizeof(XmString) * cnt);
        cnt = 0;
        break;
    }

    for (i = 0; i < g[gno].maxplot; i++)
    {
        switch (l.fflag)
        {
        case FILTER_SELECT_NONE: /* Active sets */
            if (isactive(gno, i))
            {
                sprintf(buf, "S%d (%s)", i, getcomment(gno, i));
                xms[cnt] = XmStringCreateLtoR(buf, charset);
                cnt++;
            }
            break;
        case FILTER_SELECT_ALL: /* All sets */
            sprintf(buf, "S%d (%s)", i, getcomment(gno, i));
            xms[cnt] = XmStringCreateLtoR(buf, charset);
            cnt++;
            break;
        case FILTER_SELECT_ACTIVE: /* Active sets */
            if (isactive(gno, i))
            {
                sprintf(buf, "S%d (%s)", i, getcomment(gno, i));
                xms[cnt] = XmStringCreateLtoR(buf, charset);
                cnt++;
            }
            break;
        case FILTER_SELECT_INACT: /* Inactive sets */
            if (!isactive(gno, i))
            {
                sprintf(buf, "S%d (%s)", i, getcomment(gno, i));
                xms[cnt] = XmStringCreateLtoR(buf, charset);
                cnt++;
            }
            break;
        case FILTER_SELECT_DEACT: /* Deactivated sets */
            if (!isactive(gno, i) && g[gno].p[i].deact)
            {
                sprintf(buf, "S%d (%s)", i, getcomment(gno, i));
                xms[cnt] = XmStringCreateLtoR(buf, charset);
                cnt++;
            }
            break;
        }
    }
    XmListAddItemsUnselected(l.list, xms, cnt, 0);
    for (i = 0; i < cnt; i++)
    {
        XmStringFree(xms[i]);
    }
    free(xms);
}

void update_set_lists(int gno)
{
    int i;
    for (i = 0; i < nplist; i++)
    {
        update_set_list(gno, plist[i]);
    }
}

void AddSetToLists(int gno, int setno)
{
    if (nplist)
    {
        int i;
        XmString xms = 0;
        char buf[256];
        if (isactive(gno, setno))
        {
            sprintf(buf, "S%d (%s)", setno, getcomment(gno, setno));
            xms = XmStringCreateLtoR(buf, charset);
        }
        else
        {
            fprintf(stderr, "compwin.cpp: xms is used uninitialized\n");
        }
        for (i = 0; i < nplist; i++)
        {
            XmListAddItemUnselected(plist[i].list, xms, 0);
        }
    }
}

typedef struct _Eval_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget formula_item;
    Widget *load_item;
    Widget *loadgraph_item;
    Widget *region_item;
    Widget rinvert_item;
} Eval_ui;

static Eval_ui eui;

void create_eval_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog, rc;
    // Arg args[3];
    set_wait_cursor();
    if (eui.top == NULL)
    {
        char *label2[3];
        label2[0] = (char *)"Accept";
        label2[1] = (char *)"Pick";
        label2[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        eui.top = XmCreateDialogShell(app_shell, (char *)"Evaluate expression", NULL, 0);
        handle_close(eui.top);
        XtVaSetValues(eui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(eui.top, (char *)"dialog_rc", NULL, 0);

        eui.sel = CreateSetSelector(dialog, (char *)"Apply to set:",
                                    SET_SELECT_ALL,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);

        rc = XmCreateRowColumn(dialog, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
        eui.load_item = CreatePanelChoice(rc,
                                          (char *)"Result to:", 3,
                                          (char *)"Same set", (char *)"New set", NULL, 0);
        eui.loadgraph_item = CreateGraphChoice(rc, (char *)"In graph: ", maxgraph, 1);
        XtManageChild(rc);

        /*
          rc = XmCreateRowColumn(dialog, "rc", NULL, 0);
          XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
          eui.region_item = CreatePanelChoice(rc,
                         "Restrictions:",
                         9,
                         "None",
                         "Region 0",
                         "Region 1",
                         "Region 2",
                         "Region 3",
      "Region 4",
      "Inside graph",
      "Outside graph",
      0,
      0);
      eui.rinvert_item = XmCreateToggleButton(rc, "Invert region", NULL, 0);
      XtManageChild(compute_rinvert_item);
      XtManageChild(rc);
      */

        eui.formula_item = CreateTextItem2(dialog, 30, (char *)"Formula:");

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 3, but2, label2);
        XtAddCallback(but2[0], XmNactivateCallback, (XtCallbackProc)do_compute_proc, (XtPointer)&eui);
        XtAddCallback(but2[1], XmNactivateCallback, (XtCallbackProc)do_pick_compose, (XtPointer)PICK_EXPR);
        XtAddCallback(but2[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)eui.top);

        XtManageChild(dialog);
    }
    XtRaise(eui.top);
    unset_wait_cursor();
}

/*
 * evaluate a formula
 */
static void do_compute_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt = 0;
    int setno, loadto = 0, graphto = 0;
    char fstr[256];
    if (ismaster)
    {
        Eval_ui *ui = (Eval_ui *)client_data;
        cnt = GetSelectedSets(ui->sel, &selsets);
        loadto = (int)GetChoice(ui->load_item);
        graphto = (int)GetChoice(ui->loadgraph_item) - 1;
        strcpy(fstr, (char *)xv_getstr(ui->formula_item));
        cm->sendCommand_StringMessage(DO_COMPUTE_PROC, fstr);
        cm->sendCommandMessage(DO_COMPUTE_PROC, loadto, graphto);
    }
    else
    {
        fprintf(stderr, "compwin.cpp: do_compute_proc(): cnt is used uninitialized\n");
    }

    set_wait_cursor();
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        do_compute(setno, loadto, graphto, fstr);
    }
    free(selsets);
    unset_wait_cursor();
    drawgraph();
}

typedef struct _Load_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget start_item;
    Widget step_item;
    Widget *load_item;
} Load_ui;

static Load_ui lui;

/* load a set */

void create_load_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;
    Widget rc;

    set_wait_cursor();
    if (lui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        lui.top = XmCreateDialogShell(app_shell, (char *)"Load values", NULL, 0);
        handle_close(lui.top);
        XtVaSetValues(lui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(lui.top, (char *)"dialog_rc", NULL, 0);

        lui.sel = CreateSetSelector(dialog, (char *)"Apply to set:",
                                    SET_SELECT_ALL,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);
        rc = XtVaCreateWidget((char *)"rc", xmRowColumnWidgetClass, dialog,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 3,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        XtVaCreateManagedWidget((char *)"Load to: ", xmLabelWidgetClass, rc, NULL);
        lui.load_item = CreatePanelChoice(rc,
                                          " ",
                                          7,
                                          "Set X",
                                          "Set Y",
                                          "Scratch A",
                                          "Scratch B",
                                          "Scratch C",
                                          "Scratch D", 0,
                                          0);
        lui.start_item = CreateTextItem4(rc, 10, (char *)"Start:");
        lui.step_item = CreateTextItem4(rc, 10, (char *)"Step:");
        XtManageChild(rc);

        XtVaCreateManagedWidget((char *)"sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_load_proc, (XtPointer)&lui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)lui.top);

        XtManageChild(dialog);
    }
    XtRaise(lui.top);
    unset_wait_cursor();
}

/*
 * load a set
 */
static void do_load_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno, toval;
    char startstr[256], stepstr[256];
    Load_ui *ui = (Load_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    toval = (int)GetChoice(ui->load_item) + 1;
    strcpy(stepstr, (char *)xv_getstr(ui->step_item));
    strcpy(startstr, (char *)xv_getstr(ui->start_item));
    set_wait_cursor();
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        do_load(setno, toval, startstr, stepstr);
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

/* histograms */

typedef struct _Histo_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget binw_item;
    Widget hxmin_item;
    Widget hxmax_item;
    Widget *type_item;
    Widget *graph_item;
} Histo_ui;

static Histo_ui hui;

void create_histo_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;
    Widget rc;

    set_wait_cursor();
    if (hui.top == NULL)
    {
        char *label2[3];
        label2[0] = (char *)"Accept";
        label2[1] = (char *)"Pick";
        label2[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        hui.top = XmCreateDialogShell(app_shell, (char *)"Histograms", NULL, 0);
        handle_close(hui.top);
        XtVaSetValues(hui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(hui.top, (char *)"dialog_rc", NULL, 0);

        hui.sel = CreateSetSelector(dialog, (char *)"Apply to set:",
                                    SET_SELECT_ACTIVE,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);
        rc = XtVaCreateWidget("rc", xmRowColumnWidgetClass, dialog,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 4,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        XtVaCreateManagedWidget("Bin width: ", xmLabelWidgetClass, rc, NULL);
        hui.binw_item = XtVaCreateManagedWidget("binwidth", xmTextWidgetClass, rc, NULL);
        XtVaSetValues(hui.binw_item, XmNcolumns, 10, NULL);
        XtVaCreateManagedWidget("Start value: ", xmLabelWidgetClass, rc, NULL);
        hui.hxmin_item = XtVaCreateManagedWidget("xmin", xmTextWidgetClass, rc, NULL);
        XtVaSetValues(hui.hxmin_item, XmNcolumns, 10, NULL);
        XtVaCreateManagedWidget("Ending value: ", xmLabelWidgetClass, rc, NULL);
        hui.hxmax_item = XtVaCreateManagedWidget("xmax", xmTextWidgetClass, rc, NULL);
        XtVaSetValues(hui.hxmax_item, XmNcolumns, 10, NULL);
        XtManageChild(rc);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);
        hui.type_item = CreatePanelChoice(dialog, "Compute: ",
                                          3,
                                          "Histogram",
                                          "Cumulative histogram",
                                          0,
                                          0);
        hui.graph_item = CreateGraphChoice(dialog, (char *)"Load result to graph:", maxgraph, 1);
        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 3, but2, label2);
        XtAddCallback(but2[0], XmNactivateCallback, (XtCallbackProc)do_histo_proc, (XtPointer)&hui);
        XtAddCallback(but2[1], XmNactivateCallback, (XtCallbackProc)do_pick_compose, (XtPointer)PICK_HISTO);
        XtAddCallback(but2[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)hui.top);

        XtManageChild(dialog);
    }
    XtRaise(hui.top);
    unset_wait_cursor();
}

/*
 * histograms
 */
static void do_histo_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int fromset, toset, tograph, hist_type;
    double binw, xmin, xmax;
    Histo_ui *ui = (Histo_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    toset = SET_SELECT_NEXT;
    tograph = (int)GetChoice(ui->graph_item) - 1;
    binw = atof((char *)xv_getstr(ui->binw_item));
    xmin = atof((char *)xv_getstr(ui->hxmin_item));
    xmax = atof((char *)xv_getstr(ui->hxmax_item));
    hist_type = (int)GetChoice(ui->type_item);
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommand_FloatMessage(DO_HISTO_PROC, (double)toset, (double)tograph, binw, xmin, xmax, (double)hist_type, 0.0, 0.0, 0.0, 0.0);
    }
    for (i = 0; i < cnt; i++)
    {
        fromset = selsets[i];
        do_histo(fromset, toset, tograph, binw, xmin, xmax, hist_type);
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

/* DFTs */

typedef struct _Four_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *load_item;
    Widget *window_item;
    Widget *loadx_item;
    Widget *inv_item;
    Widget *type_item;
    Widget *graph_item;
} Four_ui;

static Four_ui fui;

void create_fourier_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;
    Widget rc;
    Widget buts[5];

    set_wait_cursor();
    if (fui.top == NULL)
    {
        char *l[5];
        l[0] = (char *)"DFT";
        l[1] = (char *)"FFT";
        l[2] = (char *)"Window only";
        l[3] = (char *)"Pick";
        l[4] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        fui.top = XmCreateDialogShell(app_shell, (char *)"Fourier transforms", NULL, 0);
        handle_close(fui.top);
        XtVaSetValues(fui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(fui.top, (char *)"dialog_rc", NULL, 0);

        fui.sel = CreateSetSelector(dialog, (char *)"Apply to set:",
                                    SET_SELECT_ACTIVE,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);

        rc = XtVaCreateWidget("rc", xmRowColumnWidgetClass, dialog,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 5,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        XtVaCreateManagedWidget("Data window: ", xmLabelWidgetClass, rc, NULL);
        fui.window_item = CreatePanelChoice(rc,
                                            " ",
                                            8,
                                            "None (Rectangular)",
                                            "Triangular",
                                            "Hanning",
                                            "Welch",
                                            "Hamming",
                                            "Blackman",
                                            "Parzen",
                                            NULL,
                                            NULL);

        XtVaCreateManagedWidget("Load result as: ", xmLabelWidgetClass, rc, NULL);

        fui.load_item = CreatePanelChoice(rc,
                                          " ",
                                          4,
                                          "Magnitude",
                                          "Phase",
                                          "Coefficients",
                                          0,
                                          0);

        XtVaCreateManagedWidget("Let result X = ", xmLabelWidgetClass, rc, NULL);
        fui.loadx_item = CreatePanelChoice(rc,
                                           " ",
                                           4,
                                           "Index",
                                           "Frequency",
                                           "Period",
                                           0,
                                           0);

        XtVaCreateManagedWidget("Perform: ", xmLabelWidgetClass, rc, NULL);
        fui.inv_item = CreatePanelChoice(rc,
                                         " ",
                                         3,
                                         "Transform",
                                         "Inverse transform",
                                         0,
                                         0);

        XtVaCreateManagedWidget("Data is: ", xmLabelWidgetClass, rc, NULL);
        fui.type_item = CreatePanelChoice(rc,
                                          " ",
                                          3,
                                          "Real",
                                          "Complex",
                                          0,
                                          0);
        XtManageChild(rc);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);
        CreateCommandButtons(dialog, 5, buts, l);
        XtAddCallback(buts[0], XmNactivateCallback, (XtCallbackProc)do_fourier_proc, (XtPointer)&fui);
        XtAddCallback(buts[1], XmNactivateCallback, (XtCallbackProc)do_fft_proc, (XtPointer)&fui);
        XtAddCallback(buts[2], XmNactivateCallback, (XtCallbackProc)do_window_proc, (XtPointer)&fui);
        XtAddCallback(buts[3], XmNactivateCallback, (XtCallbackProc)do_pick_compose, (XtPointer)PICK_FOURIER);
        XtAddCallback(buts[4], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)fui.top);

        XtManageChild(dialog);
    }
    XtRaise(fui.top);
    unset_wait_cursor();
}

/*
 * DFT
 */
static void do_fourier_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno, load, loadx, invflag, type, wind;
    Four_ui *ui = (Four_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    wind = GetChoice(ui->window_item);
    load = GetChoice(ui->load_item);
    loadx = GetChoice(ui->loadx_item);
    invflag = GetChoice(ui->inv_item);
    type = GetChoice(ui->type_item);
    if (ismaster)
    {
        cm->sendCommand_ValuesMessage(DO_FOURIER_PROC, load, loadx, invflag, type, wind, 0, 0, 0, 0, 0);
    }
    set_wait_cursor();
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        do_fourier(0, setno, load, loadx, invflag, type, wind);
    }
    free(selsets);
    unset_wait_cursor();
    drawgraph();
}

/*
 * DFT by FFT
 */
static void do_fft_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno, load, loadx, invflag, type, wind;
    Four_ui *ui = (Four_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    wind = GetChoice(ui->window_item);
    load = GetChoice(ui->load_item);
    loadx = GetChoice(ui->loadx_item);
    invflag = GetChoice(ui->inv_item);
    type = GetChoice(ui->type_item);
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommand_ValuesMessage(DO_FOURIER_PROC, load, loadx, invflag, type, wind, 1, 0, 0, 0, 0);
    }
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        do_fourier(1, setno, load, loadx, invflag, type, wind);
    }
    free(selsets);
    unset_wait_cursor();
    drawgraph();
}

/*
 * Apply data window only
 */
static void do_window_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno, type, wind;
    Four_ui *ui = (Four_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    wind = GetChoice(ui->window_item);
    type = GetChoice(ui->type_item);
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommandMessage(DO_WINDOW_PROC, type, wind);
    }
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        do_window(setno, type, wind);
    }
    free(selsets);
    unset_wait_cursor();
    drawgraph();
}

/* running averages */

typedef struct _Run_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget len_item;
    Widget *type_item;
    Widget *region_item;
    Widget rinvert_item;
} Run_ui;

static Run_ui rui;

void create_run_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;
    Widget rc;

    set_wait_cursor();
    if (rui.top == NULL)
    {
        char *label2[3];
        label2[0] = (char *)"Accept";
        label2[1] = (char *)"Pick";
        label2[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        rui.top = XmCreateDialogShell(app_shell, (char *)"Running averages", NULL, 0);
        handle_close(rui.top);
        XtVaSetValues(rui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(rui.top, (char *)"dialog_rc", NULL, 0);

        rui.sel = CreateSetSelector(dialog, (char *)"Apply to set:",
                                    SET_SELECT_ACTIVE,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);

        rc = XtVaCreateWidget((char *)"rc", xmRowColumnWidgetClass, dialog,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 5,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        XtVaCreateManagedWidget((char *)"Running:", xmLabelWidgetClass, rc, NULL);
        rui.type_item = CreatePanelChoice(rc,
                                          " ",
                                          6,
                                          "Average",
                                          "Median",
                                          "Minimum",
                                          "Maximum",
                                          "Std. dev.", 0,
                                          0);
        rui.len_item = CreateTextItem4(rc, 10, (char *)"Length of average:");

        XtVaCreateManagedWidget((char *)"Restrictions:", xmLabelWidgetClass, rc, NULL);
        rui.region_item = CreatePanelChoice(rc,
                                            " ",
                                            9,
                                            "None",
                                            "Region 0",
                                            "Region 1",
                                            "Region 2",
                                            "Region 3",
                                            "Region 4",
                                            "Inside graph",
                                            "Outside graph",
                                            0,
                                            0);

        XtVaCreateManagedWidget((char *)"Invert region:", xmLabelWidgetClass, rc, NULL);
        rui.rinvert_item = XmCreateToggleButton(rc, (char *)" ", NULL, 0);
        XtManageChild(rui.rinvert_item);

        XtManageChild(rc);

        XtVaCreateManagedWidget((char *)"sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 3, but2, label2);
        XtAddCallback(but2[0], XmNactivateCallback, (XtCallbackProc)do_runavg_proc, (XtPointer)&rui);
        XtAddCallback(but2[1], XmNactivateCallback, (XtCallbackProc)do_pick_compose, (XtPointer)PICK_RUNAVG);
        XtAddCallback(but2[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)rui.top);

        XtManageChild(dialog);
    }
    XtRaise(rui.top);
    unset_wait_cursor();
}

/*
 * running averages, medians, min, max, std. deviation
 */
static void do_runavg_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int runlen, runtype, setno, rno, invr;
    Run_ui *ui = (Run_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    runlen = atoi((char *)xv_getstr(ui->len_item));
    runtype = GetChoice(ui->type_item);
    rno = GetChoice(ui->region_item) - 1;
    invr = XmToggleButtonGetState(ui->rinvert_item);
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommand_ValuesMessage(DO_RUNAVG_PROC, runlen, runtype, rno, invr, 0, 0, 0, 0, 0, 0);
    }
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        do_runavg(setno, runlen, runtype, rno, invr);
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

/* TODO finish this */
void do_eval_regress()
{
}

typedef struct _Reg_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *degree_item;
    Widget zero_item;
    Widget *resid_item;
    Widget *region_item;
    Widget rinvert_item;
    Widget start_item;
    Widget stop_item;
    Widget step_item;
    Widget method_item;
} Reg_ui;

static Reg_ui regui;

void create_reg_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;
    Widget rc;
    Widget buts[4];

    set_wait_cursor();
    if (regui.top == NULL)
    {
        char *label1[4];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Pick";
        /*
         label1[2] = "Eval...";
      */
        label1[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        regui.top = XmCreateDialogShell(app_shell, (char *)"Regression", NULL, 0);
        handle_close(regui.top);
        XtVaSetValues(regui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(regui.top, (char *)"dialog_rc", NULL, 0);

        regui.sel = CreateSetSelector(dialog, (char *)"Apply to set:",
                                      SET_SELECT_ALL,
                                      FILTER_SELECT_NONE,
                                      GRAPH_SELECT_CURRENT,
                                      SELECTION_TYPE_MULTIPLE);

        rc = XtVaCreateWidget("rc", xmRowColumnWidgetClass, dialog,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 5,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        XtVaCreateManagedWidget("Type of fit:", xmLabelWidgetClass, rc, NULL);
        regui.degree_item = CreatePanelChoice(rc,
                                              " ",
                                              16,
                                              "Linear",
                                              "Quadratic",
                                              "Cubic",
                                              "4th degree",
                                              "5th degree",
                                              "6th degree",
                                              "7th degree",
                                              "8th degree",
                                              "9th degree",
                                              "10th degree",
                                              "1-10",
                                              "Power y=A*x^B",
                                              "Exponential y=A*exp(B*x)",
                                              "Logarithmic y=A+B*ln(x)",
                                              "Inverse y=1/(A+Bx)",
                                              0,
                                              0);

        /*
          regui.zero_item = XmCreateToggleButton(rc, "Force fit through X = 0", NULL, 0);
          XtManageChild(toggle_zero_item);
                           "Evaluate fit",
      */

        XtVaCreateManagedWidget("Load:", xmLabelWidgetClass, rc, NULL);
        regui.resid_item = CreatePanelChoice(rc,
                                             " ",
                                             3,
                                             "Fitted values",
                                             "Residuals",
                                             0,
                                             0);

        /*
         regui.start_item = CreateTextItem4(rc, 10, "Start:");
         regui.stop_item = CreateTextItem4(rc, 10, "Stop:");
         regui.step_item = CreateTextItem4(rc, 10, "Number of points:");
      */

        XtVaCreateManagedWidget("Restrictions:", xmLabelWidgetClass, rc, NULL);
        regui.region_item = CreatePanelChoice(rc,
                                              " ",
                                              9,
                                              "None",
                                              "Region 0",
                                              "Region 1",
                                              "Region 2",
                                              "Region 3",
                                              "Region 4",
                                              "Inside graph",
                                              "Outside graph",
                                              0,
                                              0);
        XtVaCreateManagedWidget("Invert region:", xmLabelWidgetClass, rc, NULL);
        regui.rinvert_item = XmCreateToggleButton(rc, (char *)" ", NULL, 0);
        XtManageChild(regui.rinvert_item);
        XtVaCreateManagedWidget("Solve by AS274:", xmLabelWidgetClass, rc, NULL);
        regui.method_item = XmCreateToggleButton(rc, (char *)" ", NULL, 0);
        XtManageChild(regui.method_item);
        XtManageChild(rc);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 3, buts, label1);
        XtAddCallback(buts[0], XmNactivateCallback, (XtCallbackProc)do_regress_proc, (XtPointer)&regui);
        XtAddCallback(buts[1], XmNactivateCallback, (XtCallbackProc)do_pick_compose, (XtPointer)PICK_REG);
        /*
         XtAddCallback(buts[2], XmNactivateCallback, (XtCallbackProc) do_eval_regress, (XtPointer) &regui);
      */
        XtAddCallback(buts[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)regui.top);

        XtManageChild(dialog);
    }
    XtRaise(regui.top);
    unset_wait_cursor();
}

/*
 * regression
 */
static void do_regress_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int cnt;
    Reg_ui *ui = (Reg_ui *)client_data;
    int setno, ideg, iresid, i, j;
    int rno = GetChoice(ui->region_item) - 1;
    int invr = XmToggleButtonGetState(ui->rinvert_item);

    lsmethod = XmToggleButtonGetState(ui->method_item);

    cnt = GetSelectedSets(ui->sel, &selsets);
    ideg = (int)GetChoice(ui->degree_item) + 1;
    iresid = (int)GetChoice(ui->resid_item);
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommand_ValuesMessage(DO_REGRESS_PROC, ideg, iresid, rno, invr, 0, 0, 0, 0, 0, 0);
    }
    for (j = 0; j < cnt; j++)
    {
        setno = selsets[j];
        if (ideg == 11)
        {
            for (i = 1; i <= ideg - 1; i++)
            {
                do_regress(setno, i, iresid, rno, invr);
            }
        }
        else
        {
            do_regress(setno, ideg, iresid, rno, invr);
        }
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

/* finite differencing */

typedef struct _Diff_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *type_item;
    Widget *region_item;
    Widget rinvert_item;
} Diff_ui;

static Diff_ui dui;

void create_diff_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (dui.top == NULL)
    {
        char *label2[3];
        label2[0] = (char *)"Accept";
        label2[1] = (char *)"Pick";
        label2[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        dui.top = XmCreateDialogShell(app_shell, (char *)"Differences", NULL, 0);
        handle_close(dui.top);
        XtVaSetValues(dui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(dui.top, (char *)"dialog_rc", NULL, 0);

        dui.sel = CreateSetSelector(dialog, (char *)"Apply to set:",
                                    SET_SELECT_ACTIVE,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);
        dui.type_item = CreatePanelChoice(dialog,
                                          "Method:",
                                          4,
                                          "Forward difference",
                                          "Backward difference",
                                          "Centered difference",
                                          0,
                                          0);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 3, but2, label2);
        XtAddCallback(but2[0], XmNactivateCallback, (XtCallbackProc)do_differ_proc, (XtPointer)&dui);
        XtAddCallback(but2[1], XmNactivateCallback, (XtCallbackProc)do_pick_compose, (XtPointer)PICK_DIFF);
        XtAddCallback(but2[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)dui.top);

        XtManageChild(dialog);
    }
    XtRaise(dui.top);
    unset_wait_cursor();
}

/*
 * finite differences
 */
static void do_differ_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno, itype;
    Diff_ui *ui = (Diff_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    itype = (int)GetChoice(ui->type_item);
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommandMessage(DO_DIFFER_PROC, itype, 0);
    }
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        do_differ(setno, itype);
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

/* numerical integration */

typedef struct _Int_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *type_item;
    Widget sum_item;
    Widget *region_item;
    Widget rinvert_item;
} Int_ui;

Int_ui iui;

void create_int_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (iui.top == NULL)
    {
        char *label2[3];
        label2[0] = (char *)"Accept";
        label2[1] = (char *)"Pick";
        label2[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        iui.top = XmCreateDialogShell(app_shell, (char *)"Integration", NULL, 0);
        handle_close(iui.top);
        XtVaSetValues(iui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(iui.top, (char *)"dialog_rc", NULL, 0);
        iui.sel = CreateSetSelector(dialog, (char *)"Apply to set:",
                                    SET_SELECT_ACTIVE,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);

        iui.type_item = CreatePanelChoice(dialog,
                                          "Load:",
                                          3,
                                          "Cumulative sum",
                                          "Sum only",
                                          0,
                                          0);
        iui.sum_item = CreateTextItem2(dialog, 10, (char *)"Sum:");

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 3, but2, label2);
        XtAddCallback(but2[0], XmNactivateCallback, (XtCallbackProc)do_int_proc, (XtPointer)&iui);
        XtAddCallback(but2[1], XmNactivateCallback, (XtCallbackProc)do_pick_compose, (XtPointer)PICK_INT);
        XtAddCallback(but2[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)iui.top);

        XtManageChild(dialog);
    }
    XtRaise(iui.top);
    unset_wait_cursor();
}

/*
 * numerical integration
 */
static void do_int_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno, itype;
    double sum, do_int(int setno, int itype);
    Int_ui *ui = (Int_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    itype = GetChoice(ui->type_item);
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommandMessage(DO_INT_PROC, itype, 0);
    }
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        sum = do_int(setno, itype);
        sprintf(buf, "%lf", sum);
        xv_setstr(ui->sum_item, buf);
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

/* seasonal differencing */

typedef struct _Seas_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *type_item;
    Widget period_item;
    Widget *region_item;
    Widget rinvert_item;
} Seas_ui;

static Seas_ui sui;

void create_seasonal_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (sui.top == NULL)
    {
        char *label2[3];
        label2[0] = (char *)"Accept";
        label2[1] = (char *)"Pick";
        label2[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        sui.top = XmCreateDialogShell(app_shell, (char *)"Seasonal differences", NULL, 0);
        handle_close(sui.top);
        XtVaSetValues(sui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(sui.top, (char *)"dialog_rc", NULL, 0);

        sui.sel = CreateSetSelector(dialog, (char *)"Apply to set:",
                                    SET_SELECT_ACTIVE,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);
        sui.period_item = CreateTextItem2(dialog, 10, (char *)"Period:");

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 3, but2, label2);
        XtAddCallback(but2[0], XmNactivateCallback, (XtCallbackProc)do_seasonal_proc, (XtPointer)&sui);
        XtAddCallback(but2[1], XmNactivateCallback, (XtCallbackProc)do_pick_compose, (XtPointer)PICK_SEASONAL);
        XtAddCallback(but2[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)sui.top);

        XtManageChild(dialog);
    }
    XtRaise(sui.top);
    unset_wait_cursor();
}

/*
 * seasonal differences
 */
static void do_seasonal_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno, period;
    Seas_ui *ui = (Seas_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    period = atoi(xv_getstr(ui->period_item));
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommandMessage(DO_SEASONAL_PROC, period, 0);
    }
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        do_seasonal_diff(setno, period);
    }
    free(selsets);
    unset_wait_cursor();
    drawgraph();
}

/* interpolation */

typedef struct _Interp_ui
{
    Widget top;
    SetChoiceItem sel1;
    SetChoiceItem sel2;
    Widget *type_item;
    Widget *region_item;
    Widget rinvert_item;
} Interp_ui;

static Interp_ui interpui;

void create_interp_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (interpui.top == NULL)
    {
        char *label2[3];
        label2[0] = (char *)"Accept";
        label2[1] = (char *)"Pick";
        label2[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        interpui.top = XmCreateDialogShell(app_shell, (char *)"Interpolation", NULL, 0);
        handle_close(interpui.top);
        XtVaSetValues(interpui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(interpui.top, (char *)"dialog_rc", NULL, 0);

        interpui.sel1 = CreateSetSelector(dialog, (char *)"Interpolate from set:",
                                          SET_SELECT_ACTIVE,
                                          FILTER_SELECT_NONE,
                                          GRAPH_SELECT_CURRENT,
                                          SELECTION_TYPE_SINGLE);
        interpui.sel2 = CreateSetSelector(dialog, (char *)"To set:",
                                          SET_SELECT_ACTIVE,
                                          FILTER_SELECT_NONE,
                                          GRAPH_SELECT_CURRENT,
                                          SELECTION_TYPE_SINGLE);

        XtVaCreateManagedWidget((char *)"sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 3, but2, label2);
        XtAddCallback(but2[0], XmNactivateCallback, (XtCallbackProc)do_interp_proc, (XtPointer)&interpui);
        XtAddCallback(but2[1], XmNactivateCallback, (XtCallbackProc)do_pick_compose, (XtPointer)PICK_INTERP);
        XtAddCallback(but2[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)interpui.top);

        XtManageChild(dialog);
    }
    XtRaise(interpui.top);
    unset_wait_cursor();
}

/*
 * interpolation
 */
static void do_interp_proc(Widget, XtPointer, XtPointer)
{
    /* TODO
       cnt = GetSelectedSets(ui->sel, &selsets);
       set_wait_cursor();
       if(ismaster)
       {
           cm->sendCommandMessage(DO_INTERP_PROC,0,0);
       }
       for (i = 0; i < cnt; i++) {
      setno = selsets[i];
       }
       unset_wait_cursor();
   free(selsets);
   drawgraph();
   */
}

/* cross correlation */

typedef struct _Cross_ui
{
    Widget top;
    SetChoiceItem sel1;
    SetChoiceItem sel2;
    Widget *type_item;
    Widget lag_item;
    Widget *region_item;
    Widget rinvert_item;
} Cross_ui;

static Cross_ui crossui;

void create_xcor_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (crossui.top == NULL)
    {
        char *label2[3];
        label2[0] = (char *)"Accept";
        label2[1] = (char *)"Pick";
        label2[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        crossui.top = XmCreateDialogShell(app_shell, (char *)"X-correlation", NULL, 0);
        handle_close(crossui.top);
        XtVaSetValues(crossui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(crossui.top, (char *)"dialog_rc", NULL, 0);

        crossui.sel1 = CreateSetSelector(dialog, (char *)"Select set:",
                                         SET_SELECT_ACTIVE,
                                         FILTER_SELECT_NONE,
                                         GRAPH_SELECT_CURRENT,
                                         SELECTION_TYPE_SINGLE);
        crossui.sel2 = CreateSetSelector(dialog, (char *)"Select set:",
                                         SET_SELECT_ACTIVE,
                                         FILTER_SELECT_NONE,
                                         GRAPH_SELECT_CURRENT,
                                         SELECTION_TYPE_SINGLE);

        crossui.type_item = CreatePanelChoice(dialog,
                                              "Load:",
                                              3,
                                              "Biased estimate",
                                              "Unbiased estimate",
                                              0,
                                              0);
        crossui.lag_item = CreateTextItem2(dialog, 10, (char *)"Lag:");

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 3, but2, label2);
        XtAddCallback(but2[0], XmNactivateCallback, (XtCallbackProc)do_xcor_proc, (XtPointer)&crossui);
        XtAddCallback(but2[1], XmNactivateCallback, (XtCallbackProc)do_pick_compose, (XtPointer)PICK_XCOR);
        XtAddCallback(but2[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)crossui.top);

        XtManageChild(dialog);
    }
    XtRaise(crossui.top);
    unset_wait_cursor();
}

/*
 * cross correlation
 */
static void do_xcor_proc(Widget, XtPointer client_data, XtPointer)
{
    int set1, set2, itype, lag;
    Cross_ui *ui = (Cross_ui *)client_data;
    set1 = GetSelectedSet(ui->sel1);
    set2 = GetSelectedSet(ui->sel2);
    itype = (int)GetChoice(ui->type_item);
    lag = atoi((char *)xv_getstr(ui->lag_item));
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommand_ValuesMessage(DO_XCOR_PROC, set1, set2, itype, lag, 0, 0, 0, 0, 0, 0);
    }
    do_xcor(set1, set2, itype, lag);
    unset_wait_cursor();
}

/* splines */

typedef struct _Spline_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *type_item;
    Widget start_item;
    Widget stop_item;
    Widget step_item;
    Widget *region_item;
    Widget rinvert_item;
} Spline_ui;

static Spline_ui splineui;

void create_spline_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    static Widget dialog;
    Widget rc;

    set_wait_cursor();
    if (splineui.top == NULL)
    {
        char *label2[3];
        label2[0] = (char *)"Accept";
        label2[1] = (char *)"Pick";
        label2[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        splineui.top = XmCreateDialogShell(app_shell, (char *)"Splines", NULL, 0);
        handle_close(splineui.top);
        XtVaSetValues(splineui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(splineui.top, (char *)"dialog_rc", NULL, 0);

        splineui.sel = CreateSetSelector(dialog, (char *)"Apply to set:",
                                         SET_SELECT_ALL,
                                         FILTER_SELECT_NONE,
                                         GRAPH_SELECT_CURRENT,
                                         SELECTION_TYPE_MULTIPLE);

        rc = XtVaCreateWidget("rc", xmRowColumnWidgetClass, dialog,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 3,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        splineui.start_item = CreateTextItem4(rc, 10, (char *)"Start:");
        splineui.stop_item = CreateTextItem4(rc, 10, (char *)"Stop:");
        splineui.step_item = CreateTextItem4(rc, 10, (char *)"Number of points:");
        XtManageChild(rc);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 3, but2, label2);
        XtAddCallback(but2[0], XmNactivateCallback, (XtCallbackProc)do_spline_proc, (XtPointer)&splineui);
        XtAddCallback(but2[1], XmNactivateCallback, (XtCallbackProc)do_pick_compose, (XtPointer)PICK_SPLINE);
        XtAddCallback(but2[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)splineui.top);

        XtManageChild(dialog);
    }
    XtRaise(splineui.top);
    unset_wait_cursor();
}

/*
 * splines
 */
static void do_spline_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno, n;
    double start, stop;
    Spline_ui *ui = (Spline_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    start = atof((char *)xv_getstr(ui->start_item));
    stop = atof((char *)xv_getstr(ui->stop_item));
    n = atoi((char *)xv_getstr(ui->step_item));

    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommand_FloatMessage(DO_SPLINE_PROC, start, stop, (double)n, 0, 0, 0, 0, 0, 0, 0);
    }
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        do_spline(setno, start, stop, n);
    }
    unset_wait_cursor();

    free(selsets);
    drawgraph();
}

/* sample a set */

typedef struct _Samp_ui
{
    Widget top;
    SetChoiceItem sel;
    Widget *type_item;
    Widget start_item;
    Widget step_item;
    Widget expr_item;
    Widget *region_item;
    Widget rinvert_item;
} Samp_ui;

static Samp_ui sampui;

void create_samp_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    static Widget dialog;
    Widget rc;

    set_wait_cursor();
    if (sampui.top == NULL)
    {
        char *label2[3];
        label2[0] = (char *)"Accept";
        label2[1] = (char *)"Pick";
        label2[2] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        sampui.top = XmCreateDialogShell(app_shell, (char *)"Sample points", NULL, 0);
        handle_close(sampui.top);
        XtVaSetValues(sampui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(sampui.top, (char *)"dialog_rc", NULL, 0);

        sampui.sel = CreateSetSelector(dialog, (char *)"Apply to set:",
                                       SET_SELECT_ALL,
                                       FILTER_SELECT_NONE,
                                       GRAPH_SELECT_CURRENT,
                                       SELECTION_TYPE_MULTIPLE);

        rc = XtVaCreateWidget("rc", xmRowColumnWidgetClass, dialog,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 5,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        XtVaCreateManagedWidget("Sample type:", xmLabelWidgetClass, rc, NULL);
        sampui.type_item = CreatePanelChoice(rc,
                                             " ",
                                             3,
                                             "Start/step",
                                             "Expression",
                                             0,
                                             0);
        sampui.start_item = CreateTextItem4(rc, 10, (char *)"Start:");
        sampui.step_item = CreateTextItem4(rc, 10, (char *)"Step:");
        sampui.expr_item = CreateTextItem4(rc, 10, (char *)"Logical expression:");
        XtManageChild(rc);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 3, but2, label2);
        XtAddCallback(but2[0], XmNactivateCallback, (XtCallbackProc)do_sample_proc, (XtPointer)&sampui);
        XtAddCallback(but2[1], XmNactivateCallback, (XtCallbackProc)do_pick_compose, (XtPointer)PICK_SAMPLE);
        XtAddCallback(but2[2], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)sampui.top);

        XtManageChild(dialog);
    }
    XtRaise(sampui.top);
    unset_wait_cursor();
}

/*
 * sample a set, by start/step or logical expression
 */
static void do_sample_proc(Widget, XtPointer client_data, XtPointer)
{
    int *selsets;
    int i, cnt;
    int setno, typeno;
    char exprstr[256];
    int startno, stepno;
    Samp_ui *ui = (Samp_ui *)client_data;
    cnt = GetSelectedSets(ui->sel, &selsets);
    typeno = (int)GetChoice(ui->type_item);
    startno = atoi((char *)xv_getstr(ui->start_item));
    stepno = atoi((char *)xv_getstr(ui->step_item));
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommand_ValuesMessage(DO_SAMPLE_PROC, typeno, startno, stepno, 0, 0, 0, 0, 0, 0, 0);
        cm->sendCommand_StringMessage(DO_SAMPLE_PROC, (char *)xv_getstr(ui->expr_item));
    }
    for (i = 0; i < cnt; i++)
    {
        setno = selsets[i];
        /* exprstr gets clobbered */
        strcpy(exprstr, (char *)xv_getstr(ui->expr_item));
        do_sample(setno, typeno, exprstr, startno, stepno);
    }
    unset_wait_cursor();
    free(selsets);
    drawgraph();
}

/* apply a digital filter in set 2 to set 1 */

typedef struct _Digf_ui
{
    Widget top;
    SetChoiceItem sel1;
    SetChoiceItem sel2;
    Widget *type_item;
    Widget *region_item;
    Widget rinvert_item;
} Digf_ui;

static Digf_ui digfui;

void create_digf_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (digfui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        digfui.top = XmCreateDialogShell(app_shell, (char *)"Digital filter", NULL, 0);
        handle_close(digfui.top);
        XtVaSetValues(digfui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(digfui.top, (char *)"dialog_rc", NULL, 0);

        digfui.sel1 = CreateSetSelector(dialog, (char *)"Filter set:",
                                        SET_SELECT_ACTIVE,
                                        FILTER_SELECT_NONE,
                                        GRAPH_SELECT_CURRENT,
                                        SELECTION_TYPE_SINGLE);
        digfui.sel2 = CreateSetSelector(dialog, (char *)"With weights from set:",
                                        SET_SELECT_ACTIVE,
                                        FILTER_SELECT_NONE,
                                        GRAPH_SELECT_CURRENT,
                                        SELECTION_TYPE_SINGLE);

        XtVaCreateManagedWidget((char *)"sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_digfilter_proc, (XtPointer)&digfui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)digfui.top);

        XtManageChild(dialog);
    }
    XtRaise(digfui.top);
    unset_wait_cursor();
}

/*
 * apply a digital filter
 */
static void do_digfilter_proc(Widget, XtPointer client_data, XtPointer)
{
    int set1, set2;
    Digf_ui *ui = (Digf_ui *)client_data;
    set1 = GetSelectedSet(ui->sel1);
    set2 = GetSelectedSet(ui->sel2);
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommandMessage(DO_DIGFILTER_PROC, set1, set2);
    }
    do_digfilter(set1, set2);
    unset_wait_cursor();
}

/* linear convolution */

typedef struct _Lconv_ui
{
    Widget top;
    SetChoiceItem sel1;
    SetChoiceItem sel2;
    Widget *type_item;
    Widget lag_item;
    Widget *region_item;
    Widget rinvert_item;
} Lconv_ui;

static Lconv_ui lconvui;

void create_lconv_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;

    set_wait_cursor();
    if (lconvui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        lconvui.top = XmCreateDialogShell(app_shell, (char *)"Linear convolution", NULL, 0);
        handle_close(lconvui.top);
        XtVaSetValues(lconvui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(lconvui.top, (char *)"dialog_rc", NULL, 0);

        lconvui.sel1 = CreateSetSelector(dialog, (char *)"Convolve set:",
                                         SET_SELECT_ACTIVE,
                                         FILTER_SELECT_NONE,
                                         GRAPH_SELECT_CURRENT,
                                         SELECTION_TYPE_SINGLE);
        lconvui.sel2 = CreateSetSelector(dialog, (char *)"With set:",
                                         SET_SELECT_ACTIVE,
                                         FILTER_SELECT_NONE,
                                         GRAPH_SELECT_CURRENT,
                                         SELECTION_TYPE_SINGLE);

        XtVaCreateManagedWidget((char *)"sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_linearc_proc, (XtPointer)&lconvui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)lconvui.top);

        XtManageChild(dialog);
    }
    XtRaise(lconvui.top);
    unset_wait_cursor();
}

/*
 * linear convolution
 */
static void do_linearc_proc(Widget, XtPointer client_data, XtPointer)
{
    int set1, set2;
    Lconv_ui *ui = (Lconv_ui *)client_data;
    set1 = GetSelectedSet(ui->sel1);
    set2 = GetSelectedSet(ui->sel2);
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommandMessage(DO_LINEARC_PROC, set1, set2);
    }
    do_linearc(set1, set2);
    unset_wait_cursor();
}

/* evaluate a formula - load the next set */

typedef struct _Leval_ui
{
    Widget top;
    SetChoiceItem sel1;
    SetChoiceItem sel2;
    Widget *load_item;
    Widget x_item;
    Widget y_item;
    Widget start_item;
    Widget stop_item;
    Widget npts_item;
    Widget *region_item;
    Widget rinvert_item;
} Leval_ui;

static Leval_ui levalui;

void create_leval_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;
    Widget rc;

    set_wait_cursor();
    if (levalui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        levalui.top = XmCreateDialogShell(app_shell, (char *)"Load & evaluate", NULL, 0);
        handle_close(levalui.top);
        XtVaSetValues(levalui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(levalui.top, (char *)"dialog_rc", NULL, 0);

        rc = XtVaCreateWidget((char *)"rc", xmRowColumnWidgetClass, dialog,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 6,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        levalui.x_item = CreateTextItem4(rc, 10, (char *)"X = ");
        levalui.y_item = CreateTextItem4(rc, 10, (char *)"Y = ");

        XtVaCreateManagedWidget((char *)"Load:", xmLabelWidgetClass, rc, NULL);
        levalui.load_item = CreatePanelChoice(rc,
                                              " ",
                                              7,
                                              "Set X",
                                              "Set Y",
                                              "Scratch A",
                                              "Scratch B",
                                              "Scratch C",
                                              "Scratch D", 0,
                                              0);
        levalui.start_item = CreateTextItem4(rc, 10, (char *)"Start load at:");
        levalui.stop_item = CreateTextItem4(rc, 10, (char *)"Stop load at:");
        levalui.npts_item = CreateTextItem4(rc, 10, (char *)"# of points:");
        XtManageChild(rc);

        XtVaCreateManagedWidget((char *)"sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_compute_proc2, (XtPointer)&levalui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)levalui.top);

        XtManageChild(dialog);
    }
    XtRaise(levalui.top);
    unset_wait_cursor();
}

/*
 * evaluate a formula loading the next set
 */
static void do_compute_proc2(Widget, XtPointer client_data, XtPointer)
{
    int npts, toval;
    char fstrx[256], fstry[256];
    char startstr[256], stopstr[256];
    Leval_ui *ui = (Leval_ui *)client_data;
    npts = atoi((char *)xv_getstr(ui->npts_item));
    strcpy(fstrx, (char *)xv_getstr(ui->x_item));
    strcpy(fstry, (char *)xv_getstr(ui->y_item));
    strcpy(startstr, (char *)xv_getstr(ui->start_item));
    strcpy(stopstr, (char *)xv_getstr(ui->stop_item));
    toval = (int)GetChoice(ui->load_item) + 1;
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommand_StringMessage(DO_COMPUTE_PROC2_1, fstrx);
        cm->sendCommand_StringMessage(DO_COMPUTE_PROC2_2, fstry);
        cm->sendCommand_StringMessage(DO_COMPUTE_PROC2_3, startstr);
        cm->sendCommand_StringMessage(DO_COMPUTE_PROC2_4, stopstr);
        cm->sendCommandMessage(DO_COMPUTE_PROC2, npts, toval);
    }
    do_compute2(fstrx, fstry, startstr, stopstr, npts, toval);
    unset_wait_cursor();
}

/*
 * Compute n-tiles
 */

static Widget *ntiles_set_item;
static Widget *ntiles_nt_item;
static Widget ntiles_ntval_item;
static void do_ntiles_proc(Widget w, XtPointer client_data, XtPointer call_data);

void create_ntiles_frame(Widget, XtPointer, XtPointer)
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
        top = XmCreateDialogShell(app_shell, (char *)"N-tiles", NULL, 0);
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(top, (char *)"dialog_rc", NULL, 0);

        ntiles_set_item = CreateSetChoice(dialog, (char *)"Apply to set:", maxplot, 0);

        ntiles_nt_item = CreatePanelChoice(dialog,
                                           "Compute:",
                                           5,
                                           "Quartiles",
                                           "Deciles",
                                           "Percentiles",
                                           "N-tiles:",
                                           0,
                                           0);

        ntiles_ntval_item = CreateTextItem2(dialog, 10, (char *)"N tiles:");

        XtVaCreateManagedWidget((char *)"sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_ntiles_proc, (XtPointer)top);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)top);

        XtManageChild(dialog);
    }
    XtRaise(top);
    unset_wait_cursor();
}

/*
 * compute ntiles
 */
static void do_ntiles_proc(Widget, XtPointer, XtPointer)
{
    int setno, nt;
    char fstr[256];
    setno = (int)GetChoice(ntiles_set_item);
    nt = (int)GetChoice(ntiles_nt_item);
    strcpy(fstr, (char *)xv_getstr(ntiles_ntval_item));
    switch (nt)
    {
    case 0:
        nt = 4;
        break;
    case 1:
        nt = 10;
        break;
    case 2:
        nt = 100;
        break;
    case 3:
        nt = atoi(fstr);
        break;
    }
    set_wait_cursor();
    if (ismaster)
    {
        cm->sendCommandMessage(DO_NTILES_PROC, setno, nt);
    }
    do_ntiles(cg, setno, nt);
    unset_wait_cursor();
}

/*
 * Rotate, scale, translate
 */

typedef struct _Geom_ui
{
    Widget top;
    SetChoiceItem sel;
    SetChoiceItem sel2;
    Widget *order_item;
    Widget degrees_item;
    Widget rotx_item;
    Widget roty_item;
    Widget scalex_item;
    Widget scaley_item;
    Widget transx_item;
    Widget transy_item;
    Widget *region_item;
    Widget rinvert_item;
} Geom_ui;

static Geom_ui gui;

static void do_geom_proc(Widget, XtPointer, XtPointer);

void create_geom_frame(Widget, XtPointer, XtPointer)
{
    int x, y;
    Widget dialog;
    Widget rc;

    set_wait_cursor();
    if (gui.top == NULL)
    {
        char *label1[2];
        label1[0] = (char *)"Accept";
        label1[1] = (char *)"Close";
        XmGetPos(app_shell, 0, &x, &y);
        gui.top = XmCreateDialogShell(app_shell, (char *)"Geometric transformations", NULL, 0);
        handle_close(gui.top);
        XtVaSetValues(gui.top, XmNx, x, XmNy, y, NULL);
        dialog = XmCreateRowColumn(gui.top, (char *)"dialog_rc", NULL, 0);

        gui.sel = CreateSetSelector(dialog, (char *)"Apply to set:",
                                    SET_SELECT_ALL,
                                    FILTER_SELECT_NONE,
                                    GRAPH_SELECT_CURRENT,
                                    SELECTION_TYPE_MULTIPLE);

        rc = XtVaCreateWidget((char *)"rc", xmRowColumnWidgetClass, dialog,
                              XmNpacking, XmPACK_COLUMN,
                              XmNnumColumns, 8,
                              XmNorientation, XmHORIZONTAL,
                              XmNisAligned, True,
                              XmNadjustLast, False,
                              XmNentryAlignment, XmALIGNMENT_END,
                              NULL);

        gui.order_item = CreatePanelChoice(dialog,
                                           "Apply in order:",
                                           7,
                                           "Rotate, translate, scale",
                                           "Rotate, scale, translate",
                                           "Translate, scale, rotate",
                                           "Translate, rotate, scale",
                                           "Scale, translate, rotate",
                                           "Scale, rotate, translate",
                                           0,
                                           0);

        gui.degrees_item = CreateTextItem4(rc, 10, (char *)"Rotation (degrees):");
        gui.rotx_item = CreateTextItem4(rc, 10, (char *)"Rotate about X = :");
        gui.roty_item = CreateTextItem4(rc, 10, (char *)"Rotate about Y = :");
        gui.scalex_item = CreateTextItem4(rc, 10, (char *)"Scale X:");
        gui.scaley_item = CreateTextItem4(rc, 10, (char *)"Scale Y:");
        gui.transx_item = CreateTextItem4(rc, 10, (char *)"Translate X:");
        gui.transy_item = CreateTextItem4(rc, 10, (char *)"Translate Y:");
        XtManageChild(rc);

        XtVaCreateManagedWidget((char *)"sep", xmSeparatorWidgetClass, dialog, NULL);

        CreateCommandButtons(dialog, 2, but1, label1);
        XtAddCallback(but1[0], XmNactivateCallback, (XtCallbackProc)do_geom_proc, (XtPointer)&gui);
        XtAddCallback(but1[1], XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)gui.top);

        XtManageChild(dialog);
        xv_setstr(gui.degrees_item, (char *)"0.0");
        xv_setstr(gui.rotx_item, (char *)"0.0");
        xv_setstr(gui.roty_item, (char *)"0.0");
        xv_setstr(gui.scalex_item, (char *)"1.0");
        xv_setstr(gui.scaley_item, (char *)"1.0");
        xv_setstr(gui.transx_item, (char *)"0.0");
        xv_setstr(gui.transy_item, (char *)"0.0");
    }
    XtRaise(gui.top);
    unset_wait_cursor();
}

/*
 * compute geom
 */
static void do_geom_proc(Widget, XtPointer client_data, XtPointer)
{
    int i, j, k, order[3], setno, ord;
    int minset = 0, maxset = 0;
    double degrees, sx, sy, rotx, roty, tx, ty, xtmp, ytmp, *x, *y;
    char buf[256];
    Geom_ui *ui = (Geom_ui *)client_data;
    setno = GetSelectedSet(ui->sel);
    if (setno < g[cg].maxplot)
    {
        if (!isactive(cg, setno))
        {
            errwin("Set not active");
            return;
        }
        minset = maxset = setno;
    }
    else if (setno == SET_SELECT_ALL)
    {
        minset = 0;
        maxset = g[cg].maxplot - 1;
    }
    else if (ismaster)
    {
        fprintf(stderr, "compwin.cpp: do_geom_proc(): minset and maxset are used uninitialized\n");
    }
    ord = (int)GetChoice(ui->order_item);
    switch (ord)
    {
    case 0:
        order[0] = 0; /* rotate */
        order[1] = 1; /* translate */
        order[2] = 2; /* scale */
        break;
    case 1:
        order[0] = 0;
        order[1] = 2;
        order[2] = 1;
    case 2:
        order[0] = 1;
        order[1] = 2;
        order[2] = 0;
        break;
    case 3:
        order[0] = 1;
        order[1] = 0;
        order[2] = 2;
        break;
    case 4:
        order[0] = 2;
        order[1] = 1;
        order[2] = 0;
        break;
    case 5:
        order[0] = 2;
        order[1] = 0;
        order[2] = 1;
        break;
    }
    set_wait_cursor();
    strcpy(buf, (char *)xv_getstr(ui->degrees_item));
    degrees = atof(buf);
    strcpy(buf, (char *)xv_getstr(ui->transx_item));
    tx = atof(buf);
    strcpy(buf, (char *)xv_getstr(ui->transy_item));
    ty = atof(buf);
    strcpy(buf, (char *)xv_getstr(ui->rotx_item));
    rotx = atof(buf);
    strcpy(buf, (char *)xv_getstr(ui->roty_item));
    roty = atof(buf);
    strcpy(buf, (char *)xv_getstr(ui->scalex_item));
    sx = atof(buf);
    strcpy(buf, (char *)xv_getstr(ui->scaley_item));
    sy = atof(buf);
    degrees = M_PI / 180.0 * degrees;
    if (ismaster)
    {
        cm->sendCommand_FloatMessage(DO_GEOM_PROC, (double)minset, (double)maxset, (double)ord, degrees, tx, ty, sx, sy, rotx, roty);
    }
    for (k = minset; k <= maxset; k++)
    {
        if (isactive(cg, k))
        {
            x = getx(cg, k);
            y = gety(cg, k);
            for (j = 0; j < 3; j++)
            {
                switch (order[j])
                {
                case 0:
                    if (degrees == 0.0)
                    {
                        break;
                    }
                    for (i = 0; i < getsetlength(cg, k); i++)
                    {
                        xtmp = x[i] - rotx;
                        ytmp = y[i] - roty;
                        x[i] = rotx + cos(degrees) * xtmp - sin(degrees) * ytmp;
                        y[i] = roty + sin(degrees) * xtmp + cos(degrees) * ytmp;
                    }
                    break;
                case 1:
                    for (i = 0; i < getsetlength(cg, k); i++)
                    {
                        x[i] -= tx;
                        y[i] -= ty;
                    }
                    break;
                case 2:
                    for (i = 0; i < getsetlength(cg, k); i++)
                    {
                        x[i] *= sx;
                        y[i] *= sy;
                    }
                    break;
                } /* end case */
            } /* end for j */
            updatesetminmax(cg, k);
            update_set_status(cg, k);
        } /* end if */
    } /* end for k */
    unset_wait_cursor();
    drawgraph();
}

void execute_pick_compute(int, int, int function)
{
    // char fstr[256];
    switch (function)
    {
    case PICK_EXPR:
        /*
            SelectChoice(eui.selt_item, setno);
         */
        do_compute_proc((Widget)NULL, (XtPointer)0, (XtPointer)0);
        break;
    case PICK_RUNAVG:
        /*
            SetChoice(toggle_set_runavg_item, setno);
         */
        do_runavg_proc((Widget)NULL, (XtPointer)0, (XtPointer)0);
        break;
    case PICK_REG:
        /*
            SetChoice(toggle_set_regress_item, setno);
         */
        do_regress_proc((Widget)NULL, (XtPointer)0, (XtPointer)0);
        break;
    case PICK_SEASONAL:
        /*
            SetChoice(toggle_set_seasonal_item, setno);
         */
        do_seasonal_proc((Widget)NULL, (XtPointer)0, (XtPointer)0);
        break;
    }
}

/*
 * Create the comp Widget
 */
void create_comp_frame(Widget, XtPointer, XtPointer)
{
    static Widget top;
    Widget wbut, panel;
    int x, y;

    set_wait_cursor();
    if (top == NULL)
    {
        XmGetPos(app_shell, 0, &x, &y);
        top = XmCreateDialogShell(app_shell, (char *)"Transformations", NULL, 0);
        handle_close(top);
        XtVaSetValues(top, XmNx, x, XmNy, y, NULL);
        panel = XmCreateRowColumn(top, (char *)"comp_rc", NULL, 0);

        wbut = XtVaCreateManagedWidget("Evaluate expression...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_eval_frame, (XtPointer)NULL);

        /*  Not supported in covise wbut = XtVaCreateManagedWidget("Load values...", xmPushButtonWidgetClass, panel,
                      NULL);
      XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc) create_load_frame, (XtPointer) NULL);

      wbut = XtVaCreateManagedWidget("Load & evaluate...", xmPushButtonWidgetClass, panel,
                      NULL);
      XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc) create_leval_frame, (XtPointer) NULL); */

        wbut = XtVaCreateManagedWidget("Histograms...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_histo_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Fourier transforms...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_fourier_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Running averages...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_run_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Regression...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_reg_frame, (XtPointer)NULL);

        if (nonlflag)
        {
            wbut = XtVaCreateManagedWidget("Non-linear curve fitting...", xmPushButtonWidgetClass, panel,
                                           NULL);
            XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_nonl_frame, (XtPointer)NULL);
        }
        wbut = XtVaCreateManagedWidget("Differences...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_diff_frame, (XtPointer)NULL);
        wbut = XtVaCreateManagedWidget("Seasonal differences...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_seasonal_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Integration...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_int_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Cross/auto correlation...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_xcor_frame, (XtPointer)NULL);

        /*
         wbut = XtVaCreateManagedWidget("Interpolation...", xmPushButtonWidgetClass, panel,
                         NULL);
         XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc) create_interp_frame, (XtPointer) NULL);
      */

        wbut = XtVaCreateManagedWidget("Splines...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_spline_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Sample points...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_samp_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Digital filter...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_digf_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Linear convolution...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_lconv_frame, (XtPointer)NULL);

        if (nonlflag)
        {
            wbut = XtVaCreateManagedWidget("N-tiles...", xmPushButtonWidgetClass, panel,
                                           NULL);
            XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_ntiles_frame, (XtPointer)NULL);
        }
        wbut = XtVaCreateManagedWidget("Geometric transformations...", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)create_geom_frame, (XtPointer)NULL);

        wbut = XtVaCreateManagedWidget("Close", xmPushButtonWidgetClass, panel,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)top);
        XtManageChild(panel);
    }
    XtRaise(top);
    unset_wait_cursor();
}
