/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: motifutils.c,v 1.6 1994/11/01 07:02:39 pturner Exp pturner $
 *
 * utilities for Motif
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#include <X11/X.h>
#include <X11/Xatom.h>

#include <Xm/Xm.h>
#include <Xm/BulletinB.h>
#include <Xm/DialogS.h>
#include <Xm/Frame.h>
#include <Xm/Form.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/Text.h>
#include <Xm/Protocols.h>
#include <Xm/AtomMgr.h>

extern Widget app_shell, canvas;
extern Display *disp;
extern XmStringCharSet charset;

extern int maxcolors;

void errwin(const char *s);
void savewidget(Widget w);

/* this may no longer be needed
#ifndef HAS_TOLOWER
int tolower(int c)
{
    if (c >= 'A' && c <= 'Z') {
   return (c - ' ');
    } else {
   return c;
    }
}
#endif
*/

void SetChoice(Widget *w, int value)
{
    Arg a;

    if (w == (Widget *)NULL)
    {
        errwin("Internal error, SetChoice called with NULL argument");
        return;
    }
    if (w[value + 2] == (Widget)NULL)
    {
        errwin("Internal error SetChoice: Attempt to set NULL Widget");
        return;
    }
    XtSetArg(a, XmNmenuHistory, w[value + 2]);
    XtSetValues(w[0], &a, 1);
}

int GetChoice(Widget *w)
{
    Arg a;
    Widget warg;
    int i;

    if (w == (Widget *)NULL)
    {
        errwin("Internal error, GetChoice called with NULL argument, returning 0");
        return 0;
    }
    XtSetArg(a, XmNmenuHistory, &warg);
    XtGetValues(w[0], &a, 1);
    i = 0;
    while (w[i + 2] != warg)
    {
        if (w[i + 2] == (Widget)NULL)
        {
            errwin("Internal error GetChoice: Found NULL in Widget list, returning 0");
            return 0;
        }
        i++;
    }
    return i;
}

Widget *CreatePanelChoice(Widget parent, const char *labelstr, int nchoices, ...)
{
    va_list var;
    int i = 0;
    XmString str;
    char *s;
    Widget *retval;

    nchoices--;

    retval = (Widget *)XtMalloc((nchoices + 2) * sizeof(Widget));

    retval[1] = XmCreatePulldownMenu(parent, (char *)"pulldown", NULL, 0);

    i = 0;
    va_start(var, nchoices);
    while ((s = va_arg(var, char *)) != NULL)
    {
        retval[i + 2] = XmCreatePushButton(retval[1], s, NULL, 0);
        i++;
    }
    XtManageChildren(retval + 2, nchoices);
    va_end(var);

    str = XmStringCreate((char *)labelstr, charset);

    retval[0] = XmCreateOptionMenu(parent, (char *)"optionmenu", NULL, 0);
    XtVaSetValues(retval[0],
                  XmNlabelString, str,
                  XmNsubMenuId, retval[1],
                  XmNentryBorder, 2,
                  XmNwhichButton, 1,
                  NULL);
    XtManageChild(retval[0]);
    for (i = 0; i < nchoices + 2; i++)
    {
        retval[i] = retval[i];
    }
    return retval;
}

Widget *CreatePanelChoice0(Widget parent, const char *labelstr, int ncols, int nchoices, ...)
{
    va_list var;
    int i = 0;
    XmString str;
    char *s;
    Widget *retval;

    nchoices--;

    retval = (Widget *)XtMalloc((nchoices + 2) * sizeof(Widget));

    retval[1] = XmCreatePulldownMenu(parent, (char *)"pulldown", NULL, 0);
    XtVaSetValues(retval[1],
                  XmNorientation, XmVERTICAL,
                  XmNpacking, XmPACK_COLUMN,
                  XmNnumColumns, ncols,
                  NULL);

    i = 0;

    va_start(var, nchoices);
    while ((s = va_arg(var, char *)) != NULL)
    {
        retval[i + 2] = XmCreatePushButton(retval[1], s, NULL, 0);
        i++;
    }
    va_end(var);

    XtManageChildren(retval + 2, nchoices);

    str = XmStringCreate((char *)labelstr, charset);

    retval[0] = XmCreateOptionMenu(parent, (char *)"optionmenu", NULL, 0);
    XtVaSetValues(retval[0],
                  XmNlabelString, str,
                  XmNsubMenuId, retval[1],
                  XmNentryBorder, 2,
                  XmNwhichButton, 1,
                  NULL);
    XtManageChild(retval[0]);
    for (i = 0; i < nchoices + 2; i++)
    {
        retval[i] = retval[i];
    }
    return retval;
}

Widget *CreateGraphChoice(Widget parent, const char *labelstr, int ngraphs, int type)
{
    int i = 0;
    XmString str;
    char *name, buf[32];
    Widget *retval = NULL;

    strcpy(buf, "graphchoice");
    name = buf;

    switch (type)
    {
    case 0:
        retval = (Widget *)XtMalloc((ngraphs + 2) * sizeof(Widget));
        retval[1] = XmCreatePulldownMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[1],
                      XmNorientation, XmVERTICAL,
                      XmNpacking, XmPACK_COLUMN,
                      XmNnumColumns, 4,
                      NULL);
        i = 0;
        for (i = 0; i < ngraphs; i++)
        {
            sprintf(buf, "%d", i);
            retval[i + 2] = XmCreatePushButton(retval[1], buf, NULL, 0);
        }
        XtManageChildren(retval + 2, ngraphs);

        str = XmStringCreate((char *)labelstr, charset);

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
        retval = (Widget *)XtMalloc((ngraphs + 3) * sizeof(Widget));
        retval[1] = XmCreatePulldownMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[1],
                      XmNorientation, XmVERTICAL,
                      XmNpacking, XmPACK_COLUMN,
                      XmNnumColumns, 4,
                      NULL);
        retval[2] = XmCreatePushButton(retval[1], (char *)"Current", NULL, 0);
        for (i = 1; i <= ngraphs; i++)
        {
            sprintf(buf, "%d", i - 1);
            retval[i + 2] = XmCreatePushButton(retval[1], buf, NULL, 0);
        }
        XtManageChildren(retval + 2, ngraphs + 1);
        str = XmStringCreate((char *)labelstr, charset);
        retval[0] = XmCreateOptionMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[0],
                      XmNlabelString, str,
                      XmNsubMenuId, retval[1],
                      XmNentryBorder, 2,
                      XmNwhichButton, 1,
                      NULL);
        XtManageChild(retval[0]);
        break;
    case 2:
        retval = (Widget *)XtMalloc((ngraphs + 4) * sizeof(Widget));
        retval[1] = XmCreatePulldownMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[1],
                      XmNorientation, XmVERTICAL,
                      XmNpacking, XmPACK_COLUMN,
                      XmNnumColumns, 4,
                      NULL);
        retval[2] = XmCreatePushButton(retval[1], (char *)"Current", NULL, 0);
        for (i = 1; i <= ngraphs; i++)
        {
            sprintf(buf, "%d", i - 1);
            retval[i + 2] = XmCreatePushButton(retval[1], buf, NULL, 0);
        }
        retval[ngraphs + 3] = XmCreatePushButton(retval[1], (char *)"All", NULL, 0);

        XtManageChildren(retval + 2, ngraphs + 2);
        str = XmStringCreate((char *)labelstr, charset);
        retval[0] = XmCreateOptionMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[0],
                      XmNlabelString, str,
                      XmNsubMenuId, retval[1],
                      XmNentryBorder, 2,
                      XmNwhichButton, 1,
                      NULL);
        XtManageChild(retval[0]);
        break;
    }
    return retval;
}

Widget *CreateColorChoice(Widget parent, const char *s, int)
{
    int nchoices, ncols, i = 0;
    XmString str;
    char *name, labelstr[32], buf[32];
    Widget *retval;
    extern long colors[];
    extern int use_colors;

    if (s == NULL)
    {
        strcpy(labelstr, "");
    }
    else
    {
        strcpy(labelstr, s);
    }

    ncols = (int)sqrt((double)maxcolors);
    if (ncols == 0)
    {
        ncols = 1;
    }
    nchoices = maxcolors;
    retval = (Widget *)XtMalloc((maxcolors + 2) * sizeof(Widget));

    strcpy(buf, "colorchoice");
    name = buf;
    retval[1] = XmCreatePulldownMenu(parent, name, NULL, 0);
    XtVaSetValues(retval[1],
                  XmNorientation, XmVERTICAL,
                  XmNpacking, XmPACK_COLUMN,
                  XmNnumColumns, ncols,
                  NULL);
    for (i = 0; i < maxcolors; i++)
    {
        sprintf(buf, "%2d", i);
        retval[i + 2] = XmCreatePushButton(retval[1], buf, NULL, 0);
        if (use_colors > 2)
        {
            XtVaSetValues(retval[i + 2],
                          XmNbackground, colors[i],
                          NULL);
        }
    }
    XtManageChildren(retval + 2, nchoices);

    strcpy(buf, "coloroption");
    str = XmStringCreate(labelstr, charset);

    retval[0] = XmCreateOptionMenu(parent, buf, NULL, 0);
    XtVaSetValues(retval[0],
                  XmNlabelString, str,
                  XmNsubMenuId, retval[1],
                  XmNentryBorder, 2,
                  XmNwhichButton, 1,
                  NULL);
    XtManageChild(retval[0]);
    return retval;
}

Widget CreateTextItem2(Widget parent, int len, const char *s)
{
    Widget w;
    Widget rc;
    XmString str;
    rc = XmCreateRowColumn(parent, (char *)"rc", NULL, 0);
    XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
    str = XmStringCreateLtoR((char *)s, charset);
    XtVaCreateManagedWidget("label", xmLabelWidgetClass, rc,
                            XmNlabelString, str,
                            NULL);
    XmStringFree(str);
    w = XtVaCreateManagedWidget("text", xmTextWidgetClass, rc,
                                XmNtraversalOn, True,
                                XmNcolumns, len,
                                NULL);
    XtManageChild(rc);
    return w;
}

Widget CreateTextItem4(Widget parent, int len, const char *label)
{
    Widget retval;
    XtVaCreateManagedWidget(label, xmLabelWidgetClass, parent, NULL);
    retval = XtVaCreateManagedWidget("text", xmTextWidgetClass, parent, NULL);
    XtVaSetValues(retval, XmNcolumns, len, NULL);
    return retval;
}

char *xv_getstr(Widget w)
{
    char *s;
    static char buf[256];

    strcpy(buf, s = XmTextGetString(w));
    XtFree(s);
    return buf;
}

void xv_setstr(Widget w, const char *s)
{
    XmTextSetString(w, (char *)s);
}

/*
 * set position
 */
void XmGetPos(Widget w, int type, int *x, int *y)
{
    Arg a[4];
    int ac = 0;
    static int curpos = 0;
    Position xt, yt;
    Dimension wt, ht;

    /*
    * get the location of the whole app
    */
    ac = 0;
    XtSetArg(a[ac], XmNx, &xt);
    ac++;
    XtSetArg(a[ac], XmNy, &yt);
    ac++;
    XtSetArg(a[ac], XmNheight, &ht);
    ac++;
    XtSetArg(a[ac], XmNwidth, &wt);
    ac++;
    XtGetValues(w, a, ac);

    switch (type)
    {
    case 0: /* position at upper left */
        *x = xt - 50;
        *y = yt + ht / 4 + (curpos % 5) * 50;
        curpos++;
        if (*x < 0)
        {
            *x = 0;
        }
        if (*y < 0)
        {
            *y = 0;
        }
        break;
    case 1: /* position at upper right */
        *x = xt + 50 + wt;
        *y = yt + (curpos % 5 + 1) * 50;
        break;
    case 2: /* center on w */
        *x = xt + wt / 2;
        *y = yt + ht / 2;
        break;
    }
}

void GetWH(Widget w, Dimension *ww, Dimension *wh)
{
    Arg args;

    XtSetArg(args, XmNwidth, ww);
    XtGetValues(w, &args, 1);
    XtSetArg(args, XmNheight, wh);
    XtGetValues(w, &args, 1);
}

void GetXY(Widget w, Position *x, Position *y)
{
    Arg args;

    XtSetArg(args, XmNx, x);
    XtGetValues(w, &args, 1);
    XtSetArg(args, XmNy, y);
    XtGetValues(w, &args, 1);
}

/**********************************************************************
 * XtFlush - Flushes all Xt events.
 **********************************************************************/
void XtFlush(void)
{
    extern Widget app_shell;

    XtAppContext app;

    app = XtWidgetToApplicationContext(app_shell);
    while (XtAppPending(app) & XtIMXEvent)
    {
        XtAppProcessEvent(app, XtIMXEvent);
        XFlush(XtDisplay(app_shell));
    }
}

/*
 * generic unmanage popup routine, used elswhere
 */
void destroy_dialog(Widget, XtPointer p)
{
    XtUnmanageChild((Widget)p);
}

/*
 * handle the close item on the WM menu
 */
void handle_close(Widget w)
{
    Atom WM_DELETE_WINDOW;
    XtVaSetValues(w,
                  XmNdeleteResponse, XmDO_NOTHING,
                  NULL);
    WM_DELETE_WINDOW = XmInternAtom(XtDisplay(app_shell), (char *)"WM_DELETE_WINDOW", False);
    XmAddProtocolCallback((Widget)w, (Atom)XM_WM_PROTOCOL_ATOM(w), (Atom)WM_DELETE_WINDOW, (XtCallbackProc)destroy_dialog, (XtPointer)w);
}

/*
 * Manage and raise
 */
void XtRaise(Widget w)
{
    XtManageChild(w);
    XMapRaised(XtDisplay(w), XtWindow(w));
    savewidget(w);
}

/*
 * save dialog widgets for cursor control
 */

typedef struct _SaveDialogState
{
    Widget w;
    char *name;
    int open;
    int restore;
    int x, y;
    int width, height;
} SaveDialogState;

static Widget *savewidgets;
static int nsavedwidgets;

void savewidget(Widget w)
{
    int i;
    if (savewidgets == NULL)
    {
        savewidgets = (Widget *)malloc(sizeof(Widget));
    }
    else
    {
        for (i = 0; i < nsavedwidgets; i++)
        {
            if (w == savewidgets[i])
            {
                return;
            }
        }
        savewidgets = (Widget *)realloc(savewidgets, (nsavedwidgets + 2) * sizeof(Widget));
    }
    savewidgets[nsavedwidgets] = w;
    nsavedwidgets++;
}

void savewidgetstate(Widget)
{
    int i;
    if (savewidgets == NULL)
    {
    }
    else
    {
        for (i = 0; i < nsavedwidgets; i++)
        {
        }
    }
}

void restorewidgetstate(Widget)
{
    int i;
    if (savewidgets == NULL)
    {
    }
    else
    {
        for (i = 0; i < nsavedwidgets; i++)
        {
        }
    }
}

void deletewidget(Widget w)
{
    int i, j;
    if (savewidgets == NULL || nsavedwidgets == 0)
    {
        return;
    }
    else
    {
        /* find the widget */
        for (i = 0; i < nsavedwidgets; i++)
        {
            if (w == savewidgets[i])
            {
                break;
            }
        }
        /* shouldn't happen, widget not in the saved widget list */
        if (i == nsavedwidgets)
        {
            return;
        }
        /* remove the widget from the list */
        for (j = i; j < nsavedwidgets - 1; j++)
        {
            savewidgets[j] = savewidgets[j + 1];
        }
        if (nsavedwidgets - 1 > 0)
        {
            savewidgets = (Widget *)realloc(savewidgets, (nsavedwidgets - 1) * sizeof(Widget));
            nsavedwidgets--;
        }
        else
        {
            free(savewidgets);
            savewidgets = NULL;
        }
    }
}

void DefineDialogCursor(Cursor c)
{
    int i;
    //    XSetWindowAttributes attrs;
    for (i = 0; i < nsavedwidgets; i++)
    {
        XDefineCursor(disp, XtWindow(savewidgets[i]), c);
        XDefineCursor(disp, XtWindow(app_shell), c);
        /*
              attrs.cursor = c;
              XChangeWindowAttributes(disp, XtWindow(savewidgets[i]), CWCursor, &attrs);
      */
    }
    XFlush(disp);
}

void UndefineDialogCursor()
{
    int i;
    for (i = 0; i < nsavedwidgets; i++)
    {
        XUndefineCursor(disp, XtWindow(savewidgets[i]));
        XUndefineCursor(disp, XtWindow(app_shell));
    }
    XFlush(disp);
}

Widget *CreateSetChoice(Widget parent, const char *labelstr, int nsets, int type)
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
            sprintf(buf, "%d", i);
            retval[i + 2] = XmCreatePushButton(retval[1], buf, NULL, 0);
        }
        XtManageChildren(retval + 2, nsets);

        str = XmStringCreate((char *)labelstr, charset);

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
        for (i = 0; i < nsets; i++)
        {
            sprintf(buf, "%d", i);
            retval[i + 2] = XmCreatePushButton(retval[1], buf, NULL, 0);
        }
        retval[nsets + 2] = XmCreatePushButton(retval[1], (char *)"All", NULL, 0);
        XtManageChildren(retval + 2, nsets + 1);

        str = XmStringCreate((char *)labelstr, charset);

        retval[0] = XmCreateOptionMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[0],
                      XmNlabelString, str,
                      XmNsubMenuId, retval[1],
                      XmNentryBorder, 2,
                      XmNwhichButton, 1,
                      NULL);
        XtManageChild(retval[0]);
        break;
    case 2:
        retval = (Widget *)XtMalloc((nsets + 3) * sizeof(Widget));
        strcpy(buf, "setchoice");
        name = buf;
        retval[1] = XmCreatePulldownMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[1],
                      XmNorientation, XmVERTICAL,
                      XmNpacking, XmPACK_COLUMN,
                      XmNnumColumns, nsets / 10,
                      NULL);
        i = 0;
        for (i = 0; i < nsets; i++)
        {
            sprintf(buf, "%d", i);
            retval[i + 2] = XmCreatePushButton(retval[1], buf, NULL, 0);
        }
        retval[nsets + 2] = XmCreatePushButton(retval[1], (char *)"All", NULL, 0);
        XtManageChildren(retval + 2, nsets + 1);

        str = XmStringCreate((char *)labelstr, charset);

        retval[0] = XmCreateOptionMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[0],
                      XmNlabelString, str,
                      XmNsubMenuId, retval[1],
                      XmNentryBorder, 2,
                      XmNwhichButton, 1,
                      NULL);
        XtManageChild(retval[0]);
        break;
    /* 4 is Next */
    case 4:
        retval = (Widget *)XtMalloc((nsets + 3) * sizeof(Widget));
        strcpy(buf, "setchoice");
        name = buf;
        retval[1] = XmCreatePulldownMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[1],
                      XmNorientation, XmVERTICAL,
                      XmNpacking, XmPACK_COLUMN,
                      XmNnumColumns, nsets / 10,
                      NULL);
        retval[2] = XmCreatePushButton(retval[1], (char *)"Next", NULL, 0);
        for (i = 1; i <= nsets; i++)
        {
            sprintf(buf, "%d", i - 1);
            retval[i + 2] = XmCreatePushButton(retval[1], buf, NULL, 0);
        }
        XtManageChildren(retval + 2, nsets + 1);

        str = XmStringCreate((char *)labelstr, charset);

        retval[0] = XmCreateOptionMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[0],
                      XmNlabelString, str,
                      XmNsubMenuId, retval[1],
                      XmNentryBorder, 2,
                      XmNwhichButton, 1,
                      NULL);
        XtManageChild(retval[0]);
        break;
    /* 5 is Next, Same */
    case 6: /* All, then sets */
        nmal = nsets + 3;
        retval = (Widget *)XtMalloc(nmal * sizeof(Widget));
        retval[1] = XmCreatePulldownMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[1],
                      XmNorientation, XmVERTICAL,
                      XmNpacking, XmPACK_COLUMN,
                      XmNnumColumns, nsets / 10,
                      NULL);
        retval[2] = XmCreatePushButton(retval[1], (char *)"All", NULL, 0);
        for (i = 0; i < nsets; i++)
        {
            sprintf(buf, "%d", i);
            retval[i + 3] = XmCreatePushButton(retval[1], buf, NULL, 0);
        }
        XtManageChildren(retval + 2, nsets + 1);

        str = XmStringCreate((char *)labelstr, charset);

        retval[0] = XmCreateOptionMenu(parent, name, NULL, 0);
        XtVaSetValues(retval[0],
                      XmNlabelString, str,
                      XmNsubMenuId, retval[1],
                      XmNentryBorder, 2,
                      XmNwhichButton, 1,
                      NULL);
        XtManageChild(retval[0]);
        break;
    }
    return retval;
}

void CreateCommandButtons(Widget parent, int n, Widget *buts, char **labels)
{
    int i;
    Widget form;
    form = XtVaCreateWidget("form", xmFormWidgetClass, parent,
                            XmNfractionBase, n,
                            NULL);

    for (i = 0; i < n; i++)
    {
        buts[i] = XtVaCreateManagedWidget(labels[i],
                                          xmPushButtonWidgetClass, form,
                                          XmNtopAttachment, XmATTACH_FORM,
                                          XmNbottomAttachment, XmATTACH_FORM,
                                          XmNleftAttachment, XmATTACH_POSITION,
                                          XmNleftPosition, i,
                                          XmNrightAttachment, XmATTACH_POSITION,
                                          XmNrightPosition, i + 1,
                                          XmNshowAsDefault, i == 0 ? True : False,
                                          XmNdefaultButtonShadowThickness, 1,
                                          NULL);
    }
    XtManageChild(form);
    {
        Dimension h;
        XtVaGetValues(buts[0], XmNheight, &h, NULL);
        XtVaSetValues(form, XmNpaneMaximum, h, XmNpaneMinimum, h, NULL);
    }
}
