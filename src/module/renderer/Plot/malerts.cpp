/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: malerts.c,v 1.1 1994/05/13 01:29:47 pturner Exp $
 *
 * alerts for Motif
 */

#include <stdio.h>
#include <math.h>
#include "extern.h"
#include "globals.h"

#include <Xm/Xm.h>
#include <Xm/DialogS.h>
#include <Xm/MessageB.h>

static Widget error_popup;
static Widget yesno_popup;

extern Widget app_shell;
extern XmStringCharSet charset;
extern int inwin;
extern int noask;

extern XtAppContext app_con;

static int yesno_retval = 0;
static Boolean keep_grab = True;

static char *ht; /* help text */
void infowin(char *s);

XtCallbackProc create_help_frame(Widget w,
                                 XtPointer client_data,
                                 XtPointer call_data);

void yesnoCB(Widget w, Boolean *keep_grab, XmAnyCallbackStruct *reason)
{
    int why = reason->reason;

    *keep_grab = False;
    XtRemoveGrab(XtParent(w));
    XtUnmanageChild(w);
    switch (why)
    {
    case XmCR_OK:
        yesno_retval = 1;
        /* process ok action */
        break;
    case XmCR_CANCEL:
        yesno_retval = 0;
        /* process cancel action */
        break;
    case XmCR_HELP:
        yesno_retval = 0;
        infowin(ht);
        /*
            create_help_frame(w, (XtPointer) ht, (XtPointer) NULL);
         */
        break;
    }
}

int yesno(const char *msg1, const char *s1, const char *s2, const char *helptext)
{
    char buf[256];
    static XmString str, str1, str2;
    XEvent event;

    ht = (char *)helptext;

    keep_grab = True;

    if (noask)
    {
        return 1;
    }
    if (!inwin)
    {
        fprintf(stderr, "%s\n", msg1);
        fprintf(stderr, "%s\n", "(y)es/(n)o:");
        fgets(buf, 256, stdin);
        if (buf[0] == 'y')
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
    if (yesno_popup)
    {
        XmStringFree(str);
        XmStringFree(str1);
        XmStringFree(str2);
        str = XmStringCreateLtoR((char *)msg1, charset);
        XtVaSetValues(yesno_popup,
                      XmNmessageString, str,
                      NULL);
        if (s1 != NULL)
        {
            XtVaSetValues(yesno_popup,
                          XmNokLabelString, str1 = XmStringCreateLtoR((char *)s1, charset),
                          NULL);
        }
        else
        {
            XtVaSetValues(yesno_popup,
                          XmNokLabelString, str1 = XmStringCreateLtoR((char *)"OK", charset),
                          NULL);
        }
        if (s2 != NULL)
        {
            XtVaSetValues(yesno_popup,
                          XmNcancelLabelString, str2 = XmStringCreateLtoR((char *)s2, charset),
                          NULL);
        }
        else
        {
            XtVaSetValues(yesno_popup,
                          XmNcancelLabelString, str2 = XmStringCreateLtoR((char *)"Cancel", charset),
                          NULL);
        }
    }
    else
    {
        str = XmStringCreateLtoR((char *)msg1, charset);
        yesno_popup = XmCreateErrorDialog(app_shell, (char *)"warndlg", NULL, 0);
        XtVaSetValues(yesno_popup,
                      XmNmessageString, str,
                      XmNdialogTitle, XmStringCreateLtoR((char *)"Warning", charset),
                      NULL);
        if (s1 != NULL)
        {
            XtVaSetValues(yesno_popup, XmNokLabelString, str1 = XmStringCreateLtoR((char *)s1, charset),
                          NULL);
        }
        else
        {
        }
        if (s2 != NULL)
        {
            XtVaSetValues(yesno_popup, XmNcancelLabelString, str2 = XmStringCreateLtoR((char *)s2, charset),
                          NULL);
        }
        else
        {
        }
        XtAddCallback(yesno_popup, XmNokCallback, (XtCallbackProc)yesnoCB, (XtPointer)&keep_grab);
        XtAddCallback(yesno_popup, XmNcancelCallback, (XtCallbackProc)yesnoCB, (XtPointer)&keep_grab);

        XtAddCallback(yesno_popup, XmNhelpCallback, (XtCallbackProc)yesnoCB, (XtPointer)&keep_grab);
    }
    XtManageChild(yesno_popup);
    XtAddGrab(XtParent(yesno_popup), True, False);
    while (keep_grab || XtAppPending(app_con))
    {
        XtAppNextEvent(app_con, &event);
        XtDispatchEvent(&event);
    }
    return yesno_retval;
}

void error_helpCB(Widget w, XtPointer, XtPointer)
{
    XtUnmanageChild(error_popup);
    create_help_frame(w, (XtPointer)NULL, (XtPointer)NULL);
}

void errwin(const char *s)
{
    static XmString str;
    log_results((char *)s);
    if ((!inwin) || (!ismaster))
    {
        fprintf(stderr, "%s\n", s);
        return;
    }
    if (error_popup)
    {
        XmStringFree(str);
        str = XmStringCreateLtoR((char *)s, charset);
        XtVaSetValues(error_popup,
                      XmNmessageString, str,
                      NULL);
        XtManageChild(error_popup);
        return;
    }
    str = XmStringCreateLtoR((char *)s, charset);
    error_popup = XmCreateErrorDialog(app_shell, (char *)"errordlg", NULL, 0);
    XtVaSetValues(error_popup,
                  XmNmessageString, str,
                  XmNdialogTitle, XmStringCreateLtoR((char *)"Error", charset),
                  XmNdialogStyle, XmDIALOG_APPLICATION_MODAL,
                  NULL);
    XtAddCallback(error_popup, XmNhelpCallback, (XtCallbackProc)error_helpCB,
                  (XtPointer)NULL);
    XtUnmanageChild(XmMessageBoxGetChild(error_popup, XmDIALOG_CANCEL_BUTTON));
    XtManageChild(error_popup);
}

static Widget info_popup;

void infowin(char *s)
{
    static XmString str;
    char *buf = (char *)"Sorry, no help available for this item";
    if (s == NULL)
    {
        s = buf;
    }
    if (!inwin)
    {
        fprintf(stderr, "%s\n", s);
        return;
    }
    if (info_popup)
    {
        XmStringFree(str);
        str = XmStringCreateLtoR(s, charset);
        XtVaSetValues(info_popup,
                      XmNmessageString, str,
                      NULL);
        XtManageChild(info_popup);
        return;
    }
    str = XmStringCreateLtoR(s, charset);
    info_popup = XmCreateInformationDialog(app_shell, (char *)"Info", NULL, 0);
    XtVaSetValues(info_popup,
                  XmNmessageString, str,
                  XmNdialogTitle, XmStringCreateLtoR((char *)"Info", charset),
                  XmNdialogStyle, XmDIALOG_APPLICATION_MODAL,
                  NULL);
    /*
       XtAddCallback(info_popup, XmNhelpCallback, (XtCallbackProc) info_helpCB,
           (XtPointer) NULL);
   */
    XtUnmanageChild(XmMessageBoxGetChild(info_popup, XmDIALOG_CANCEL_BUTTON));
    XtUnmanageChild(XmMessageBoxGetChild(info_popup, XmDIALOG_HELP_BUTTON));
    XtManageChild(info_popup);
}
