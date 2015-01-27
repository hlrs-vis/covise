/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: helpwin.c,v 1.2 1994/09/29 00:51:40 pturner Exp $
 *
 * Help
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <Xm/Xm.h>
#include <Xm/DialogS.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/ScrolledW.h>
#include <Xm/ScrollBar.h>
#include <Xm/Separator.h>
#include <Xm/List.h>
#include <Xm/ToggleB.h>
#include <Xm/Text.h>
#include <Xm/TextF.h>

#include <Xm/Protocols.h>
#include <X11/keysym.h>

#include "globals.h"
#include "motifinc.h"

#ifndef USE_HTMLW

void create_help_frame(Widget,
                       XtPointer,
                       XtPointer)
{
}

#else

#include "htmlw/HTML.h"

static Widget help_frame, help_panel;

static Widget scrolled_win;

void anchorcb(Widget w, XtPointer cd, XtPointer cld)
{
    int id;
    WbAnchorCallbackData *d = (WbAnchorCallbackData *)cld;
    id = HTMLAnchorToId(w, d->href);
    HTMLGotoId(w, id);
    printf("Called %d %s %s\n", d->element_id, d->text, d->href);
}

/*
 * Create the help Panel
 */
void create_help_frame(Widget w,
                       XtPointer client_data,
                       XtPointer call_data)
{
    FILE *fp;
    int x, y;
    Widget wbut, rc;
    char *t = (char *)client_data;
    char *tbuf;
    struct stat sb;
    set_wait_cursor();
    if (help_frame == NULL)
    {

        if (stat("doc/ACEgr.html", &sb))
        {
            char tmpbuf[256];
            sprintf(tmpbuf, "Can't stat file %s", "doc/ACEgr.html");
            errwin(tmpbuf);
            return;
        }
        if (!S_ISREG(sb.st_mode))
        {
            char tmpbuf[256];
            sprintf(tmpbuf, "File %s is not a regular file", "doc/ACEgr.html");
            errwin(tmpbuf);
            return;
        }
        fp = fopen("doc/ACEgr.html", "r");
        if (fp == NULL)
        {
            errwin("Unable to open help file");
            return;
        }
        if (!(tbuf = (char *)malloc((unsigned)(sb.st_size + 1))))
        {
            errwin("Can't allocate memory for helpfile");
            return;
        }
        if (!fread(tbuf, sizeof(char), sb.st_size + 1, fp))
        {
        }
        tbuf[sb.st_size] = 0;
        fclose(fp);

        XmGetPos(app_shell, 0, &x, &y);
        help_frame = XmCreateDialogShell(app_shell, "Help", NULL, 0);
        handle_close(help_frame);
        XtVaSetValues(help_frame, XmNx, x, XmNy, y, NULL);
        help_panel = XmCreateRowColumn(help_frame, "help_rc", NULL, 0);

        scrolled_win = XtVaCreateWidget("view", htmlWidgetClass,
                                        help_panel,
                                        XmNresizePolicy, XmRESIZE_ANY,
                                        WbNfancySelections, True,
                                        WbNverticalScrollOnRight, True,
                                        WbNdelayImageLoads, True,
                                        XmNshadowThickness, 2,

                                        XmNwidth, 500,
                                        XmNheight, 500,
                                        NULL);
        XtAddCallback(scrolled_win, WbNanchorCallback, anchorcb, 0);
        XtManageChild(scrolled_win);

        XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, help_panel, NULL);
        rc = XmCreateRowColumn(help_panel, "rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);
        wbut = XtVaCreateManagedWidget("Close", xmPushButtonWidgetClass, rc,
                                       NULL);
        XtAddCallback(wbut, XmNactivateCallback, (XtCallbackProc)destroy_dialog, (XtPointer)help_frame);
        XtManageChild(rc);

        HTMLSetText(scrolled_win, tbuf, "Header text",
                    "Footer text", 1,
                    "Anchor", NULL);
        XtManageChild(help_panel);
    }
    XtRaise(help_frame);
    if (t == NULL)
    {
        t = "Sorry, no help available for this item";
    }
    unset_wait_cursor();
}
#endif
