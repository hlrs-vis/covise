/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: blockwin.c,v 1.4 1994/09/29 03:37:37 pturner Exp pturner $
 *
 * read block data files
 *
 */

#include <stdio.h>

#include <Xm/Xm.h>
#include <Xm/DialogS.h>
#include <Xm/BulletinB.h>
#include <Xm/FileSB.h>
#include <Xm/Frame.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/SelectioB.h>
#include <Xm/ToggleB.h>

#include "globals.h"
#include "motifinc.h"

static Widget block_dialog; /* read data popup */

static int blocksrc = DISK;

static void set_src_proc(Widget, XtPointer client_data, XtPointer);
static void block_proc(Widget, XtPointer, XtPointer);

static void set_src_proc(Widget, XtPointer client_data, XtPointer)
{
    int data = (long)client_data;
    switch (data)
    {
    case 0:
        blocksrc = DISK;
        break;
    case 1:
        blocksrc = PIPE;
        break;
    }
}

static void block_proc(Widget, XtPointer, XtPointer)
{
    Arg args;
    XmString list_item;
    char *s;
    XtSetArg(args, XmNtextString, &list_item);
    XtGetValues(block_dialog, &args, 1);
    XmStringGetLtoR(list_item, charset, &s);
    if (getdata(cg, s, blocksrc, BLOCK))
    {
        if (blocklen == 0)
        {
            errwin("Block data length = 0");
        }
        else if (blockncols == 0)
        {
            errwin("Number of columns in block data = 0");
        }
        else
        {
            XtUnmanageChild(block_dialog);
            create_eblock_frame(NULL, NULL, NULL);
        }
    }
    XtFree(s);
}

void create_block_popup(Widget, XtPointer, XtPointer)
{
    long i;
    Widget lab, rc, fr, rb, rw[5];

    set_wait_cursor();
    if (block_dialog == NULL)
    {
        block_dialog = XmCreateFileSelectionDialog(app_shell, (char *)"read_block_data", NULL, 0);
        XtVaSetValues(XtParent(block_dialog), XmNtitle, "Read block data", NULL);

        XtAddCallback(block_dialog, XmNcancelCallback, (XtCallbackProc)destroy_dialog, block_dialog);
        XtAddCallback(block_dialog, XmNokCallback, (XtCallbackProc)block_proc, 0);

        fr = XmCreateFrame(block_dialog, (char *)"frame", NULL, 0);

        rc = XmCreateRowColumn(fr, (char *)"rc", NULL, 0);
        XtVaSetValues(rc, XmNorientation, XmHORIZONTAL, NULL);

        lab = XmCreateLabel(rc, (char *)"Data source:", NULL, 0);
        XtManageChild(lab);

        rb = XmCreateRadioBox(rc, (char *)"rb", NULL, 0);
        XtVaSetValues(rb, XmNorientation, XmHORIZONTAL, NULL);

        rw[0] = XmCreateToggleButton(rb, (char *)"Disk", NULL, 0);
        rw[1] = XmCreateToggleButton(rb, (char *)"Pipe", NULL, 0);
        for (i = 0; i < 2; i++)
        {
            XtAddCallback(rw[i], XmNvalueChangedCallback, (XtCallbackProc)set_src_proc, (XtPointer)i);
        }

        XtManageChildren(rw, 2);
        XtManageChild(rb);
        XtManageChild(rc);
        XtManageChild(fr);
        XmToggleButtonSetState(rw[0], True, False);
    }
    XtRaise(block_dialog);
    unset_wait_cursor();
}
