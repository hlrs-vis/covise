/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * for Motif specific items
 */

extern Widget app_shell; /* defined in xmgr.c */
extern XmStringCharSet charset; /* defined in xmgr.c */

/* set selection gadget */
typedef struct _SetChoiceItem
{
    int type;
    int display;
    int gno;
    int spolicy;
    int fflag; /* if 0, no filter gadgets */
    Widget list;
    Widget rb;
    Widget but[8];
} SetChoiceItem;

void update_set_list(int gno, SetChoiceItem l);
void save_set_list(SetChoiceItem l);
SetChoiceItem CreateSetSelector(Widget parent, const char *label, int type, int ff, int gtype, int stype);
void SetSelectorFilterCB(Widget parent, XtPointer cld, XtPointer calld);
int GetSelectedSet(SetChoiceItem l);

#include "xprotos.h"
#include "noxprotos.h"
