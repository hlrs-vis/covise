/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "VRBPopupMenu.h"

VRBPopupMenu::VRBPopupMenu(QWidget *parent)
    : QMenu(parent)
    , item(0)
{
}

void VRBPopupMenu::setItem(QTreeWidgetItem *entry)
{
    item = entry;
}

QTreeWidgetItem *VRBPopupMenu::getItem()
{
    return item;
}

VRBPopupMenu::~VRBPopupMenu()
{
}
