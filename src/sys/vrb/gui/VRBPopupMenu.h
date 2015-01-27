/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRBPOPUPMENU_H
#define VRBPOPUPMENU_H

#include <QMenu>

class QTreeWidgetItem;

class VRBPopupMenu : public QMenu
{
    Q_OBJECT

public:
    VRBPopupMenu(QWidget *parent);
    ~VRBPopupMenu();

    QTreeWidgetItem *getItem();
    void setItem(QTreeWidgetItem *);

private:
    QTreeWidgetItem *item;
};
#endif
