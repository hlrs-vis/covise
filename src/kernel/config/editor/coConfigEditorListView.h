/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGEDITORLISTVIEW_H
#define COCONFIGEDITORLISTVIEW_H

#include <qdom.h>
#include <q3listview.h>
//Added by qt3to4:
#include <Q3PopupMenu>

#include "coDomListViewItem.h"
#include <util/coTypes.h>

class QPoint;
class Q3PopupMenu;

class CONFIGEDITOREXPORT coConfigEditorListView : public Q3ListView
{

    Q_OBJECT

public:
    coConfigEditorListView(const QDomNode &root,
                           QWidget *parent = 0,
                           const char *name = 0);

    ~coConfigEditorListView();

public slots:
    void requestContextMenu(Q3ListViewItem *item, const QPoint &pos, int c);
    void editCurrentItem();
    void newItem();

private:
    void buildTree();
    Q3ListViewItem *buildTreeBranch(Q3ListViewItem *parent, QDomElement node);
    Q3PopupMenu *makeContextMenu();

    QDomNode root;

    Q3PopupMenu *contextMenu;
    DomListViewItem *currentItem;
};

#endif
