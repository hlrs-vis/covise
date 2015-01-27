/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coConfigEditorListView.h"
//#include "coConfigEditorListView.moc"

#include <qpoint.h>
#include <q3popupmenu.h>
#include <QMenuItem>

coConfigEditorListView::coConfigEditorListView(const QDomNode &root,
                                               QWidget *parent,
                                               const char *name)
    : Q3ListView(parent, name)
{

    contextMenu = 0;
    currentItem = 0;

    this->root = root;
    buildTree();

    connect(this, SIGNAL(contextMenuRequested(Q3ListViewItem *, const QPoint &, int)),
            this, SLOT(requestContextMenu(Q3ListViewItem *, const QPoint &, int)));
}

coConfigEditorListView::~coConfigEditorListView()
{
}

void coConfigEditorListView::buildTree()
{

    addColumn("Node");
    addColumn("Value");
    addColumn("Config");
    //addColumn("Config Name");
    setAllColumnsShowFocus(true);

    DomListViewItem *rootItem = DomListViewItem::getInstance(root.toElement(), this);
    insertItem(rootItem);
    rootItem->getPrototype()->updateColors();

    QDomNode child = root.firstChild();

    while (!child.isNull())
    {
        if (child.isElement())
        {
            buildTreeBranch(rootItem, child.toElement());
        }
        child = child.nextSibling();
    }

    setOpen(firstChild(), true);
}

Q3ListViewItem *coConfigEditorListView::buildTreeBranch(Q3ListViewItem *parent,
                                                        QDomElement node)
{

    DomListViewItem *item = DomListViewItem::getInstance(node, parent);

    if (node.hasChildNodes())
    {
        QDomNode child = node.firstChild();
        while (!child.isNull())
        {
            if (child.isElement())
            {
                buildTreeBranch(item, child.toElement());
            }
            child = child.nextSibling();
        }
    }

    return item;
}

void coConfigEditorListView::requestContextMenu(Q3ListViewItem *item,
                                                const QPoint &pos,
                                                int)
{

    //cerr << "coConfigEditorListView::contextMenuRequested info: showing menu" << endl;

    if (!contextMenu)
        contextMenu = makeContextMenu();

    currentItem = (DomListViewItem *)item;

    QMenu *scopeMenu = contextMenu->findItem(2)->menu();
    scopeMenu->setItemEnabled(3, coConfig::getInstance()->isAdminMode());
    scopeMenu->setItemEnabled(4, coConfig::getInstance()->isAdminMode());

    contextMenu->popup(pos);
}

Q3PopupMenu *coConfigEditorListView::makeContextMenu()
{

    Q3PopupMenu *menu = new Q3PopupMenu(this);
    Q3PopupMenu *scopeMenu = new Q3PopupMenu(menu);

    scopeMenu->insertItem("Global", 1);
    scopeMenu->insertItem("Host", 2);

    //menu->insertItem("New Variable", this, SLOT(newItem()), 0, 3);
    menu->insertItem("Edit Variable", this, SLOT(editCurrentItem()), 0, 1);
    menu->insertItem("Move Scope", scopeMenu, 2);

    return menu;
}

void coConfigEditorListView::editCurrentItem()
{
    if (currentItem->renameEnabled(1))
    {
        currentItem->startRename(1);
    }
    else if (currentItem->renameEnabled(0))
    {
        currentItem->startRename(0);
    }
}

void coConfigEditorListView::newItem()
{

    QDomElement node = root.ownerDocument().createElement(tr("New Item"));
    node.setAttribute("type", "attribute");
    node.setAttribute("value", QString());
    node.setAttribute("scope", currentItem->getScope());
    node.setAttribute("config", currentItem->getConfigScope());
    node.setAttribute("configname", currentItem->getConfigName());

    //cerr << currentItem->getScope() << "  " << currentItem->getConfigScope() << endl;

    DomListViewItem::getInstance(node, currentItem);

    update();
}
