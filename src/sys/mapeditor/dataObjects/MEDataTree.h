/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_DATATREE_H
#define ME_DATATREE_H

#include <QTreeWidget>

class QTreeWidgetItem;

class MEDataTree;
class MEDataObject;
class MEDataArray;
class MEDataPort;

namespace covise
{
class coRecvBuffer;
}

class MEDataTreeItem : public QTreeWidgetItem
{

public:
    MEDataTreeItem(MEDataTree *, const QString &, const QColor &color);
    MEDataTreeItem(MEDataTreeItem *, const QString &);
    MEDataTreeItem(MEDataTreeItem *, const QString &, MEDataPort *port);
    MEDataTreeItem(MEDataTreeItem *, const QString &, int);

    ~MEDataTreeItem();

    void updateItem();
    void showItemContent();

    MEDataObject *getDataObject()
    {
        return m_dataObject;
    };
    MEDataArray *getDataArray()
    {
        return m_dataArray;
    };

    void setDataObject(MEDataObject *obj)
    {
        m_dataObject = obj;
    };
    void setDataArray(MEDataArray *array)
    {
        m_dataArray = array;
    };

    void setObjType(int type)
    {
        m_dataObjectType = type;
    };
    int getObjType()
    {
        return m_dataObjectType;
    };

    int getIndex()
    {
        return m_index;
    };
    QColor getColor()
    {
        return m_color;
    };
    MEDataPort *getPort()
    {
        return m_port;
    };

private:
    MEDataObject *m_dataObject;
    MEDataArray *m_dataArray;
    MEDataPort *m_port;

    int m_dataObjectType; // POINTER (0) or ARRAY (1)
    int m_index;

    QColor m_color;
};

class MEDataTree : public QTreeWidget
{
    Q_OBJECT

public:
    MEDataTree(QWidget *parent = 0);
    ~MEDataTree();

    static MEDataTree *instance();

    void init();
    int getDepth(QTreeWidgetItem *);

    MEDataTreeItem *lostfound;
    MEDataTreeItem *findObject(int, int, int);
    MEDataTreeItem *findArray(MEDataTreeItem *, int);
    MEDataTreeItem *search(QTreeWidgetItem *);

private:
    int m_id1, m_id2, m_id3;

private slots:

    void doubleClicked(QTreeWidgetItem *, int);
    void activated(QTreeWidgetItem *, int);
    void collapsed(QTreeWidgetItem *);
    void expanded(QTreeWidgetItem *);
};
#endif
