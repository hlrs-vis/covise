/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_DATAVIEWER_H
#define ME_DATAVIEWER_H

#include <QWidget>

class QSplitter;
class QStackedWidget;
class QScrollArea;

class MEDataArray;
class MEDataObject;
class MEDataTreeItem;

namespace covise
{
class coRecvBuffer;
}

//================================================
class MEDataViewer : public QWidget
//================================================
{
    Q_OBJECT

public:
    MEDataViewer(QWidget *parent = 0);

    static MEDataViewer *instance();

    void createObject(MEDataTreeItem *item, covise::coRecvBuffer &rb);
    void createArray(MEDataTreeItem *item, covise::coRecvBuffer &rb);
    MEDataObject *createObject(MEDataTreeItem *item);
    MEDataArray *createArray(MEDataTreeItem *item);

    void showObject(MEDataObject *object);
    void showArray(MEDataArray *array);
    void reset();

public slots:

    void spinCB(int);

private:
    int m_maxArray;
    QScrollArea *m_scrolling;
    QSplitter *m_splitter, *m_left, *m_right;
    QStackedWidget *m_infoView;
    QWidget *m_widget;
};
#endif
