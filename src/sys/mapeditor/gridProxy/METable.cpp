/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QKeyEvent>

#include "gridProxy/METable.h"

/*****************************************************************************
 *
 * Class METable
 *
 *****************************************************************************/
METable::METable(QWidget *parent)
    : QTableWidget(parent)
{
    setSelectionBehavior(QAbstractItemView::SelectRows);
    setSelectionMode(QAbstractItemView::SingleSelection);
}

METable::METable(int numRows, int numCols, QWidget *parent)
    : QTableWidget(numRows, numCols, parent)
{
    setSelectionBehavior(QAbstractItemView::SelectRows);
    setSelectionMode(QAbstractItemView::SingleSelection);
}

/*
 * Delete the selected row when <delete> is pressed
 */
void METable::keyPressEvent(QKeyEvent *e)
{

    int n = 0;
    QList<QTableWidgetItem *> list = selectedItems();

    if (!list.isEmpty() && e->key() == Qt::Key_Delete)
    {
        n = row(list[0]);
        removeRow(n);
    }
    /*switch (e->key()) {

       case Qt::Key_Delete:
          while (n < rowCount()) {
             if (isRowSelected(n)) {
                removeRow(n);
                break;
             }
             n++;
          }
          break;

       default:
          break;
   }*/
}
