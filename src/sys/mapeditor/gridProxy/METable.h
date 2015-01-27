/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_TABLE_H
#define ME_TABLE_H

#include <QTableWidget>

class QKeyEvent;

class METable : public QTableWidget
{

    Q_OBJECT

public:
    METable(QWidget *parent = 0);
    METable(int numRows, int numCols, QWidget *parent = 0);

protected:
    void keyPressEvent(QKeyEvent *e);
};
#endif
