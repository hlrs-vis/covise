/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_REGISTER_H
#define ME_REGISTER_H

#include <QFrame>

class QTableWidget;
class QTableWidgetItem;

//================================================
class MERegistry : public QFrame
//================================================
{
    Q_OBJECT

public:
    MERegistry(QWidget *parent = 0);
    ~MERegistry();

    static MERegistry *instance();

    void updateEntry(const QString &, int, const QString &, const QString &);
    void removeEntry(const QString &, int, const QString &, const QString &);

private:
    QTableWidgetItem *getEntry(QString, int, QString, QString);
    QTableWidget *itemsTable;
};
#endif
