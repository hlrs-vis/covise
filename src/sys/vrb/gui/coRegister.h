/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COREGISTER_H
#define COREGISTER_H

#include <QFrame>

class QTreeWidget;

class coRegister : public QFrame
{
    Q_OBJECT

public:
    coRegister(QWidget *parent = 0, const char *name = 0);
    ~coRegister();

    QTreeWidget *table;

    void updateEntry(QString, int, QString, QString);
    void removeEntry(QString, int, QString);
    void removeEntries(int);

private:
    int getEntry(QString, int, QString);
};
#endif
