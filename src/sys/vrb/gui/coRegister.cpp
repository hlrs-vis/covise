/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QTreeWidget>
#include <QHeaderView>
#include <QString>
#include <QFrame>
#include <QTextEdit>
#include <QGridLayout>

#include "coRegister.h"

//======================================================================

coRegister::coRegister(QWidget *parent, const char * /*name*/)
    : QFrame(parent)
{
    auto layout = new QGridLayout(this);
    setLayout(layout);

    table = new QTreeWidget(this);
    layout->addWidget(table);
    table->setSortingEnabled(true);
    // connect (table,  SIGNAL (currentChanged ( int , int )),
    //        this,   SLOT(tableCB ( int , int )));

    QStringList headerLabels;
    headerLabels << tr("ClassName") << tr("ID") << tr("Variable") << tr("Value");
    table->setHeaderLabels(headerLabels);
    table->setColumnWidth(1, 20);
    table->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

#if 0
   //no aequivalent present in qt4
   table->setColumnReadOnly(0, true);
   table->setColumnReadOnly(1, true);
   table->setColumnReadOnly(2, true);
#endif
}

coRegister::~coRegister()
{
}

/****************************************************************+
 * update entry on table
 */

void coRegister::updateEntry(QString cname, int id, QString name, QString value)
{

    // look if registry entry already exist
    int found = getEntry(cname, id, name);

    if (found == -1)
    {
        // add entry

        QStringList entries;
        entries << cname << QString::number(id) << name << value;
        new QTreeWidgetItem(table, entries);
    }

    else
    {
        // update entry
        QTreeWidgetItem *entry = table->invisibleRootItem()->child(found);
        entry->setText(1, QString().setNum(id));
        entry->setText(2, name);
        entry->setText(3, value);
    }
}

/****************************************************************+
 * remove entry in table
 */

void coRegister::removeEntry(QString cname, int id, QString name)
{
    // look if registry entry already exist
    int found = getEntry(cname, id, name);

    if (found == 0)
    {
        //mw->infoWin->append("Register entry to delete not found");
    }

    else
    {
        table->invisibleRootItem()->takeChild(found);
    }
}

void coRegister::removeEntries(int id)
{
    QList<QTreeWidgetItem *> matches = table->findItems(QString::number(id),
                                                        Qt::MatchFixedString, 1);

    foreach (QTreeWidgetItem *match, matches)
    {
        int found = table->invisibleRootItem()->indexOfChild(match);
        table->invisibleRootItem()->takeChild(found);
    }
}

/****************************************************************+
 * get the entry for a given registry record
 */

int coRegister::getEntry(QString cname, int id, QString name)
{
    QList<QTreeWidgetItem *> matches = table->findItems(QString::number(id),
                                                        Qt::MatchFixedString, 1);

    foreach (QTreeWidgetItem *match, matches)
        if (match->text(0) == cname && match->text(2) == name)
            return table->invisibleRootItem()->indexOfChild(match);

    return -1;
}
