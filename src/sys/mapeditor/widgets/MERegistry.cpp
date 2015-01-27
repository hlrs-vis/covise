/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QTableWidget>
#include <QHBoxLayout>

#include "MERegistry.h"
#include "MEUserInterface.h"
#include "color/MEColorMap.h"

static int last = 0;

/*!
   \class MERegistry
   \brief is class provides a widget for displaying the registry for YAC
*/

/*****************************************************************************
 *
 * Class MERegistry
 *
 *****************************************************************************/

MERegistry::MERegistry(QWidget *parent)
    : QFrame(parent)
{

    QHBoxLayout *hb = new QHBoxLayout(this);

    // create a table for the registry disply
    itemsTable = new QTableWidget(20, 4);

    // set proper table header
    QStringList labels;
    labels << "ClassName"
           << "ID"
           << "Variable"
           << "Value";
    itemsTable->setHorizontalHeaderLabels((labels));

    hb->addWidget(itemsTable, 1);
}

MERegistry *MERegistry::instance()
{
    static MERegistry *singleton = 0;
    if (singleton == 0)
        singleton = new MERegistry();

    return singleton;
}

MERegistry::~MERegistry()
{
}

//!
//! update entry in table
//!
void MERegistry::updateEntry(const QString &cname, int id,
                             const QString &name, const QString &value)
{

    // look if registry entry already exist
    QTableWidgetItem *item = getEntry(cname, id, name, value);

    // add new entry
    if (item == NULL)
    {
        QTableWidgetItem *item1 = new QTableWidgetItem(cname);
        QTableWidgetItem *item2 = new QTableWidgetItem(QString::number(id));
        QTableWidgetItem *item3 = new QTableWidgetItem(name);
        QTableWidgetItem *item4 = new QTableWidgetItem(value);

        itemsTable->setItem(last, 0, item1);
        itemsTable->setItem(last, 1, item2);
        itemsTable->setItem(last, 2, item3);
        itemsTable->setItem(last, 3, item4);

        last++;
    }

    // update current entry
    else
        item->setText(value);

    if (cname == "CMap")
        MEUserInterface::instance()->getColorMap()->updateColorMap(name, value);

    itemsTable->resizeColumnsToContents();
    itemsTable->resizeRowsToContents();
}

//!
//! remove entry in table
//!
void MERegistry::removeEntry(const QString &cname, int id,
                             const QString &name, const QString &value)
{

    // look if registry entry already exist
    QTableWidgetItem *item = getEntry(cname, id, name, value);

    if (item == NULL)
        MEUserInterface::instance()->printMessage("Register entry to delete not found ");

    else
    {
        int row = itemsTable->row(item);
        itemsTable->takeItem(row, 0);
        itemsTable->takeItem(row, 1);
        itemsTable->takeItem(row, 2);
        itemsTable->takeItem(row, 3);
        last--;
    }

    itemsTable->resizeColumnsToContents();
    itemsTable->resizeRowsToContents();
}

//!
//! get the entry for a given registry record
//!
QTableWidgetItem *MERegistry::getEntry(QString cname, int id, QString name, QString)
{
    // find all items containing name
    QList<QTableWidgetItem *> list3 = itemsTable->findItems(name, Qt::MatchExactly);

    // check if other components will fit
    // return current value item
    foreach (QTableWidgetItem *it, list3)
    {
        int row = itemsTable->row(it);
        if (itemsTable->item(row, 0)->text() == cname && itemsTable->item(row, 1)->text() == QString::number(id))
            return itemsTable->item(row, 3);
    }

    return NULL;
}
