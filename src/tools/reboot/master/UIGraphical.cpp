/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <UIGraphical.h>
#include <UIGraphicalInputDialog.h>

#include "RemoteRebootMaster.h"

#include <qapplication.h>
#include <qgroupbox.h>
#include <qlayout.h>
#include <qlistbox.h>
#include <qpushbutton.h>

#include <iostream>
using namespace std;

#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

UIGraphical::UIGraphical(RemoteRebootMaster *master)
    : UI(master)
{
}

UIGraphical::~UIGraphical()
{
}

QString UIGraphical::getUserInput(const QString &name, const QString &type, const QString &defaultValue)
{

    QStringList list;
    list.append(name);
    list.append(type);
    list.append(defaultValue);

    return getUserInputs(list)["name"];
}

QMap<QString, QString> UIGraphical::getUserInputs(const QStringList &inputList)
{

    UIGraphicalInputDialog dialog;
    return dialog.getUserInputs(inputList);
}

void UIGraphical::exec()
{

    QGridLayout *layout = new QGridLayout(this, 2, 2);
    layout->setAutoAdd(false);

    QGroupBox *listGroupBox = new QGroupBox(1, Qt::Horizontal, tr("Select Boot Image"), this);
    layout->addMultiCellWidget(listGroupBox, 0, 0, 0, 1);

    QStringList items = master->getBootEntries();
    int defaultItem = master->getDefaultBoot();

    QListBox *list = new QListBox(listGroupBox);
    list->insertStringList(items);
    list->setSelected(defaultItem, true);
    list->centerCurrentItem();

    int width = (int)min(QApplication::desktop()->width() - 50, list->maxItemWidth() + 5);
    int height = (int)min(QApplication::desktop()->height() - 50, list->itemHeight() * items.size() + 5);

    //cerr << list->maxItemWidth() << " " << list->itemHeight() * items.size() << endl;
    list->setMinimumSize(width, height);

    QPushButton *rebootButton = new QPushButton(tr("Reboot"), this);
    QPushButton *cancelButton = new QPushButton(tr("Cancel"), this);

    layout->addWidget(rebootButton, 1, 0);
    layout->addWidget(cancelButton, 1, 1);

    connect(rebootButton, SIGNAL(clicked()),
            this, SLOT(accept()));

    connect(cancelButton, SIGNAL(clicked()),
            this, SLOT(reject()));

    QDialog::exec();

    if (result() == Accepted)
    {

        master->setDefaultBoot(list->index(list->selectedItem()));
        master->reboot();

        qApp->exit(0);
    }
    else
    {
        qApp->exit(-1);
    }
}
