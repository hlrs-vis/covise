/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "UIGraphicalInputDialog.h"

#include <qapplication.h>
#include <qgroupbox.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qlineedit.h>
#include <qptrlist.h>
#include <qpushbutton.h>
#include <qstringlist.h>

UIGraphicalInputDialog::UIGraphicalInputDialog(QWidget *parent)
    : QDialog(parent)
{
}

UIGraphicalInputDialog::~UIGraphicalInputDialog()
{
}

QMap<QString, QString> UIGraphicalInputDialog::getUserInputs(const QStringList &inputList)
{

    QGridLayout *layout = new QGridLayout(this, 2, 2);
    layout->setAutoAdd(false);

    QGroupBox *editsBox = new QGroupBox(2, Qt::Horizontal, this);

    QStringList names;
    QPtrList<QLineEdit> edits;

    QString inputMask;

    for (QStringList::const_iterator i = inputList.begin(); i != inputList.end(); ++i)
    {

        QString name = *i;
        QString type = *(++i);
        QString defaultValue = *(++i);

        new QLabel(name + " (" + type + ") " + (defaultValue.isEmpty() ? "" : QString("[%1]").arg(defaultValue).latin1()), editsBox);

        if (type == "int")
        {
            inputMask = "#00000000000000000000000000000000000000000000000000";
        }
        else
        {
            inputMask = "";
        }

        QLineEdit *edit = new QLineEdit(defaultValue, inputMask, editsBox);

        if (type == "passwd")
        {
            edit->setEchoMode(QLineEdit::Password);
        }

        names.append(name);
        edits.append(edit);
    }

    layout->addMultiCellWidget(editsBox, 1, 1, 1, 2);

    QPushButton *okButton = new QPushButton(tr("OK"), this);
    QPushButton *cancelButton = new QPushButton(tr("Cancel"), this);

    layout->addWidget(okButton, 2, 1);
    layout->addWidget(cancelButton, 2, 2);

    connect(okButton, SIGNAL(clicked()),
            this, SLOT(accept()));

    connect(cancelButton, SIGNAL(clicked()),
            this, SLOT(reject()));

    exec();

    QMap<QString, QString> returnValues;

    if (result() == QDialog::Accepted)
    {
        QStringList::iterator n = names.begin();
        QPtrList<QLineEdit>::iterator e = edits.begin();

        while (n != names.end())
        {
            returnValues.insert(*n, (*e)->text());
            ++n;
            ++e;
        }
    }
    else
    {
        qApp->exit(1);
    }

    return returnValues;
}
