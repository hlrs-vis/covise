/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/********************************************************************************
 ** Form generated from reading ui file 'clientDialog.ui'
 **
 ** Created: Thu 15. Mar 14:29:24 2007
 **      by: Qt User Interface Compiler version 4.2.2
 **
 ** WARNING! All changes made in this file will be lost when recompiling ui file!
 ********************************************************************************/

#ifndef REMOTECLIENTDIALOG_H
#define REMOTECLIENTDIALOG_H

#include <QVariant>
#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QDialog>
#include <QDialogButtonBox>
#include <QListWidget>

class Ui_RemoteClients : public QDialog
{

public:
    QListWidget *listWidget;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *RemoteClients)
    {
        RemoteClients->setObjectName(QString::fromUtf8("RemoteClients"));
        RemoteClients->setWindowModality(Qt::ApplicationModal);
        QSizePolicy sizePolicy(static_cast<QSizePolicy::Policy>(0), static_cast<QSizePolicy::Policy>(0));

        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(RemoteClients->sizePolicy().hasHeightForWidth());
        RemoteClients->setSizePolicy(sizePolicy);
        RemoteClients->setWindowFlags(Qt::MSWindowsFixedSizeDialogHint);
        listWidget = new QListWidget(RemoteClients);
        listWidget->setObjectName(QString::fromUtf8("listWidget"));
        listWidget->setGeometry(QRect(10, 10, 271, 401));
        buttonBox = new QDialogButtonBox(RemoteClients);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setGeometry(QRect(290, 10, 81, 241));
        buttonBox->setOrientation(Qt::Vertical);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel | QDialogButtonBox::NoButton | QDialogButtonBox::Ok);

        retranslateUi(RemoteClients);

        QSize size(385, 420);
        size = size.expandedTo(RemoteClients->minimumSizeHint());
        RemoteClients->resize(size);

        QMetaObject::connectSlotsByName(RemoteClients);
    } // setupUi

    void retranslateUi(QDialog *RemoteClients)
    {
        RemoteClients->setWindowTitle(QApplication::translate("RemoteClients", "Remote Clients", 0));
        Q_UNUSED(RemoteClients);
    } // retranslateUi

    QString mSelectedClient;

    void setClients(QStringList list)
    {
        this->listWidget->clear();
        this->listWidget->reset();
        this->listWidget->addItems(list);
    };
};

namespace Ui
{
class RemoteClients : public Ui_RemoteClients
{
};
} // namespace Ui
#endif // REMOTECLIENTDIALOG_H
