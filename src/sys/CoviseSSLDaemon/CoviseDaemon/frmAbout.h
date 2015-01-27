/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/********************************************************************************
** Form generated from reading ui file 'About.ui'
**
** Created: Thu 29. May 12:19:50 2008
**      by: Qt User Interface Compiler version 4.3.0
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef FRMABOUT_H
#define FRMABOUT_H

#include <QVariant>
#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QDialog>
#include <QLabel>
#include <QPushButton>

class frmAbout : public QDialog
{
    Q_OBJECT

public:
    QLabel *label;
    QPushButton *btnClose;

    frmAbout(QApplication *app);
    ~frmAbout();

    void setupUi(QDialog *frmAbout);
    void retranslateUi(QDialog *frmAbout);

protected slots:
    void handleClose(bool);

private:
    void connectSlots();
    QApplication *mApplication;
};

#endif // FRMABOUT_H
