/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/********************************************************************************
** Form generated from reading ui file 'RequestDialog.ui'
**
** Created: Thu 29. May 12:20:13 2008
**      by: Qt User Interface Compiler version 4.3.0
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef FRMREQUESTDIALOG_H
#define FRMREQUESTDIALOG_H

#include <QVariant>
#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QWidget>
#include <QCheckBox>
#include "SSLDaemon.h"

class SSLDaemon;

class frmRequestDialog : public QDialog
{
    Q_OBJECT
public:
    QGroupBox *grpBorder;
    QLabel *lblMessage;
    QWidget *layoutWidget;
    QHBoxLayout *hboxLayout;
    QPushButton *btnAllow;
    QPushButton *btnTempAllow;
    QPushButton *btnDeny;

    enum DialogMode
    {
        MachineLevel,
        SubjectLevel
    };

    frmRequestDialog();
    ~frmRequestDialog();

    void setupUi(QDialog *frmRequestConfirm);
    void retranslateUi(QDialog *frmRequestConfirm);
    void setHandleObject(SSLDaemon *object);
    void setDaemon(SSLDaemon *ssl);
    void setMessage(const char *message, DialogMode mode);
    DialogMode getCurrentMode();

public slots:
    void handlePermanent(bool);
    void handleTemporary(bool);
    void handleDeny(bool);

private:
    SSLDaemon *mSSLObject;
    DialogMode mCurrentMode;
};

#endif // FRMREQUESTDIALOG_H
