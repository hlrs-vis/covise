/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SETTINGSDIALOG_H
#define SETTINGSDIALOG_H

#include <QDialog>
#include "ui_SettingsDialog.h"

class SettingsDialog : public QDialog
{
    Q_OBJECT

public:
    SettingsDialog(QWidget *parent = 0);
    ~SettingsDialog();
    void setupConnects();
    void setupLineEdits();
    void setupGui();
    void update();

public slots:
    //allgemein
    void btnBlendingPathClicked();
    void btnFragFileClicked();
    void btnVertexFileClicked();
    void btnOkClicked();
    void btnCancelClicked();

private:
    Ui::SettingsDialog ui;
};

#endif // SETTINGSDIALOG_H
