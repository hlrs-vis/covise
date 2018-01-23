/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_SETTINGS_DIALOG
#define ME_SETTINGS_DIALOG

#include <QDialog>

class QCheckBox;
class QComboBox;
class QLineEdit;
class QPushButton;
class QWidget;
class QVBoxLayout;

//================================================
class MESessionSettings : public QDialog
//================================================
{
    Q_OBJECT

public:
    MESessionSettings(QWidget *parent = 0, Qt::WindowFlags f = 0);
    ~MESessionSettings();

private:
    QCheckBox *errorHandlingCheckBox;
    QCheckBox *storeWindowConfigBox;
    QCheckBox *hideUnusedModulesBox;
    QCheckBox *autoConnectBox;
    QCheckBox *browserBox;
    QCheckBox *tabletUITabsBox;
    QCheckBox *developerModeCheckBox;
    //QCheckBox * imbeddedRenderBox;

    QComboBox *qtStyleComboBox;

    QLineEdit *autoSaveTimeEdit;
    QLineEdit *undoBufferEdit;
    QLineEdit *highlightColorEdit;

    QPushButton *saveButton;
    QPushButton *cancelButton;
    QPushButton *resetButton;

    void createFormLayout(QVBoxLayout *fbox);
    void initState();

private slots:

    void save();
    void resetValues();
    void cancel();
};
#endif
