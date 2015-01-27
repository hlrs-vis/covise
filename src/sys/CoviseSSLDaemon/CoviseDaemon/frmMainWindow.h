/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/********************************************************************************
** Form generated from reading ui file 'MainWindow.ui'
**
** Created: Thu 29. May 12:18:38 2008
**      by: Qt User Interface Compiler version 4.3.0
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef FRMMAINWINDOW_H
#define FRMMAINWINDOW_H

#include <QObject>
#include <QVariant>
#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QColumnView>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QListView>
#include <QTextEdit>
#include <QListWidget>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QStatusBar>
#include <QTabWidget>
#include <QWidget>
#include <QPushButton>

#include "SSLDaemon.h"

class SSLDaemon;

class frmMainWindow : public QMainWindow
{
    Q_OBJECT
public:
    frmMainWindow(QApplication *app);
    ~frmMainWindow();
    void setupUi();
    void retranslateUi();
    void connectSlots();
    void setPort(int port);
    QTextEdit *getLog();
    QListWidget *getHostList();
    QListWidget *getUserList();

protected:
    QAction *actionAbout;
    QAction *actionExit;
    QAction *actionSave;
    QAction *actionLoad;
    QAction *actionRespawn;
    QWidget *centralwidget;
    QTabWidget *tabWidCovise;
    QWidget *tabMonitor;
    QGroupBox *grpLog;
    QTextEdit *lstLog;
    QWidget *tabConfig;
    QGroupBox *grpPartner;
    QListWidget *columnPartners;
    QGroupBox *grpHost;
    QListWidget *lstHost;
    QGroupBox *grpCovise;
    QLabel *lblPort;
    QLineEdit *txtPort;
    QPushButton *btnOnOff;
    QMenuBar *menubar;
    QMenu *menuFile;
    QMenu *menuHelp;
    QStatusBar *statusbar;
    SSLDaemon *mDaemon;

protected slots:
    void handleAboutShow();
    void handleRespawn();

public slots:
    void handleOnOff(bool);

private:
    QApplication *mApplication;
    bool mIsOn;
};

#endif // FRMMAINWINDOW_H
