/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRBAPPLICATION_H
#define VRBAPPLICATION_H

#include <QCloseEvent>
#include <QGridLayout>
#include <QFrame>
#include <QMainWindow>

class QDialog;
class QTabWidget;
class QTextEdit;
class QTreeWidget;
class QTreeWidgetItem;
class QListWidget;
class QSplitter;

class ApplicationWindow;

class VRBSClient;
class VRBPopupMenu;
class VRBFileDialog;
class VRBCurve;
class coRegister;

extern ApplicationWindow *appwin;

class ApplicationWindow : public QMainWindow
{
    Q_OBJECT
    enum
    {
        MaxCurve = 5
    };

public:
    ApplicationWindow();
    ~ApplicationWindow();
    QTreeWidget *table;
    QTextEdit *msg;
    coRegister *registry;
    void addMessage(char *);
    void createCurves(VRBSClient *);
    void removeCurves(VRBSClient *);

protected:
    void closeEvent(QCloseEvent *);

private slots:
    void choose();
    void about();
    void newDoc();
    void timerDone();
    void showMsg();
    void deleteItem();
    void configItem();
    void popupCB(QTreeWidgetItem *item, const QPoint &, int);
    void showBPS(QTreeWidgetItem *item);
    void setStyle(const QString &);

private:
    void createMenubar();
    void createToolbar();
    void createTabWidget(QSplitter *);

    QAction *showMessageAreaAction;
    QDialog *dialog;
    QListWidget *plugins;
    QFrame *msgFrame;
    QTimer *timer, *start;
    QTabWidget *tabs;

    QGridLayout *grid[MaxCurve];
    QFrame *wtab[MaxCurve];

    VRBFileDialog *browser;
    VRBPopupMenu *popup;
    VRBSClient *currClient;
    VRBCurve *curve[4];
};
#endif
