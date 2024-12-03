/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TUI_MAINFRAME_H
#define CO_TUI_MAINFRAME_H

#include <list>
#include <set>
#include <QMainWindow>
#include <QFrame>
#include <QFont>

class QGridLayout;
class QWidget;
class QTimer;
class QDialog;
class QTabWidget;
class QSplitter;
class QSocketNotifier;
class QFrame;

namespace covise
{
class Message;
}

#include <tui/TUIElement.h>
#include <tui/TUIMain.h>

class TUIMainFrame :
    public QFrame
    , public TUIMain
{
    Q_OBJECT

public:
    TUIMainFrame(QWidget *parent = nullptr, QTabWidget *mainFolder=nullptr);

    ~TUIMainFrame();

    QSplitter *split, *central;
    QFont mainFont;

    void removeElement(TUIElement *e) override;
    TUIMainFrame *getMain()
    {
        return this;
    }

signals:
    void removeTabletUI();

protected:
    void closeEvent(QCloseEvent *) override;
    void notifyRemoveTabletUI() override;
    void registerElement(TUIElement *e, QWidget *parent) override;

private slots:
    void timerDone();
    void closeServer();
    void processMessages();
    bool handleClient(covise::Message *msg) override;

private:
    int firstTabFolderID = -1;
    QTabWidget *mainFolder = nullptr;

    TUIElement *createElement(int id, TabletObjectType type, QWidget *w, int parent, QString name);


    QAction *_exit = nullptr, *_help = nullptr;
};
#endif
