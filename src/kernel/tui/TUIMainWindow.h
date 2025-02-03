#ifndef CO_TUI_MAINWINDOW_H
#define CO_TUI_MAINWINDOW_H

#include "export.h"


#ifdef HAVE_WIRINGPI
#include "Thyssen.h"
#endif

#include <list>
#include <set>
#include <QMainWindow>
#include <QFrame>
#include <QFont>
#ifdef HAVE_WIRINGPI
#include <ThyssenButton.h>
#endif

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

#include "TUIElement.h"
#include "TUIMain.h"

class TUIEXPORT TUIMainWindow :
    public QMainWindow
    , public TUIMain
{
    Q_OBJECT

public:
    TUIMainWindow(QWidget *parent = nullptr, QTabWidget *mainFolder=nullptr);

    ~TUIMainWindow();

    QSplitter *split, *central;
    QFont mainFont;
#ifdef HAVE_WIRINGPI
   ThyssenPanel * thyssenPanel;
   QTimer *thyssenTimer;
#endif

    TUIMainWindow *getMain()
    {
        return this;
    }

signals:
    void removeTabletUI();

public slots:
    void storeGeometry();

protected:
    void closeEvent(QCloseEvent *) override;

private slots:
    void timerDone();
#ifdef HAVE_WIRINGPI
    void thyssenTimerDone();
#endif
    void fontCB(const QString &);
    void styleCB(const QString &);
    void about();
    void closeServer();
    void processMessages();
    bool handleClient(covise::Message *msg) override;
    void notifyRemoveTabletUI() override;

private:
    void createMenubar();
    QToolBar *createToolbar();
    QTabWidget *mainFolder = nullptr;

    QAction *_exit = nullptr, *_help = nullptr;
    QToolBar *toolbar = nullptr;
    bool toolbarVisible = true;
};
#endif
