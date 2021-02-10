/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifdef HAVE_WIRINGPI
#include "Thyssen.h"
#endif
#ifndef TUIAPPLICATION_H
#define TUIAPPLICATION_H
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

class TUIMainWindow;
namespace covise
{
class ServerConnection;
class Connection;
class ConnectionList;
class Message;
class TokenBuffer;
}

class TUITab;

#include "TUIElement.h"

class TUIMainWindow :
#ifdef TABLET_PLUGIN
    public QFrame
#else
    public QMainWindow
#endif
{
    Q_OBJECT

public:
    TUIMainWindow(QWidget *parent = nullptr, QTabWidget *mainFolder=nullptr);

    ~TUIMainWindow();

    void setPort(int port);

    static TUIMainWindow *getInstance();
    QGridLayout *mainGrid;
    QSplitter *split, *central;
    QFrame *mainFrame;
    QFont mainFont;
#ifdef HAVE_WIRINGPI
   ThyssenPanel * thyssenPanel;
#endif

    bool serverRunning();
    int openServer();
    void addElement(TUIElement *);
    void removeElement(TUIElement *e);
    void deActivateTab(TUITab *activedTab);
    void addElementToLayout(TUIElement *);
    void send(covise::TokenBuffer &tb);
    TUIMainWindow *getMain()
    {
        return this;
    }
    TUIElement *getElement(int ID);
    QWidget *getWidget(int ID);

    covise::Connection *toCOVERSG = nullptr;

protected:
    void closeEvent(QCloseEvent *);
    std::vector<TUIElement *> elements; // sorted by ID
    std::set<TUITab *> tabs;

private slots:
    void timerDone();
    void fontCB(const QString &);
    void styleCB(const QString &);
    void about();
    void closeServer();
    void processMessages();
    bool handleClient(covise::Message *msg);

private:
#ifndef TABLET_PLUGIN
    void createMenubar();
    void createToolbar();
#else
    int firstTabFolderID = -1;
#endif
    QTabWidget *mainFolder = nullptr;

    static TUIMainWindow *appwin;
    TUIElement *createElement(int id, TabletObjectType type, QWidget *w, int parent, QString name);

    int smsg;
    int port;

    QSocketNotifier *serverSN = nullptr, *clientSN = nullptr;
    QTimer *timer = nullptr;
    QAction *_exit = nullptr, *_help = nullptr;

    const covise::ServerConnection *sConn = nullptr;
    const covise::Connection *clientConn = nullptr;
    covise::ConnectionList connections;
    covise::Message *msg = nullptr;
    QTimer *m_periodictimer = nullptr;

    covise::ServerConnection *sgConn = nullptr;
    int numberOfColumns = 5;
};
#endif
