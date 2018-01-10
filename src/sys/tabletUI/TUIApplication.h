/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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
    TUIMainWindow(QWidget *parent = 0);

    ~TUIMainWindow();

    void setPort(int port);

    static TUIMainWindow *getInstance();
    QGridLayout *mainGrid;
    QSplitter *split, *central;
    QFrame *mainFrame;
    QFont mainFont;

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

    covise::Connection *toCOVERTexture;
    covise::Connection *toCOVERSG;

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
#endif

    static TUIMainWindow *appwin;
    TUIElement *createElement(int id, int type, QWidget *w, int parent, QString name);
    void createTabWidget(QSplitter *);

    int smsg;
    int port;
    int lastID;

    QSocketNotifier *serverSN, *clientSN;
    QTimer *timer;
    QAction *_exit, *_help;

    covise::ServerConnection *sConn;
    covise::Connection *clientConn;
    covise::ConnectionList *connections;
    covise::Message *msg;
    TUIElement *lastElement;
    QTimer *m_periodictimer;

    covise::ServerConnection *texConn;
    covise::ServerConnection *sgConn;
    int numberOfColumns = 5;
};
#endif
