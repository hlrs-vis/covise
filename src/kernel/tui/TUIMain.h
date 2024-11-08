/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef CO_TUI_MAIN_H
#define CO_TUI_MAIN_H

#include "export.h"

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

class TUIMain;
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

class TUIEXPORT TUIMain
{
public:
    static TUIMain *getInstance();
    void addElement(TUIElement *);
    virtual void removeElement(TUIElement *e);
    void addElementToLayout(TUIElement *);
    void send(covise::TokenBuffer &tb);

    void setPort(int port);

    void closeServer();
    void processMessages();
    bool handleClient(covise::Message *msg);

    bool serverRunning();
    int openServer();
    void deActivateTab(TUITab *activedTab);
    TUIElement *getElement(int ID);
    TUIElement *createElement(int id, TabletObjectType type, QWidget *w, int parent, QString name);
    QWidget *getWidget(int ID);

    covise::Connection *toCOVERSG = nullptr;

protected:
    TUIMain(QWidget *w);

    void timerDone();
    virtual void notifyRemoveTabletUI();
    virtual void registerElement(TUIElement *e, QWidget *parent);

    QWidget *widget = nullptr;
    QTimer *timer = nullptr;

    int smsg = 0;
    int port = 31802;

    QGridLayout *mainGrid = nullptr;
    QFrame *mainFrame = nullptr;
    std::set<TUITab *> tabs;
    QSocketNotifier *serverSN = nullptr, *clientSN = nullptr;
    QTimer *m_periodictimer = nullptr;

    const covise::Connection *clientConn = nullptr;
    std::vector<TUIElement *> elements; // sorted by ID
    const covise::ServerConnection *sConn = nullptr;
    covise::ConnectionList connections;
    covise::Message *msg = nullptr;
    covise::ServerConnection *sgConn = nullptr;

    int numberOfColumns = 5;
private:
    static TUIMain *tuimain;
};
#endif
