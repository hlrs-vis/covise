/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if 0
#define setfill qtsetfill
#define setprecision qtsetprecision
#define setw qtsetw
#endif
#include "TUIMain.h"
#include <QApplication>
#if 0
#undef setfill
#undef setprecision
#undef setw
#endif


#include <QSocketNotifier>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QCloseEvent>

#include "TUIElement.h"
#define IOMANIPH

#include <QAction>
#include <QComboBox>
#include <QFont>
#include <QImage>
#include <QLabel>
#include <QLayout>
#include <QMenuBar>
#include <QMessageBox>
#include <QPixmap>
#include <QSettings>
#include <QSignalMapper>
#include <QSplitter>
#include <QString>
#include <QStyleFactory>
#include <QTabWidget>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>
#include <QToolTip>
#include <QWidget>
#include <QFont>
#include <QtGlobal>
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <config/CoviseConfig.h>
#include <config/coConfig.h>
#endif


#include "TUITab.h"
#include "TUISGBrowserTab.h"
#include "TUIAnnotationTab.h"
#ifndef WITHOUT_VIRVO
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include "TUIFunctionEditorTab.h"
#endif
#endif
#include "TUIColorTab.h"
#include "TUITabFolder.h"
#include "TUIButton.h"
#include "TUIColorTriangle.h"
#include "TUIColorButton.h"
#include "TUIToggleButton.h"
#include "TUIToggleBitmapButton.h"
#include "TUIComboBox.h"
#include "TUIListBox.h"
#include "TUIFloatSlider.h"
#include "TUISlider.h"
#include "TUINavElement.h"
#include "TUIFloatEdit.h"
#include "TUIProgressBar.h"
#include "TUIIntEdit.h"
#include "TUILabel.h"
#include "TUIFrame.h"
#include "TUIGroupBox.h"
#include "TUIScrollArea.h"
#include "TUISplitter.h"
#include "TUIFileBrowserButton.h"
#include "TUIMap.h"
#include "TUIEarthMap.h"
//#include "TUITextSpinEdit.h"
#include "TUILineEdit.h"
#include "TUITextEdit.h"
#include "TUIPopUp.h"
#ifdef USE_UITOOLS
#include "TUIUITab.h"
#endif
#include "TUIWebview.h"

#include <net/tokenbuffer.h>
#include <net/message_types.h>
#include <util/threadname.h>

#ifndef _WIN32
#include <signal.h>
#include <sys/socket.h>
#endif

#include <cassert>
#ifdef HAVE_WIRINGPI
#include "Thyssen.h"
#include "ThyssenButton.h"
#endif


TUIMain *TUIMain::tuimain = 0;

TUIMain::TUIMain(QWidget *w)
: widget(w)
{
    assert(tuimain == nullptr);
    tuimain = this;
}

TUIMain::~TUIMain()
{
    tuimain = nullptr;
}

TUIMain *TUIMain::getInstance()
{
    assert(tuimain != nullptr);
    return tuimain;
}

void TUIMain::addElement(TUIElement *e)
{
    auto iter = elements.end();
    if (!elements.empty() && e->getID() < elements.back()->getID())
    {
        iter = std::lower_bound(elements.begin(), elements.end(), e->getID(), [](const TUIElement *el, int id){ return el->getID()<id;});
    }
    elements.insert(iter, e);
}

void TUIMain::removeElement(TUIElement *e)
{
    tabs.erase(static_cast<TUITab *>(e));
    /*auto iter = std::lower_bound(elements.begin(), elements.end(), e->getID(), [](const TUIElement* el, int id) { return el->getID()<id; });
    if (iter == elements.end())
        return;
    if (*iter != e)
        return;*/

    auto iter = std::find(elements.begin(), elements.end(), e);
    if (iter == elements.end())
    {
        std::cerr << "element not found in removeElement" << e->getName().toLatin1().toStdString() << std::endl;
        return;

    }
    elements.erase(iter);
}

void TUIMain::addElementToLayout(TUIElement *elem)
{
    if (elem->getWidget())
        mainGrid->addWidget(elem->getWidget(), elem->getXpos(), elem->getYpos());
    else if (elem->getLayout())
        mainGrid->addLayout(elem->getLayout(), elem->getXpos(), elem->getYpos());
}

bool TUIMain::send(covise::TokenBuffer &tb)
{
    if (clientConn == NULL)
        return false;
    covise::Message m(tb);
    m.type = covise::COVISE_MESSAGE_TABLET_UI;
    if (!clientConn->sendMessage(&m))
    {
        std::cerr << "send: could not send message" << std::endl;
        return false;
    }
    return true;
}

void TUIMain::setPort(int p)
{
    port = p;
}

void TUIMain::setFds(int fd, int fdSg)
{
    closeServer();

    port = 0;

    auto preconnectedClientConn = std::make_unique<covise::Connection>(fd);
    preconnectedClientConn->set_sendertype(0);
    auto preconnectedSgConn = std::make_unique<covise::Connection>(fdSg);
    preconnectedSgConn->set_sendertype(0);

    toCOVERSG = std::move(preconnectedSgConn);
    clientConn = connections.add(std::move(preconnectedClientConn)); //add new connection;

    clientSN = new QSocketNotifier(clientConn->get_id(NULL), QSocketNotifier::Read);
    QObject::connect(clientSN, SIGNAL(activated(int)), widget, SLOT(processMessages()));
}

//------------------------------------------------------------------------
// show this message after 2 sec
// wait 2 more sec to disconnect clients or exit
//------------------------------------------------------------------------
void TUIMain::timerDone()
{
    timer->stop();
}

//------------------------------------------------------------------------
void TUIMain::closeServer()
//------------------------------------------------------------------------
{
    delete serverSN;
    serverSN = NULL;

    if (sConn)
    {
        connections.remove(sConn);
        sConn = NULL;
    }

    if (clientConn)
    {
        connections.remove(clientConn);
        clientConn = NULL;
    }

    //remove all UI Elements
    while (!elements.empty())
    {
        TUIElement *ele = &*elements.back();
        delete ele;
    }

    toCOVERSG.reset();

    if (!tabs.empty())
    {
        std::cerr << "TUIMain::closeEvent: not all tabs erased: still " << tabs.size() << " remaining" << std::endl;
    }
    assert(tabs.empty());
}

//------------------------------------------------------------------------
int TUIMain::openServer()
//------------------------------------------------------------------------
{
    connections.remove(sConn);

    //cerr << "listening on port " << port << endl;
// weil unter windows manchmal Messages verloren gehen
// der SocketNotifier wird nicht oft genug aufgerufen)
#if defined(_WIN32) || defined(__APPLE__)
    m_periodictimer = new QTimer;
    QObject::connect(m_periodictimer, SIGNAL(timeout()), widget, SLOT(processMessages()));
    m_periodictimer->start(1000);
#endif

    if (port == 0)
        return 0;

    sConn = connections.tryAddNewListeningConn<covise::ServerConnection>(port, 0, 0);
    if (!sConn)
    {
        return (-1);
    }

    serverSN = new QSocketNotifier(sConn->get_id(NULL), QSocketNotifier::Read);
    QObject::connect(serverSN, SIGNAL(activated(int)), widget, SLOT(processMessages()));
    return 0;
}

//------------------------------------------------------------------------
bool TUIMain::serverRunning()
//------------------------------------------------------------------------
{
    return sConn && sConn->is_connected();
}

bool TUIMain::makeSGConnection(covise::Connection *conn)
{
    if (!conn->is_connected())
    {
        std::cerr << "makeSGConnection: not connected" << std::endl;
        return false;
    }

    // create connections for SceneGraph Browser Thread
    auto sgConn = std::make_unique<covise::ServerConnection>(&port, 0, (covise::sender_type)0);
    sgConn->listen();

    covise::TokenBuffer stb;
    stb << port;
    covise::Message m(stb);
    m.type = covise::COVISE_MESSAGE_TABLET_UI;
    if (!conn->sendMessage(&m))
    {
        std::cerr << "makeSGConnection: could not send port " << port << " to client" << std::endl;
        return false;
    }

    std::cerr << "add connection: waiting for SGBrowser connection on port " << port << std::endl;

    if (sgConn->acceptOne(5) < 0)
    {
        fprintf(stderr, "Could not accept connection to sg port in time %d\n", port);
        return false;
    }
    if (!sgConn->getSocket())
    {
        fprintf(stderr, "sg connection closed during connection %d\n", port);
        return false;
    }

    struct linger linger;
    linger.l_onoff = 0;
    linger.l_linger = 0;
    setsockopt(sgConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

    if (!sgConn->is_connected()) // could not open server port
    {
        fprintf(stderr, "Could not open server port %d\n", port);
        return false;
    }

    toCOVERSG = std::move(sgConn);
    std::cerr << "add connection: SGBrowser connected to port " << port << std::endl;
    return true;
}

template<class Conn>
bool TUIMain::checkNewClient(std::unique_ptr<Conn> &conn)
{
    struct linger linger;
    linger.l_onoff = 0;
    linger.l_linger = 0;
    setsockopt(conn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

    clientSN = new QSocketNotifier(conn->get_id(NULL), QSocketNotifier::Read);
    QObject::connect(clientSN, SIGNAL(activated(int)), widget, SLOT(processMessages()));

    return makeSGConnection(conn.get());
}

//------------------------------------------------------------------------
void TUIMain::processMessages()
//------------------------------------------------------------------------
{
    covise::Message msg;
    const covise::Connection *conn = nullptr;
    while ((conn = connections.check_for_input(0.0001f)))
    {
        if (conn == sConn) // connection to server port
        {
            auto newConn = sConn->spawn_connection();
            if (!clientConn)
            {
                if (checkNewClient(newConn))
                {
                    clientConn = connections.add(std::move(newConn)); //add new connection;
                }
            }
            return;
        }
        else if (conn->recv_msg(&msg))
        {
            if (handleClient(&msg))
            {
                return; // we have been deleted, exit immediately
            }
        }
    }
}

TUIElement *TUIMain::createElement(int id, TabletObjectType type, QWidget *w, int parent, QString name)
{
    //cerr << "TUIMain::createElement info: creating '" << name.toStdString()
    //         << "' of type " << type << " for parent " << id << endl;
    switch (type)
    {
    case TABLET_TEXT_FIELD:
        return new TUILabel(id, type, w, parent, name);
    case TABLET_BUTTON:
        {
        if(name.mid(0,7) == "thyssen")
        {
#ifdef HAVE_WIRINGPI
            return new ThyssenButton(id,type,w,parent,name.mid(7));
#else
            return new TUIButton(id, type, w, parent, name.mid(7));
#endif
        }
        return new TUIButton(id, type, w, parent, name);
        }
    case TABLET_FILEBROWSER_BUTTON:
        return new TUIFileBrowserButton(id, type, w, parent, name);
    case TABLET_TAB: {
        auto tab = new TUITab(id, type, w, parent, name);
        tabs.insert(tab);
        tab->setNumberOfColumns(numberOfColumns);
        return tab;
    }
    case TABLET_BROWSER_TAB:
        return new TUISGBrowserTab(id, type, w, parent, name);
    case TABLET_ANNOTATION_TAB:
        return new TUIAnnotationTab(id, type, w, parent, name);
#ifndef WITHOUT_VIRVO
#if !defined _WIN32_WCE && !defined ANDROID_TUI
    case TABLET_FUNCEDIT_TAB:
        return new TUIFunctionEditorTab(id, type, w, parent, name);
#endif
#endif
    case TABLET_COLOR_TAB:
        return new TUIColorTab(id, type, w, parent, name);
    case TABLET_FRAME:
        return new TUIFrame(id, type, w, parent, name);
    case TABLET_GROUPBOX:
        return new TUIGroupBox(id, type, w, parent, name);
    case TABLET_SCROLLAREA:
        return new TUIScrollArea(id, type, w, parent, name);
    case TABLET_SPLITTER:
        return new TUISplitter(id, type, w, parent, name);
    case TABLET_TOGGLE_BUTTON:
        return new TUIToggleButton(id, type, w, parent, name);
    case TABLET_BITMAP_TOGGLE_BUTTON:
        return new TUIToggleBitmapButton(id, type, w, parent, name);
    case TABLET_COMBOBOX:
        return new TUIComboBox(id, type, w, parent, name);
    case TABLET_LISTBOX:
        return new TUIListBox(id, type, w, parent, name);
    case TABLET_FLOAT_SLIDER:
        return new TUIFloatSlider(id, type, w, parent, name);
    case TABLET_SLIDER:
        return new TUISlider(id, type, w, parent, name);
    case TABLET_FLOAT_EDIT_FIELD:
        return new TUIFloatEdit(id, type, w, parent, name);
    case TABLET_INT_EDIT_FIELD:
        return new TUIIntEdit(id, type, w, parent, name);
    case TABLET_TAB_FOLDER:
        return new TUITabFolder(id, type, w, parent, name);
    case TABLET_MAP:
        return new TUIMap(id, type, w, parent, name);
    case TABLET_EARTHMAP:
#ifdef HAVE_TUIEARTHMAP
        return new TUIEarthMap(id, type, w, parent, name);
#else
        std::cerr << "TUIapplication::createElement info: TUIEarthMap requires Qt >= 5.9" << std::endl;
        break;
#endif
    case TABLET_PROGRESS_BAR:
        return new TUIProgressBar(id, type, w, parent, name);
    case TABLET_NAV_ELEMENT:
        return new TUINavElement(id, type, w, parent, name);
    //     case TABLET_TEXT_SPIN_EDIT_FIELD:
    //       return new TUITextSpinEdit(id, type, w, parent, name);
    case TABLET_EDIT_FIELD:
        return new TUILineEdit(id, type, w, parent, name);
    case TABLET_COLOR_TRIANGLE:
        return new TUIColorTriangle(id, type, w, parent, name);
    case TABLET_COLOR_BUTTON:
        return new TUIColorButton(id, type, w, parent, name);
    case TABLET_TEXT_EDIT_FIELD:
        return new TUITextEdit(id, type, w, parent, name);
    case TABLET_POPUP:
        return new TUIPopUp(id, type, w, parent, name);
    case TABLET_UI_TAB:
#ifdef USE_UITOOLS
        return new TUIUITab(id, type, w, parent, name);
#else
        std::cerr << "TUIUiTab requires Qt Uitools, which is not available" << std::endl;
        break;
#endif
    case TABLET_WEBVIEW:
#ifdef USE_WEBENGINE
        return new TUIWebview(id, type, w, parent, name);
#else
        std::cerr << "TUIWebview is for Qt versions older than 5.4 not available" << std::endl;
        break;
#endif
    default:
        std::cerr << "TUIapplication::createElement info: unknown element type: " << type << std::endl;
        break;
    }
    return NULL;
}

//------------------------------------------------------------------------
void TUIMain::deActivateTab(TUITab *activedTab)
//------------------------------------------------------------------------
{
    for (auto el: elements)
    {
        el->deActivate(activedTab);
    }
}

//------------------------------------------------------------------------
TUIElement *TUIMain::getElement(int ID)
//------------------------------------------------------------------------
{
    auto iter = std::lower_bound(elements.begin(), elements.end(), ID, [](const TUIElement *el, int id){ return el->getID()<id;});
    if (iter != elements.end())
    {
        if ((*iter)->getID() == ID)
            return *iter;
        std::cerr << "TUIMain::getElement(ID=" << ID << "), got " << (*iter)->getID() << std::endl;
    }
    iter = std::find_if(elements.begin(), elements.end(), [ID](const TUIElement *el) { return el->getID() == ID; });
    if (iter != elements.end())
    {
        std::cerr << "TUIMain::getElement(ID=" << ID << "), STILL found !!!!!!" << std::endl;
        return *iter;
    }
    return nullptr;
}

//------------------------------------------------------------------------
QWidget *TUIMain::getWidget(int ID)
//------------------------------------------------------------------------
{
    if (auto el = getElement(ID))
        return el->getWidget();
    std::cerr << "TUIMain::getWidget(ID=" << ID << "): mainFrame" << std::endl;
    return mainFrame;
}

//------------------------------------------------------------------------
bool TUIMain::handleClient(covise::Message *msg)
//------------------------------------------------------------------------
{
    if((msg->type == covise::COVISE_MESSAGE_SOCKET_CLOSED) || (msg->type == covise::COVISE_MESSAGE_CLOSE_SOCKET))
    {
        if (msg->type == covise::COVISE_MESSAGE_SOCKET_CLOSED)
            std::cerr << "TUIMain: socket closed" << std::endl;
        else
            std::cerr << "TUIMain: closing socket" << std::endl;

        delete clientSN;
        clientSN = NULL;
        connections.remove(msg->conn); //remove connection;
        msg->conn = NULL;
        clientConn = NULL;

        //remove all UI Elements
        while (!elements.empty())
        {
            TUIElement *ele = &*elements.back(); // destructor removes the element from the list
            delete ele;
        }

        toCOVERSG.reset();

        notifyRemoveTabletUI();
        return true; // we have been deleted, exit immediately
    }

    covise::TokenBuffer tb(msg);
    switch (msg->type)
    {
    case covise::COVISE_MESSAGE_TABLET_UI:
    {
        int type;
        tb >> type;
        int ID;
        //if(ID >= 0)
        // {
        switch (type)
        {

        case TABLET_CREATE:
        {
            tb >> ID;
            //if (ID > 304)
            //    return true;
            int elementTypeInt, parent;
            const char *name;
            tb >> elementTypeInt;
            tb >> parent;
            tb >> name;
            enum TabletObjectType elementType = static_cast<TabletObjectType>(elementTypeInt);
            //cerr << "TUIMain::handleClient info: Create: ID: " << ID << " Type: " << elementType << " name: "<< name << " parent: " << parent << std::endl;
            TUIElement *parentElement = getElement(parent);
            TUIContainer *parentElem = dynamic_cast<TUIContainer *>(parentElement);
            if (parentElement && !parentElem)
            {
                std::cerr << "TUIMain::handleClient warn: parent element " << parent << " is not a container: " << ID
                          << std::endl;
            }
            else if (!parentElement)
            {
                std::cerr << "TUIMain::handleClient warn: did not find parent " << parent << " for: " << ID
                          << std::endl;
            }

            QWidget *parentWidget = mainFrame;
            if (parentElem)
                parentWidget = parentElem->getWidget();

            TUIElement *newElement = createElement(ID, elementType, parentWidget, parent, name);
            if (newElement)
            {
                if (parentElem)
                    parentElem->addElement(newElement);
            }

            registerElement(newElement, parentWidget);
        }
        break;
        case TABLET_SET_VALUE:
        {
            int typeInt;
            tb >> typeInt;
            tb >> ID;
            auto type = static_cast<TabletValue>(typeInt);
            //std::cerr << "TUIMain::handleClient info: Set Value ID: " << ID <<" Type: "<< type << std::endl;
            TUIElement *ele = getElement(ID);
            if (ele)
            {
                ele->setValue(type, tb);
            }
            else
            {
                std::cerr << "TUIMain::handleClient warn: element not available in setValue: " << ID
                          << ", value type=" << type << std::endl;
            }
        }
        break;
        case TABLET_REMOVE:
        {
            tb >> ID;
            TUIElement *ele = getElement(ID);
            if (ele)
            {
                delete ele;
            }
            else
            {
#ifdef DEBUG
                std::cerr << "TUIMain::handleClient warn: element not available in remove: " << ID << std::endl;
#endif
            }
        }
        break;

        default:
        {
            std::cerr << "TUIMain::handleClient err: unhandled message type " << type << std::endl;
        }
        break;
        }
        //}
    }
    break;
    default:
    {
        if (msg->type >= 0 && msg->type < covise::COVISE_MESSAGE_LAST_DUMMY_MESSAGE)
            std::cerr << "TUIMain::handleClient err: unknown COVISE message type " << msg->type << " "
                      << covise::covise_msg_types_array[msg->type] << std::endl;
        else
            std::cerr << "TUIMain::handleClient err: unknown COVISE message type " << msg->type << std::endl;
    }
    break;
    }
    return false;
}

void TUIMain::notifyRemoveTabletUI()
{
}

void TUIMain::registerElement(TUIElement *e, QWidget *parent)
{
}
