/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if 0
#define setfill qtsetfill
#define setprecision qtsetprecision
#define setw qtsetw
#endif
#include "TUIApplication.h"
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

#include <QImage>
#include <QPixmap>
#include <QToolBar>
#include <QToolButton>
#include <QMenuBar>
#include <QWidget>
#include <QLabel>
#include <QComboBox>
#include <QMessageBox>
#include <QString>
#include <QSplitter>
#include <QTimer>
#include <QLayout>
#include <QStyleFactory>
#include <QAction>
#include <QSignalMapper>
#include <QTabWidget>
#include <QToolTip>
#include <QFont>
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
#include "TUIUITab.h"
#include "TUIWebview.h"

#ifdef TABLET_PLUGIN
#include "../mapeditor/handler/MEMainHandler.h"
#include "../mapeditor/widgets/MEUserInterface.h"
#else
#include "icons/exit.xpm"
#include "icons/covise.xpm"
#endif

#include <net/tokenbuffer.h>
#include <net/message_types.h>

#ifndef _WIN32
#include <signal.h>
#include <sys/socket.h>
#endif

#include <cassert>
#ifdef HAVE_WIRINGPI
#include "Thyssen.h"
#include "ThyssenButton.h"
#endif


TUIMainWindow *TUIMainWindow::appwin = 0;

#if QT_VERSION >= QT_VERSION_CHECK(5,14,0)
#define ACTIVATED textActivated
#else
#define ACTIVATED activated
#endif

//======================================================================

TUIMainWindow *TUIMainWindow::getInstance()
{
    if (appwin)
        return appwin;

    appwin = new TUIMainWindow(0);
    return appwin;
}

//======================================================================
#ifdef TABLET_PLUGIN

TUIMainWindow::TUIMainWindow(QWidget *parent, QTabWidget *mainFolder)
    : QFrame(parent)
    , mainFolder(mainFolder)
    , port(31803)
    , serverSN(NULL)
    , clientSN(NULL)
    , sConn(NULL)
    , clientConn(NULL)
{

    setFrameStyle(QFrame::Panel | QFrame::Sunken);
    setContentsMargins(2, 2, 2, 2);
    setFont(mainFont);
    setWindowTitle("COVISE:TabletUI");
    mainFrame = this;

    // main layout
    mainGrid = new QGridLayout(mainFrame);
#ifdef TABLET_PLUGIN
    mainFrame->setVisible(false);
#endif

    // init some values
    appwin = this;

    port = covise::coCoviseConfig::getInt("port", "COVER.TabletUI", port);
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN); // otherwise writes to a closed socket kill the application.
#endif

    // initialize two timer
    // timer.....waits for disconneting vrb clients
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(timerDone()));
#ifdef HAVE_WIRINGPI
    thyssenTimer = new QTimer(this);
    connect(thyssenTimer, SIGNAL(timeout()), this, SLOT(thyssenTimerDone()));
    thyssenTimer->start(100);
#endif

    resize(500, 200);
}

#else

/// ============================================================

TUIMainWindow::TUIMainWindow(QWidget *parent, QTabWidget *mainFolder)
    : QMainWindow(parent)
    , mainFolder(mainFolder)
    , port(31802)
    , serverSN(NULL)
    , clientSN(NULL)
    , sConn(NULL)
    , clientConn(NULL)
{
    // init some values
    appwin = this;

#ifdef HAVE_WIRINGPI
    ThyssenPanel::instance()->led->setLED(0,true);
#endif

#if !defined _WIN32_WCE && !defined ANDROID_TUI
    port = covise::coCoviseConfig::getInt("port", "COVER.TabletUI", port);
#endif
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN); // otherwise writes to a closed socket kill the application.
#endif

    // initialize two timer
    // timer.....waits for disconneting vrb clients

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(timerDone()));
#ifdef HAVE_WIRINGPI
    thyssenTimer = new QTimer(this);
    connect(thyssenTimer, SIGNAL(timeout()), this, SLOT(thyssenTimerDone()));
    thyssenTimer->start(100);
#endif

// create the menus and toolbar buttons
//createMenubar();
#ifndef _WIN32_WCE
    createToolbar();
#endif

    // widget that contains the main windows(mainFrame)

    QWidget *w = nullptr;
#ifdef _WIN32_WCE
    QScrollArea *scrollArea = new QScrollArea;
    scrollArea->setBackgroundRole(QPalette::Dark);
    w = scrollArea;
#endif

// main windows
#ifdef _WIN32_WCE
    mainFrame = new QFrame(w);
    mainFrame->setContentsMargins(1, 1, 1, 1);
#else
    mainFrame = new QFrame(this);
    mainFrame->setFrameStyle(QFrame::NoFrame | QFrame::Plain);
    mainFrame->setContentsMargins(0, 0, 0, 0);
#endif

    // main layout
    mainGrid = new QGridLayout(mainFrame);

    setFont(mainFont);

#ifdef _WIN32_WCE
    setWindowTitle("COVISE: PocketUI");
    setCentralWidget(w);
#else
    setWindowTitle("COVISE: TabletUI");
    setCentralWidget(mainFrame);
#endif

    // set a logo &size
    setWindowIcon(QPixmap(logo));
#ifdef _WIN32_WCE
    setMaximumWidth(480);
    setMaximumHeight(480);
#else
    resize(1066, 600);
#endif
}
#endif

TUIMainWindow::~TUIMainWindow()
{
#ifdef TABLET_PLUGIN
    closeServer();
#endif
}

void TUIMainWindow::setPort(int p)
{
    port = p;
}

//------------------------------------------------------------------------
// show this message after 2 sec
// wait 2 more sec to disconnect clients or exit
//------------------------------------------------------------------------
void TUIMainWindow::timerDone()
{
    timer->stop();
}

#ifdef HAVE_WIRINGPI
void TUIMainWindow::thyssenTimerDone()
{
    ThyssenPanel::instance()->update();
}
#endif

//------------------------------------------------------------------------
void TUIMainWindow::closeServer()
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

    delete  toCOVERSG;
    toCOVERSG = NULL;
    sgConn = NULL;

    if (!tabs.empty())
    {
        std::cerr << "TUIMainWindow::closeEvent: not all tabs erased: still " << tabs.size() << " remaining" << std::endl;
    }
    assert(tabs.empty());
}

//------------------------------------------------------------------------
int TUIMainWindow::openServer()
//------------------------------------------------------------------------
{
    connections.remove(sConn);
    sConn = connections.tryAddNewListeningConn<covise::ServerConnection>(port, 0, 0);
    if (!sConn)
    {
        return (-1);
    }

    msg = new covise::Message;

    serverSN = new QSocketNotifier(sConn->get_id(NULL), QSocketNotifier::Read);

    //cerr << "listening on port " << port << endl;
// weil unter windows manchmal Messages verloren gehen
// der SocketNotifier wird nicht oft genug aufgerufen)
#if defined(_WIN32) || defined(__APPLE__)
    m_periodictimer = new QTimer;
    QObject::connect(m_periodictimer, SIGNAL(timeout()), this, SLOT(processMessages()));
    m_periodictimer->start(1000);
#endif

    QObject::connect(serverSN, SIGNAL(activated(int)), this, SLOT(processMessages()));
    return 0;
}

//------------------------------------------------------------------------
bool TUIMainWindow::serverRunning()
//------------------------------------------------------------------------
{
    return sConn && sConn->is_connected();
}

//------------------------------------------------------------------------
void TUIMainWindow::processMessages()
//------------------------------------------------------------------------
{
    //qDebug() << "process message called";
    const covise::Connection *conn;
    while ((conn = connections.check_for_input(0.0001f)))
    {
        if (conn == sConn) // connection to server port
        {
            if (clientConn == NULL) // only accept connections if not already connected to a COVER
            {
                auto conn = sConn->spawn_connection();
                struct linger linger;
                linger.l_onoff = 0;
                linger.l_linger = 0;
                setsockopt(conn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

                clientSN = new QSocketNotifier(conn->get_id(NULL), QSocketNotifier::Read);
                QObject::connect(clientSN, SIGNAL(activated(int)),
                                 this, SLOT(processMessages()));

                // create connections for SceneGraph Browser Thread
                sgConn = new covise::ServerConnection(&port, 0, (covise::sender_type)0);
                sgConn->listen();

                covise::TokenBuffer stb;
                stb << port;
                covise::Message m(stb);
                m.type = covise::COVISE_MESSAGE_TABLET_UI;
                conn->sendMessage(&m);

                std::cerr << "SGBrowser port: " << port << std::endl;

                if (sgConn->acceptOne(5) < 0)
                {
                    fprintf(stderr, "Could not accept connection to sg port in time %d\n", port);
                    delete sgConn;
                    sgConn = nullptr;
                    return;
                }
                if (!sgConn->getSocket())
                {
					fprintf(stderr, "sg connection closed during connection %d\n", port);
                    delete sgConn;
                    sgConn = nullptr;
                    return;
                }

                linger.l_onoff = 0;
                linger.l_linger = 0;
                setsockopt(sgConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

                if (!sgConn->is_connected()) // could not open server port
                {
                    fprintf(stderr, "Could not open server port %d\n", port);
                    delete sgConn;
                    sgConn = NULL;
                    return;
                }

                toCOVERSG = sgConn;

                clientConn = connections.add(std::move(conn)); //add new connection;
            }
            else
            {
                sConn->spawn_connection();
            }
        }
        else
        {
            if (conn->recv_msg(msg))
            {
                if (msg)
                {
                    if (handleClient(msg))
                    {
                        return; // we have been deleted, exit immediately
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------
void TUIMainWindow::addElementToLayout(TUIElement *elem)
//------------------------------------------------------------------------
{
    if (elem->getWidget())
        mainGrid->addWidget(elem->getWidget(), elem->getXpos(), elem->getYpos());
    else if (elem->getLayout())
        mainGrid->addLayout(elem->getLayout(), elem->getXpos(), elem->getYpos());
}

//------------------------------------------------------------------------
void TUIMainWindow::addElement(TUIElement *e)
//------------------------------------------------------------------------
{
    //std::cerr << "new element: ID=" << e->getID() << ", name=" << e->getName().toStdString() << std::endl;
    auto iter = elements.end();
    if (!elements.empty() && e->getID() < elements.back()->getID())
    {
        iter = std::lower_bound(elements.begin(), elements.end(), e->getID(), [](const TUIElement *el, int id){ return el->getID()<id;});
    }
    elements.insert(iter, e);
}

//------------------------------------------------------------------------
void TUIMainWindow::removeElement(TUIElement *e)
//------------------------------------------------------------------------
{
#ifdef TABLET_PLUGIN
    if (e->getID() == firstTabFolderID)
        firstTabFolderID = -1;
#endif
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

//------------------------------------------------------------------------
void TUIMainWindow::send(covise::TokenBuffer &tb)
//------------------------------------------------------------------------
{
    if (clientConn == NULL)
        return;
    covise::Message m(tb);
    m.type = covise::COVISE_MESSAGE_TABLET_UI;
    clientConn->sendMessage(&m);
}

//------------------------------------------------------------------------
TUIElement *TUIMainWindow::createElement(int id, TabletObjectType type, QWidget *w, int parent, QString name)
//------------------------------------------------------------------------
{

    //cerr << "TUIMainWindow::createElement info: creating '" << name.toStdString()
    //         << "' of type " << type << " for parent " << id << endl;
    switch (type)
    {
    case TABLET_TEXT_FIELD:
        return new TUILabel(id, type, w, parent, name);
    case TABLET_BUTTON:
        {
#ifdef HAVE_WIRINGPI
        if(name.mid(0,7) == "thyssen")
        {
              return new ThyssenButton(id,type,w,parent,name.mid(7));
        }
#endif
        
        if(name.mid(0,7) == "thyssen")
        {
            return new TUIButton(id, type, w, parent, name.mid(7));
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
#ifdef TABLET_PLUGIN
        if (parent==1 && firstTabFolderID<0)
        {
            firstTabFolderID = id;
            return new TUITabFolder(id, type, w, parent, name, mainFolder);
        }
#endif
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
        return new TUIUITab(id, type, w, parent, name);
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
void TUIMainWindow::deActivateTab(TUITab *activedTab)
//------------------------------------------------------------------------
{
    for (auto el: elements)
    {
        el->deActivate(activedTab);
    }
}

//------------------------------------------------------------------------
TUIElement *TUIMainWindow::getElement(int ID)
//------------------------------------------------------------------------
{
    auto iter = std::lower_bound(elements.begin(), elements.end(), ID, [](const TUIElement *el, int id){ return el->getID()<id;});
    if (iter != elements.end())
    {
        if ((*iter)->getID() == ID)
            return *iter;
        std::cerr << "TUIMainWindow::getElement(ID=" << ID << "), got " << (*iter)->getID() << std::endl;
    }
    return nullptr;
}

//------------------------------------------------------------------------
QWidget *TUIMainWindow::getWidget(int ID)
//------------------------------------------------------------------------
{
    if (auto el = getElement(ID))
        return el->getWidget();
    std::cerr << "TUIMainWindow::getWidget(ID=" << ID << "): mainFrame" << std::endl;
    return mainFrame;
}

//------------------------------------------------------------------------
bool TUIMainWindow::handleClient(covise::Message *msg)
//------------------------------------------------------------------------
{
    if((msg->type == covise::COVISE_MESSAGE_SOCKET_CLOSED) || (msg->type == covise::COVISE_MESSAGE_CLOSE_SOCKET))
    {
        std::cerr << "TUIMainWindow: socket closed" << std::endl;

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

        delete  toCOVERSG;
        toCOVERSG = NULL;
        sgConn = NULL;

#ifdef TABLET_PLUGIN
        MEUserInterface::instance()->removeTabletUI();
#endif
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
            //cerr << "TUIApplication::handleClient info: Create: ID: " << ID << " Type: " << elementType << " name: "<< name << " parent: " << parent << std::endl;
            TUIElement *parentElement = getElement(parent);
            TUIContainer *parentElem = dynamic_cast<TUIContainer *>(parentElement);
            if (parentElement && !parentElem)
                std::cerr << "TUIApplication::handleClient warn: parent element " << parent << " is not a container: " << ID << std::endl;
#if 0
            else if (!parentElement)
                std::cerr << "TUIApplication::handleClient warn: no parent for: " << ID << std::endl;
#endif

            QWidget *parentWidget = mainFrame;
            if (parentElem)
                parentWidget = parentElem->getWidget();

            TUIElement *newElement = createElement(ID, elementType, parentWidget, parent, name);
            if (newElement)
            {
                if (parentElem)
                    parentElem->addElement(newElement);
            }

#ifdef TABLET_PLUGIN
            if (newElement->getID() != firstTabFolderID && parentWidget == mainFrame)
            {
                if (mainFolder)
                {
                    if (mainFolder->indexOf(this) == -1)
                    {
                        mainFolder->addTab(this, "Tablet UI");
                        mainFolder->setTabToolTip(mainFolder->indexOf(this), "This is the new beautiful OpenCOVER user interface");
                    }

                    mainFolder->setCurrentWidget(this);
                    mainFrame->setVisible(true);
                }
            }
#endif
        }
        break;
        case TABLET_SET_VALUE:
        {
            int typeInt;
            tb >> typeInt;
            tb >> ID;
            auto type = static_cast<TabletValue>(typeInt);
            //std::cerr << "TUIApplication::handleClient info: Set Value ID: " << ID <<" Type: "<< type << std::endl;
            TUIElement *ele = getElement(ID);
            if (ele)
            {
                ele->setValue(type, tb);
            }
            else
            {
                std::cerr << "TUIApplication::handleClient warn: element not available in setValue: " << ID << std::endl;
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
                std::cerr << "TUIApplication::handleClient warn: element not available in remove: " << ID << std::endl;
#endif
            }
        }
        break;

        default:
        {
            std::cerr << "TUIApplication::handleClient err: unhandled message type " << type << std::endl;
        }
        break;
        }
        //}
    }
    break;
    default:
    {
        if (msg->type >= 0 && msg->type < covise::COVISE_MESSAGE_LAST_DUMMY_MESSAGE)
            std::cerr << "TUIApplication::handleClient err: unknown COVISE message type " << msg->type << " " << covise::covise_msg_types_array[msg->type] << std::endl;
        else
            std::cerr << "TUIApplication::handleClient err: unknown COVISE message type " << msg->type << std::endl;
    }
    break;
    }
    return false;
}

//------------------------------------------------------------------------
// close the application
//------------------------------------------------------------------------
void TUIMainWindow::closeEvent(QCloseEvent *ce)
{

    closeServer();

    ce->accept();
}

//------------------------------------------------------------------------
// short info
//------------------------------------------------------------------------
void TUIMainWindow::about()
{
    QMessageBox::about(this, "Tablet PC UI for COVER ",
                       "This is the new beautiful COVER User Interface");
}

//------------------------------------------------------------------------
// font callback
//------------------------------------------------------------------------
#ifndef TABLET_PLUGIN
void TUIMainWindow::fontCB(const QString &string)
{
    mainFont.setPixelSize(string.toInt());
    this->setFont(mainFont);
}

#else
void TUIMainWindow::fontCB(const QString &)
{
}
#endif

//------------------------------------------------------------------------
// style callback
//------------------------------------------------------------------------
#ifndef TABLET_PLUGIN
void TUIMainWindow::styleCB(const QString &string)
{
    QStyle *s = QStyleFactory::create(string);
    if (s)
        QApplication::setStyle(s);
}

#else
void TUIMainWindow::styleCB(const QString &)
{
}
#endif

#ifndef TABLET_PLUGIN

//------------------------------------------------------------------------
// create all stuff for the menubar
//------------------------------------------------------------------------
void TUIMainWindow::createMenubar()
{

    // File menu
    QMenu *file = menuBar()->addMenu("&File");
    _exit = new QAction(QPixmap(qexit), "&Quit", this);
    _exit->setShortcut(Qt::CTRL | Qt::Key_Q);
    _exit->setToolTip("Close the tablet UI");
    connect(_exit, SIGNAL(triggered()), qApp, SLOT(closeAllWindows()));
    file->addAction(_exit);

    // Help menu
    QMenu *help = menuBar()->addMenu("&Help");
    _help = new QAction("&About", this);
    _help->setShortcut(Qt::Key_F1);
    connect(_help, SIGNAL(triggered()), this, SLOT(about()));
    help->addAction(_help);
}

//------------------------------------------------------------------------
// create all stuff for the toolbar
//------------------------------------------------------------------------
void TUIMainWindow::createToolbar()
{
    QToolBar *toolbar = addToolBar("TabletUI Toolbar");

#if !defined _WIN32_WCE && !defined ANDROID_TUI
    bool visible = covise::coCoviseConfig::isOn("toolbar", "COVER.TabletUI", true);
#else
    bool visible = false;
#endif
    toolbar->setVisible(visible);

    // quit
    //toolbar->addAction(_exit);
    //toolbar->addSeparator();

    // fontsizes
    // label
    QLabel *l = new QLabel("Font size: ", toolbar);
    l->setFont(mainFont);
    l->setToolTip("Select a new font size");
    toolbar->addWidget(l);

    // content
    QStringList list;
    list << "9"
         << "10"
         << "12"
         << "14"
         << "16"
         << "18"
         << "20"
         << "22"
         << "24";

    // combobox
    QComboBox *fontsize = new QComboBox();
#ifndef TABLET_PLUGIN
#if !defined _WIN32_WCE && !defined ANDROID_TUI
    std::string configFontsize = covise::coCoviseConfig::getEntry("fontsize", "COVER.TabletUI");
#else
    std::string configFontsize;
#endif
    if (!configFontsize.empty())
    {
        QString qfs = QString::fromStdString(configFontsize);
        if (!list.contains(qfs))
            list << qfs;
        mainFont.setPixelSize(qfs.toInt());
        this->setFont(mainFont);
    }
#endif
    QString ss;
    ss.setNum(mainFont.pointSize());
    if (!list.contains(ss))
    {
        list << ss;
    }
    fontsize->insertItems(0, list);

    int index = fontsize->findText(ss);
    fontsize->setCurrentIndex(index);
    toolbar->addWidget(fontsize);
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
    connect(fontsize, SIGNAL(textActivated(const QString &)), this, SLOT(fontCB(const QString &)));
#else
    connect(fontsize, SIGNAL(activated(const QString &)), this, SLOT(fontCB(const QString &)));
#endif
    toolbar->addSeparator();

    //styles
    // label
    l = new QLabel("Qt style: ", toolbar);
    l->setFont(mainFont);
    l->setToolTip("Select another style");
    toolbar->addWidget(l);

    // content
    QStringList styles = QStyleFactory::keys();
    if (!styles.contains("Default"))
        styles.append("Default");
    styles.sort();

    // combobox
    QComboBox *qtstyles = new QComboBox(toolbar);
    qtstyles->insertItems(0, styles);
    toolbar->addWidget(qtstyles);

#if !defined _WIN32_WCE && !defined ANDROID_TUI
    // yac/covise configuration environment
    covise::coConfigGroup *mapConfig = new covise::coConfigGroup("Map Editor");
    mapConfig->addConfig(covise::coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "mapqt.xml", "local", true);
    covise::coConfig::getInstance()->addConfig(mapConfig);

    auto currStyle = mapConfig->getValue("System.UserInterface.QtStyle").entry;
    if (!currStyle.empty())
    {
        QStyle *s = QStyleFactory::create(currStyle.c_str());
        if (s)
            QApplication::setStyle(s);
        qtstyles->setCurrentIndex(qtstyles->findText(currStyle.c_str()));
    }
    else
#endif
    {
        QStyle *s = QStyleFactory::create("Default");
        if (s)
            QApplication::setStyle(s);
        qtstyles->setCurrentIndex(qtstyles->findText("Default"));
    }

#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
    connect(qtstyles, SIGNAL(textActivated(const QString &)), this, SLOT(styleCB(const QString &)));
#else
    connect(qtstyles, SIGNAL(activated(const QString &)), this, SLOT(styleCB(const QString &)));
#endif


    toolbar->addSeparator();
    l = new QLabel("Columns: ", toolbar);
    l->setFont(mainFont);
    l->setToolTip("Number of columns in flexible layouts");
    toolbar->addWidget(l);

    list.clear();
    list << "Unlimited" << "1" << "2" << "3" << "4" << "5" << "6" << "8" << "10" << "12" << "15" << "20";
    QComboBox *numColumns = new QComboBox();
    numColumns->insertItems(0, list);
    int idx = numColumns->findText(QString::number(numberOfColumns));
    if (idx < 0 && numberOfColumns > 0)
    {
        numColumns->addItem(QString::number(numberOfColumns));
        idx = numColumns->findText(QString::number(numberOfColumns));
    }
    numColumns->setCurrentIndex(idx);
    void (QComboBox::*activated)(const QString &) = &QComboBox::ACTIVATED;
    connect(numColumns, activated, [this](QString num){
        int ncol = -1;
        if (num != "Unlimited")
            ncol = num.toInt();
        numberOfColumns = ncol;
        for (auto t: tabs)
            t->setNumberOfColumns(ncol);
    });
    toolbar->addWidget(numColumns);

}
#endif
