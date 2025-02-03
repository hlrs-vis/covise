/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TUIMainFrame.h"
#include <QApplication>


#include <QSocketNotifier>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QCloseEvent>

#include <tui/TUIElement.h>
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


#include <tui/TUITab.h>
#include <tui/TUITabFolder.h>
#include <tui/TUISGBrowserTab.h>

#include <tui/icons/exit.xpm>
#include <tui/icons/covise.xpm>

#include <net/tokenbuffer.h>
#include <net/message_types.h>

#ifndef _WIN32
#include <signal.h>
#include <sys/socket.h>
#endif

#include <cassert>

#include "MEUserInterface.h"



#if QT_VERSION >= QT_VERSION_CHECK(5,14,0)
#define ACTIVATED textActivated
#else
#define ACTIVATED activated
#endif

//======================================================================

//======================================================================

TUIMainFrame::TUIMainFrame(QWidget *parent, QTabWidget *mainFolder)
    : QFrame(parent)
    , TUIMain(this)
    , mainFolder(mainFolder)
{

    setFrameStyle(QFrame::Panel | QFrame::Sunken);
    setContentsMargins(2, 2, 2, 2);
    setFont(mainFont);
    setWindowTitle("COVISE:TabletUI");
    mainFrame = this;

    // main layout
    mainGrid = new QGridLayout(mainFrame);
    mainFrame->setVisible(false);

    // init some values

    port = covise::coCoviseConfig::getInt("port", "COVER.TabletUI", 31803);
    TUIMain::setPort(port);
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN); // otherwise writes to a closed socket kill the application.
#endif

    // initialize two timer
    // timer.....waits for disconneting vrb clients
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(timerDone()));

    resize(500, 200);
}

TUIMainFrame::~TUIMainFrame()
{
    closeServer();
}

void TUIMainFrame::timerDone()
{
    TUIMain::timerDone();
}

void TUIMainFrame::closeServer()
{
    TUIMain::closeServer();
}

void TUIMainFrame::processMessages()
{
    TUIMain::processMessages();
}

bool TUIMainFrame::handleClient(covise::Message *msg)
{
    return TUIMain::handleClient(msg);
}

//------------------------------------------------------------------------
void TUIMainFrame::removeElement(TUIElement *e)
//------------------------------------------------------------------------
{
    if (e->getID() == firstTabFolderID)
        firstTabFolderID = -1;
    TUIMain::removeElement(e);
}

//------------------------------------------------------------------------
TUIElement *TUIMainFrame::createElement(int id, TabletObjectType type, QWidget *w, int parent, QString name)
//------------------------------------------------------------------------
{
    switch (type)
    {
    case TABLET_TAB_FOLDER:
        if (parent==1 && firstTabFolderID<0)
        {
            firstTabFolderID = id;
            return new TUITabFolder(id, type, w, parent, name, mainFolder);
        }
        break;
    default:
        break;
    }

    return TUIMain::createElement(id, type, w, parent, name);
}

//------------------------------------------------------------------------
// close the application
//------------------------------------------------------------------------
void TUIMainFrame::closeEvent(QCloseEvent *ce)
{
    closeServer();

    ce->accept();
}

void TUIMainFrame::notifyRemoveTabletUI()
{
    MEUserInterface::instance()->removeTabletUI();
}

void TUIMainFrame::registerElement(TUIElement *newElement, QWidget *parentWidget)
{
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
}
