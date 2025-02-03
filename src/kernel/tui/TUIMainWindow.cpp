/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TUIMainWindow.h"
#include <QApplication>


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
#include "TUIUITab.h"
#include "TUIWebview.h"

#include "icons/exit.xpm"
#include "icons/covise.xpm"

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



#if QT_VERSION >= QT_VERSION_CHECK(5,14,0)
#define ACTIVATED textActivated
#else
#define ACTIVATED activated
#endif

static const char *organization = "HLRS";
static const char *application = "tabletUI";

//======================================================================

//======================================================================
/// ============================================================

TUIMainWindow::TUIMainWindow(QWidget *parent, QTabWidget *mainFolder)
    : QMainWindow(parent)
    , TUIMain(this)
    , mainFolder(mainFolder)
{
    // init some values

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
    toolbar = createToolbar();
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
    setWindowTitle("COVER: PocketUI");
    setCentralWidget(w);
#else
    setWindowTitle("COVER: TabletUI");
    setCentralWidget(mainFrame);
#endif

    // set a logo &size
    setWindowIcon(QPixmap(logo));
#ifdef _WIN32_WCE
    setMaximumWidth(480);
    setMaximumHeight(480);
#else
    QSettings s(organization, application);
    auto geo = s.value("geometry");
    auto state = s.value("state");
    if (geo.isValid() && state.isValid())
    {
        std::cerr << "restoreGeometry from " << s.fileName().toStdString() << std::endl;
        restoreGeometry(geo.toByteArray());
        restoreState(state.toByteArray());
    }
    else
    {
        resize(1066, 600);
    }
#endif
    if (toolbar)
    {
        toolbarVisible = toolbar->isVisible();
        toolbar->setVisible(true);
    }
}

TUIMainWindow::~TUIMainWindow()
{
}

void TUIMainWindow::timerDone()
{
    TUIMain::timerDone();
}

void TUIMainWindow::closeServer()
{
    TUIMain::closeServer();
}

void TUIMainWindow::processMessages()
{
    TUIMain::processMessages();
}

bool TUIMainWindow::handleClient(covise::Message *msg)
{
    if (toolbar)
    {
        toolbar->setVisible(toolbarVisible);
    }
    return TUIMain::handleClient(msg);
}

void TUIMainWindow::notifyRemoveTabletUI()
{
    if (toolbar)
    {
        toolbarVisible = toolbar->isVisible();
        toolbar->setVisible(true);
    }
    TUIMain::notifyRemoveTabletUI();
}

#ifdef HAVE_WIRINGPI
void TUIMainWindow::thyssenTimerDone()
{
    ThyssenPanel::instance()->update();
}
#endif

void TUIMainWindow::storeGeometry()
{
    QSettings s(organization, application);
    s.setValue("geometry", saveGeometry());
    s.setValue("state", saveState());
    //std::cerr << "storeGeometry to " << s.fileName().toStdString() << std::endl;
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
void TUIMainWindow::fontCB(const QString &string)
{
    mainFont.setPixelSize(string.toInt());
    this->setFont(mainFont);
}
//------------------------------------------------------------------------
// style callback
//------------------------------------------------------------------------
void TUIMainWindow::styleCB(const QString &string)
{
    QStyle *s = QStyleFactory::create(string);
    if (s)
        QApplication::setStyle(s);
}

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
QToolBar *TUIMainWindow::createToolbar()
{
    QToolBar *toolbar = addToolBar("TabletUI Toolbar");
    toolbar->setObjectName("tabletUI.Toolbar");
    //toolbar->toggleViewAction()->setVisible(false);

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

    return toolbar;
}
