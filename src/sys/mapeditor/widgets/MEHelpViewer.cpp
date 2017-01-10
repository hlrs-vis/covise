/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MEHelpViewer.h"

#include <QComboBox>
#include <QPrinter>
#include <QMenuBar>
#include <QToolBar>
#include <QPrintDialog>
#include <QFileDialog>
#include <QDesktopServices>
#include <QDebug>

#if defined(USE_WEBENGINE)
#include <QWebEngineView>
#include <QWebEngineHistory>
#include "MEWebEnginePage.h"
#elif defined(USE_WEBKIT)
#include <QWebView>
#include <QtWebKit>
#elif defined(USE_TEXTBROWSER)
#include "METextBrowser.h"
#endif

#include "handler/MEMainHandler.h"

/*!
   \class MEHelpViewer
   \brief Simple help viewer shows local HTML text, files or content of a certain web browser
*/

MEHelpViewer::MEHelpViewer()
    : QMainWindow()
    , m_browser(NULL)
{
}

MEHelpViewer *MEHelpViewer::instance()
{
    static MEHelpViewer *singleton = 0;
    if (singleton == 0)
        singleton = new MEHelpViewer();

    return singleton;
}

//!
//! initialisation, make menus etc.
//!
void MEHelpViewer::init()
{

// create the central widget, the m_browser window
#if defined(USE_WEBENGINE)
    m_browser = new QWebEngineView(this);
    m_browser->setPage(new MEWebEnginePage);

#elif defined(USE_WEBKIT)
    m_browser = new QWebView(this);
    m_browser->page()->setLinkDelegationPolicy(QWebPage::DelegateExternalLinks);
    connect(m_browser, SIGNAL(linkClicked(const QUrl &)), this, SLOT(linkClicked(const QUrl &)));

    connect(m_browser->page()->networkAccessManager(),
            SIGNAL(sslErrors(QNetworkReply *, const QList<QSslError> &)),
            MEMainHandler::instance(),
            SLOT(handleSslErrors(QNetworkReply *, const QList<QSslError> &)));

#elif defined(USE_TEXTBROWSER)
    m_browser = new METextBrowser(this);
#endif

    // make menus
    makeMenu();

    // set the logo & resize
    setCentralWidget(m_browser);
    setWindowIcon(MEMainHandler::instance()->pm_logo);
    resize(900, 700);
}

//!
//! create the menu and the toolbar
//!
void MEHelpViewer::makeMenu()
{

    // create toolbar
    QToolBar *toolbar = addToolBar("Toolbar");
    toolbar->setIconSize(QSize(24, 24));

    // create some actions

    m_backwardId = toolbar->addAction(QPixmap(":/icons/previous32.png"), tr("&Backward"));
    m_forwardId = toolbar->addAction(QPixmap(":/icons/next32.png"), tr("&Forward"), m_browser, SLOT(forward()));
    m_homeId = toolbar->addAction(QPixmap(":/icons/home32.png"), tr("&Home"), this, SLOT(home()));
#ifndef USE_WEBENGINE
    m_printId = toolbar->addAction(QPixmap(":/icons/fileprint32.png"), tr("&Print"), this, SLOT(print()));
    m_printId->setShortcut(QKeySequence::Print);
#endif
    m_backwardId->setShortcut(QKeySequence::Back);
    m_forwardId->setShortcut(QKeySequence::Forward);

#if defined(USE_WEBENGINE)
    m_browser->page()->history()->canGoForward();
    connect(m_backwardId, SIGNAL(triggered()), m_browser, SLOT(back()));
#elif defined(USE_WEBKIT)
    m_browser->page()->history()->canGoForward();
    connect(m_backwardId, SIGNAL(triggered()), m_browser, SLOT(back()));
#elif defined(USE_TEXTBROWSER)
    m_backwardId->setEnabled(false);
    m_forwardId->setEnabled(false);
    connect(m_backwardId, SIGNAL(triggered()), m_browser, SLOT(backward()));
    connect(m_browser, SIGNAL(backwardAvailable(bool)), this, SLOT(setBackwardAvailable(bool)));
    connect(m_browser, SIGNAL(forwardAvailable(bool)), this, SLOT(setForwardAvailable(bool)));
    connect(m_browser, SIGNAL(backwardAvailable(bool)), m_backwardId, SLOT(setEnabled(bool)));
    connect(m_browser, SIGNAL(forwardAvailable(bool)), m_forwardId, SLOT(setEnabled(bool)));
#endif

    QAction *toolbar_action = toolbar->toggleViewAction();
    toolbar_action->setVisible(false);

    QAction *closeAction = new QAction("close", this);
    closeAction->setShortcut(QKeySequence::Close);
    connect(closeAction, SIGNAL(triggered(bool)), this, SLOT(close()));
    this->addAction(closeAction);
}

MEHelpViewer::~MEHelpViewer()
{
}

//!
//! check if url has to be opened in a seperate browser (used with WebKit)
//!
void MEHelpViewer::linkClicked(const QUrl &url)
{
#if defined(USE_WEBENGINE) || defined(USE_WEBKIT)
    if (url.host() != "fs.hlrs.de" || !url.path().startsWith("/projects/covise/doc"))
        QDesktopServices::openUrl(url);
    else
        m_browser->load(url);
#else
    (void)url;
#endif
}

//!
//! start index for help
//!
void MEHelpViewer::home()
{
    MEMainHandler::instance()->help();
}

void MEHelpViewer::newSource(const QString &newPath)
{
    if (newPath.startsWith("http"))
#if defined(USE_WEBENGINE) || defined(USE_WEBKIT)
        m_browser->load(QUrl(newPath));
    else
        m_browser->load(QUrl("file:///" + newPath));
#elif defined(USE_TEXTBROWSER)
        m_browser->setSource(QUrl(newPath));
    else
        m_browser->setSource(QUrl::fromLocalFile(newPath));
#endif

    show();
    raise();
}

void MEHelpViewer::setBackwardAvailable(bool b)
{
    m_backwardId->setEnabled(b);
}

void MEHelpViewer::setForwardAvailable(bool b)
{
    m_forwardId->setEnabled(b);
}

//!
//! print content of help viewer
//!
void MEHelpViewer::print()
{
    QPrinter printer;
    QPrintDialog *dialog = new QPrintDialog(&printer, this);
    dialog->setWindowTitle(tr("Print Document"));
    if (dialog->exec() != QDialog::Accepted)
        return;

#if defined(USE_WEBENGINE)
#elif defined(USE_WEBKIT)
    m_browser->print(&printer);
#elif defined(USE_TEXTBROWSER)
    QTextEdit *editor = static_cast<QTextBrowser *>(m_browser);
    QTextDocument *document = editor->document();
    document->print(&printer);
#endif
}
