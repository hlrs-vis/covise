/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <QApplication>

#include "VRBapplication.h"
#include "VRBPopupMenu.h"
#include "VRBFileDialog.h"
#include "VRBCurve.h"

#include "VRBClientList.h"
#include <vrbclient/VRBMessage.h>
#include <net/tokenbuffer.h>
#include "VRBServer.h"

#include <QImage>
#include <QPixmap>
#include <QToolBar>
#include <QToolButton>
#include <QMenuBar>
#include <QSplitter>
#include <QWidget>
#include <QFrame>
#include <QTimer>
#include <QLabel>
#include <QMessageBox>
#include <QTreeWidget>
#include <QListWidgetItem>
#include <QSignalMapper>
#include <QStyleFactory>
#include <QWhatsThis>
#include <QHeaderView>
//#include <QAccel>
#include <QTextEdit>

#include "icons/filenew.xpm"
#include "icons/fileopen.xpm"
#include "icons/quit.xpm"

#include "coRegister.h"

using namespace covise;

const char *ltext[4] = {
    "TCP sent bytes", "TCP received bytes",
    "UDP sent bytes", "UDP received bytes"
};

extern VRBServer server;
static VRBSClient *oldClient = NULL;
static int currRow = 0;
ApplicationWindow *appwin = NULL;

//======================================================================

ApplicationWindow::ApplicationWindow()
    : QMainWindow()
{
    setWindowTitle("VRB - Virtual Reality Request Broker");

    // init some values

    QLabel *label;
    QFrame *w;
    QVBoxLayout *box;

    appwin = this;
    currClient = NULL;
    dialog = NULL;
    plugins = NULL;

    // set a proper font & layout

    // initialize two timer
    // timer.....waits for disconneting vrb clients

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(timerDone()));

    // create the menus and toolbar buttons

    createMenubar();
    createToolbar();

    //make a central widget to contain the other widgets

    QSplitter *central = new QSplitter(Qt::Vertical, this);
    central->setObjectName("main");
    central->setOpaqueResize(true);

    // create a horizontal splitter window

    QSplitter *split = new QSplitter(Qt::Horizontal, central);
    split->setObjectName("top");
    split->setOpaqueResize(true);

    // create the table list for the left side
    // contains one entry for each vrb client

    w = new QFrame(split);
    w->setFrameStyle(QFrame::WinPanel | QFrame::Sunken);
    box = new QVBoxLayout(w);
    box->setMargin(5);

    table = new QTreeWidget(w);
    QStringList labels;
    // keep in sync with VRBSClient::Columns enum
    labels << "Master"
           << "ID"
           << "Group"
           << "User"
           << "Host"
           << "Email"
           << "URL"
           << "IP";
    table->setHeaderLabels(labels);
    table->setMinimumSize(table->sizeHint());
    table->header()->resizeSections(QHeaderView::ResizeToContents);
    connect(table, SIGNAL(itemClicked(QTreeWidgetItem *,int)),
            this, SLOT(showBPS(QTreeWidgetItem *)));
#if 0
    connect(table, SIGNAL(rightButtonClicked(QTreeWidgetItem *, const QPoint &, int)),
            this, SLOT(popupCB(QTreeWidgetItem *, const QPoint &, int)));
#else
    table->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(table, SIGNAL(customContextMenuRequested(const QPoint &)),
            this, SLOT(popupCB(const QPoint &)));
#endif

    box->addWidget(table);

    // create the tabwidget for the right side

    createTabWidget(split);

    // create a message area for the bottom

    msgFrame = new QFrame(central);
    msgFrame->setFrameStyle(QFrame::WinPanel | QFrame::Sunken);
    box = new QVBoxLayout(msgFrame);
    box->setMargin(5);
    label = new QLabel(msgFrame);
    label->setText("Message Area");
    label->setAlignment(Qt::AlignCenter);
    //label->setFont( boldfont );

    msg = new QTextEdit(msgFrame);
    box->addWidget(label);
    box->addWidget(msg);
    msgFrame->hide();

    // create a file browser

    browser = new VRBFileDialog(this);

    // create a popupmenu -- available for each vrb client in the table list

    popup = new VRBPopupMenu(table);
    popup->addAction("&Delete", this, SLOT(deleteItem()));
    popup->addAction("&Configure", this, SLOT(configItem()));

    // set a proper size

    setCentralWidget(central);
    resize(600, 450);
}

ApplicationWindow::~ApplicationWindow()
{
}

//------------------------------------------------------------------------
// create all stuff for the menubar
//------------------------------------------------------------------------
void ApplicationWindow::createMenubar()
{
    QPixmap newIcon(filenew);
    QPixmap openIcon(fileopen);
    QPixmap quitIcon(quit);

    // File menu

    QMenu *file = new QMenu(tr("&File"), this);
    file->addAction(newIcon, "&New", this, SLOT(newDoc()), Qt::CTRL + Qt::Key_N);
    file->addAction(openIcon, "&Open", this, SLOT(choose()), Qt::CTRL + Qt::Key_O);
    file->addSeparator();
    file->addAction(quitIcon, "&Quit", qApp, SLOT(closeAllWindows()), Qt::CTRL + Qt::Key_Q);

    // Preference menu

    QMenu *pref = new QMenu(tr("&Preference"), this);
    showMessageAreaAction = pref->addAction("Message Area");
    showMessageAreaAction->setCheckable(true);
    showMessageAreaAction->setChecked(false);
    connect(showMessageAreaAction, SIGNAL(toggled(bool)), SLOT(showMsg(bool)));

    // Style menu

    QMenu *styleMenu = new QMenu(tr("&Style"), this);

    QActionGroup *ag = new QActionGroup(this);
    ag->setExclusive(true);

    QSignalMapper *styleMapper = new QSignalMapper(this);
    connect(styleMapper, SIGNAL(mapped(const QString &)),
            this, SLOT(setStyle(const QString &)));

    // Help menu
    QMenu *help = new QMenu(tr("&Help"), this);
    help->addAction("&About", this, SLOT(about()), Qt::Key_F1);
    help->addSeparator();
    help->addAction("What's &This", this, SLOT(enterWhatsThis()), Qt::SHIFT + Qt::Key_F1);

    menuBar()->addMenu(file);
    menuBar()->addMenu(pref);
    menuBar()->addMenu(styleMenu);
    menuBar()->addMenu(help);
}

//------------------------------------------------------------------------
// create all stuff for the toolbar
//------------------------------------------------------------------------
void ApplicationWindow::createToolbar()
{
    QPixmap openIcon(fileopen);
    QPixmap newIcon(filenew);
    QPixmap quitIcon(quit);

    QToolBar *fileTools = new QToolBar(tr("VRB Toolbar"), this);
    fileTools->layout()->setSpacing(2);
    fileTools->layout()->setMargin(0);
    addToolBar(fileTools);
    QAction *exitAction = fileTools->addAction(quitIcon, tr("Exit"), qApp, SLOT(closeAllWindows()));
    exitAction->setToolTip(tr("exit"));

    QAction *newAction = fileTools->addAction(newIcon, tr("New"), this, SLOT(newDoc()));
    newAction->setToolTip(tr("new"));

    QAction *openAction = fileTools->addAction(openIcon, tr("Open File"), this, SLOT(choose()));
    openAction->setToolTip(tr("open file"));
}

//------------------------------------------------------------------------
// create the tab widgets containing the transfer rates
//------------------------------------------------------------------------
void ApplicationWindow::createTabWidget(QSplitter *split)
{
    QLabel *label, *text;
    int row;

    tabs = new QTabWidget(split);

    // show the transfer rates for a selected vrb client

    wtab[0] = new QFrame(tabs);
    wtab[0]->setFrameStyle(QFrame::WinPanel | QFrame::Sunken);
    grid[0] = new QGridLayout(wtab[0]);

    row = 0;
    for (int j = 0; j < 4; j++)
    {
        label = new QLabel(wtab[0]);
        label->setText(ltext[j]);
        label->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
        grid[0]->addWidget(label, row, 0);

        text = new QLabel(wtab[0]);
        text->setText("0 KBit/s");
        text->setAlignment(Qt::AlignVCenter | Qt::AlignRight);
        grid[0]->addWidget(text, row, 1);
        row++;

        curve[j] = new VRBCurve(wtab[0]);
        curve[j]->setLabel(text);
        grid[0]->addWidget(curve[j], row, 0, 1, 2, 0);
        row++;
    }

    tabs->addTab(wtab[0], tr("Current VRB"));
    tabs->setTabToolTip(tabs->indexOf(wtab[0]), tr("Transfer rates for selected vrb client"));

    // show the sent TCP transfer rates for a all vrb clients

    wtab[1] = new QFrame(tabs);
    wtab[1]->setFrameStyle(QFrame::WinPanel | QFrame::Sunken);
    grid[1] = new QGridLayout(wtab[1]);
    tabs->addTab(wtab[1], tr("TCP sent"));
    tabs->setTabToolTip(tabs->indexOf(wtab[1]), tr("Sent TCP rates for all clients"));

    // show the received TCP transfer rates for a all vrb clients

    wtab[2] = new QFrame(tabs);
    wtab[2]->setFrameStyle(QFrame::WinPanel | QFrame::Sunken);
    grid[2] = new QGridLayout(wtab[2]);
    tabs->addTab(wtab[2], tr("TCP received"));
    tabs->setTabToolTip(tabs->indexOf(wtab[2]), tr("Received TCP rates for all clients"));

    // show the sent UDP transfer rates for a all vrb clients

    wtab[3] = new QFrame(tabs);
    wtab[3]->setFrameStyle(QFrame::WinPanel | QFrame::Sunken);
    grid[3] = new QGridLayout(wtab[3]);
    tabs->addTab(wtab[3], tr("UDP sent"));
    tabs->setTabToolTip(tabs->indexOf(wtab[3]), tr("Sent UDP rates for all clients"));

    // show the received UDP transfer rates for a all vrb clients

    wtab[4] = new QFrame(tabs);
    wtab[4]->setFrameStyle(QFrame::WinPanel | QFrame::Sunken);
    grid[4] = new QGridLayout(wtab[4]);
    tabs->addTab(wtab[4], tr("UDP received"));
    tabs->setTabToolTip(tabs->indexOf(wtab[4]), tr("Received UDP rates for all clients"));

    // register area
    registry = new coRegister(tabs);
    tabs->addTab(registry, "Registry");
}

//------------------------------------------------------------------------
// add a message to the message window
//------------------------------------------------------------------------
void ApplicationWindow::addMessage(char *text)
{
    msg->append(text);
    msg->append("\n");
}

//------------------------------------------------------------------------
// remove all 4 curve types for current vrb client
//------------------------------------------------------------------------
void ApplicationWindow::removeCurves(VRBSClient *vrb)
{
    for (int j = 0; j < 4; j++)
    {
        delete vrb->myCurves[j];
        delete vrb->myLabels[j * 2];
        delete vrb->myLabels[j * 2 + 1];
    }
}

//------------------------------------------------------------------------
// add all 4 curve types for for current vrb client
//------------------------------------------------------------------------
void ApplicationWindow::createCurves(VRBSClient *vrb)
{
    QLabel *label, *text;
    QString s;
    VRBCurve *curve;

    tabs->setCurrentIndex(0);
    for (int j = 0; j < 4; j++)
    {
        label = new QLabel(wtab[j + 1]);
        s = vrb->myItem->text(3);
        label->setText(s);
        label->setAlignment(Qt::AlignVCenter | Qt::AlignLeft);
        grid[j + 1]->addWidget(label, currRow, 0);

        text = new QLabel(wtab[j + 1]);
        text->setText("0 KBit/s");
        text->setAlignment(Qt::AlignVCenter | Qt::AlignRight);
        grid[j + 1]->addWidget(text, currRow, 1);

        curve = new VRBCurve(wtab[j + 1]);

        vrb->myCurves[j] = curve;
        vrb->myLabels[j * 2] = label;
        vrb->myLabels[j * 2 + 1] = text;
        curve->setClient(vrb);
        curve->setLabel(text);
        grid[j + 1]->addWidget(curve, currRow + 1, 0, 1, 2, 0);
        curve->run();
    }

    currRow = currRow + 2;
}

//------------------------------------------------------------------------
// hide/show the message window
//------------------------------------------------------------------------
void ApplicationWindow::showMsg(bool show)
{
    if (show)
    {
        msgFrame->show();
        showMessageAreaAction->setChecked(true);
    }
    else
    {
        msgFrame->hide();
        showMessageAreaAction->setChecked(false);
    }
}

//------------------------------------------------------------------------
// set a new style for the layout
//------------------------------------------------------------------------
void ApplicationWindow::setStyle(const QString &style)
{
    QStyle *s = QStyleFactory::create(style);
    if (s)
        QApplication::setStyle(s);
}

void ApplicationWindow::enterWhatsThis()
{
    QWhatsThis::enterWhatsThisMode();
}

//------------------------------------------------------------------------
// open the file browser
// choose a file
//-----------------------------------------------------------------------
void ApplicationWindow::choose()
{
    QString s;

    // show the current groups
    //
    // ????????????????????????
    //

    // show the filebrowser
    browser->show();
    if (browser->exec() == QDialog::Accepted)
        s = browser->selectedFiles()[0];

    // send selected file  to clients
    if (!s.isEmpty())
    {
        msg->append(s);
        msg->append("\n");
        TokenBuffer tb;
        tb << LOAD_FILE;
        tb << s.toStdString().c_str();
        clients.sendMessage(tb);
    }
}

//------------------------------------------------------------------------
// open a new file
//------------------------------------------------------------------------
void ApplicationWindow::newDoc()
{
    TokenBuffer tb;
    tb << NEW_FILE;
    clients.sendMessage(tb);
}

//------------------------------------------------------------------------
// short info
//------------------------------------------------------------------------
void ApplicationWindow::about()
{
    QMessageBox::about(this, "VRB User Interface",
                       "This is the GUI for the COVISE Virtual Reality Request Broker (VRB).");
}

//------------------------------------------------------------------------
// close the application
//------------------------------------------------------------------------
void ApplicationWindow::closeEvent(QCloseEvent *ce)
{

    if (clients.num())
    {
        switch (QMessageBox::information(this, "VRB User Interface",
                                         "There are clients connected to this VRB.\nDo you want to quit anyway?",
                                         "Quit", "Cancel",
                                         0, 1, 1))
        {
        case 0:
            // start a timer to allow all clients to disconnect
            server.closeServer();
            timer->setSingleShot(true);
            timer->start(2000);
            break;

        case 1:
            ce->ignore();
            break;

        case 2:
        default: // just for sanity
            ce->ignore();
            break;
        }
    }

    else
    {
        clients.deleteAll();
        ce->accept();
    }
}

//------------------------------------------------------------------------
// show this message after 2 sec
// wait 2 more sec to disconnect clients or exit
//------------------------------------------------------------------------
void ApplicationWindow::timerDone()
{
    timer->stop();
    switch (QMessageBox::warning(this, "VRB User Interface name",
                                 "There are still clients connected to this VRB server.\n"
                                 "Do you want to quit all processes?\n\n",
                                 "Retry", "Quit", 0, 0, 1))
    {
    case 0: // The user clicked the Retry again button or pressed Enter
        timer->setSingleShot(true);
        timer->start(2000);
        break;

    case 1: // The user clicked the Quit or pressed Escape
        qApp->exit(0);
        break;
    }
}

void ApplicationWindow::popupCB(const QPoint &pos)
{
    auto item = table->itemAt(pos);
    if (!item)
        return;
    int col = table->columnAt(pos.x());
    popupCB(item, pos, col);
}

//------------------------------------------------------------------------
// show the popup menu at the clicked vrb client
//------------------------------------------------------------------------
void ApplicationWindow::popupCB(QTreeWidgetItem *item, const QPoint &pos, int col)
{
    if (col == -1)
        return;
    popup->setItem(item);
    popup->popup(table->mapToGlobal(pos));
}

//------------------------------------------------------------------------
// show tha plugins for selected vrb client
//------------------------------------------------------------------------
void ApplicationWindow::configItem()
{
    QString s = (popup->getItem())->text(1);
    //int ID  = s.toInt ();
    if (dialog == NULL)
    {
        dialog = new QDialog(this);
        dialog->setMinimumSize(200, 200);
        QFrame *f = new QFrame(dialog);
        f->setFrameStyle(QFrame::WinPanel | QFrame::Sunken);
        plugins = new QListWidget(f);
    }

    dialog->show();
}

//------------------------------------------------------------------------
// delete the selected vrb client
//------------------------------------------------------------------------
void ApplicationWindow::deleteItem()
{
    QString s = (popup->getItem())->text(1);
    int ID = s.toInt();

    // send delete to client
    if (!s.isEmpty())
    {
        msg->append("Deleted client:");
        msg->append(s);
        msg->append("\n");

        TokenBuffer tb;
        tb << DO_QUIT;
        tb << s.toStdString().c_str();
        clients.sendMessageToID(tb, ID);
    }
}

//------------------------------------------------------------------------
// show the progressbars for a selected vrb client
//------------------------------------------------------------------------
void ApplicationWindow::showBPS(QTreeWidgetItem *item)
{
    if (item)
    {
        // get current vrb client
        QString s = item->text(1);
        int ID = s.toInt();
        currClient = clients.get(ID);

        // stop old timer
        if (currClient == oldClient)
        {
            curve[0]->stop();
            curve[1]->stop();
            curve[2]->stop();
            curve[3]->stop();
            oldClient = NULL;
        }

        else
        {
            curve[0]->setClient(currClient);
            curve[1]->setClient(currClient);
            curve[2]->setClient(currClient);
            curve[3]->setClient(currClient);
            curve[0]->run();
            curve[1]->run();
            curve[2]->run();
            curve[3]->run();
        }
    }
    oldClient = currClient;
}
