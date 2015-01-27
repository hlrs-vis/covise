/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifdef _WIN32
#include <direct.h>
#include <util/common.h>
#endif

#include <QComboBox>
#include <QListWidget>
#include <QPushButton>
#include <QSplitter>
#include <QDrag>
#include <QUrl>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QDialogButtonBox>
#include <QAction>
#include <QMessageBox>
#include <QLineEdit>
#include <QKeyEvent>
#include <QMimeData>

#include <covise/covise_msg.h>

#include "MEFileBrowser.h"
#include "MEMessageHandler.h"
#include "handler/MEMainHandler.h"
#include "handler/MEHostListHandler.h"
#include "handler/MEFileBrowserListHandler.h"
#include "hosts/MEHost.h"
#include "nodes/MENode.h"
#include "ports/MEFileBrowserPort.h"

int MEFileBrowser::instance = 0;

/*!
    \class MEFileTreeItem
    \brief Class sorts file list items (case sensitive)
*/

class MEFileTreeItem : public QListWidgetItem
{
public:
    MEFileTreeItem(const QString &text, QListWidget *parent = 0, int type = Type)
        : QListWidgetItem(text, parent, type)
    {
    }
    virtual ~MEFileTreeItem() {}

    virtual bool operator<(const QListWidgetItem &other) const
    {
        QString myData = data(Qt::DisplayRole).toString();
        QString otherData = other.data(Qt::DisplayRole).toString();
        return myData.compare(otherData, Qt::CaseInsensitive) < 0;
    }
};

/*!
    \class MEFileBrowser
    \brief Class provides a Remote Filebrowser
*/

MEFileBrowser::MEFileBrowser(QWidget *parent, MEFileBrowserPort *p)
    : QDialog(parent)
    , shouldClose(true)
    , currentMode(0)
    , savePath("")
    , saveFile("")
    , saveFilter("*")
    , openPath("")
    , filter(NULL)
{

    // set unique identifier variables
    ident = instance;

    // set some values depending on a given port
    QString title;
    if (p)
    {
        port = p;
        netType = MODULEPORT;
        node = port->getNode();
        host = node->getHost();
        title = node->getNodeTitle() + "@" + host->getShortname() + "[" + port->getName() + "]";
        //title = MEMainHandler::instance()->generateTitle(QString (node->getTitle() + ":" + port->getName()) );

        // catch the red book signals from a module node
        connect(node, SIGNAL(bookClose()), this, SLOT(closeFileBrowserCB()));
    }

    else
    {
        port = NULL;
        node = NULL;
        netType = OPENNET;

        QString longname;
        longname = MEHostListHandler::instance()->getIPAddress(MEMainHandler::instance()->localHost);
        host = MEHostListHandler::instance()->getHost(longname);
        title = "COVISE: File Browser";
    }

    // set a poper title
    setWindowTitle(title);

    // make central widget and layout
    // make the main layout for this windows
    mainLayout = new QVBoxLayout(this);

    makeFirstLine();
    makeExplorer();
    makeLastLine();

    if (!port)
        saveFilter = getFilter();

    // set logo & central widget
    setWindowIcon(MEMainHandler::instance()->pm_logo);

    // add browser to list
    MEFileBrowserListHandler::instance()->addFileBrowser(this);

    setLayout(mainLayout);
    setFocusProxy(m_fileBox);

    setAcceptDrops(true);
}

//------------------------------------------------------------------------
MEFileBrowser::~MEFileBrowser()
//------------------------------------------------------------------------
{
    MEFileBrowserListHandler::instance()->removeFileBrowser(this);
}

//------------------------------------------------------------------------
// this designs the first line in the browser widget
// at the moment this is a tool bar
//------------------------------------------------------------------------
void MEFileBrowser::makeFirstLine()
{
    QFrame *frame = new QFrame(this);
    tool2 = new QHBoxLayout(frame);

    // home button
    homeB = new myPathLabel("");
    homeB->setPixmap(QPixmap(":/icons/home32.png"));
    homeB->setToolTip("Go to default directory");
    connect(homeB, SIGNAL(clicked()), this, SLOT(homePressed()));
    tool2->addWidget(homeB);

    // cwd button
    cwdB = new myPathLabel(".");
    cwdB->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    cwdB->setToolTip("Show current directory");
    cwdB->setFixedWidth(32);
    connect(cwdB, SIGNAL(clicked()), this, SLOT(cwdPressed()));
    tool2->addWidget(cwdB);

    tool2->addSpacing(10);

    // root button
    rootB = new myPathLabel("/");
    rootB->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    rootB->setToolTip("Show root directories");
    rootB->setFixedWidth(32);
    connect(rootB, SIGNAL(clicked()), this, SLOT(rootPressed()));
    tool2->addWidget(rootB);

    tool2->insertStretch(-1, 2);
    mainLayout->addWidget(frame);
}

class LineEdit : public QLineEdit
{
    void keyPressEvent(QKeyEvent *ev)
    {
        if (ev->key() == Qt::Key_Enter || ev->key() == Qt::Key_Return)
        {
            ev->accept();
            emit returnPressed();
        }
        else
            QLineEdit::keyPressEvent(ev);
    }
};

//------------------------------------------------------------------------
// this designs the last line in the browser widget
//------------------------------------------------------------------------
void MEFileBrowser::makeLastLine()
{

    // 1. line
    //----------------------------------------------------
    // main horizontal container layout
    QHBoxLayout *box = new QHBoxLayout();
    mainLayout->addLayout(box);
    box->setMargin(2);
    box->setSpacing(2);

    // filename input field
    QLabel *l = new QLabel("File:", this);
    l->setFont(MEMainHandler::s_boldFont);

    box->addWidget(l);

    // path history
    m_fileBox = new QComboBox(this);
    m_fileBox->setDuplicatesEnabled(false);
    m_fileBox->setEditable(true);
    m_fileBox->setInsertPolicy(QComboBox::InsertAtTop);
    m_fileBox->setLineEdit(new LineEdit);
    m_fileBox->setFocusPolicy(Qt::StrongFocus);
    m_fileBox->setFocus();
    connect(m_fileBox, SIGNAL(activated(const QString &)), this, SLOT(historyCB(const QString &)));
    box->addWidget(m_fileBox, 1);

    l->setFocusProxy(m_fileBox);

    // 2. line
    //----------------------------------------------------
    box = new QHBoxLayout();
    mainLayout->addLayout(box);
    box->setMargin(2);
    box->setSpacing(2);

    // filter input field
    l = new QLabel("Filetype:", this);
    l->setFont(MEMainHandler::s_boldFont);
    box->addWidget(l);

    //filter combo box
    filter = new QComboBox(this);
    filter->setDuplicatesEnabled(false);
    filter->setEditable(true);
    filter->setInsertPolicy(QComboBox::InsertAtTop);
    filter->setLineEdit(new LineEdit);
    connect(filter, SIGNAL(activated(const QString &)), this, SLOT(filterCB(const QString &)));
    box->addWidget(filter, 1);

    l->setFocusProxy(filter);

    // two push button for actions
    QDialogButtonBox *bb = new QDialogButtonBox(QDialogButtonBox::Cancel | QDialogButtonBox::Ok);
    connect(bb, SIGNAL(accepted()), this, SLOT(applyCB()));
    connect(bb, SIGNAL(rejected()), this, SLOT(cancelCB()));
    box->addWidget(bb);
    applyButton = bb->button(QDialogButtonBox::Ok);
}

//------------------------------------------------------------------------
// central designs the content of the explorer for directories and files
//------------------------------------------------------------------------
void MEFileBrowser::makeExplorer()
{
    QSplitter *split = new QSplitter(Qt::Horizontal, this);

    // create the the directory list

    QWidget *w1 = new QWidget();
    split->addWidget(w1);
    QVBoxLayout *l1 = new QVBoxLayout(w1);
    l1->setMargin(2);
    l1->setSpacing(2);

    QLabel *label = new QLabel("Directory List", w1);
    label->setFont(MEMainHandler::s_boldFont);
    l1->addWidget(label);

    directory = new QListWidget(w1);
    directory->setSortingEnabled(true);
    directory->setFocusPolicy(Qt::NoFocus);
    //directory->setAlternatingRowColors(true);
    l1->addWidget(directory, 1);

    connect(directory, SIGNAL(itemClicked(QListWidgetItem *)), this, SLOT(dir2Clicked(QListWidgetItem *)));
    connect(directory, SIGNAL(itemDoubleClicked(QListWidgetItem *)), this, SLOT(dir2Clicked(QListWidgetItem *)));

    // create the file list view

    QWidget *w2 = new QWidget();
    split->addWidget(w2);
    QVBoxLayout *l2 = new QVBoxLayout(w2);
    l2->setMargin(2);
    l2->setSpacing(2);

    label = new QLabel("File List", w2);
    label->setFont(MEMainHandler::s_boldFont);
    l2->addWidget(label);

    filetable = new QListWidget(w2);
    filetable->setSortingEnabled(true);
    filetable->setFocusPolicy(Qt::NoFocus);
    filetable->setSelectionMode(QAbstractItemView::SingleSelection);
    l2->addWidget(filetable, 1);

    connect(filetable, SIGNAL(itemDoubleClicked(QListWidgetItem *)), this, SLOT(file2Clicked(QListWidgetItem *)));
    connect(filetable, SIGNAL(itemSelectionChanged()), this, SLOT(fileSelection()));
    connect(filetable, SIGNAL(itemClicked(QListWidgetItem *)), this, SLOT(file1Clicked(QListWidgetItem *)));

    //more space for main browser

    directory->setMinimumSize(250, 250);
    filetable->setMinimumSize(150, 250);

    if (port && !MEMainHandler::instance()->cfg_TopLevelBrowser)
    {
        directory->setMinimumSize(200, 200);
        filetable->setMinimumSize(120, 200);
    }

    mainLayout->addWidget(split, 1);
}

//------------------------------------------------------------------------
// update push buttons containing the pathname
//------------------------------------------------------------------------
void MEFileBrowser::updateButtonList(const QString &path)
{
    foreach (QWidget *w, buttonList)
    {
        tool2->removeWidget(w);
        delete w;
    }
    buttonList.clear();

    QStringList list = path.split("/", QString::SkipEmptyParts);
    int no = list.count();
    if (!path.endsWith("/"))
        no--;

    myPathLabel *b;
    QLabel *l;
    int index = 4;
    for (int i = 0; i < no; i++)
    {
        if (i == 0 && path[0] == '/' && path[1] == '/')
        {
            QString tmplabel = "/";
            tmplabel += list[i];
            b = new myPathLabel(tmplabel);
        }
        else
            b = new myPathLabel(list[i]);
        buttonList << b;
        connect(b, SIGNAL(clicked()), this, SLOT(buttonCB()));
        tool2->insertWidget(index, b);
        index++;

        if (i != no - 1)
        {
            l = new QLabel("/");
            l->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
            buttonList << l;
            tool2->insertWidget(index, l);
            index++;
        }
    }
}

//------------------------------------------------------------------------
// set a new filter
//------------------------------------------------------------------------
void MEFileBrowser::setFilter(const QString &text)
{
    if (!filterList.count(text))
    {
        filterList << text;
        filter->addItem(text);
    }

    filter->setCurrentIndex(filter->findText(text));
}

//------------------------------------------------------------------------
// get the current filter
//------------------------------------------------------------------------
QString MEFileBrowser::getFilter()
{
    return filter->currentText();
}

//------------------------------------------------------------------------
void MEFileBrowser::setFilterList(const QStringList &list)
//------------------------------------------------------------------------
{
    filterList = list;
    filter->clear();
    filter->addItems(filterList);
    setCurrentFilter(0);
}

//------------------------------------------------------------------------
void MEFileBrowser::setFullFilename(const QString &text)
//------------------------------------------------------------------------
{
    QString tmp = text;
    tmp.replace("\\", "/");

    int np = tmp.count("/");
    if (np == 0)
    {
        savePath = tmp;
        saveFile = "";
    }
    else
    {
        saveFile = tmp.section("/", np, np);
        savePath = tmp.section("/", 0, np - 1);
    }
    if (!savePath.endsWith("/"))
        savePath.append("/");
}

//------------------------------------------------------------------------
void MEFileBrowser::setCurrentFilter(int num)
//------------------------------------------------------------------------
{
    filter->setCurrentIndex(num);
}

//------------------------------------------------------------------------
QStringList &MEFileBrowser::getFilterList()
//------------------------------------------------------------------------
{
    return filterList;
}

//------------------------------------------------------------------------
QString MEFileBrowser::getFilename()
//------------------------------------------------------------------------
{
    return saveFile;
}

//------------------------------------------------------------------------
QString MEFileBrowser::getPath()
//------------------------------------------------------------------------
{
    return savePath + saveFile;
}

//------------------------------------------------------------------------
QString MEFileBrowser::getPathname()
//------------------------------------------------------------------------
{
    return savePath;
}

//------------------------------------------------------------------------
int MEFileBrowser::getCurrentFilter()
//------------------------------------------------------------------------
{
    return filter->currentIndex();
}

//------------------------------------------------------------------------
// if browser has a port it belongs to a node parameter
//------------------------------------------------------------------------
bool MEFileBrowser::hasPort()
{
    if (port == NULL)
        return false;
    else
        return true;
}

//------------------------------------------------------------------------
// set the right type of action
// show default directories and/or file
//------------------------------------------------------------------------
void MEFileBrowser::setNetType(int mode)
{
    netType = mode;

    switch (netType)
    {
    case OPENNET:
    case CMAPPORTOPEN:
    case CMAPMAINOPEN:
        applyButton->setText("Open");
        break;

    case SAVENET:
    case CMAPPORTSAVE:
    case CMAPMAINSAVE:
        applyButton->setText("Save");
        break;

    case SAVEASNET: // SAVE AS
        applyButton->setText("Save As");
        break;
    }

    // set a proper filter
    if (netType == CMAPPORTSAVE || netType == CMAPPORTOPEN || netType == CMAPMAINSAVE || netType == CMAPMAINOPEN)
    {
        QStringList list;
        list << "*.cmap"
             << "*";
        setFilterList(list);
        sendRequest(".", getFilter());
    }

    else if (netType == SAVEASNET || netType == SAVENET)
    {
        QStringList list;
        list << "*.net"
             << "*.py"
             << "*";
        setFilterList(list);
        sendRequest(savePath, getFilter());
    }

    else if (netType == OPENNET)
    {
        QStringList list;
        list << "*.net"
             << "*";
        setFilterList(list);
        sendRequest(savePath, getFilter());
    }
}

//------------------------------------------------------------------------
// request information of directories & files
//------------------------------------------------------------------------
void MEFileBrowser::sendDefaultRequest()
{
    sendRequest(savePath, getFilter());
}

//------------------------------------------------------------------------
// request information of directories & files
//------------------------------------------------------------------------
void MEFileBrowser::sendRequest(const QString &path, const QString &filter)
{
    if (!MEMainHandler::instance()->isMaster())
        return;

    QStringList buffer;
    buffer << "FILE_SEARCH" << host->getIPAddress() << host->getUsername();

    if (node == NULL)
    {
        buffer << "none";
        buffer << "none";
        buffer << "none";
    }
    else
    {
        buffer << node->getName();
        buffer << node->getNumber();
        buffer << port->getName();
    }

    buffer << path;
    buffer << filter;
    QString data = buffer.join("\n");

    // disable file browser
    //setEnabled(false);

    //qDebug() << "MEFileBrowserBrowser::lookupFile   " << data ;

    if (MEMainHandler::instance()->isMaster())
        MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
}

//------------------------------------------------------------------------
// answer for FILE_SEARCH
//------------------------------------------------------------------------
void MEFileBrowser::updateTree(const QStringList &items)
{
    int ndir, nfile, j, ip;
    QString name, tmp;

    // init start point in message list
    ip = 6;

    // number of directories
    ndir = items[ip].toInt();
    ip++;

    // enable browser
    setEnabled(true);

    // clear tables
    filetable->clear();
    directory->clear();

    // display directories in tree
    QString getPath;
    for (j = 0; j < ndir; j++)
    {
        // get new path
        if (getPath.isEmpty())
        {
            int np = items[ip].count("/");
            getPath = items[ip].section("/", 0, np - 1);
            if (!getPath.endsWith("/"))
                getPath.append("/");
        }

        // show items without . & .. & .*
        if (!items[ip].endsWith("/.") && !items[ip].endsWith("/.."))
        {
            tmp = items[ip].section("/", -1, -1);

            // ignore hidden directories
            if (!tmp.startsWith("."))
            {
                directory->addItem(new MEFileTreeItem(tmp, directory));
            }
        }
        ip++;
    }

    directory->sortItems(Qt::AscendingOrder);

    // display new path in buttons
    if (!getPath.isEmpty())
    {
        savePath = getPath;
        updateButtonList(savePath);
    }

    // number of files
    nfile = items[ip].toInt();
    ip++;
    if (nfile == 0)
        return;

    for (j = 0; j < nfile; j++)
    {
        // ignore hidden files
        if (!items[ip].startsWith("."))
        {
            filetable->addItem(new MEFileTreeItem(items[ip], filetable));
        }
        ip++;
    }
    filetable->sortItems(Qt::AscendingOrder);

    // show old selection
    QList<QListWidgetItem *> list = filetable->findItems(m_fileBox->currentText(), Qt::MatchExactly);
    if (!list.isEmpty())
    {
        filetable->setCurrentItem(list[0]);
        filetable->scrollToItem(list[0]);
        list[0]->setSelected(true);
    }
    else
        filetable->scrollToTop();

    m_fileBox->setFocus();
}

//------------------------------------------------------------------------
// fill empty directory and filetable windows with items
//------------------------------------------------------------------------
void MEFileBrowser::updateContent()
{
    if (directory->count() == 0 && filetable->count() == 0)
        lookupFile(savePath, saveFile, FB_OPEN);
}

//------------------------------------------------------------------------
// look if a certain file exists
//------------------------------------------------------------------------
void MEFileBrowser::lookupFile(const QString &currPath, const QString &filename, int mode)
{
    currentMode = mode;

    QStringList buffer;
    buffer << "FILE_LOOKUP" << host->getIPAddress() << host->getUsername();

    if (node == NULL)
    {
        buffer << "none"
               << "none"
               << "none";
    }
    else
    {
        buffer << node->getName();
        buffer << node->getNumber();
        buffer << port->getName();
    }

    buffer << currPath << filename;

    QString data = buffer.join("\n");
    //qDebug() << "MEFileBrowserBrowser::lookupFile   " << data ;

    // disable file browser
    setEnabled(false);

    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
}

//------------------------------------------------------------------------
// answer for FILE_LOOKUP
//------------------------------------------------------------------------
void MEFileBrowser::lookupResult(const QString & /*requested*/, const QString &filename, QString &type)
{
    QString tmp = filename;
    tmp.replace("\\", "/");

    if (type == "FILE")
    {
        int np = tmp.count("/");
        if (np == 0)
        {
            savePath = tmp;
            saveFile = "";
        }
        else
        {
            saveFile = tmp.section("/", np, np);
            savePath = tmp.section("/", 0, np - 1);
        }
        if (!savePath.endsWith("/"))
            savePath.append("/");
        updateButtonList(savePath);
        setFilename(saveFile);

        switch (currentMode)
        {
        case FB_APPLY:
        case FB_APPLY2:
            shouldClose = true;
            apply();
            break;

        case FB_OPEN:
            sendRequest(savePath, getFilter());
            break;
        }
    }

    else if (type == "DIRECTORY")
    {
        savePath = tmp;
        saveFile = "";
        if (!savePath.endsWith("/"))
            savePath.append("/");
        openPath = savePath + saveFile;
        emit currentPath(openPath);
        sendRequest(savePath, getFilter());
        QPalette palette;
        palette.setBrush(foregroundRole(), Qt::black);
        m_fileBox->setPalette(palette);
        m_fileBox->setCurrentIndex(m_fileBox->findText(saveFile));
    }

    else if (type == "NOENT")
    {
        int np = tmp.count("/");
        saveFile = tmp.section("/", np, np);
        savePath = tmp.section("/", 0, np - 1);

        if (!savePath.endsWith("/"))
            savePath.append("/");

        m_fileBox->setCurrentIndex(m_fileBox->findText(saveFile));
        sendRequest(savePath, getFilter());

        if (netType == SAVEASNET || netType == CMAPMAINSAVE || netType == CMAPPORTSAVE || currentMode == FB_APPLY2)
        {
            shouldClose = true;
            apply();
        }

        else
        {
            QPalette palette;
            palette.setBrush(foregroundRole(), Qt::red);
            m_fileBox->setPalette(palette);
        }
    }

    // enable browser
    m_fileBox->setFocus();
    setEnabled(true);
}

//------------------------------------------------------------------------
// build the message for controller
//------------------------------------------------------------------------
void MEFileBrowser::buildMessage(QString filename)
{

    // get a proper filter name if no ending is given

    if ((netType == SAVENET || netType == SAVEASNET || netType == CMAPMAINSAVE || netType == CMAPPORTSAVE) && !filename.count("."))
    {
        QString tmp = filter->currentText();
        tmp = tmp.section(';', 0, 0);
        if (tmp.count("."))
        {
            tmp = tmp.section('.', 1);
            if (tmp.count("*"))
            {
                QMessageBox::information(this,
                                         "COVISE: Remote File Browser",
                                         "filters containing * after the . are not possible for storage");
            }

            else
                filename.append(".").append(tmp);
        }

        int np = filename.count("/");
        saveFile = filename.section("/", np, np);
        m_fileBox->setCurrentIndex(m_fileBox->findText(saveFile));
    }

    // send clear network message to controller

    if (netType == OPENNET)
    {
        QString tmp = "NEW\n";
        MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, tmp);
        MEMainHandler::instance()->reset();
    }

    // build message for controller

    sendMessage(filename);
    m_fileBox->addItem(filename);
    m_fileBox->setCurrentIndex(m_fileBox->findText(filename));

    // update title in MEMainHandler::instance() with loaded map

    if (netType == OPENNET || netType == SAVENET || netType == SAVEASNET)
    {
        QString tmp = "UPDATE_LOADED_MAPNAME\n" + filename;
        MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, tmp);
    }

    // refresh files in browser
    else
        sendRequest(filename, getFilter());
}

//------------------------------------------------------------------------
// send the right message type to controller
//------------------------------------------------------------------------
void MEFileBrowser::sendMessage(QString data)
{
    switch (netType)
    {
    case MODULEPORT:
        // generate new string values for sending message
        port->sendParamMessage();
        break;

    case OPENNET:
        MEMainHandler::instance()->openNetworkFile(data);
        break;

    case SAVENET: // save
    case SAVEASNET: // save as
        MEMainHandler::instance()->saveNetwork(data);
        break;
    }
}

//------------------------------------------------------------------------
// insert item into history
//------------------------------------------------------------------------
void MEFileBrowser::updateHistory(const QString &text)
{
    if (!port)
        MEMainHandler::instance()->insertNetworkInHistory(text);

    int count = m_fileBox->count();

    QString tmp;
    for (int i = 0; i < count; i++)
    {
        tmp = m_fileBox->itemText(i);
        if (tmp == text)
            return;
    }

    m_fileBox->addItem(text);
    m_fileBox->setFocus();
}

// CALLBACKS

//------------------------------------------------------------------------
// CALLBACK: goto default directory
//------------------------------------------------------------------------
void MEFileBrowser::homePressed()
{
    sendRequest("~", getFilter());
}

//------------------------------------------------------------------------
// CALLBACK: go up one directory
//------------------------------------------------------------------------
void MEFileBrowser::upPressed()
{

    QString path = getPath();

    if ((path.length() > 2 && path.endsWith("/.")))
    {
        path = path.left(path.length() - 2);
    }

    if (path.length() > 1 && path.endsWith("/"))
    {
        path = path.left(path.length() - 1);
    }

    int n1 = path.count("/");

    // get all window drives is only a path like e: is given
    QString tmp;
    if (n1 == 0)
        tmp = "";

    else
    {
        tmp = path.section("/", 0, n1 - 1);
        if (!tmp.endsWith("/"))
            tmp.append("/");
    }

    sendRequest(tmp, getFilter());
}

//------------------------------------------------------------------------
// CALLBACK: click on a path button
//------------------------------------------------------------------------
void MEFileBrowser::buttonCB()
{
    // object that sent the signal
    myPathLabel *ac = (myPathLabel *)sender();

    int index = buttonList.indexOf(ac);

    QLabel *l;
    QString text = "/";

    for (int i = 0; i < index + 1; i++)
    {
        l = (QLabel *)buttonList.at(i);
        text = text + l->text();
    }

    if (!text.endsWith("/"))
        text.append("/");

    if (index != -1)
    {
        sendRequest(text, getFilter());
    }
}

//------------------------------------------------------------------------
// CALLBACK: close filebrowser
//------------------------------------------------------------------------
void MEFileBrowser::closeFileBrowserCB()
{
    if (port)
        port->fileBrowserClosed();

    hide();
}

//------------------------------------------------------------------------
// CALLBACK: finish filebrowser
//------------------------------------------------------------------------
void MEFileBrowser::cancelCB()
{
    setFilter(saveFilter);
    lookupFile("", openPath, FB_OPEN);

    closeFileBrowserCB();
    // check if this is the last save before quitting COVISE
    if (MEMainHandler::instance()->isWaitingForClose())
        MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_QUIT, "");
}

//------------------------------------------------------------------------
// CALLBACK: close filebrowser, apply user selection
//------------------------------------------------------------------------
void MEFileBrowser::applyCB()
{
    QString text = m_fileBox->currentText();
    if (text != saveFile && !text.isEmpty())
        checkInput(text, FB_APPLY2);

    else
    {
        shouldClose = true;
        apply();
    }
}

//------------------------------------------------------------------------
// CALLBACK: send message to controller
//------------------------------------------------------------------------
void MEFileBrowser::apply()
{

    // ask user again if type is SAVE AS
    QList<QListWidgetItem *> items = filetable->findItems(getFilename(), Qt::MatchExactly);

    if (!items.isEmpty() && (netType == SAVEASNET || netType == CMAPPORTSAVE || netType == CMAPMAINSAVE))
    {
        switch (QMessageBox::question(this,
                                      "COVISE: File Browser",
                                      QString("Do you really want to overwrite %1?").arg(getFilename()),
                                      "Overwrite", "Cancel", "", 0, 1))
        {
        case 0:
            buildMessage(getPath());
            break;

        case 1:
            return;
            break;
        }
    }

    else
        buildMessage(getPath());

    // store current value
    openPath = savePath + saveFile;
    if (!port)
        MEMainHandler::instance()->insertNetworkInHistory(openPath);
    else
        emit currentPath(openPath);

    // close file browser
    if (shouldClose)
        closeFileBrowserCB();

    // check if this is the last save before quitting COVISE
    if (MEMainHandler::instance()->isWaitingForClose())
        MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_QUIT, "");
}

//------------------------------------------------------------------------
void MEFileBrowser::filterCB(const QString &filter)
//------------------------------------------------------------------------
{
    sendRequest(savePath, filter);
}

//------------------------------------------------------------------------
void MEFileBrowser::historyCB(const QString &text)
//------------------------------------------------------------------------
{
    m_fileBox->removeItem(0);
    checkInput(text, FB_APPLY);
}

//------------------------------------------------------------------------
void MEFileBrowser::checkInput(const QString &text, int mode)
//------------------------------------------------------------------------
{
    lookupFile(savePath, text, mode);
}

//------------------------------------------------------------------------
// CALLBACK: user wants to show the root directories
//-------------------------------------------------------------------------------------------------------------
void MEFileBrowser::rootPressed()
{
    sendRequest("", getFilter());
}

//------------------------------------------------------------------------
// CALLBACK: user wants to show the current working directory
//-------------------------------------------------------------------------------------------------------------
void MEFileBrowser::cwdPressed()
{
    sendRequest(".", getFilter());
}

//------------------------------------------------------------------------
// CALLBACK: a selection was changed
//------------------------------------------------------------------------
void MEFileBrowser::fileSelection()
{
    QList<QListWidgetItem *> list = filetable->selectedItems();
    if (list.isEmpty())
        return;

    QListWidgetItem *item = list.first();
    if (item)
        file1Clicked(item);
}

//------------------------------------------------------------------------
// CALLBACK: a filename was clicked
//------------------------------------------------------------------------
void MEFileBrowser::file1Clicked(QListWidgetItem *item)
{
    if (item)
    {
        setFilename(item->text());
    }
}

void MEFileBrowser::setFilename(const QString &filename)
{
    QPalette palette;
    palette.setBrush(foregroundRole(), Qt::black);
    m_fileBox->setPalette(palette);
    m_fileBox->setCurrentIndex(m_fileBox->findText(filename));
    m_fileBox->setEditText(filename);
    saveFile = filename;
}

//------------------------------------------------------------------------
// CALLBACK: a filename was double clicked
//------------------------------------------------------------------------
void MEFileBrowser::file2Clicked(QListWidgetItem *item)
{
    if (item)
    {
        setFilename(item->text());
        lookupFile("", savePath + saveFile, FB_APPLY);
    }
}

//------------------------------------------------------------------------
// CALLBACK: callback: a directory was double clicked
//------------------------------------------------------------------------
void MEFileBrowser::dir2Clicked(QListWidgetItem *item)
{
    if (item)
    {
        QPalette palette;
        palette.setBrush(foregroundRole(), Qt::black);
        m_fileBox->setPalette(palette);
        QString text = item->text();

        if (text.startsWith("/") || text[1] == ':')
            text.append("/");

        else
            text = savePath + text + "/";

        sendRequest(text, getFilter());
    }
}

void MEFileBrowser::dropEvent(QDropEvent *ev)
{
    if (!ev->mimeData()->hasUrls() || ev->mimeData()->urls().isEmpty())
        return;

    ev->accept();
    QUrl url = ev->mimeData()->urls()[0];
    QString pathname = url.toLocalFile();
    if (!pathname.isEmpty())
    {
        QStringList components = pathname.split('/');
        QString filename = components.back();
        components.pop_back();
        QString path = components.join("/");
        path += "/";
        lookupFile(path, filename, FB_OPEN);
    }
}

void MEFileBrowser::dragEnterEvent(QDragEnterEvent *ev)
{
    if (ev->mimeData()->hasUrls() && !ev->mimeData()->urls().isEmpty())
        ev->accept();
}

/*****************************************************************************
 *
 * Class myLabel
 *
 *
 *****************************************************************************/

myPathLabel::myPathLabel(const QString &text, QWidget *parent)
    : QLabel(text, parent)
{
    setFrameStyle(QFrame::Panel | QFrame::Raised);
    setLineWidth(1);
    setMargin(4);
    setFocusPolicy(Qt::NoFocus);
    setAutoFillBackground(true);
}

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
void myPathLabel::mousePressEvent(QMouseEvent *)
{
    setFrameStyle(QFrame::Panel | QFrame::Sunken);
}

void myPathLabel::mouseReleaseEvent(QMouseEvent *)
{
    setFrameStyle(QFrame::Panel | QFrame::Raised);
    emit clicked();
}
