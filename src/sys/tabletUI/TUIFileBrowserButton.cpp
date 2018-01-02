/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <assert.h>
#include <stdio.h>

#include <QPushButton>
#include <QLabel>
#include <QPixmap>
#include <QMessageBox>

#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#else
#include <wce_msg.h>
#endif
#include "TUIFileBrowserButton.h"
#include "TUIApplication.h"

/// Constructor
TUIFileBrowserButton::TUIFileBrowserButton(int id, int type, QWidget *w, int parent, QString name)
    : TUIElement(id, type, w, parent, name)
{
    QPushButton *b = new QPushButton(w);
    if (name.contains("."))
    {
        QPixmap pm(name);
        if (pm.isNull())
        {
            QString covisedir = QString(getenv("COVISEDIR"));
            QPixmap pm(covisedir + "/" + name);
            if (pm.isNull())
            {
                b->setText(name);
            }
            else
            {
                b->setIcon(pm);
            }
        }
        else
        {
            b->setIcon(pm);
        }
    }
    else
        b->setText(name);

    //b->setFixedSize(b->sizeHint());
    //std::cerr << "Creating new Filebrowser instance!" << std::endl;
    this->mFileBrowser = new FileBrowser(this, 0, id);
    connect(this, SIGNAL(dirUpdate(QStringList)), this->mFileBrowser, SLOT(handleDirUpdate(QStringList)));
    connect(this, SIGNAL(fileUpdate(QStringList)), this->mFileBrowser, SLOT(handleFileUpdate(QStringList)));
    connect(this, SIGNAL(clientUpdate(QStringList)), this->mFileBrowser, SLOT(handleClientUpdate(QStringList)));
    /*connect(this,SIGNAL(filterUpdate(QStringList)),this->mFileBrowser, SLOT(handleFilterUpdate(QStringList)));*/
    connect(this, SIGNAL(curDirUpdate(QString)), this->mFileBrowser, SLOT(handleCurDirUpdate(QString)));
    connect(this, SIGNAL(driveUpdate(QStringList)), this->mFileBrowser, SLOT(handleDriveUpdate(QStringList)));
    connect(this, SIGNAL(updateMode(int)), this->mFileBrowser, SLOT(handleUpdateMode(int)));
    connect(this, SIGNAL(updateFilterList(char *)), this->mFileBrowser, SLOT(handleUpdateFilterList(char *)));
    connect(this, SIGNAL(locationUpdate(QString)), this->mFileBrowser, SLOT(handleLocationUpdate(QString)));
    connect(this, SIGNAL(updateRemoteButtonState(int)), this->mFileBrowser, SLOT(handleupdateRemoteButtonState(int)));
    connect(this, SIGNAL(updateLoadCheckBox(bool)), this->mFileBrowser, SLOT(handleUpdateLoadCheckBox(bool)));

    mFileBrowser->setWindowTitle("TabletUI VRML - Remote File Browser");

    widget = b;

    connect(b, SIGNAL(pressed()), this, SLOT(onPressed()));
}

/// Destructor
TUIFileBrowserButton::~TUIFileBrowserButton()
{
    this->mFileBrowser->hide();
    delete mFileBrowser;
    delete widget;
}

void TUIFileBrowserButton::handleClientRequest()
{

    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_REQ_CLIENTS;

    TUIMainWindow::getInstance()->send(tb);
}

void TUIFileBrowserButton::onPressed()
{
    //check for available VRB
    // Commented out until it is known why it causes a crash
    // in master-slave setups on slave machines
    covise::TokenBuffer vtb;
    vtb << ID;
    vtb << TABLET_REQ_VRBSTAT;

    TUIMainWindow::getInstance()->send(vtb);

    mFileBrowser->raise();
    mFileBrowser->setVisible(!hidden);

    //Get current location
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_REQ_LOCATION;

    TUIMainWindow::getInstance()->send(tb);

    tb.reset();

    //Check whether OpenCOVER client is master
    tb << ID;
    tb << TABLET_REQ_MASTER;
    TUIMainWindow::getInstance()->send(tb);
}

//void TUIFileBrowserButton::released( )
//{
//
//}

/** Set activation state of this container and all its children.
  @param en true = elements enabled
*/
void TUIFileBrowserButton::setEnabled(bool en)
{
    TUIElement::setEnabled(en);
}

/** Set highlight state of this container and all its children.
  @param hl true = element highlighted
*/
void TUIFileBrowserButton::setHighlighted(bool hl)
{
    TUIElement::setHighlighted(hl);
}

const char *TUIFileBrowserButton::getClassName() const
{
    return "TUIFileBrowserButton";
}

bool TUIFileBrowserButton::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return TUIElement::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

void TUIFileBrowserButton::sendSelectedFile(QString file, QString dir, bool loadAll)
{
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_FB_FILE_SEL;

    QByteArray ba = file.toUtf8();
    tb << ba.data();
    ba = dir.toUtf8();
    tb << ba.data();

    tb << loadAll;

    TUIMainWindow::getInstance()->send(tb);
}

void TUIFileBrowserButton::setValue(int type, covise::TokenBuffer &tb)
{
    int size;
    char *entry;
    QString strEntry;
    QStringList list;

    if (type == TABLET_SET_DIRLIST)
    {
#ifdef MB_DEBUG
        std::cerr << "Received DirList!" << std::endl;
#endif
        tb >> size;
        for (int i = 0; i < size; i++)
        {
            tb >> entry;
            strEntry = entry;
            list.append(entry);
        }
        emit dirUpdate(list);
    }
    else if (type == TABLET_SET_FILELIST)
    {
#ifdef MB_DEBUG
        std::cerr << "Received FileList!" << std::endl;
#endif
        tb >> size;
        for (int i = 0; i < size; i++)
        {
            tb >> entry;
            strEntry = entry;
            list.append(entry);
        }
        emit fileUpdate(list);
    }
    else if (type == TABLET_SET_CURDIR)
    {
#ifdef MB_DEBUG
        std::cerr << "Received CurDirList!" << std::endl;
#endif
        tb >> entry;
        //std::cerr << "Received dir update command!" << std::endl;
        strEntry = entry;
        emit curDirUpdate(entry);
    }
    else if (type == TABLET_SET_CLIENTS)
    {
#ifdef MB_DEBUG
        std::cerr << "Received ClientList!" << std::endl;
#endif
        tb >> size;
        for (int i = 0; i < size; i++)
        {
            tb >> entry;
            strEntry = entry;
            list.append(entry);
        }
        emit clientUpdate(list);
    }
    else if (type == TABLET_SET_DRIVES)
    {
#ifdef MB_DEBUG
        std::cerr << "Received DriveList!" << std::endl;
#endif
        tb >> size;
        for (int i = 0; i < size; i++)
        {
            tb >> entry;
            strEntry = entry;
            list.append(entry);
        }
        emit driveUpdate(list);
    }
    else if (type == TABLET_SET_MODE)
    {
#ifdef MB_DEBUG
        std::cerr << "Received SetMode!" << std::endl;
#endif
        int mode = 0;
        tb >> mode;
        emit updateMode(mode);
    }
    else if (type == TABLET_SET_FILTERLIST)
    {
#ifdef MB_DEBUG
        std::cerr << "Received FilterList!" << std::endl;
#endif
        char *filterList = NULL;
        tb >> filterList;
        emit updateFilterList(filterList);
    }
    else if (type == TABLET_SET_LOCATION)
    {
#ifdef MB_DEBUG
        std::cerr << "Received Location!" << std::endl;
#endif
        tb >> entry;
        strEntry = entry;
        emit locationUpdate(strEntry);
    }
    else if (type == TABLET_SET_VRBSTAT)
    {
#ifdef MB_DEBUG
        std::cerr << "Received VRBSTAT!" << std::endl;
#endif
        int state = 0;
        tb >> state;
        emit updateRemoteButtonState(state);
    }
    else if (type == TABLET_SET_MASTER)
    {
        int state = false;
        tb >> state;
        emit updateLoadCheckBox((bool)state);
    }
    else
    {
        //Call setVal of base class
        TUIElement::setValue(type, tb);
    }
}

void TUIFileBrowserButton::sendVal(int type)
{
    Q_UNUSED(type);
}

void TUIFileBrowserButton::handleRequestLists(QString filter, QString location)
{
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_REQ_FILELIST;

    QByteArray ba = filter.toUtf8();
    tb << ba.data();
    ba = location.toUtf8();
    tb << ba.data();

    TUIMainWindow::getInstance()->send(tb);

    covise::TokenBuffer rtb;
    rtb << ID;
    rtb << TABLET_REQ_DIRLIST;
    ba = filter.toUtf8();
    rtb << ba.data();
    ba = location.toUtf8();
    rtb << ba.data();

    TUIMainWindow::getInstance()->send(rtb);
}

void TUIFileBrowserButton::handleFilterUpdate(QString filter)
{
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_REQ_FILTERCHANGE;
    QByteArray ba = filter.toUtf8();
    tb << ba.data();

    TUIMainWindow::getInstance()->send(tb);
}

void TUIFileBrowserButton::handleDirChange(QString dir)
{
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_REQ_DIRCHANGE;
    QByteArray ba = dir.toUtf8();
    tb << ba.data();

    TUIMainWindow::getInstance()->send(tb);
}

void TUIFileBrowserButton::handleLocationChange(QString location)
{
    //TODO: Implement selected location
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_SET_LOCATION;
    QByteArray ba = location.toUtf8();
    tb << ba.data();

    TUIMainWindow::getInstance()->send(tb);
}

void TUIFileBrowserButton::handleLocalHome()
{
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_REQ_HOME;

    TUIMainWindow::getInstance()->send(tb);
}

void TUIFileBrowserButton::handleReqDriveList()
{
    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_REQ_DRIVES;

    TUIMainWindow::getInstance()->send(tb);
}

void TUIFileBrowserButton::handleReqDirUp(QString path)
{
    Q_UNUSED(path);
    //covise::TokenBuffer tb;
    //tb << ID;
    //tb << TABLET_REQ_DIRUP;
    //std::string spath = path.toStdString();
    //tb <<  spath.c_str();

    //TUIMainWindow::getInstance()->send(tb);
}

void TUIFileBrowserButton::handlePathSelected(QString file, QString path)
{
    std::string spath = path.toStdString();
    std::string sfile = file.toStdString();

    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_FB_PATH_SELECT;
    tb << spath.c_str();
    tb << sfile.c_str();

    TUIMainWindow::getInstance()->send(tb);
}
