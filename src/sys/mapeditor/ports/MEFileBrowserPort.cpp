/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QPushButton>
#include <QHBoxLayout>
#include <QDebug>

#include "MEFileBrowserPort.h"
#include "MELineEdit.h"
#include "MEExtendedPart.h"
#include "MEFileBrowser.h"
#include "MEMessageHandler.h"
#include "handler/MEMainHandler.h"
#include "nodes/MENode.h"
#include "controlPanel/MEControlParameter.h"
#include "controlPanel/MEControlParameterLine.h"

/*****************************************************************************
 *
 * Class MEFileBrowserPort
 *
 *****************************************************************************/

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
MEFileBrowserPort::MEFileBrowserPort(MENode *node, QGraphicsScene *scene,
                                     const QString &portname,
                                     const QString &paramtype,
                                     const QString &description)
    : MEParameterPort(node, scene, portname, paramtype, description)
    , fileOpen(false)
    , browser(NULL)
{
    extendedPart[0] = extendedPart[1] = NULL;
    folderAction[0] = folderAction[1] = NULL;
    editLine[0] = editLine[1] = NULL;
}

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
MEFileBrowserPort::MEFileBrowserPort(MENode *node, QGraphicsScene *scene,
                                     const QString &portname,
                                     int paramtype,
                                     const QString &description,
                                     int porttype)
    : MEParameterPort(node, scene, portname, paramtype, description, porttype)
    , fileOpen(false)
    , browser(NULL)
{
    extendedPart[0] = extendedPart[1] = NULL;
    folderAction[0] = folderAction[1] = NULL;
    editLine[0] = editLine[1] = NULL;
}

//------------------------------------------------------------------------
MEFileBrowserPort::~MEFileBrowserPort()
//------------------------------------------------------------------------
{
    if (browser)
    {
        delete browser;
        browser = NULL;
    }
}

//------------------------------------------------------------------------
// restore saved parameters after
// the user pressed cancel in module parameter window
//------------------------------------------------------------------------
void MEFileBrowserPort::restoreParam()
{
#ifdef YAC
    browser->setFullFilename(filenameold);
    sendParamMessage();
#endif
}

int MEFileBrowserPort::getCurrentFilterNum()
{
    return browser->getCurrentFilter();
}

void MEFileBrowserPort::setCurrentFilterNum(int curr)
{
    browser->setCurrentFilter(curr);
}

void MEFileBrowserPort::setBrowserFilter(const QStringList &list)
{
    browser->setFilterList(list);
}

void MEFileBrowserPort::setCurrentFilter(const QString &filt)
{
    return browser->setFilter(filt);
}

QString MEFileBrowserPort::getCurrentFilter()
{
    return browser->getFilter();
}

QStringList &MEFileBrowserPort::getBrowserFilter()
{
    return browser->getFilterList();
}

#ifdef YAC
void MEFileBrowserPort::setPath(const QString &name)
{
    browser->setPath(name, false);
}

void MEFileBrowserPort::setPathname(const QString &name)
{
    browser->setPathname(name, false);
}

void MEFileBrowserPort::setFilename(const QString &name)
{
    browserFile = name;
}

QString MEFileBrowserPort::getPathname()
{
    return browser->getPathname();
}

QString MEFileBrowserPort::getFullFilename()
{
    return browser->getPathname();
}

QString MEFileBrowserPort::getPath()
{
    return browser->getPath();
}
#else
QString MEFileBrowserPort::getFullFilename()
{
    return browser->getPath();
}

QString MEFileBrowserPort::getFilename()
{
    return browser->getFilename();
}

QString MEFileBrowserPort::getPathname()
{
    return browser->getPathname();
}
#endif

//------------------------------------------------------------------------
//save current value for further use
//------------------------------------------------------------------------
void MEFileBrowserPort::storeParam()
{
    filenameold = getFullFilename();
}

//------------------------------------------------------------------------
// module has requested parameter
//------------------------------------------------------------------------
void MEFileBrowserPort::moduleParameterRequest()
{
    sendParamMessage();
}

//------------------------------------------------------------------------
void MEFileBrowserPort::separatePath(QString all)
//------------------------------------------------------------------------
{
#ifdef YAC
    int i;
    i = all.length();
    browserPath = "";
    browserFile = all;
    while (i >= 0)
    {
        if (all[i] == '\\' || all[i] == '/')
        {
            browserPath = all.left(i);
            browserFile = all.mid(i + 1);
            break;
        }
        i--;
    }
#else
    Q_UNUSED(all);
#endif
}

//------------------------------------------------------------------------
// set the filter
//------------------------------------------------------------------------
void MEFileBrowserPort::setFilter(QString value)
{
#ifdef YAC

    Q_UNUSED(value);

#else

    // set filter if parameter is given
    QStringList filterList;
    filterList = value.split("/", QString::SkipEmptyParts);
    browser->setFilterList(filterList);

#endif
}

//------------------------------------------------------------------------
// define one parameter
//------------------------------------------------------------------------
void MEFileBrowserPort::defineParam(QString value, int apptype)
{
#ifdef YAC

    Q_UNUSED(value);
    Q_UNUSED(apptype);

#else

    browser = new MEFileBrowser(0, this);

    // check filename
    if (!MEMainHandler::instance()->isInMapLoading())
        browser->lookupFile("", value, MEFileBrowser::FB_OPEN);

    fileOpen = false;

    MEParameterPort::defineParam(value, apptype);
#endif
}

//------------------------------------------------------------------------
// modify one parameter
//------------------------------------------------------------------------
void MEFileBrowserPort::modifyParam(QStringList list, int noOfValues, int istart)
{
#ifdef YAC

    Q_UNUSED(list);
    Q_UNUSED(noOfValues);
    Q_UNUSED(istart);

#else

    Q_UNUSED(noOfValues);

    // BROWSER
    browser->setFullFilename(list[istart]);

    // modify module line content
    if (editLine[MODULE])
        editLine[MODULE]->setText(getFullFilename());

    if (editLine[CONTROL])
        editLine[CONTROL]->setText(getFullFilename());
#endif
}

//------------------------------------------------------------------------
// modify one parameter
//------------------------------------------------------------------------
void MEFileBrowserPort::modifyParameter(QString lvalue)
{
#ifdef YAC

    Q_UNUSED(lvalue);

#else

    browser->setFullFilename(lvalue);

    // modify module line content
    if (editLine[MODULE])
        editLine[MODULE]->setText(getFullFilename());

    if (editLine[CONTROL])
        editLine[CONTROL]->setText(getFullFilename());
#endif
}

void MEFileBrowserPort::sendParamMessage()
//------------------------------------------------------------------------
/* send a PARAM message to controller				                        */
/* key	    ______	    keyword for message			                  	*/
//------------------------------------------------------------------------
{
    MEParameterPort::sendParamMessage(getFullFilename());
}

//------------------------------------------------------------------------
// make the layout for the module parameter widget
//------------------------------------------------------------------------
void MEFileBrowserPort::makeLayout(layoutType type, QWidget *container)
{

    int stretch = 4;

    //create a vertical layout for 2 rows

    QVBoxLayout *vb = new QVBoxLayout(container);
    vb->setMargin(1);
    vb->setSpacing(1);

    // create first container widgets

    QWidget *w1 = new QWidget(container);
    vb->addWidget(w1);

    // create for each widget a horizontal layout

    QHBoxLayout *controlBox = new QHBoxLayout(w1);
    controlBox->setMargin(2);
    controlBox->setSpacing(2);

    // pixmap button

    folderAction[type] = new QPushButton();
    folderAction[type]->setFlat(true);
    folderAction[type]->setFocusPolicy(Qt::NoFocus);
    connect(folderAction[type], SIGNAL(released()), this, SLOT(folderCB()));

    if (fileOpen)
        folderAction[type]->setIcon(MEMainHandler::instance()->pm_folderopen);
    else
        folderAction[type]->setIcon(MEMainHandler::instance()->pm_folderclosed);

    controlBox->addWidget(folderAction[type]);

    // lineedit

    editLine[type] = new MELineEdit(w1);
    editLine[type]->setMinimumWidth(300);
    editLine[type]->setText(getFullFilename());

    connect(editLine[type], SIGNAL(contentChanged(const QString &)), this, SLOT(applyCB(const QString &)));
    connect(editLine[type], SIGNAL(focusChanged(bool)), this, SLOT(setFocusCB(bool)));
    connect(browser, SIGNAL(currentPath(const QString &)), this, SLOT(showPath(const QString &)));

    controlBox->addWidget(editLine[type], stretch);

    if (MEMainHandler::instance()->cfg_TopLevelBrowser)
        extendedPart[type] = NULL;

    else
    {
        // create second widget and layout for browser
        extendedPart[type] = new MEExtendedPart(container, this);
        vb->addWidget(extendedPart[type]);

        if (mapped && folderAction[MODULE])
            folderAction[MODULE]->setEnabled(false);
    }
}

//------------------------------------------------------------------------
// open/close the colormap
//------------------------------------------------------------------------
void MEFileBrowserPort::folderCB()
{
    fileOpen = !fileOpen;
    changeFolderPixmap();
    switchExtendedPart();
    if (fileOpen)
        browser->updateContent();
}

//------------------------------------------------------------------------
// filebrowser was closed by user
//------------------------------------------------------------------------
void MEFileBrowserPort::fileBrowserClosed()
{
    fileOpen = false;
    changeFolderPixmap();
    switchExtendedPart();
}

//------------------------------------------------------------------------
// show the path (initiated from MEFileBrowser, nornally no filename was selected)
//------------------------------------------------------------------------
void MEFileBrowserPort::showPath(const QString &text)
{
    if (editLine[MODULE])
        editLine[MODULE]->setText(text);
    if (editLine[CONTROL])
        editLine[CONTROL]->setText(text);
    sendParamMessage();
}

//------------------------------------------------------------------------
// this routine is always called when the user clicks the folder pixmap
//------------------------------------------------------------------------
void MEFileBrowserPort::changeFolderPixmap()
{
    // disable pixmap when mapped && embedded window exist
    if (mapped && !MEMainHandler::instance()->cfg_TopLevelBrowser)
    {
        if (folderAction[MODULE])
            folderAction[MODULE]->setEnabled(false);
    }

    else
    {
        if (folderAction[MODULE])
            folderAction[MODULE]->setEnabled(true);
    }

    // change pixmap
    if (!fileOpen)
    {
        if (folderAction[MODULE])
            folderAction[MODULE]->setIcon(MEMainHandler::instance()->pm_folderclosed);
        if (folderAction[CONTROL])
            folderAction[CONTROL]->setIcon(MEMainHandler::instance()->pm_folderclosed);
    }

    else
    {
        if (folderAction[MODULE])
            folderAction[MODULE]->setIcon(MEMainHandler::instance()->pm_folderopen);
        if (folderAction[CONTROL])
            folderAction[CONTROL]->setIcon(MEMainHandler::instance()->pm_folderopen);
    }
}

//------------------------------------------------------------------------
// open/close the browser
//------------------------------------------------------------------------
void MEFileBrowserPort::switchExtendedPart()
{

    // open extended part or top level widget
    if (fileOpen)
    {
        if (mapped)
        {
            if (extendedPart[CONTROL])
                extendedPart[CONTROL]->addBrowser();

            else
                browser->show();
        }

        else
        {
            if (extendedPart[MODULE])
                extendedPart[MODULE]->addBrowser();

            else
                browser->show();
        }
    }

    // close extended part or top level widget
    else
    {
        if (mapped)
        {
            if (extendedPart[CONTROL])
                extendedPart[CONTROL]->hide();

            else
                browser->hide();
        }

        else
        {
            if (extendedPart[MODULE])
                extendedPart[MODULE]->hide();

            else
                browser->hide();
        }
    }
}

//------------------------------------------------------------------------
// read new parameter and send content to Controller
//------------------------------------------------------------------------
void MEFileBrowserPort::applyCB(const QString &text)
{
    QString tmp;

#ifdef YAC

    covise::coSendBuffer sb;
    sb << node->getNodeID() << portname;
    sb << node->getHostID();
    sb << text;
    sb << getCurrentFilter();
    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_SET_PARAMETER, sb);

#else

    browser->lookupFile("", text, MEFileBrowser::FB_APPLY2);
    if (editLine[MODULE])
    {
        if (text != editLine[MODULE]->text())
            sendParamMessage();
    }

    else if (editLine[CONTROL])
    {
        if (text != editLine[CONTROL]->text())
            sendParamMessage();
    }
#endif
}

//------------------------------------------------------------------------
// map parameter to control panel
// special treatment for filebrowser and colormap
//------------------------------------------------------------------------
void MEFileBrowserPort::addToControlPanel()
{

    fileBrowserClosed();

    // create a module parameter window for the node
    if (!node->getControlInfo())
        node->createControlPanelInfo();

    // remove extended parts for filebrowser
    // close extended parts
    if (browser && moduleLine)
    {
        fileOpen = false;
        if (extendedPart[MODULE])
            extendedPart[MODULE]->removeBrowser();
    }

    // create a control parameter line for this port
    if (controlLine == NULL)
    {
        QWidget *w = node->getControlInfo()->getContainer();
        controlLine = new MEControlParameterLine(w, this);
    }

    node->getControlInfo()->insertParameter(controlLine);
}

//------------------------------------------------------------------------
// unmap parameter from control panel
// remove extended parts for filebrowser
// close extended parts
//------------------------------------------------------------------------
void MEFileBrowserPort::removeFromControlPanel()
{

    fileBrowserClosed();

    if (controlLine != NULL)
    {
        if (browser)
        {
            fileOpen = false;
            if (extendedPart[CONTROL])
                extendedPart[CONTROL]->removeBrowser();
        }

        node->getControlInfo()->removeParameter(controlLine);
        extendedPart[CONTROL] = NULL;
        folderAction[CONTROL] = NULL;
        editLine[CONTROL] = NULL;
        controlLine = NULL;
    }
}

#ifdef YAC

//------------------------------------------------------------------------
void MEFileBrowserPort::setValues(covise::coRecvBuffer &tb)
//------------------------------------------------------------------------
{
    if (!browser)
        browser = new MEFileBrowser(0, this);

    browser->setCurrentFilter(0);

    int hostid;
    tb >> hostid;

#ifdef YAC
// Ruth, das fehlt noch, das sollte man bestimmt noch implementieren, kannst Du mal danach schauen? Gruss Uwe   browser->setHost(hostid);
#endif

    const char *name, *filter;
    tb >> name;
    browser->setPathname(name, true);

    tb >> filter;
    QString tmp = filter;
    browserFilter = tmp.split(" ", QString::SkipEmptyParts);
    currentFilter = 0;
    browser->setFilterList(browserFilter);

    fileOpen = false;

    // modify module line content
    if (editLine[MODULE])
        editLine[MODULE]->setText(getFullFilename());

    if (editLine[CONTROL])
        editLine[CONTROL]->setText(getFullFilename());
}
#endif
