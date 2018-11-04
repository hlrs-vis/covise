/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QDebug>
#include <QProgressBar>
#include <QDateTime>
#include <QGraphicsSceneContextMenuEvent>
#include <QUrl>
#include <QMimeData>

#include <covise/covise_msg.h>

#include "MENode.h"
#include "MEMessageHandler.h"
#include "widgets/MEGraphicsView.h"
#include "widgets/MEUserInterface.h"
#include "modulePanel/MEModulePanel.h"
#include "modulePanel/MEModuleParameter.h"
#include "controlPanel/MEControlPanel.h"
#include "controlPanel/MEControlParameter.h"
#include "handler/MEMainHandler.h"
#include "handler/MEHostListHandler.h"
#include "handler/MELinkListHandler.h"
#include "ports/MEDataPort.h"
#include "ports/MEParameterPort.h"
#include "ports/MEColorMapPort.h"
#include "ports/MEColormapChoicePort.h"
#include "ports/MEMaterialPort.h"
#include "ports/MEFileBrowserPort.h"
#include "ports/MEColorPort.h"
#include "ports/MEBooleanPort.h"
#include "ports/MEStringPort.h"
#include "ports/MEChoicePort.h"
#include "ports/MEIntVectorPort.h"
#include "ports/MEIntScalarPort.h"
#include "ports/MEIntSliderPort.h"
#include "ports/MEFloatScalarPort.h"
#include "ports/MEFloatVectorPort.h"
#include "ports/MEFloatSliderPort.h"
#include "dataObjects/MEDataTree.h"

static int zLevel = 4;
static int hLevel = 10;
static QPen highPen(Qt::red, 4, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
static QPen activePen(Qt::green, 4, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
qreal MENode::distance = 3;

/*!
   \class MENode
   \brief This class provides a graphics item for handling module node stuff
*/

//======================================================================
MENode::MENode(MEGraphicsView *graphWidget)
    : QGraphicsItem()
    , defaultFileBrowserPort(NULL)
    , graph(graphWidget)
    , xx(20)
    , yy(20)
    , isActive(false)
    , isSynced(false)
    , showDetails(false)
    , isModified(false)
    , isDoubleClicked(false)
    , isLocal(true)
    , isMoved(false)
    , bookclosed(true)
    , shown(true)
    , dead(false)
    , multihost(false)
    , collapsed(true)
    , showing(false)
    , clonedFromNode(NULL)
    , leftnode(NULL)
    , rightnode(NULL)
    , openBookItem(NULL)
    , closedBookItem(NULL)
    , textItem(NULL)
    , m_dataTreeItem(NULL)
    , m_moduleInfo(NULL)
    , m_controlInfo(NULL)
    , pb(NULL)
//======================================================================

{

    setFlag(ItemIsMovable);
    setFlag(ItemIsSelectable);
    setFlag(ItemIsFocusable);
    setZValue(zLevel);
    setAcceptHoverEvents(true);

    setCursor(Qt::OpenHandCursor);

    multiItem = NULL;
}

MENode::~MENode()
{

    if (!isCloned())
    {
        // remove data tree item
        if (m_dataTreeItem)
        {
            delete m_dataTreeItem;
            m_dataTreeItem = NULL;
        }

        // remove moduleinfo
        if (m_moduleInfo)
        {
            MEModulePanel::instance()->hideModuleInfo(m_moduleInfo);
            delete m_moduleInfo;
            m_moduleInfo = NULL;
        }

        // remove mapped parameters
        if (m_controlInfo)
        {
            delete m_controlInfo;
            m_controlInfo = NULL;
        }
    }

    else
    {
        //MEMainHandler::instance()->mirrorList.remove(MEMainHandler::instance()->mirrorList.indexOf(this));
    }

    // delete ports
    qDeleteAll(dinlist);
    qDeleteAll(doutlist);
    qDeleteAll(chanlist);
    qDeleteAll(pinlist);
    qDeleteAll(poutlist);
}

QRectF MENode::boundingRect() const
{
    QRectF rect = childrenBoundingRect();
    qreal localDistance = 2 * distance;
    rect.setWidth(rect.width() + localDistance);
    rect.setHeight(rect.height() + localDistance);
    rect.setX(rect.x() - localDistance);
    rect.setY(rect.y() - localDistance);
    return rect;
}

void MENode::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *)
{
    Q_UNUSED(option);

    QRectF rect = boundingRect();
    QLinearGradient gradient(rect.topLeft(), rect.bottomRight());

    if (dead)
    {
        painter->setBrush(Qt::darkGray);
        painter->setPen(Qt::black);
    }

    else
    {
        gradient.setColorAt(0, color);
        gradient.setColorAt(1, color_dark);
        if (isActive)
        {
            painter->setPen(activePen);
            setZValue(hLevel);
        }
        else if (isSelected())
        {
            highPen.setColor(MEMainHandler::s_highlightColor);
            painter->setPen(highPen);
            setZValue(hLevel);
        }

        else
        {
            painter->setPen(*normalPen);
            setZValue(zLevel);
        }

        if (isActive)
        {
            gradient.setColorAt(0, color_active);
            gradient.setColorAt(1, color);
        }

        else
        {
            gradient.setColorAt(0, color);
            gradient.setColorAt(1, color_dark);
        }

        painter->setBrush(gradient);
    }

    rect = childrenBoundingRect();
    rect.setWidth(rect.width() + distance);
    rect.setHeight(rect.height() + distance);
    rect.setX(rect.x() - distance);
    rect.setY(rect.y() - distance);
    painter->drawRoundRect(rect, 6, 6);
}

void MENode::mouseMoveEvent(QGraphicsSceneMouseEvent *e)
{
    ensureVisible();
    if (!isMoved)
        graph->wasMoved(this, e);
    isMoved = true;
    QGraphicsItem::mouseMoveEvent(e);
}

void MENode::baseMouseMoveEvent(QGraphicsSceneMouseEvent *e)
{
    QGraphicsItem::mouseMoveEvent(e);
}

void MENode::mouseReleaseEvent(QGraphicsSceneMouseEvent *e)
{
    QGraphicsItem::mouseReleaseEvent(e);

    if (isDoubleClicked)
    {
        setSelected(false);
        isDoubleClicked = false;
    }
    isMoved = false;
}

void MENode::mousePressEvent(QGraphicsSceneMouseEvent *e)
{
    QGraphicsItem::mousePressEvent(e);
}

void MENode::dragEnterEvent(QGraphicsSceneDragDropEvent *ev)
{
    if (defaultFileBrowserPort && ev->mimeData()->hasUrls() && ev->mimeData()->urls().isEmpty())
        ev->accept();
}

void MENode::dropEvent(QGraphicsSceneDragDropEvent *ev)
{
    if (!defaultFileBrowserPort || !ev->mimeData()->hasUrls() || ev->mimeData()->urls().isEmpty())
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
        defaultFileBrowserPort->modifyParameter(pathname);
        defaultFileBrowserPort->sendParamMessage();
    }
}

//------------------------------------------------------------------------
//execute node
//------------------------------------------------------------------------
void MENode::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *e)
{
    QGraphicsItem::mouseDoubleClickEvent(e);

    sendExec();

    isDoubleClicked = true;
}

//------------------------------------------------------------------------
// inform node that a parameter has been changed
//------------------------------------------------------------------------
void MENode::setModified(bool flag)
{
    isModified = flag;
    if (m_moduleInfo)
        m_moduleInfo->paramChanged(flag);
}

//------------------------------------------------------------------------
// set a new description
//------------------------------------------------------------------------
void MENode::setDesc(const QString &text)
{
    description = text;

    // set new text in module information widget
    if (m_moduleInfo != NULL)
    {
        QString title = text;
        title.append("@");
        title.append(getHostname());
        title.append("(");
        title.append(getCategory());
        title.append("):");
        m_moduleInfo->setWindowTitle(title);
    }
}

//------------------------------------------------------------------------
// switch a node to active mode
// draw a frame around the node
// used when executing
//------------------------------------------------------------------------
void MENode::setActive(bool flag)
{
    isActive = flag;
    update(boundingRect());
}

//------------------------------------------------------------------------
// show node in a different way if module has crashed
//------------------------------------------------------------------------
void MENode::setDead(bool flag)
{
    dead = flag;
    update();
}

bool MENode::isDead() const
{
    return dead;
}

//------------------------------------------------------------------------
// set a new title
//------------------------------------------------------------------------
void MENode::setLabel(const QString &name)
{
    moduleTitle = name;

    textItem->setText(name);

    // change label in module info
    if (m_moduleInfo)
        MEModulePanel::instance()->changeModuleInfoTitle(m_moduleInfo, getNodeTitle());

    // change label in control panel
    if (m_controlInfo)
        m_controlInfo->setNodeTitle(getNodeTitle());

    // change label in data viewer
    if (m_dataTreeItem)
        m_dataTreeItem->setText(0, getNodeTitle());

    layoutItem();
    update();
}

//------------------------------------------------------------------------
// create the control panel info
//------------------------------------------------------------------------
void MENode::createControlPanelInfo()
{
    m_controlInfo = new MEControlParameter(this);
    MEControlPanel::instance()->addControlInfo(m_controlInfo);
}

//------------------------------------------------------------------------
// restore saved parameter values after user pressed cancel
//------------------------------------------------------------------------
void MENode::restoreParameter()
{
    foreach (MEParameterPort *ptr, pinlist)
        ptr->restoreParam();
}

//------------------------------------------------------------------------
// restore saved parameter values after user pressed cancel
//------------------------------------------------------------------------
void MENode::storeParameter()
{
    foreach (MEParameterPort *ptr, pinlist)
        ptr->storeParam();
}

//------------------------------------------------------------------------
// the book icon was clicked
//------------------------------------------------------------------------
void MENode::bookClick()
{

    if (bookclosed)
    {
        // module info stuff
        if (m_moduleInfo == NULL)
        {
            if (pinlist.size() == 0)
            {
                MEUserInterface::instance()->printMessage("Module has no output parameter");
                return;
            }
            else
                m_moduleInfo = MEModulePanel::instance()->addModuleInfo(this);
        }

        else
        {
            MEModulePanel::instance()->showModuleInfo(m_moduleInfo);
        }

        // store current parameter
        foreach (MEParameterPort *ptr, pinlist)
            ptr->storeParam();

        // control info stuff
        bookclosed = false;
        if (openBookItem)
            layoutItem();
        emit bookIconChanged();
        MEModulePanel::instance()->raise();
    }

    // if the book is open, close it and the module parameter window
    else
    {
        // module info stuff
        MEModulePanel::instance()->hideModuleInfo(m_moduleInfo);

        // control info stuff
        bookclosed = true;
        if (openBookItem)
            layoutItem();
        emit bookIconChanged();
        emit bookClose();
    }
}

//------------------------------------------------------------------------
// create the node name	for label in node widget
//------------------------------------------------------------------------
QString MENode::getNodeTitle()
{

    QString object = moduleTitle;

    return (object);
}

//------------------------------------------------------------------------
// get the entry in the parameterlist
//------------------------------------------------------------------------
int MENode::getIndex(MEParameterPort *port)
{
    int index = pinlist.indexOf(port);
    return index;
}

//------------------------------------------------------------------------
// get the port for a given index
//------------------------------------------------------------------------
MEParameterPort *MENode::getPort(int index)
{
    MEParameterPort *port = pinlist.at(index);
    return port;
}

//------------------------------------------------------------------------
// get a port for a portname
//------------------------------------------------------------------------
MEPort *MENode::getPort(const QString &pname)
{
    MEDataPort *dp = getDataPort(pname);
    if (dp)
        return dp;

    else
    {
        MEParameterPort *pp = getParameterPort(pname);
        if (pp)
            return pp;
        else
            return NULL;
    }
}

//------------------------------------------------------------------------
// get a port for a portname
//------------------------------------------------------------------------
MEParameterPort *MENode::getParameterPort(const QString &pname)
{

    foreach (MEParameterPort *ptr, pinlist)
    {
        if (ptr->getName() == pname)
            return ptr;
    }

    return (NULL);
}

//------------------------------------------------------------------------
// get a port for a portname
//------------------------------------------------------------------------
MEDataPort *MENode::getDataPort(const QString &pname)
{

    foreach (MEDataPort *ptr, dinlist)
    {
        if (ptr->getName() == pname)
            return ptr;
    }

    foreach (MEDataPort *ptr, doutlist)
    {
        if (ptr->getName() == pname)
            return ptr;
    }

    return (NULL);
}

//------------------------------------------------------------------------
// module help
//------------------------------------------------------------------------
void MENode::helpCB()
{
    MEMainHandler::instance()->showModuleHelp(category, moduleName);
}

//------------------------------------------------------------------------
// popup menu callbacks
// execute pipeline
//------------------------------------------------------------------------
void MENode::executeCB()
{
    emit execute();
    sendExec();
}

//------------------------------------------------------------------------
// mouse events
//------------------------------------------------------------------------

//------------------------------------------------------------------------
// analyze mouse presse event
//------------------------------------------------------------------------
void MENode::contextMenuEvent(QGraphicsSceneContextMenuEvent *e)
{
    if (pinlist.size() == 0)
        MEGraphicsView::instance()->disableParameterItem(false);
    if (!isSelected())
        setSelected(true);
    MEGraphicsView::instance()->showNodeActions(this, e);
}

//------------------------------------------------------------------------
// the multi icon was clicked
//------------------------------------------------------------------------
void MENode::multiClick()
{
}

//------------------------------------------------------------------------
// print the node info
//------------------------------------------------------------------------
void MENode::printSibling()
{
}

//------------------------------------------------------------------------
// add port data
//------------------------------------------------------------------------
void MENode::addPort(MEPort *port)
{
    port->hide();
    switch (port->getPortType())
    {
    case MEPort::DIN:
    case MEPort::MULTY_IN:
        dinlist << static_cast<MEDataPort *>(port);
        break;

    case MEPort::CHAN:
        chanlist << static_cast<MEDataPort *>(port);
        break;

    case MEPort::DOUT:
        doutlist << static_cast<MEDataPort *>(port);
        break;

    case MEPort::PIN:
        pinlist << static_cast<MEParameterPort *>(port);
        break;

    case MEPort::POUT:
        poutlist << static_cast<MEParameterPort *>(port);
        ;
        break;
    }
}

//------------------------------------------------------------------------
// remove a port
//------------------------------------------------------------------------
void MENode::removePort(MEPort *port)
{

    port->hide();
    switch (port->getPortType())
    {
    case MEPort::DIN:
    case MEPort::MULTY_IN:
    {
        MEDataPort *dp = static_cast<MEDataPort *>(port);
        dinlist.remove(dinlist.indexOf(dp));
    }
    break;

    case MEPort::DOUT:
    {
        MEDataPort *dp = static_cast<MEDataPort *>(port);
        doutlist.remove(doutlist.indexOf(dp));
    }
    break;

    case MEPort::CHAN:
    {
        MEDataPort *dp = static_cast<MEDataPort *>(port);
        chanlist.remove(chanlist.indexOf(dp));
    }
    break;

    case MEPort::PIN:
    {
        MEParameterPort *pp = static_cast<MEParameterPort *>(port);
        pinlist.remove(pinlist.indexOf(pp));
    }
    break;

    case MEPort::POUT:
    {
        MEParameterPort *pp = static_cast<MEParameterPort *>(port);
        poutlist.remove(poutlist.indexOf(pp));
    }
    break;
    }
}

//------------------------------------------------------------------------
// add the multi icon with proper pixmap
// remove if no longer needed
//------------------------------------------------------------------------
void MENode::managemultiItem()
{
    if (multihost)
    {
        // change pixmap
        if (collapsed)
            multiItem->setPixmap(MEMainHandler::instance()->pm_expand);
        else
            multiItem->setPixmap(MEMainHandler::instance()->pm_collapse);
        multiItem->show();
    }

    else
    {
        multiItem->hide();
    }

    layoutItem();
}

//------------------------------------------------------------------------
// expand/collapse a node
//------------------------------------------------------------------------
void MENode::expand(int state)
{
    MENode *node = this->getLeftmost();
    while (node != NULL)
    {
        node->collapsed = !state;
        node->showing = false;
        node->managemultiItem();
        node = node->rightnode;
    }
    showing = true;
    managemultiItem();
}

//------------------------------------------------------------------------
// get the most left node in the sibling list
//------------------------------------------------------------------------
MENode *MENode::getLeftmost()
{
    if (leftnode == NULL)
        return (this);
    else
        return (leftnode->getLeftmost());
}

//------------------------------------------------------------------------
// get the most right node in the sibling list
//------------------------------------------------------------------------
MENode *MENode::getRightmost()
{
    if (rightnode == NULL)
        return (this);
    else
        return (rightnode->getRightmost());
}

//------------------------------------------------------------------------
// remove node from sibling list
// if only one node is left, reset status
//------------------------------------------------------------------------
void MENode::unlink()
{
    if (leftnode != NULL)
    {
        leftnode->rightnode = rightnode;
        if ((rightnode == NULL) && (leftnode->leftnode == NULL))
        {
            leftnode->multihost = false;
            leftnode->collapsed = false;
            leftnode->showing = false;
            leftnode->managemultiItem();
        }
    }

    if (rightnode != NULL)
    {
        rightnode->leftnode = leftnode;
        if ((leftnode == NULL) && (rightnode->rightnode == NULL))
        {
            rightnode->multihost = false;
            rightnode->collapsed = false;
            rightnode->showing = false;
            rightnode->managemultiItem();
        }
    }
}

//------------------------------------------------------------------------
// insert node into sibling list
//------------------------------------------------------------------------
bool MENode::link(MENode *m_newNode)
{
    MENode *current = m_newNode->getLeftmost();
    MENode *current_left = current;
    MENode *current_right = m_newNode->getRightmost();

    // search sibling list from left to right
    // if node is already in the list return
    do
    {
        if (getLeftmost()->findSibling(m_newNode))
            return false;
        current = current->rightnode;
    } while (current != NULL);

    // insert node n2
    m_newNode->rightnode = rightnode;
    m_newNode->leftnode = this;

    // update neighbours of this node
    if (rightnode != NULL)
        rightnode->leftnode = current_right;
    rightnode = current_left;

    // set multi host status
    if (!multihost)
    {
        multihost = true;
        managemultiItem();
    }

    m_newNode->multihost = true;
    m_newNode->collapsed = collapsed;
    m_newNode->managemultiItem();

    return true;
}

//------------------------------------------------------------------------
//  find a sibling (right)
//  returns false, if node n2 is not in the list
//------------------------------------------------------------------------
bool MENode::findSibling(MENode *n2)
{
    if (n2 == this)
        return (true);

    else if (rightnode != NULL)
        return (rightnode->findSibling(n2));

    else
        return (false);
}

//------------------------------------------------------------------------
// is a node visible
//------------------------------------------------------------------------
bool MENode::isShown()
{
    if (!multihost)
        return true;

    /*else
   {
      // if a multihost module is expanded
      if(!collapsed)
      {
         setBrush(color);
         return true;
      }
      // if collapsed this node is the only one currently visible
      else
      {
         if(showing)
         {
            setBrush(Qt::lightGray);
            return true;
         }

         else
            return false;
      }
   }*/
    return true;
}

//------------------------------------------------------------------------
// move a node to a new position
//------------------------------------------------------------------------
void MENode::moveNode(int x, int y)
{
    xx = x;
    yy = y;
    setPos(xx, yy);
}

//------------------------------------------------------------------------
// show/hide the progressbar
//------------------------------------------------------------------------
void MENode::showProgressBar(int state)
{
    if (state)
        pb->show();

    else
        pb->hide();
}

//------------------------------------------------------------------------
/* sync a module node   						                              */
/* modulename	______  name of module					                     */
/* num		   ______  current number				                        */
/* nodename	   ______  name of host					                        */
/* x,y		   ______  position of widget				                     */
/* cnode	      ______  node record for copy of parameter		            */
//------------------------------------------------------------------------
void MENode::syncNode(const QString &modulename, const QString &num,
                      const QString &nodename, int x, int y,
                      MENode *cnode)
{
    Q_UNUSED(modulename);
    Q_UNUSED(num);
    Q_UNUSED(nodename);
    Q_UNUSED(x);
    Q_UNUSED(y);
    Q_UNUSED(cnode);

    /*
      // init node
      moduleTitle = modulename;
      moduleName  = modulename;
      number      = num;
      nodeid      = id;
      xx          = x;
      yy          = y;
      host        = MEMainHandler::instance()->getHostOfNode(nodename);
      color       = host->getColor();
      int h, s, v;
      color.getHsv(&h, &s, &v);
      color_dark.setHsv(h, 255, v);
      color_active.setHsv(h, 255, 100);
      isSynced    = true;
      if(cnode)
      {
         clonedFromNode = cnode;
         cnode->m_syncList << this;
         MEMainHandler::instance()->newNodeList.append(this);
      }

      setPalette(QPalette(color));

      // create text label
      text = new MENodeLabel(this);
      text->setText( moduleTitle + " (" + QString::number(cnode->m_syncList.count()) +")" );

      // insert node in data tree
      m_dataTreeItem = new MEDataTreeItem(host->getDataRoot(), getNodeTitle());*/
}

//------------------------------------------------------------------------
// read the module description list from controller
//------------------------------------------------------------------------
void MENode::createPorts(QStringList token)
{

    if (token.size() > 6)
    {
        // start position in list
        int it = 2;

        // read list
        category = token[it];
        it = it + 2;
        description = token[it];
        it++;
        textItem->setToolTip(description);
        int d1 = token[it].toInt();
        it++;
        int d2 = token[it].toInt();
        it++;
        int p1 = token[it].toInt();
        it++;
        int p2 = token[it].toInt();
        it++;

        // store input data records
        for (int j = 0; j < d1; j++)
        {
            MEDataPort *port = new MEDataPort(this, graph->scene(), token[it], token[it + 1], token[it + 2], token[it + 3], MEPort::DIN, isSynced);
            dinlist << port;
            it = it + 4;
        }

        // store output data records
        for (int j = 0; j < d2; j++)
        {
            MEDataPort *port = new MEDataPort(this, graph->scene(), token[it], token[it + 1], token[it + 2], token[it + 3], MEPort::DOUT, isSynced);
            doutlist << port;
            it = it + 4;
        }

        // store input parameter records
        for (int j = 0; j < p1; j++)
        {
            createParameterPort(token[it], token[it + 1], token[it + 2], token[it + 3], -1);
            it = it + 5;
        }

        // disable book icon if no parameters are given
        // create the module info button
        if (p1 != 0)
        {
            closedBookItem = new MENodeBook(this, graph->scene(), ":/icons/bookclosed.svg");
            openBookItem = new MENodeBook(this, graph->scene(), ":/icons/bookopen.svg");
        }

        // store output parameter records
        for (int j = 0; j < p2; j++)
        {
            it = it + 5;
        }
    }

    // create the widget for the current module node
    layoutItem();

    // write node to mapeditor.xml
    // -> value
    QString tmp = QString("%1:%2(%3)").arg(QDateTime::currentDateTime().toTime_t()).arg(moduleName).arg(category);
    MEMainHandler::instance()->insertModuleInHistory(tmp);

    // insert node in data tree
    // do this only if output ports are available
    if (!doutlist.isEmpty())
        m_dataTreeItem = new MEDataTreeItem(host->getDataRoot(), getNodeTitle());
}

//------------------------------------------------------------------------
// create a parameter port
//------------------------------------------------------------------------
MEParameterPort *MENode::createParameterPort(const QString &pname, const QString &paramtype, const QString &description,
                                             QString value, int apptype)
{
    if (paramtype == "Colormap")
    {
        MEColorMapPort *port = new MEColorMapPort(this, graph->scene(), pname, paramtype, description);
        pinlist << port;
        port->defineParam(value, apptype);
        return port;
    }

    else if (paramtype == "Color")
    {
        MEColorPort *port = new MEColorPort(this, graph->scene(), pname, paramtype, description);
        pinlist << port;
        port->defineParam(value, apptype);
        return port;
    }

    else if (paramtype == "ColormapChoice")
    {
        MEColormapChoicePort *port = new MEColormapChoicePort(this, graph->scene(), pname, paramtype, description);
        pinlist << port;
        port->defineParam(value, apptype);
        return port;
    }

    else if (paramtype == "Material")
    {
        MEMaterialPort *port = new MEMaterialPort(this, graph->scene(), pname, paramtype, description);
        pinlist << port;
        port->defineParam(value, apptype);
        return port;
    }

    else if (paramtype == "Browser")
    {
        MEFileBrowserPort *port = new MEFileBrowserPort(this, graph->scene(), pname, paramtype, description);
        if (!defaultFileBrowserPort)
        {
            defaultFileBrowserPort = port;
            setAcceptDrops(true);
        }
        pinlist << port;
        port->defineParam(value, apptype);
        return port;
    }

    // this routine adds the filter for a given browser port
    // the parameter "BrowserFilter" exists only in the mapeditor and is only used by the mapeditor
    // it is send bz the api when a module calls "setValue" for the coFileBrowserParameter
    else if (paramtype == "BrowserFilter")
    {
        addBrowserFilter(value);
    }

    else if (paramtype == "Boolean")
    {
        MEBooleanPort *port = new MEBooleanPort(this, graph->scene(), pname, paramtype, description);
        pinlist << port;
        port->defineParam(value, apptype);
        return port;
    }

    else if (paramtype == "String")
    {
        MEStringPort *port = new MEStringPort(this, graph->scene(), pname, paramtype, description);
        pinlist << port;
        port->defineParam(value, apptype);
        return port;
    }

    else if (paramtype == "FloatVector")
    {
        MEFloatVectorPort *port = new MEFloatVectorPort(this, graph->scene(), pname, paramtype, description);
        pinlist << port;
        port->defineParam(value, apptype);
        return port;
    }

    else if (paramtype == "IntVector")
    {
        MEIntVectorPort *port = new MEIntVectorPort(this, graph->scene(), pname, paramtype, description);
        pinlist << port;
        port->defineParam(value, apptype);
        return port;
    }

    else if (paramtype == "Choice")
    {
        MEChoicePort *port = new MEChoicePort(this, graph->scene(), pname, paramtype, description);
        pinlist << port;
        port->defineParam(value, apptype);
        return port;
    }

    else if (paramtype == "IntScalar")
    {
        MEIntScalarPort *port = new MEIntScalarPort(this, graph->scene(), pname, paramtype, description);
        pinlist << port;
        port->defineParam(value, apptype);
        return port;
    }

    else if (paramtype == "FloatScalar")
    {
        MEFloatScalarPort *port = new MEFloatScalarPort(this, graph->scene(), pname, paramtype, description);
        pinlist << port;
        port->defineParam(value, apptype);
        return port;
    }

    else if (paramtype == "IntSlider")
    {
        MEIntSliderPort *port = new MEIntSliderPort(this, graph->scene(), pname, paramtype, description);
        pinlist << port;
        port->defineParam(value, apptype);
        return port;
    }

    else if (paramtype == "FloatSlider")
    {
        MEFloatSliderPort *port = new MEFloatSliderPort(this, graph->scene(), pname, paramtype, description);
        pinlist << port;
        port->defineParam(value, apptype);
        return port;
    }

    // unknown
    else
    {
        QString msg = "Port::defineParam: " + getNodeTitle() + ": Parameter type (" + paramtype + ") not yet supported";
        qWarning() << msg;
    }
    return NULL;
}

void MENode::addBrowserFilter(const QString &value)
{
    QStringList list = value.split(" ");
    MEFileBrowserPort *port = qobject_cast<MEFileBrowserPort *>(getParameterPort(list[0]));
    if (port && list.size() == 2)
        port->setFilter(list[1]);
}

//------------------------------------------------------------------------
// calculate and assemble parts for the node graphics item
//------------------------------------------------------------------------
void MENode::layoutItem()
{
    prepareGeometryChange();

    int index = 0;
    qreal dx = distance;
    qreal dy = distance + 1;
    qreal xsize = MEMainHandler::instance()->getPortSize();
    qreal ysize = xsize;

    qreal xstart = xsize + dx;
    qreal ystart = 0;

    // layout for input ports
    foreach (MEDataPort *port, dinlist)
    {
        if (port->isVisible())
        {
            port->setPos(index * xstart, ystart);
            index++;
        }
    }
    // layout for parameter ports
    foreach (MEParameterPort *port, pinlist)
    {
        if (port->isVisible())
        {
            port->setPos(index * xstart, ystart);
            index++;
        }
    }

    index = 0;
    xstart = xsize + dx;
    if (!dinlist.isEmpty())
        ystart = ystart + ysize + dy + textItem->boundingRect().height() + dy;
    else
        ystart = ystart + dy + textItem->boundingRect().height() + dy;

    // layout for output ports
    foreach (MEDataPort *port, doutlist)
    {
        if (port->isVisible())
        {
            port->setPos(index * xstart, ystart);
            index++;
        }
    }

    // layout for parameter ports
    foreach (MEParameterPort *port, poutlist)
    {
        if (port->isVisible())
        {
            port->setPos(index * xstart, ystart);
            index++;
        }
    }

    if (!dinlist.isEmpty())
        ystart = ysize + dy;
    else
        ystart = dy;
    // create layout for middle line
    textItem->setPos(dx, ystart);

    if (closedBookItem)
    {
        closedBookItem->setPos(0, 0);
        openBookItem->setPos(0, 0);
        if (childrenBoundingRect().width() > textItem->boundingRect().width() + closedBookItem->boundingRect().width() + dx)
            xstart = childrenBoundingRect().width() - closedBookItem->boundingRect().width();
        else
            xstart = textItem->boundingRect().width() + dx + dx;
        closedBookItem->setPos(xstart, ystart);
        openBookItem->setPos(xstart, ystart);

        if (bookclosed)
        {
            openBookItem->hide();
            closedBookItem->show();
        }

        else
        {
            openBookItem->show();
            closedBookItem->hide();
        }
    }

    /*h2->add(text);
   if(bookicon)
      h2->add(bookicon);
   h2->add(multiicon);
   multiicon->hide();
   h2->addStretch(1);

   pb->setFixedSize(h2->sizeHint());*/

    // set position of node and show it
    setPos(xx, yy);
    update();
}

void MENode::sendExec()
{
    emit execute();
    sendMessage("EXEC");
    MEMainHandler::instance()->execTriggered();
}

//------------------------------------------------------------------------
// send a  message used from ports
//------------------------------------------------------------------------
void MENode::sendMessage(const QString &key)
{
    QStringList list;
    QString data;

    list << key << moduleName << number << getHostname();
    data = list.join("\n");
    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
    list.clear();
}

//------------------------------------------------------------------------
// send a node message
//------------------------------------------------------------------------
void MENode::sendNodeMessage(const QString &key)
{
    QStringList buffer;
    QString data, tmp;

    buffer << key << QString::number(1) << moduleName << number << getHostname();
    if (key == "MOV")
    {
        tmp.setNum(xx);
        buffer << tmp;
        tmp.setNum(yy);
        buffer << tmp;
        MEMainHandler::instance()->mapWasChanged("MOV");
    }

    data = buffer.join("\n");
    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
    buffer.clear();

    int cnt = 0;
    foreach (MENode *nn, m_syncList)
    {
        buffer << key << nn->getName() << nn->getNumber() << nn->getHostname();
        if (key == "MOV")
        {
            tmp.setNum(x() + (cnt + 1) * 200);
            buffer << tmp;
            tmp.setNum(y());
            buffer << tmp;
        }
        data = buffer.join("\n");
        MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
        buffer.clear();
        cnt++;
    }
}

//------------------------------------------------------------------------
// send a message
//------------------------------------------------------------------------
void MENode::sendRenameMessage(const QString &text)
{
    QStringList buffer;
    buffer << "MODULE_TITLE" << moduleName << number << getHostname() << text;
    QString data = buffer.join("\n");
    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
    buffer.clear();
    MEMainHandler::instance()->mapWasChanged("MODULE_TITLE");
}

//------------------------------------------------------------------------
// init a module node
//------------------------------------------------------------------------
void MENode::init(const QString &modulename, const QString &num,
                  const QString &nodename, int x, int y)
{

    // init node

    moduleName = modulename;
    moduleTitle = moduleName + "_" + num;
    number = num;
    nodeid = -1;
    xx = x;
    yy = y;
    host = MEHostListHandler::instance()->getHost(nodename);
    color = host->getColor();
    int h, s, v;
    color.getHsv(&h, &s, &v);
    color_dark.setHsv(h, 255, v);
    color_active.setHsv(h, 255, 100);
    isSynced = false;

    normalPen = new QPen(color_dark, 3, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);

    // create text label

    textItem = new MENodeText(this, graph->scene());
    textItem->setText(getNodeTitle());
    textItem->setToolTip(description);

    setPos(xx, yy);

    if (getHostname() != MEHostListHandler::instance()->getIPAddress(MEMainHandler::instance()->localHost))
        isLocal = false;
}

int MENode::getX() const
{
    return xx;
}

int MENode::getY() const
{
    return yy;
}

/*****************************************************************************
 *
 * Class MENodeBook
 *
 *****************************************************************************/

MENodeBook::MENodeBook(MENode *n, QGraphicsScene *, const QString &name)
    : QGraphicsSvgItem(name, n)
    , node(n)
{
    setScale(0.033);
    setCursor(Qt::PointingHandCursor);
    setToolTip("Open/Close the Module Parameter Window");
}

MENodeBook::~MENodeBook()
{
    // nothing necessary
}

void MENodeBook::mousePressEvent(QGraphicsSceneMouseEvent *)
{
    // do nothing also no base class processing e.g. selection
}

void MENodeBook::mouseReleaseEvent(QGraphicsSceneMouseEvent *)
{
    node->bookClick();
}

/*****************************************************************************
 *
 * Class MENodeMulti
 *
 *****************************************************************************/

MENodeMulti::MENodeMulti(MENode *n, QGraphicsScene *scene)
    : QGraphicsPixmapItem(n)
    , node(n)
{
    setCursor(Qt::PointingHandCursor);
    setPixmap(MEMainHandler::instance()->pm_expand);
    hide();
}

MENodeMulti::~MENodeMulti()
{
}

void MENodeMulti::mousePressEvent(QGraphicsSceneMouseEvent *)
{
    node->multiClick();
}

void MENodeMulti::mouseReleaseEvent(QGraphicsSceneMouseEvent *)
{
    // do nothing also no base class processing e.g. selection
}

/*****************************************************************************
 *
 * Class MENodetext
 *
 *****************************************************************************/

MENodeText::MENodeText(MENode *n, QGraphicsScene *scene)
    : QGraphicsSimpleTextItem(n)
    , node(n)
{
}

MENodeText::~MENodeText()
{
}

void MENodeText::contextMenuEvent(QGraphicsSceneContextMenuEvent *e)
{
    node->contextMenuEvent(e);
}

void MENodeText::mousePressEvent(QGraphicsSceneMouseEvent *e)
{
    node->mousePressEvent(e);
}

void MENodeText::mouseReleaseEvent(QGraphicsSceneMouseEvent *e)
{
    node->mouseReleaseEvent(e);
}

void MENodeText::mouseMoveEvent(QGraphicsSceneMouseEvent *e)
{
    ensureVisible();
    if (!node->isMoved)
        node->graph->wasMoved(node, e);
    node->isMoved = true;
    node->baseMouseMoveEvent(e);
}

void MENodeText::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *e)
{
    node->mouseDoubleClickEvent(e);
}
