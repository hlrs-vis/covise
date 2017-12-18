/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cmath>

#include <QGraphicsScene>
#include <QFrame>
#include <QDebug>
#include <QAction>
#include <QMenu>
#include <QScrollBar>
#include <QApplication>
#include <QClipboard>
#include <QGraphicsSceneContextMenuEvent>
#include <QStatusBar>
#include <QUrl>
#include <QMimeData>

#include <covise/covise_msg.h>
#ifdef YAC
#include "yac/coQTSendBuffer.h"
using covise::coErr;
#endif

#include "MEGraphicsView.h"
#include "MEUserInterface.h"
#include "MEFavorites.h"
#include "MEDialogTools.h"
#include "MEMessageHandler.h"
#include "nodes/MENode.h"
#include "handler/MEMainHandler.h"
#include "handler/MEPortSelectionHandler.h"
#include "handler/MENodeListHandler.h"
#include "handler/MELinkListHandler.h"
#include "handler/MEHostListHandler.h"

#include <ports/MEDataPort.h>

#include <climits>

class MEGraphicsScene : public QGraphicsScene
{
public:
    MEGraphicsScene()
        : QGraphicsScene()
    {
    }

    void dragEnterEvent(QGraphicsSceneDragDropEvent *ev)
    {
        ev->ignore();
        QGraphicsScene::dragEnterEvent(ev);
    }

    void dragMoveEvent(QGraphicsSceneDragDropEvent *ev)
    {
        ev->ignore();
        QGraphicsScene::dragMoveEvent(ev);
    }

    void dropEvent(QGraphicsSceneDragDropEvent *ev)
    {
        ev->ignore();
        QGraphicsScene::dropEvent(ev);
    }
};

/*!
   \class MEGraphicsView
   \brief This class provides the Visual Programming Area displaying module nodes and connection lines
*/

;

static const float viewportOversize = 1.f;

//======================================================================
MEGraphicsView::MEGraphicsView(QWidget *parent)
    : QGraphicsView(parent)
    , m_portIsMoving(false)
    , m_nodeIsMoving(false)
    , m_replaceMode(false)
    , m_dragMode(false)
    , m_clickedLine(NULL)
    , m_clickedPort(NULL)
    , m_connectionSourcePort(NULL)
    , m_renameBox(NULL)
    , m_selectedHostAction(NULL)
    , m_selectedCategoryAction(NULL)
    , m_lastPos(QPoint())
    , m_autoSelectNewNodes(false)
//======================================================================
{
}

//!
//! set the default values, create some actions
//!
void MEGraphicsView::init()
{
    const char *text1 = "<h4>Overview</h4>"
                        "<p>The <b>Visual Programming Area</b> (canvas) is used to show the module network. "
                        "Module icons, that can be moved around, and connection lines between module ports can be seen.</p>"
                        "<p>The execution of modules is indicated by highlighting the icon boundaries of currently executing modules.</p>"

                        "<h4>How to group modules</h4>"
                        "There are two methods for grouping module icons on the canvas: "

                        "<ol>"
                        "<li><em>Specific Selection</em><br>"

                        "Press the SHIFT-Key and click on a module icon. The icon background changes to "
                        "the selected highlight color."
                        "Doing the same action on an already selected module icon deselects it."

                        "<li><em>Selection via a rubberband</em><br>"

                        "Click on an empty part of the canvas to determine the startpoint of the "
                        "rubberband rectangle. Keep the mouse button pressed and move the mouse "
                        "so that module icons which should be grouped together lie completely inside "
                        "the rectangle. After the mouse button is released each of the icons inside the group"
                        "become red (the current hihglight color). Clicking on an empty part of the canvas ungroups the group. "

                        "<li><em>A selects all modules on the canvas</em>"
                        "</ol>";

    m_currViewRect.setCoords(-500, -500, 500, 500);
    setSceneRect(m_currViewRect);

    QGraphicsScene *scene = new MEGraphicsScene();
    connect(scene, SIGNAL(selectionChanged()), this, SLOT(selectionChangedCB()));
    scene->setItemIndexMethod(QGraphicsScene::NoIndex);
    setScene(scene);

    setSceneRect(m_currViewRect);
    setAcceptDrops(true);
    setCursor(Qt::ArrowCursor);
    setDragMode(QGraphicsView::RubberBandDrag);
    setRenderHints(QPainter::Antialiasing);
    setInteractive(true);
#if QT_VERSION >= 0x040400
    setViewportUpdateMode(BoundingRectViewportUpdate);
#endif
    setBackgroundBrush(QColor(230, 230, 230));
    //setRubberBandSelectionMode ( Qt::ContainsItemBoundingRect );

    // init all menus and actions
    initPopupStuff();

    // help text
    setWhatsThis(text1);

    // create a rubber line for selecting nodes
    m_rubberLine = scene->addLine(QLineF(0, 0, 1, 1));
    scene->removeItem(m_rubberLine);

    // store scroll bar
    m_scrollLines = QApplication::wheelScrollLines();

    connect(horizontalScrollBar(), SIGNAL(valueChanged(int)), this, SLOT(valueChangedCB(int)));
    connect(verticalScrollBar(), SIGNAL(valueChanged(int)), this, SLOT(valueChangedCB(int)));

    updateSceneRect();
}

MEGraphicsView::~MEGraphicsView()
{
}

void MEGraphicsView::developerMode(bool devmode)
{
    m_devmode = devmode;
    m_restartAction->setVisible(devmode);
    m_restartDebugAction->setVisible(devmode);
    m_restartMemcheckAction->setVisible(devmode);
}

MEGraphicsView *MEGraphicsView::instance()
{
    static MEGraphicsView *singleton = 0;
    if (singleton == 0)
    {
        singleton = new MEGraphicsView();
    }

    return singleton;
}

//!
//! init all popup and actions used inside the GraphicsView for nodes
//!
void MEGraphicsView::initPopupStuff()
{
    // init a popup menu for connection line deletion
    m_linePopup = new QMenu(0);
    m_deleteLineAction = m_linePopup->addAction("Delete Link");
    connect(m_deleteLineAction, SIGNAL(triggered()), this, SLOT(deleteLink()));

    // create all popup menus
    m_hostPopup = new QMenu(0);
    m_nodePopup = new QMenu(0);
    m_copyMovePopup = new QMenu(0);
    m_viewportPopup = new QMenu(0);

    // fill the popup menu for module nodes with actions
    m_execAction = m_nodePopup->addAction("Execute", this, SLOT(executeCB()));
    m_nodePopup->addSeparator();
    m_deleteAction = m_nodePopup->addAction("Delete", this, SLOT(deleteNodesCB()), QKeySequence::Delete);
    m_cutAction = m_nodePopup->addAction("Cut", this, SLOT(cut()), QKeySequence::Cut);
    m_copyAction = m_nodePopup->addAction("Copy", this, SLOT(copy()), QKeySequence::Copy);
    m_moveToAction = m_nodePopup->addAction("Move to");
    m_copyToAction = m_nodePopup->addAction("Copy to");
    m_nodePopup->addSeparator();
    m_restartAction = m_nodePopup->addAction("Restart", this, SLOT(restartCB()));
    m_restartDebugAction = m_nodePopup->addAction("Restart in Debugger", this, SLOT(restartDebugCB()));
    m_restartMemcheckAction = m_nodePopup->addAction("Restart with Memory Check", this, SLOT(restartMemcheckCB()));

    m_cloneAction = m_nodePopup->addAction("Clone", this, SLOT(cloneCB()));
    m_replaceAction = m_nodePopup->addAction("Replace with");
    m_renameAction = m_nodePopup->addAction("Rename...", this, SLOT(renameNodesCB()));
    m_nodePopup->addSeparator();
    m_paramAction = m_nodePopup->addAction("Parameters...", this, SLOT(paramCB()));
    m_nodePopup->addSeparator();
    m_helpAction = m_nodePopup->addAction("Help", this, SLOT(helpCB()));

    developerMode(MEMainHandler::instance()->cfg_DeveloperMode);
    connect(MEMainHandler::instance(), SIGNAL(developerMode(bool)), this, SLOT(developerMode(bool)));

    // associate submenus
    connect(m_copyToAction, SIGNAL(hovered()), this, SLOT(copyModuleCB()));
    connect(m_moveToAction, SIGNAL(hovered()), this, SLOT(moveModuleCB()));

    m_copyToAction->setMenu(m_copyMovePopup);
    m_moveToAction->setMenu(m_copyMovePopup);
    m_replaceAction->setMenu(m_hostPopup);

    // fill the global popup menu of the GraphicsView
    m_viewportPopup->addAction(MEUserInterface::instance()->m_exec_a);
    m_viewportPopup->addSeparator();
    m_viewportPopup->addAction(MEUserInterface::instance()->m_viewAll_a);
    m_viewportPopup->addAction(MEUserInterface::instance()->m_layoutMap_a);
    m_viewportPopup->addSeparator();
    m_viewportPopup->addAction(MEUserInterface::instance()->m_actionCut);
    m_viewportPopup->addAction(MEUserInterface::instance()->m_actionCopy);
    m_viewportPopup->addAction(MEUserInterface::instance()->m_actionPaste);
    m_viewportPopup->addAction(MEUserInterface::instance()->m_keyDelete_a);
    m_viewportPopup->addSeparator();
    m_viewportPopup->addAction(MEUserInterface::instance()->m_selectAll_a);
    m_viewportPopup->addAction(MEUserInterface::instance()->m_deleteAll_a);
}

//
//
void MEGraphicsView::dragEnterEvent(QDragEnterEvent *event)
{
    QGraphicsView::dragEnterEvent(event);
    if (event->isAccepted())
        return;

    if (event->mimeData()->hasFormat("application/x-qabstractitemmodeldatalist")
        || event->mimeData()->hasText()
        || event->mimeData()->hasUrls())
        event->accept();
}

//
//
void MEGraphicsView::dragMoveEvent(QDragMoveEvent *event)
{
    QGraphicsView::dragMoveEvent(event);
    if (event->isAccepted())
        return;

    if (event->mimeData()->hasFormat("application/x-qabstractitemmodeldatalist")
        || event->mimeData()->hasText()
        || event->mimeData()->hasUrls())
        event->accept();
}

//
//
void MEGraphicsView::dragLeaveEvent(QDragLeaveEvent *event)
{
    event->accept();
}

//!
//! decode drop events from favorite list, module tree or a network file
//!
void MEGraphicsView::dropEvent(QDropEvent *event)
{
    QGraphicsView::dropEvent(event);
    if (event->isAccepted())
        return;

    // from module tree
    if (event->mimeData()->hasFormat("application/x-qabstractitemmodeldatalist"))
    {
        QByteArray encodedData = event->mimeData()->data("application/x-qabstractitemmodeldatalist");
        QDataStream stream(&encodedData, QIODevice::ReadOnly);

        while (!stream.atEnd())
        {
            QString text;
            stream >> text;
            decodeMessage(event, text);
            event->accept();
        }
    }

    else if (event->mimeData()->hasUrls())
    {
        QString pathname = event->mimeData()->urls()[0].toLocalFile();
        if (!pathname.isEmpty())
        {
            event->accept();
            MEMainHandler::instance()->openDroppedMap(pathname);
        }
    }

    else if (event->mimeData()->hasText())
    {
        // user has dropped perhaps a network file
        if (event->source() == NULL)
        {
            MEMainHandler::instance()->openDroppedMap(event->mimeData()->text());
            event->accept();
        }

        // from favorite list
        else
        {
            QString text = event->mimeData()->text();
            decodeMessage(event, text);
            MEFavorites *fav = static_cast<MEFavorites *>(event->source());
            fav->setDown(false);
            event->accept();
        }
    }

    else
        event->ignore();
}

//!
//! scale the viewport, called when wheel mouse is used
//!
void MEGraphicsView::wheelEvent(QWheelEvent *e)
{
    // CTRL + scroll --> scale viewport
    if (e->modifiers() == Qt::ControlModifier)
        scaleView(pow(2., -e->delta() / 1000.0));

    else
        QGraphicsView::wheelEvent(e);
}

//!
//! scale the viewport, called also from MEMainHandler
//!
void MEGraphicsView::scaleView(qreal scaleFactor)
{
    QMatrix m = matrix();
    qreal factor = m.scale(scaleFactor, scaleFactor).mapRect(QRectF(0, 0, 1, 1)).width();
    factor = qMin(factor, 2.0);
    factor = qMax(factor, 0.25);

    emit factorChanged(factor);

    QMatrix m2(factor, m.m12(), m.m21(), factor, m.dx(), m.dy());

    QPointF viewsize = m2.map(QPoint(viewport()->size().width(), viewport()->size().height())) - m2.map(QPoint(0, 0));
    if (m_currViewRect.width() < viewsize.x() * viewportOversize)
    {
        m_currViewRect.setWidth(viewsize.x() * viewportOversize);
    }
    if (m_currViewRect.height() < viewsize.y() * viewportOversize)
    {
        m_currViewRect.setHeight(viewsize.y() * viewportOversize);
    }
    setSceneRect(m_currViewRect);

    setMatrix(m2);
}

//!
//! clipboard content changed
//!
void MEGraphicsView::selectionChangedCB()
{
    QList<QGraphicsItem *> list = scene()->selectedItems();
    if (list.empty())
        MEUserInterface::instance()->changeEditItems(false);
    else
        MEUserInterface::instance()->changeEditItems(true);
}

//!
//! scene content changed, get new position for module nodes
//!
void MEGraphicsView::valueChangedCB(int)
{
    m_freePos = mapToScene(viewport()->size().width() / 2, viewport()->size().height() / 2);

    // snap position to grid
    int gs = MEMainHandler::instance()->getGridSize();
    if (gs != 0)
    {
        m_freePos.rx() = ((int)((((float)m_freePos.rx()) / (float)gs) + (sign(int(m_freePos.rx())) * 0.5f))) * gs;
        m_freePos.ry() = ((int)((((float)m_freePos.ry()) / (float)gs) + (sign(int(m_freePos.rx())) * 0.5f))) * gs;
    }
}

//!
//! set a new drag mode either rubberbanding or moving the whole scene
//!
void MEGraphicsView::mousePressEvent(QMouseEvent *e)
{
    if (e->button() == Qt::RightButton)
    {
        clearPortSelections(MEPortSelectionHandler::Connectable);
        clearPortSelections(MEPortSelectionHandler::Clicked);
    }

    else if (e->button() == Qt::MidButton)
    {
        setCursor(Qt::OpenHandCursor);
        setDragMode(QGraphicsView::ScrollHandDrag);
        m_dragMode = true;
        QMouseEvent *mouseEvent = new QMouseEvent(QEvent::MouseButtonPress,
                                                  e->pos(),
                                                  e->globalPos(),
                                                  Qt::LeftButton,
                                                  Qt::LeftButton,
                                                  e->modifiers());
        e->accept();
        QGraphicsView::mousePressEvent(mouseEvent);
    }

    else
    {
        m_nodeIsMoving = false;
        m_oldViewRect = sceneRect();
        movedItemList.clear();
        QGraphicsView::mousePressEvent(e);
    }
}

//!
//! reset drag mode, send new node position or send a new connection line
//!
void MEGraphicsView::mouseReleaseEvent(QMouseEvent *e)
{
    if (e->button() == Qt::MidButton)
    {
        m_dragMode = false;
        QMouseEvent *mouseEvent = new QMouseEvent(QEvent::MouseButtonRelease,
                                                  e->pos(),
                                                  e->globalPos(),
                                                  Qt::LeftButton,
                                                  Qt::LeftButton,
                                                  e->modifiers());
        QGraphicsView::mouseReleaseEvent(mouseEvent);
    }

    // event worked by base class
    QGraphicsView::mouseReleaseEvent(e);

    // reset standard behaviour
    setCursor(Qt::ArrowCursor);
    setDragMode(QGraphicsView::RubberBandDrag);
    m_dragMode = false;

    // send new positions
    if (m_nodeIsMoving)
    {
        sendMoveMessage();
        m_nodeIsMoving = false;
        repaint();
    }

    // a connection line is possibly made with a rubber line
    else if (m_portIsMoving)
    {
        QPointF pp = mapToScene(e->pos());
        MEDataPort *port = qgraphicsitem_cast<MEDataPort *>(scene()->itemAt(pp, QTransform()));
        if (dynamic_cast<MEDataPort *>(m_connectionSourcePort)
            && port
            && port != m_connectionSourcePort
            && MEDataPort::arePortsCompatible(static_cast<MEDataPort *>(m_connectionSourcePort), port, false))
        {
            if (!port->isConnectable())
            {
                // this list has only one entry because the input port of a module can be connected only once
                // except for a Render module which is always connectable
                QVector<MENodeLink *> links = MELinkListHandler::instance()->getLinksIn(port);
                foreach (MENodeLink *link, links)
                    link->removeLines();
            }
            sendLine(m_connectionSourcePort, port);
        }
        if (port != m_connectionSourcePort)
        {
            clearPortSelections(MEPortSelectionHandler::Clicked);
            clearPortSelections(MEPortSelectionHandler::Connectable);
        }
    }
}

//!
//! either move nodes, the rubber liner or reset situation
//!
void MEGraphicsView::mouseMoveEvent(QMouseEvent *e)
{
    if (!MEMainHandler::instance()->isMaster())
        return;

    if (m_portIsMoving)
    {
        if (!m_rubberLine->scene())
            scene()->addItem(m_rubberLine);
        QPointF pp = mapToScene(e->pos().x(), e->pos().y());
        m_rubberLine->setLine(m_portPressedPosition.x(), m_portPressedPosition.y(), pp.x(), pp.y());
    }

    else if (m_dragMode)
    {
        QMouseEvent *mouseEvent = new QMouseEvent(QEvent::MouseMove,
                                                  e->pos(),
                                                  e->globalPos(),
                                                  Qt::LeftButton,
                                                  Qt::LeftButton,
                                                  e->modifiers());
        QGraphicsView::mouseMoveEvent(mouseEvent);
    }

    else if (m_nodeIsMoving)
    {
        QPointF pp = mapToScene(e->pos().x(), e->pos().y());
        m_currViewRect = sceneRect().united(QRectF(pp, QSizeF(1., 1.)));
        if (pp.x() > m_oldViewRect.right() + viewport()->width() / 4 * 3 - offxr)
            return;
        if (pp.x() < m_oldViewRect.left() - viewport()->width() / 4 * 3 + offxl)
            return;
        if (pp.y() > m_oldViewRect.bottom() + viewport()->height() / 4 * 3 - offyb)
            return;
        if (pp.y() < m_oldViewRect.top() - viewport()->height() / 4 * 3 + offyt)
            return;

        // move also the connection lines
        if (movedItemList.isEmpty())
            movedItemList = scene()->selectedItems();
        foreach (QGraphicsItem *item, movedItemList)
        {
            MENode *node = qgraphicsitem_cast<MENode *>(item);
            MELinkListHandler::instance()->resetLinks(node);
        }
        setSceneRect(m_currViewRect);
        QGraphicsView::mouseMoveEvent(e);
    }

    else
        QGraphicsView::mouseMoveEvent(e);
}

//!
//! react on a port hover event from MEPort class
//!
void MEGraphicsView::hoverEnterPort(MEPort *port)
{
    if (!m_connectionSourcePort)
        highlightMatchingPorts(MEPortSelectionHandler::HoverConnectable, port);
}

//!
//! react on a port hover leave from MEPort class
//!
void MEGraphicsView::hoverLeavePort(MEPort *)
{
    clearPortSelections(MEPortSelectionHandler::HoverConnected);
    clearPortSelections(MEPortSelectionHandler::HoverConnectable);
}

//!
//! react on a port press event from MEPort class
//!
void MEGraphicsView::portPressed(MEPort *port, QGraphicsSceneMouseEvent *e)
{
    // highlight corresponding ports
    // store current cursor position if user wants to use a rubber line
    // same port was clicked again --> deselect all ports
    if (port == m_clickedPort || port == m_connectionSourcePort)
    {
        clearPortSelections(MEPortSelectionHandler::Clicked);
        clearPortSelections(MEPortSelectionHandler::Connectable);
        m_clickedPort = NULL;
        m_connectionSourcePort = NULL;
    }

    else if (m_connectionSourcePort == NULL)
    {
        // delete old list & create a new one with matching ports
        clearPortSelections(MEPortSelectionHandler::Clicked);
        clearPortSelections(MEPortSelectionHandler::Connectable);
        highlightMatchingPorts(MEPortSelectionHandler::Connectable, port);

        m_clickedPort = port;

        // store source port && position
        if (MEPortSelectionHandler::instance()->count(MEPortSelectionHandler::Connectable) > 0)
        {
            m_connectionSourcePort = port;
            m_portPressedPosition = e->scenePos();
            MEPortSelectionHandler::instance()->addPort(MEPortSelectionHandler::Clicked, port);
            m_portIsMoving = true;

            MELinkListHandler::instance()->highlightPortAndLinks(port, false);
            MEGraphicsView::instance()->hoverLeavePort(port);
        }
    }
}

//!
//! react on a port release event from MEPort class
//!
void MEGraphicsView::portReleased(MEPort *port, QGraphicsSceneMouseEvent *)
{

    if (dynamic_cast<MEDataPort *>(m_connectionSourcePort)
        && dynamic_cast<MEDataPort *>(port))
    {
        if (port != m_connectionSourcePort
            && MEDataPort::arePortsCompatible(static_cast<MEDataPort *>(m_connectionSourcePort),
                                              static_cast<MEDataPort *>(port),
                                              false))
        {
            // remove existing connection line if input port is already connected
            if (!port->isConnectable())
            {
                // this list has only one entry because the input port of a module can be connected only once
                // except for a Render module which is always connectable
                QVector<MENodeLink *> links = MELinkListHandler::instance()->getLinksIn(port);
                foreach (MENodeLink *link, links)
                    link->removeLines();
            }
            sendLine(m_connectionSourcePort, port);
            clearPortSelections(MEPortSelectionHandler::Clicked);
            clearPortSelections(MEPortSelectionHandler::Connectable);
        }
        else
        {
            MEUserInterface::instance()->statusBar()->showMessage("Connection not possible");
        }
    }
}

//!
//! initiate port moving
//!
void MEGraphicsView::portMoved(MEPort *port)
{
    if (port == m_clickedPort)
        m_portIsMoving = true;
}

//!
//! node move event was catched
//!
void MEGraphicsView::wasMoved(QGraphicsItem *item, QGraphicsSceneMouseEvent *e)
{
    m_nodeIsMoving = true;
    QRectF box = item->boundingRect();
    qreal x = e->pos().x();
    qreal y = e->pos().y();
    offxr = box.width() - x;
    offxl = x;
    offyb = box.height() - y;
    offyt = y;
}

//!
//! compute length of longest path from bottom ("Renderer") to node
//!
void MEGraphicsView::computeDepths(MENode *node1, QMap<MENode *, int> *depthMap)
{
    foreach (MEPort *port1, node1->doutlist)
    {
        QVector<MENodeLink *> links = MELinkListHandler::instance()->getLinksOut(port1);
        foreach (MENodeLink *link, links)
        {
            MEPort *port2 = link->port2;
            MENode *node2 = port2->getNode();
            computeDepths(node2, depthMap);
            if ((*depthMap)[node1] < (*depthMap)[node2] + 1)
                (*depthMap)[node1] = (*depthMap)[node2] + 1;
        }
    }
}

//!
//! lay out module graph vertically
//!
void MEGraphicsView::layoutVertical(QMap<MENode *, int> *depth, int maxDepth, int maxY)
{
    const int gs = MEMainHandler::instance()->getGridSize();
    const int nodeheight = (90 + gs - 1) / gs * gs;
    QVector<MENode *> nodes = MENodeListHandler::instance()->getNodes();
    foreach (MENode *node, nodes)
    {
        // move all nodes w/o input to the top
        bool noinput = true;
        foreach (MEPort *port2, node->dinlist)
        {
            QVector<MENodeLink *> links = MELinkListHandler::instance()->getLinksIn(port2);
            if (links.count() > 0)
            {
                noinput = false;
                break;
            }
        }
        if (noinput)
            (*depth)[node] = maxDepth;

        // position according to depth
        node->moveNode(node->getX(), maxY - (*depth)[node] * nodeheight);
        node->sendNodeMessage("MOV");
        MELinkListHandler::instance()->resetLinks(node);
    }
}

//!
//! lay out module graph horizontally
//!
void MEGraphicsView::layoutHorizontal(const QMap<MENode *, int> depth, int maxDepth)
{
    QVector<MENode *> nodes = MENodeListHandler::instance()->getNodes();
    const int gs = MEMainHandler::instance()->getGridSize();
    const int nodewidth = (150 + gs - 1) / gs * gs;

    // look for the level with the most modules
    int maxLevel = 0;
    int maxModules = 0;
    for (int curdepth = 0; curdepth <= maxDepth; ++curdepth)
    {
        int numModules = 0;
        foreach (MENode *node, nodes)
        {
            if (depth[node] == curdepth)
            {
                bool connected = false;
                foreach (MEPort *p, node->doutlist)
                {
                    if (MELinkListHandler::instance()->getLinksOut(p).count() > 0)
                    {
                        connected = true;
                        break;
                    }
                }
                if (!connected)
                {
                    foreach (MEPort *p, node->dinlist)
                    {
                        if (MELinkListHandler::instance()->getLinksIn(p).count() > 0)
                        {
                            connected = true;
                            break;
                        }
                    }
                }
                if (connected)
                    ++numModules;
            }
        }
        if (numModules > maxModules)
        {
            maxModules = numModules;
            maxLevel = curdepth;
        }
    }

    // space the modules equally in the level with the most modules
    QMap<MENode *, int> horpos;
    int xx = 0;
    foreach (MENode *node, nodes)
    {
        if (depth[node] != maxLevel)
            continue;
        horpos[node] = xx;
        node->moveNode(xx, node->getY());
        node->sendNodeMessage("MOV");
        MELinkListHandler::instance()->resetLinks(node);
        xx += nodewidth + gs;
    }

    // loop over all other levels, starting with the neighbours of the most populated level
    for (int i = 2; i < (maxDepth + 1) * 2; ++i)
    {
        int curdepth = (i % 2) ? maxLevel + i / 2 : maxLevel - i / 2;

        if (curdepth < 0)
            continue;
        if (curdepth > maxDepth)
            continue;
        if (curdepth == maxLevel)
            continue;

        // collect all nodes of current level
        int maxLinks = 0;
        QMap<MENode *, int> curnodes, unprocessedNodes;
        QVector<MENode *> processedNodes;
        foreach (MENode *node, nodes)
        {
            if (depth[node] != curdepth)
                continue;
            curnodes[node] = 0;
            if (curdepth > maxLevel)
            {
                foreach (MEPort *port1, node->doutlist)
                {
                    QVector<MENodeLink *> links = MELinkListHandler::instance()->getLinksOut(port1);
                    curnodes[node] += links.count() ? 1 : 0;
                }
            }
            else
            {
                foreach (MEPort *port2, node->dinlist)
                {
                    QVector<MENodeLink *> links = MELinkListHandler::instance()->getLinksIn(port2);
                    curnodes[node] += links.count() ? 1 : 0;
                }
            }
            if (curnodes[node] > maxLinks)
                maxLinks = curnodes[node];
        }

        //std::cerr << "current depth: " << curdepth << ", " << curnodes.count() << " nodes to process" << std::endl;

        // position those nodes with the most connected ports first
        int maxpos = INT_MIN;
        int minpos = INT_MAX;
        for (int i = maxLinks * 2; i >= 0; --i)
        {
            // in a first pass, try to position all modules such that at least one straight
            // connection is achieved,
            // otherwise put the node onto unprocessedNodes and place it in a second pass
            while (MENode *node = i > maxLinks
                                      ? curnodes.key(i - maxLinks, NULL)
                                      : i == 0 ? curnodes.key(0, NULL) : unprocessedNodes.key(i, NULL))
            {
                //std::cerr << "positioning " << node->getTitle().toStdString() << std::endl;
                unprocessedNodes[node] = curnodes[node];
                curnodes.remove(node);
                int otherPortIndex = -1, portIndex = -1;
                MENode *otherNode = NULL;
                bool available = false;
                bool out = curdepth > maxLevel;
                int connectioncenter = 0;
                int numconn = 0;
                foreach (MEPort *port, out ? node->doutlist : node->dinlist)
                {
                    QVector<MENodeLink *> links = out
                                                      ? MELinkListHandler::instance()->getLinksOut(port)
                                                      : MELinkListHandler::instance()->getLinksIn(port);
                    available = false;
                    numconn += links.count();
                    foreach (MENodeLink *link, links)
                    {
                        // determine index of port in current module
                        portIndex = 0;
                        foreach (MEPort *p, out ? node->doutlist : node->dinlist)
                        {
                            if (p == port)
                                break;
                            ++portIndex;
                        }

                        // determine index of port in connected module
                        otherNode = out ? link->port2->getNode() : link->port1->getNode();
                        otherPortIndex = 0;
                        foreach (MEPort *p, out ? otherNode->dinlist : otherNode->doutlist)
                        {
                            if (p == (out ? link->port2 : link->port1))
                                break;
                            ++otherPortIndex;
                        }
                        //std::cerr << "   link " << portIndex << " -> " << otherPortIndex << std::endl;
                        connectioncenter += gs * otherPortIndex + otherNode->getX();

                        // try to position such that connection is a straight line
                        int pos = otherNode->getX() + (otherPortIndex - portIndex) * gs;
                        available = true;
                        foreach (MENode *n, processedNodes)
                        {
                            //std::cerr << "      testing against " << n->getTitle().toStdString() << std::endl;
                            if (abs(horpos[n] - pos) < nodewidth)
                            {
                                // space already occupied by another module
                                //std::cerr << "      Collision!!!" << std::endl;
                                available = false;
                                break;
                            }
                        }
                        if (available)
                            break;
                    }
                    if (available)
                        break;
                }

                int pos = 0;
                if (!otherNode)
                {
                    if (minpos != INT_MAX)
                        pos = abs(minpos - nodewidth) < abs(maxpos + nodewidth)
                                  ? minpos - nodewidth
                                  : maxpos + nodewidth;
                }
                else
                {
                    pos = otherNode->getX() + (otherPortIndex - portIndex) * gs;

                    if (!available && i < maxLinks)
                    {
                        // as a last resort, try to find a position that is as close as
                        // possible to the connected ports
                        bool occupied = true;
                        for (int i = 0; i < maxpos - minpos; i += gs)
                        {
                            occupied = false;
                            pos = (connectioncenter / numconn + gs / 2 + i * gs) / gs * gs;
                            foreach (MENode *n, processedNodes)
                            {
                                if (abs(horpos[n] - pos) < nodewidth)
                                {
                                    occupied = true;
                                    break;
                                }
                            }
                            if (occupied)
                            {
                                occupied = false;
                                pos = (connectioncenter / numconn + gs / 2 - i * gs) / gs * gs;
                                foreach (MENode *n, processedNodes)
                                {
                                    if (abs(horpos[n] - pos) < nodewidth)
                                    {
                                        occupied = true;
                                        break;
                                    }
                                }
                            }
                            if (!occupied)
                                break;
                        }
                        if (occupied)
                        {
                            pos = (connectioncenter / numconn + gs / 2) / gs * gs;
                            if (abs(pos - minpos) < abs(pos - maxpos))
                                pos = (minpos - nodewidth) / gs * gs;
                            else
                                pos = (maxpos + nodewidth + gs - 1) / gs * gs;
                        }
                    }
                }

                // actually move the module
                if (available || i < maxLinks)
                {
                    if (pos > maxpos)
                        maxpos = pos;
                    if (pos < minpos)
                        minpos = pos;
                    horpos[node] = pos;
                    node->moveNode(pos, node->getY());
                    node->sendNodeMessage("MOV");
                    MELinkListHandler::instance()->resetLinks(node);
                    unprocessedNodes.remove(node);
                    processedNodes.push_back(node);
                }
            }
        }
    }
}

//!
//! automatically position module nodes
//!
void MEGraphicsView::layoutMap()
{
    QVector<MENode *> nodes = MENodeListHandler::instance()->getNodes();

    int maxY = INT_MIN;
    QMap<MENode *, int> depth;
    foreach (MENode *node, nodes)
    {
        depth[node] = 0;
        if (node->getY() > maxY)
            maxY = node->getY();
    }

    int maxDepth = 0;
    foreach (MENode *node, nodes)
    {
        computeDepths(node, &depth);
        if (depth[node] > maxDepth)
            maxDepth = depth[node];
    }

    layoutVertical(&depth, maxDepth, maxY);
    layoutHorizontal(depth, maxDepth);

    QCoreApplication::processEvents();
    viewAll();
}

//!
//! show all items in the GraphicsView
//!
void MEGraphicsView::viewAll(qreal scaleMax)
{
    QRectF r = scene()->itemsBoundingRect();
    m_currViewRect = r;
    setSceneRect(r);
    fitInView(r, Qt::KeepAspectRatio);
    const QMatrix &m = matrix();
    qreal factor = m.m11();
    factor = qMin(factor, 2.0);
    if (scaleMax > 0.)
        factor = qMin(factor, scaleMax);
    QMatrix m2(factor, 0., 0., factor, r.center().x(), r.center().y());
    setMatrix(m2);
    MEUserInterface::instance()->updateScaleViewCB(factor);
    updateSceneRect();
}

//!
//! scale to 100%
//!

void MEGraphicsView::scaleView100()
{
    MEUserInterface::instance()->scaleViewCB("100%");
}

//!
//! scale to 50%
//!

void MEGraphicsView::scaleView50()
{
    MEUserInterface::instance()->scaleViewCB("50%");
}

//!
//! show the map after reading
//!
void MEGraphicsView::showMap()
{
    QPointF pos = (scene()->itemsBoundingRect()).topLeft();
    QPointF viewPoint = matrix().map(pos);

    horizontalScrollBar()->setValue((int)viewPoint.x());
    verticalScrollBar()->setValue((int)viewPoint.y());

    viewAll(1.0);
}

//!
//! update scene rectangle after adding  or moving nodes
//!
void MEGraphicsView::updateSceneRect()
{
    m_currViewRect = m_currViewRect.united(scene()->itemsBoundingRect());
    QPointF viewsize = mapToScene(viewport()->size().width(), viewport()->size().height()) - mapToScene(0, 0);
    QPointF center = m_currViewRect.center();
    bool updateCenter = false;
    if (m_currViewRect.width() < viewsize.x() * viewportOversize)
    {
        m_currViewRect.setWidth(viewsize.x() * viewportOversize);
        updateCenter = true;
    }
    if (m_currViewRect.height() < viewsize.y() * viewportOversize)
    {
        m_currViewRect.setHeight(viewsize.y() * viewportOversize);
        updateCenter = true;
    }
    if (updateCenter)
        m_currViewRect.moveCenter(center);
    setSceneRect(m_currViewRect);
}

//!
//! return a free position for module nodes
//!
QPointF MEGraphicsView::getFreePos()
{
    m_freePos.rx() = m_freePos.rx() - 20.;
    m_freePos.ry() = m_freePos.ry() - 20.;
    return m_freePos;
}

//!
//! change the master/slave status
//!
void MEGraphicsView::setMasterState(bool state)
{
    // set the right permission for master/slave
    m_deleteLineAction->setEnabled(state);
    m_execAction->setEnabled(state);
    m_deleteAction->setEnabled(state);
    m_restartAction->setEnabled(state);
    m_restartDebugAction->setEnabled(state);
    m_restartMemcheckAction->setEnabled(state);
    m_cloneAction->setEnabled(state);
    m_renameAction->setEnabled(state);
    m_replaceAction->setEnabled(state);
    m_copyToAction->setEnabled(state);
    m_moveToAction->setEnabled(state);
    m_copyAction->setEnabled(state);
    m_cutAction->setEnabled(state);
}

//!
//! enable/disable copy/move items in node popup, used for nodes in mirror state
//!
void MEGraphicsView::disableNodePopupItems(bool state)
{
    m_restartAction->setEnabled(state);
    m_restartDebugAction->setEnabled(state);
    m_restartMemcheckAction->setEnabled(state);
    m_cloneAction->setEnabled(state);
    m_replaceAction->setEnabled(state);
}

//!
//! enable/disable parameter item in node popup, used if mode has no parameter
//!
void MEGraphicsView::disableParameterItem(bool state)
{
    m_paramAction->setEnabled(state);
}

//!
//! set node popup for a single host
//!
void MEGraphicsView::setSingleHostNodePopup()
{
    m_copyToAction->setVisible(false);
    m_moveToAction->setVisible(false);
    m_cloneAction->setVisible(true);
    m_cutAction->setVisible(true);
    m_copyAction->setVisible(true);

    QMenu *menu = (MEHostListHandler::instance()->getFirstHost())->getMenu();
    m_selectedHostAction = (MEHostListHandler::instance()->getFirstHost())->getHostAction();
    m_replaceAction->setMenu(menu);
}

//!
//! set node popup for a single host
//!
void MEGraphicsView::setMultiHostNodePopup()
{
    m_copyToAction->setVisible(true);
    m_moveToAction->setVisible(true);
    m_cloneAction->setVisible(false);
    m_cutAction->setVisible(false);
    m_copyAction->setVisible(false);

    m_replaceAction->setMenu(m_hostPopup);
}

//!
//! react on a port click with right mouse button, show possible ports for connecting
//!
void MEGraphicsView::showPossiblePorts(MEDataPort *port, QGraphicsSceneContextMenuEvent *e)
{
    MEPortSelectionHandler::instance()->showPossiblePorts(port, e);
}

//!
//! react on a node click with right mouse button, show a node menu
//!
void MEGraphicsView::showNodeActions(MENode *node, QGraphicsSceneContextMenuEvent *e)
{
    m_popupNode = node;
    m_replaceAction->setEnabled(false);
    m_renameAction->setText("Add prefix...");
    m_restartAction->setVisible(m_devmode || node->isDead());

    // look how many nodes are selected
    QList<QGraphicsItem *> list = scene()->selectedItems();

    // don't allow replace operation for more than one selected node
    if (list.count() == 1 && list.contains(node))
    {
        m_renameAction->setText("Rename...");
        if (MEMainHandler::instance()->isMaster())
            m_replaceAction->setEnabled(true);
    }

    // show popup menu
    if (!node->isCloned())
        m_nodePopup->popup(e->screenPos());
}

//!
//! show context menu
//!
void MEGraphicsView::contextMenuEvent(QContextMenuEvent *e)
{
    if (itemAt(e->pos()))
        QGraphicsView::contextMenuEvent(e);
    else
        m_viewportPopup->popup(e->globalPos());
}

//!
//! action callback copy
//!
void MEGraphicsView::copy()
{
#ifndef YAC
    QList<QGraphicsItem *> copyList = scene()->selectedItems();
    if (copyList.isEmpty())
        return;

    QStringList buffer;
    buffer << "SETCLIPBOARD";
    buffer << QString::number(copyList.count());

    foreach (QGraphicsItem *item, copyList)
    {
        MENode *node = qgraphicsitem_cast<MENode *>(item);
        if (node)
            buffer << node->getName() << node->getNumber() << node->getHostname();
    }

    QString data = buffer.join("\n");
    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
    buffer.clear();
#endif
}

//!
//! action callback cut
//!
void MEGraphicsView::cut()
{
    copy();
    deleteNodesCB();
}

//!
//! action callback paste
//!
void MEGraphicsView::paste()
{
#ifndef YAC
    QByteArray stream = QApplication::clipboard()->mimeData()->data("covise/clipboard");
    QString text(stream);
    if (text.startsWith("SETCLIPBOARD"))
    {
        text.replace(0, 1, "G");
        MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, text);
    }
#endif
}

//!
//! popup callback execute
//!
void MEGraphicsView::executeCB()
{
    m_popupNode->executeCB();
}

//!
//! popup callback delete selected nodes
//!
void MEGraphicsView::deleteNodesCB()
{

    // get list of selected items
    QList<QGraphicsItem *> list = scene()->selectedItems();

#ifdef YAC

    covise::coSendBuffer sb;
    sb << list.count();
    foreach (QGraphicsItem *item, list)
    {
        MENode *node = qgraphicsitem_cast<MENode *>(item);
        if (node)
            sb << node->getNodeID();
    }
    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_NODE_DELETE, sb);

#else

    QStringList buffer;
    buffer << "DEL" << QString::number(list.count());
    foreach (QGraphicsItem *item, list)
    {
        MENode *node = qgraphicsitem_cast<MENode *>(item);
        if (node)
            buffer << node->getName() << node->getNumber() << node->getHostname();
    }

    QString data = buffer.join("\n");
    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
    buffer.clear();
#endif

    MEMainHandler::instance()->mapWasChanged("DEL");
}

//!
//! popup callback restart
//!
void MEGraphicsView::restartCB()
{
    MEHost *host = MEHostListHandler::instance()->getFirstHost();
    sendCopyList(host, MEMainHandler::MOVE);
}

//!
//! popup callback restart memcheck
//!
void MEGraphicsView::restartMemcheckCB()
{
    MEHost *host = MEHostListHandler::instance()->getFirstHost();
    sendCopyList(host, MEMainHandler::MOVE_MEMCHECK);
}

//!
//! popup callback restart debug
//!
void MEGraphicsView::restartDebugCB()
{
    MEHost *host = MEHostListHandler::instance()->getFirstHost();
    sendCopyList(host, MEMainHandler::MOVE_DEBUG);
}

//!
//! popup callback clone
//!
void MEGraphicsView::cloneCB()
{
    MEHost *host = MEHostListHandler::instance()->getFirstHost();
    sendCopyList(host, MEMainHandler::COPY);
}

//!
//! popup callback rename node name, create a dialog box
//!
void MEGraphicsView::renameNodesCB()
{
    MENode *node = m_popupNode;

    if (m_renameBox)
        delete m_renameBox;

    QList<QGraphicsItem *> list = scene()->selectedItems();
    if (list.count() == 1)
        m_renameBox = new MERenameDialog(MERenameDialog::SINGLE, node->getTitle());

    else
        m_renameBox = new MERenameDialog(MERenameDialog::GROUP, " ");

    m_renameBox->exec();
}

//!
//! popup callback parameter
//!
void MEGraphicsView::paramCB()
{
    m_popupNode->bookClick();
}

//!
//! popup callback help
//!
void MEGraphicsView::helpCB()
{
    m_popupNode->helpCB();
}

//!
//! callback from replace popup
//!
void MEGraphicsView::replaceModulesCB()
{
    // get action, text contains the modulename
    QAction *ac = (QAction *)sender();
    QString newName = ac->text();

    // get host
    if (!m_selectedHostAction)
        return;

    QVariant var = m_selectedHostAction->data();
    MEHost *host = var.value<MEHost *>();

    if (!host)
        return;

    // assemble message
    QStringList buffer;
    buffer << "REPLACE";
    int key = -1;
    int xx = int(m_popupNode->x() / positionScaleX());
    int yy = int(m_popupNode->y() / positionScaleY());
    buffer << newName << QString::number(key) << host->getIPAddress();
    buffer << QString::number(xx) << QString::number(yy);
    buffer << m_popupNode->getName() << m_popupNode->getNumber() << m_popupNode->getHostname();

#ifndef YAC
    QString data = buffer.join("\n");
    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
    buffer.clear();
#endif
}

//!
//! callback from move action
//!
void MEGraphicsView::moveModuleCB()
{
    // store mode when action is hovered
    m_currentMode = MEMainHandler::MOVE;
}

//!
//! callback from copy action
//!
void MEGraphicsView::copyModuleCB()
{
    // store mode when action is hovered
    m_currentMode = MEMainHandler::COPY;
}

//!
//! callback from category action
//!
void MEGraphicsView::hoveredCategoryCB()
{
    m_selectedCategoryAction = (QAction *)sender();
}

//!
//! callback from host action (replace mode)
//!
void MEGraphicsView::hoveredHostCB()
{
    m_selectedHostAction = (QAction *)sender();
}

//!
//! callback from host action (copy-move mode)
//!
void MEGraphicsView::triggeredHostCB()
{
    QAction *ac = (QAction *)sender();
    QVariant var = ac->data();
    MEHost *host = var.value<MEHost *>();
    sendCopyList(host, m_currentMode);
}

//!
//! react on a connection link click with right mouse button, event from MENodeLink
//!
void MEGraphicsView::deletePossibleLink(MENodeLink *link, QGraphicsSceneContextMenuEvent *e)
{
    m_clickedLine = link;
    m_linePopup->popup(e->screenPos());
}

//!
//! delete a connection link
//!
void MEGraphicsView::deleteLink()
{
    m_clickedLine->removeLines();
    m_clickedLine = NULL;
}

//!
//! decode the dropped mesage, initialise a module node
//!
void MEGraphicsView::decodeMessage(QDropEvent *event, QString text)
{
    const int gs = MEMainHandler::instance()->getGridSize();

    // snap position to grid
    QPointF pp = mapToScene(event->pos());
    if (gs != 0)
    {
        pp.setX(((int)((((float)pp.x()) / (float)gs) + (sign(int(pp.x())) * 0.5f))) * gs);
        pp.setY(((int)((((float)pp.y()) / (float)gs) + (sign(int(pp.y())) * 0.5f))) * gs);
    }

    qreal x = pp.x();
    qreal y = pp.y();

    // get normal info out of textstring

    QString hostname = text.section(':', 0, 0);
    QString username = text.section(':', 1, 1);
    QString category = text.section(':', 2, 2);
    QString modulename = text.section(':', 3, 3);

    // send signal which is caught from MEModuleTree
    emit usingNode(category + ":" + modulename);

// send message to controller to start the module
#ifdef YAC

    covise::coSendBuffer sb;
    int id = hostname.toInt();
    sb << id << category << modulename << int(x) << int(y);
    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_START_MODULE, sb);
    MEMainHandler::instance()->mapWasChanged("UI_START_MODULE");

#else

    MEMainHandler::instance()->requestNode(modulename, hostname, int(x), int(y), NULL, MEMainHandler::NORMAL);
#endif
}

//!
//! reset internal lists
//!
void MEGraphicsView::reset()
{
    scene()->clearSelection();
    clearPortSelections(MEPortSelectionHandler::Clicked);
    clearPortSelections(MEPortSelectionHandler::HoverConnected);
    clearPortSelections(MEPortSelectionHandler::HoverConnectable);
    clearPortSelections(MEPortSelectionHandler::Connectable);
}

//!
//! reset selected port
//!
void MEGraphicsView::clearPortSelections(MEPortSelectionHandler::Type type)
{
    MEPortSelectionHandler::instance()->clear(type);
    if (type == MEPortSelectionHandler::Clicked)
    {
        m_connectionSourcePort = NULL;
        m_clickedPort = NULL;
        m_portIsMoving = false;
        if (m_rubberLine->scene())
            scene()->removeItem(m_rubberLine);
    }
}

//!
//! highlight matching ports, event from MEPort class
//!
void MEGraphicsView::highlightMatchingPorts(MEPortSelectionHandler::Type type, MEPort *port)
{
    // search for matching ports
    MENodeListHandler::instance()->searchMatchingPorts(type, port);

    if (type == MEPortSelectionHandler::Connectable)
    {
        // no matching port found at all, reset ports
        if (MEPortSelectionHandler::instance()->isEmpty(type))
            clearPortSelections(type);
        // highlight port
        else
        {
            MEPortSelectionHandler::instance()->clear(MEPortSelectionHandler::Clicked);
            MEPortSelectionHandler::instance()->addPort(MEPortSelectionHandler::Clicked, port);
        }
    }
}

//!
//! send a message to connect two ports
//!
void MEGraphicsView::sendLine(MEPort *from, MEPort *to)
{
#ifdef YAC

    covise::coSendBuffer sb;

    if (from == NULL || to == NULL)
    {
        LOGWARNING("NULL -- watch out!");
        return;
    }

    if (from->getPortType() == MEPort::DIN || from->getPortType() == MEPort::MULTY_IN || from->getPortType() == MEPort::PIN)
        sb << to->getNode()->getNodeID() << to->getName() << from->getNode()->getNodeID() << from->getName();
    else
        sb << from->getNode()->getNodeID() << from->getName() << to->getNode()->getNodeID() << to->getName();

    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_PORT_CONNECT, sb);
    MEMainHandler::instance()->mapWasChanged("UI_PORT_CONNECT");

#else
    QStringList buffer;
    QString data;
    MEPort *port1, *port2;
    MENode *node1, *node2;

    if (from == NULL || to == NULL)
    {
        qWarning() << "MEGraphicsView::sendLine __ NULL -- watch out!";
        return;
    }

    if (from->getPortType() == MEPort::DOUT)
    {
        port1 = from;
        port2 = to;
    }

    else
    {
        port2 = from;
        port1 = to;
    }

    node1 = port1->getNode();
    node2 = port2->getNode();
    if (node1 == node2)
        return;

    if (MEMainHandler::instance()->isMaster())
    {

        buffer << "OBJCONN";
        buffer << node1->getName() << node1->getNumber() << node1->getHostname() << port1->getName();
        buffer << node2->getName() << node2->getNumber() << node2->getHostname() << port2->getName();
        data = buffer.join("\n");

        MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
        buffer.clear();
        MEMainHandler::instance()->mapWasChanged("OBJCONN");

        // look if there are synced nodes
        /*int imax = qMin( node1->m_syncList.count(), node2->m_syncList.count());
      for( int i = 0; i< imax; i++)
      {
         MENode *n1 = node1->m_syncList.at(i);
         MENode *n2 = node2->m_syncList.at(i);
         if(n1 && n2)
         {
            MEDataPort *p1 = n1->getDataPort(port1->getName());
            MEDataPort *p2 = n2->getDataPort(port2->getName());
            if(p1 && p2)
            {
               buffer << "OBJCONN";
               buffer << n1->getName() << n1->getNumber() << n1->getHostname() << p1->getName();
               buffer << n2->getName() << n2->getNumber() << n2->getHostname() << p2->getName() ;
               data = buffer.join("\n");

               MEMessageHandler::instance()->sendMessage(COVISE_MESSAGE_UI, data);
               buffer.clear();
            }
         }
      }*/
    }
#endif

    clearPortSelections(MEPortSelectionHandler::Connectable);
    clearPortSelections(MEPortSelectionHandler::Clicked);
}

//!
//! rename selected nodes
//!
void MEGraphicsView::renameNodes(const QString &text)
{
    QList<QGraphicsItem *> list = scene()->selectedItems();
    if (list.count() == 1)
    {
        m_popupNode->sendRenameMessage(text);
    }

    else
    {
        foreach (QGraphicsItem *item, list)
        {
            MENode *node = qgraphicsitem_cast<MENode *>(item);
            if (node)
            {
                QString tmp = text + "-" + node->getNodeTitle();
                node->sendRenameMessage(tmp);
            }
        }
    }
}

//!
//! end of moving nodes, send message to controller, snap position to grid
//!
void MEGraphicsView::sendMoveMessage()
{
    const int gs = MEMainHandler::instance()->getGridSize();

#ifdef YAC

    covise::coSendBuffer sb;

    sb << movedItemList.count();

    foreach (QGraphicsItem *item, movedItemList)
    {
        MENode *node = qgraphicsitem_cast<MENode *>(item);
        if (node)
        {
            int x = (int)node->x();
            int y = (int)node->y();
            x = (int)(x / positionScaleX());
            y = (int)(y / positionScaleY());
            if (gs != 0)
            {
                x = ((int)((((float)x) / (float)gs) + (sign(x) * 0.5f))) * gs;
                y = ((int)((((float)y) / (float)gs) + (sign(y) * 0.5f))) * gs;
            }
            sb << node->getNodeID() << x << y;
            MELinkListHandler::instance()->resetLinks(node);
        }
    }
    MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_NODE_MOVE, sb);

#else

    // snap position to grid

    QStringList buffer;
    buffer << "MOV" << QString::number(movedItemList.count());
    foreach (QGraphicsItem *item, movedItemList)
    {
        MENode *node = qgraphicsitem_cast<MENode *>(item);
        if (node)
        {
            int x = (int)node->x();
            int y = (int)node->y();
            x = (int)(x / positionScaleX());
            y = (int)(y / positionScaleY());
            if (gs != 0)
            {
                x = ((int)((((float)x) / (float)gs) + (sign(x) * 0.5f))) * gs;
                y = ((int)((((float)y) / (float)gs) + (sign(y) * 0.5f))) * gs;
            }
            node->moveNode((int)(x * positionScaleX()), (int)(y * positionScaleY()));
            buffer << node->getName() << node->getNumber() << node->getHostname();
            buffer << QString::number(x) << QString::number(y);
        }
    }
    QString data = buffer.join("\n");
    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
    buffer.clear();

#endif
}

//!
//! add a host to the popup menu
//!
void MEGraphicsView::addHost(MEHost *host)
{
    m_hostPopup->addAction(host->getHostAction());
    m_copyMovePopup->addAction(host->getCopyMoveAction());
}

//!
//! delete a host from the popup menu
//!
void MEGraphicsView::removeHost(MEHost *host)
{
    m_hostPopup->removeAction(host->getHostAction());
    m_copyMovePopup->addAction(host->getCopyMoveAction());
}

#ifdef YAC

//!
//! send a list of nodes that have to be copied/moved to the controller
//!
void MEGraphicsView::sendCopyList(MEHost *host, int mode)
{

    covise::coSendBuffer sb;

    // host id
    if (host)
        sb << host->getID();
    else
        sb << 0;

    // get selected list
    QList<QGraphicsItem *> copyList = scene()->selectedItems();

    // no. of nodes
    sb << (int)copyList.count();

    // list of node ids
    foreach (QGraphicsItem *item, copyList)
    {

        MENode *node = qgraphicsitem_cast<MENode *>(item);
        if (node)
            sb << node->getNodeID();
    }

    if (mode == MEMainHandler::COPY)
        MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_COPY, sb);
    else
        MEMessageHandler::instance()->sendMessage(covise::coUIMsg::UI_MOVE, sb);
}

#else

//!
//! send a list of nodes that have to be copied/moved to the controller
//!
void MEGraphicsView::sendCopyList(MEHost *host, int mode)
{

    QStringList buffer;
    if (mode == MEMainHandler::COPY)
        buffer << "COPY2";

    else if (mode == MEMainHandler::MOVE)
        buffer << "MOVE2";

    else if (mode == MEMainHandler::MOVE_DEBUG)
        buffer << "MOVE2_DEBUG";

    else if (mode == MEMainHandler::MOVE_MEMCHECK)
        buffer << "MOVE2_MEMCHECK";

    QList<QGraphicsItem *> copyList = scene()->selectedItems();

    if (mode == MEMainHandler::COPY)
    {
        scene()->clearSelection();
        m_autoSelectNewNodes = true;
    }

    buffer << QString::number(copyList.count());
    buffer << QString::number(mode == MEMainHandler::COPY ? MEMainHandler::COPY : MEMainHandler::MOVE);

    foreach (QGraphicsItem *item, copyList)
    {
        MENode *node = qgraphicsitem_cast<MENode *>(item);
        if (node)
        {
            int key = -1;
            int xx = int(node->x() / positionScaleX());
            int yy = int(node->y() / positionScaleY());
            if (mode == MEMainHandler::COPY)
                xx = xx + 20 + int(node->boundingRect().width());
            buffer << node->getName() << QString::number(key) << host->getIPAddress();
            buffer << QString::number(xx) << QString::number(yy);
            buffer << node->getName() << node->getNumber() << node->getHostname();
        }
    }

    QString data = buffer.join("\n");
    MEMessageHandler::instance()->sendMessage(covise::COVISE_MESSAGE_UI, data);
    buffer.clear();
}
#endif

float MEGraphicsView::positionScaleX() const
{
    return 1.0;
}

float MEGraphicsView::positionScaleY() const
{
    return 1.3f;
}
