/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   04.02.2010
**
**************************************************************************/

#include "graphview.hpp"

#include "topviewgraph.hpp"
#include "graphscene.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/scenerysystem/scenerysystem.hpp"

// Items //
//
#include "items/view/ruler.hpp"
#include "items/scenerysystem/scenerysystemitem.hpp"

// Tools //
//
#include "src/gui/tools/toolaction.hpp"
#include "src/gui/tools/zoomtool.hpp"
#include "src/gui/tools/selectiontool.hpp"
#include "src/gui/tools/maptool.hpp"
#include "src/gui/tools/junctioneditortool.hpp"

// Qt //
//
#include <QWheelEvent>
#include <QMouseEvent>
#include <QFileDialog>
#include <QApplication>
#include <QUndoStack>

// Utils //
//
#include "math.h"
#include "src/util/colorpalette.hpp"

// DEFINES //
//
//#define USE_MIDMOUSE_PAN

GraphView::GraphView(GraphScene *graphScene, TopviewGraph *topviewGraph)
    : QGraphicsView(graphScene, topviewGraph)
    , topviewGraph_(topviewGraph)
    , graphScene_(graphScene)
    , doPan_(false)
    , doKeyPan_(false)
    , doBoxSelect_(BBOff)
    , doCircleSelect_(CircleOff)
    , radius_(0.0)
    , horizontalRuler_(NULL)
    , verticalRuler_(NULL)
    , rulersActive_(false)
    , rubberBand_(NULL)
    , circleItem_(NULL)
    , additionalSelection_(false)
{
    // ScenerySystem //
    //
    scenerySystemItem_ = new ScenerySystemItem(topviewGraph_, topviewGraph_->getProjectData()->getScenerySystem());
    scene()->addItem(scenerySystemItem_);

    // Zoom to mouse pos //
    //
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
}

GraphView::~GraphView()
{
    activateRulers(false);
}

/** Returns the scaling of the GraphicsView. Returns 0.0 if view is stretched.
*/
double
GraphView::getScale() const
{
    QMatrix vm = matrix();
    if (fabs(vm.m11()) != fabs(vm.m22()))
    {
        qDebug("View stretched! getScale() returns 0.0!");
        return 0.0;
    }
    else
    {
        return fabs(vm.m11());
    }
}

/*! \brief Resets the transformation of the view.
*
* \note The default view matrix is rotated 180 degrees around the x-Axis,
* because OpenDRIVE and Qt use different coordinate systems.
*/
void
GraphView::resetViewTransformation()
{
    QTransform trafo;
    trafo.rotate(180.0, Qt::XAxis);

    resetMatrix();
    setTransform(trafo);
}

//################//
// SLOTS          //
//################//

/*! \brief .
*
*/
void
GraphView::toolAction(ToolAction *toolAction)
{
    // Zoom //
    //
    ZoomToolAction *zoomToolAction = dynamic_cast<ZoomToolAction *>(toolAction);
    if (zoomToolAction)
    {
        ZoomTool::ZoomToolId id = zoomToolAction->getZoomToolId();

        if (id == ZoomTool::TZM_ZOOMTO)
        {
            zoomTo(zoomToolAction->getZoomFactor());
        }
        else if (id == ZoomTool::TZM_ZOOMIN)
        {
            zoomIn();
        }
        else if (id == ZoomTool::TZM_ZOOMOUT)
        {
            zoomOut();
        }
        else if (id == ZoomTool::TZM_ZOOMBOX)
        {
            zoomBox();
        }
        else if (id == ZoomTool::TZM_VIEW_SELECTED)
        {
            viewSelected();
        }
        else if (id == ZoomTool::TZM_RULERS)
        {
            activateRulers(zoomToolAction->isToggled());
        }
    }

    // Selection //
    //
    SelectionToolAction *selectionToolAction = dynamic_cast<SelectionToolAction *>(toolAction);
    if (selectionToolAction)
    {
        SelectionTool::SelectionToolId id = selectionToolAction->getSelectionToolId();

        if (id == SelectionTool::TSL_BOUNDINGBOX)
        {
            if ((doBoxSelect_ == BBOff) && selectionToolAction->isToggled())
            {
                doBoxSelect_ = BBActive;
            }
            else if ((doBoxSelect_ == BBActive) && !selectionToolAction->isToggled())
            {
                doBoxSelect_ = BBOff;
                if (rubberBand_)
                {
                    rubberBand_->hide();
                }
            }
        }
    }

    // Circular Cutting Tool
    //
    JunctionEditorToolAction *junctionEditorAction = dynamic_cast<JunctionEditorToolAction *>(toolAction);
    if (junctionEditorAction)
    {
        ODD::ToolId id = junctionEditorAction->getToolId();

        if (id == ODD::TJE_CIRCLE)
        {
            if ((doCircleSelect_ == CircleOff) && junctionEditorAction->isToggled())
            {
                doCircleSelect_ = CircleActive;
                radius_ = junctionEditorAction->getThreshold();

                QPen pen(Qt::DashLine);
                pen.setColor(ODD::instance()->colors()->brightBlue());

                circleItem_ = new QGraphicsPathItem();
                circleItem_->setPen(pen);
                scene()->addItem(circleItem_);
            }
            else if ((circleItem_) && !junctionEditorAction->isToggled())
            {
                doCircleSelect_ = CircleOff;
                scene()->removeItem(circleItem_);
                delete circleItem_;
            }
        }
        else if (id == ODD::TJE_THRESHOLD)
        {
            radius_ = junctionEditorAction->getThreshold();
        }
    }

    // Map //
    //
    MapToolAction *mapToolAction = dynamic_cast<MapToolAction *>(toolAction);
    if (mapToolAction)
    {
        MapTool::MapToolId id = mapToolAction->getMapToolId();

        if (id == MapTool::TMA_LOAD)
        {
            loadMap();
            lockMap(true);
        }
        else if (id == MapTool::TMA_DELETE)
        {
            deleteMap();
        }
        else if (id == MapTool::TMA_LOCK)
        {
            lockMap(mapToolAction->isToggled());
        }
        else if (id == MapTool::TMA_OPACITY)
        {
            setMapOpacity(mapToolAction->getOpacity());
        }
        //		else if(id == MapTool::TMA_X)
        //		{
        //			setMapX(mapToolAction->getX());
        //		}
        //		else if(id == MapTool::TMA_Y)
        //		{
        //			setMapY(mapToolAction->getY());
        //		}
        //		else if(id == MapTool::TMA_WIDTH)
        //		{
        //			setMapWidth(mapToolAction->getWidth(), mapToolAction->isKeepRatio());
        //		}
        //		else if(id == MapTool::TMA_HEIGHT)
        //		{
        //			setMapHeight(mapToolAction->getHeight(), mapToolAction->isKeepRatio());
        //		}
    }
}

/**
*/
void
GraphView::rebuildRulers()
{
    if (!rulersActive_)
    {
        return;
    }
    //	qDebug() << viewportTransform();
    //	qDebug() << size();
    //	qDebug() << viewport()->size();
    //	testrect_->setPos(viewportTransform().inverted().map(QPointF(1.0, 1.0)));
    QPointF pos = viewportTransform().inverted().map(QPointF(0.0, 0.0));
    double width = viewport()->size().width() / matrix().m11();
    double height = viewport()->size().height() / matrix().m22();
    //	testrect_->setRectF(pos.x(), pos.y(), width, height);
    horizontalRuler_->updateRect(QRectF(pos.x(), pos.y(), width, height), matrix().m11(), matrix().m22());
    verticalRuler_->updateRect(QRectF(pos.x(), pos.y(), width, height), matrix().m11(), matrix().m22());
    update();
}

/**
*/
void
GraphView::activateRulers(bool activate)
{
    if (activate)
    {
        // Activate rulers //
        //
        if (!horizontalRuler_)
        {
            horizontalRuler_ = new Ruler(Qt::Horizontal);
            scene()->addItem(horizontalRuler_);
        }
        if (!verticalRuler_)
        {
            verticalRuler_ = new Ruler(Qt::Vertical);
            scene()->addItem(verticalRuler_);
        }
    }
    else
    {
        // Deactivate rulers //
        //
        if (horizontalRuler_)
        {
            scene()->removeItem(horizontalRuler_);
            delete horizontalRuler_;
            horizontalRuler_ = NULL;
        }
        if (verticalRuler_)
        {
            scene()->removeItem(verticalRuler_);
            delete verticalRuler_;
            verticalRuler_ = NULL;
        }
    }

    rulersActive_ = activate;
    rebuildRulers();
}

/**
*/
void
GraphView::zoomTo(const QString &zoomFactor)
{
    // Erase %-sign and parse to double
    double scaleFactor = zoomFactor.left(zoomFactor.indexOf(tr("%"))).toDouble() / 100.0;
    //	const QMatrix & vm = matrix();
    //	resetMatrix();
    //	translate(vm.dx(), vm.dy()); // this is 0.0 anyway!

    resetViewTransformation();
    scale(scaleFactor, scaleFactor);
}

/**
*/
void
GraphView::zoomIn()
{
    zoomIn(1.25);
}

/**
*/
void
GraphView::zoomIn(double zoom)
{
    if (zoom < 1.0)
        zoom = 1.0;

#define MAX_ZOOM_IN 50.0f
    if (getScale() * zoom >= MAX_ZOOM_IN)
    {
        zoom = MAX_ZOOM_IN / getScale();
    }
    scale(zoom, zoom);
    //		update();
    rebuildRulers();
#undef MAX_ZOOM_IN
}

/**
*/
void
GraphView::zoomOut()
{
    scale(0.8, 0.8);
    //	update();
    rebuildRulers();
}

/**
*/
void
GraphView::zoomBox()
{

    qDebug("GraphView::zoomBox() not yet implemented");
}

/**
*/
void
GraphView::viewSelected()
{
    QList<QGraphicsItem *> selectList = scene()->selectedItems();
    QRectF boundingRect = QRectF();

    foreach (QGraphicsItem *item, selectList)
    {
        boundingRect.operator|=(item->sceneBoundingRect());
    }

    fitInView(boundingRect, Qt::KeepAspectRatio);
}

/*! \brief .
*/
void
GraphView::loadMap()
{
    // File //
    //
    QString filename = QFileDialog::getOpenFileName(this, tr("Open Image File"));
    if (filename.isEmpty())
    {
        return;
    }

    scenerySystemItem_->loadMap(filename, mapToScene(10.0, 10.0)); // place pic near top left corner
}

/*! \brief .
*/
void
GraphView::deleteMap()
{
    scenerySystemItem_->deleteMap();
}

/*! \brief Locks all the MapItems (no selecting/moving).
*/
void
GraphView::lockMap(bool locked)
{
    scenerySystemItem_->lockMaps(locked);
}

/*! \brief Sets the opacity of the selected MapItems.
*/
void
GraphView::setMapOpacity(const QString &opacity)
{
    double opacityValue = opacity.left(opacity.indexOf(tr("%"))).toDouble() / 100.0;
    scenerySystemItem_->setMapOpacity(opacityValue);
}

/*! \brief Sets the x-coordinate of the selected MapItems.
*/
void
GraphView::setMapX(double x)
{
    scenerySystemItem_->setMapX(x);
}

/*! \brief Sets the y-coordinate of the selected MapItems.
*/
void
GraphView::setMapY(double y)
{
    scenerySystemItem_->setMapY(y);
}

/*! \brief Sets the width of the selected MapItems.
*/
void
GraphView::setMapWidth(double width, bool keepRatio)
{
    scenerySystemItem_->setMapWith(width, keepRatio);
}

/*! \brief Sets the height of the selected MapItems.
*/
void
GraphView::setMapHeight(double height, bool keepRatio)
{
    scenerySystemItem_->setMapHeight(height, keepRatio);
}

//################//
// EVENTS         //
//################//

/*! \brief Mouse events for panning, etc.
*
*/
void
GraphView::resizeEvent(QResizeEvent *event)
{
    QGraphicsView::resizeEvent(event);
    rebuildRulers();
}

/*! \brief Mouse events for panning, etc.
*
*/
void
GraphView::scrollContentsBy(int dx, int dy)
{
    QGraphicsView::scrollContentsBy(dx, dy);
    rebuildRulers();
}

void
GraphView::mousePressEvent(QMouseEvent *event)
{

    if (doBoxSelect_ == BBActive)
    {
        if ((event->modifiers() & (Qt::ControlModifier | Qt::AltModifier)) != 0)
        {
            additionalSelection_ = true;
        }

        mp_ = event->pos();
        if (!rubberBand_)
        {
            rubberBand_ = new QRubberBand(QRubberBand::Rectangle, this);
        }
        rubberBand_->setGeometry(QRect(mp_, QSize()));
        rubberBand_->show();
    }
    else if (doCircleSelect_ == CircleActive)
    {

        if (event->button() == Qt::LeftButton)
        {
            circleCenter_ = mapToScene(event->pos());
            QPainterPath circle = QPainterPath();
            circle.addEllipse(circleCenter_, radius_, radius_);
            circleItem_->setPath(circle);

            // Select roads intersecting with circle
            //
            scene()->setSelectionArea(circle);
        }

        doCircleSelect_ = CircleOff;
        scene()->removeItem(circleItem_);
        delete circleItem_;
        circleItem_ = NULL;
    }
    else if (doKeyPan_)
    {
        setDragMode(QGraphicsView::ScrollHandDrag);
        setInteractive(false); // this prevents the event from being passed to the scene
        QGraphicsView::mousePressEvent(event); // pass to baseclass
    }
#ifdef USE_MIDMOUSE_PAN
    else if (event->button() == Qt::MidButton)
    {
        doPan_ = true;

        setDragMode(QGraphicsView::ScrollHandDrag);
        setInteractive(false); // this prevents the event from being passed to the scene

        // Harharhar Hack //
        //
        // Qt wants a LeftButton event for dragging, so feed Qt what it wants!
        QMouseEvent *newEvent = new QMouseEvent(QEvent::MouseMove, event->pos(), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
        QGraphicsView::mousePressEvent(newEvent); // pass to baseclass
        delete newEvent;
        return;
    }
#endif

    else if ((event->modifiers() & (Qt::AltModifier | Qt::ControlModifier)) != 0)
    {

        // Deselect element from the previous selection

        QList<QGraphicsItem *> oldSelection = scene()->selectedItems();

        QGraphicsView::mousePressEvent(event); // pass to baseclass

        QGraphicsItem *selectedItem = scene()->mouseGrabberItem();

        foreach (QGraphicsItem *item, oldSelection)
        {
            item->setSelected(true);
        }
        if (selectedItem)
        {
            if (((event->modifiers() & Qt::ControlModifier) != 0) && !oldSelection.contains(selectedItem))
            {
                selectedItem->setSelected(true);
            }
            else
            {
                selectedItem->setSelected(false);
            }
        }
    }
    else
    {

        QGraphicsView::mousePressEvent(event); // pass to baseclass
    }
}

void
GraphView::mouseMoveEvent(QMouseEvent *event)
{
    if ((doBoxSelect_ == BBActive) && rubberBand_)
    {

        // Check for enough drag distance
        if ((mp_ - event->pos()).manhattanLength() < QApplication::startDragDistance())
        {

            return;
        }
        QPoint ep = event->pos();

        rubberBand_->setGeometry(QRect(qMin(mp_.x(), ep.x()), qMin(mp_.y(), ep.y()),
                                       qAbs(mp_.x() - ep.x()) + 1, qAbs(mp_.y() - ep.y()) + 1));
    }
    else if (doCircleSelect_ == CircleActive)
    {
        // Draw circle with radius and mouse pos center
        //
        circleCenter_ = mapToScene(event->pos());
        QPainterPath circle = QPainterPath();
        circle.addEllipse(circleCenter_, radius_, radius_);
        circleItem_->setPath(circle);
    }
    else if (doKeyPan_)
    {
        QGraphicsView::mouseMoveEvent(event); // pass to baseclass
    }
#ifdef USE_MIDMOUSE_PAN
    else if (doPan_)
    {
        QMouseEvent *newEvent = new QMouseEvent(QEvent::MouseMove, event->pos(), Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
        QGraphicsView::mouseMoveEvent(newEvent); // pass to baseclass
        delete newEvent;
    }
#endif
    else
    {
        QGraphicsView::mouseMoveEvent(event); // pass to baseclass
    }
}

void
GraphView::mouseReleaseEvent(QMouseEvent *event)
{
    if (doKeyPan_)
    {
        setDragMode(QGraphicsView::NoDrag);
        setInteractive(true);
        if (doBoxSelect_ == BBPressed)
        {
            doBoxSelect_ = BBActive;
        }
    }

#ifdef USE_MIDMOUSE_PAN
    if (doPan_)
    {
        setDragMode(QGraphicsView::NoDrag);
        setInteractive(true);
        if (doBoxSelect_ == BBPressed)
        {
            doBoxSelect_ = BBActive;
        }
        doPan_ = false;
    }
#endif

    //	setDragMode(QGraphicsView::RubberBandDrag);

    else
    {
        if ((event->modifiers() & (Qt::AltModifier | Qt::ControlModifier)) == 0)
        {
            QGraphicsView::mouseReleaseEvent(event);
        }

        if ((doBoxSelect_ == BBActive) && rubberBand_)
        {
            QList<QGraphicsItem *> oldSelection;

            if (additionalSelection_)
            {
                // Save old selection

                oldSelection = scene()->selectedItems();
            }

            // Set the new selection area

            QPainterPath selectionArea;

            selectionArea.addPolygon(mapToScene(QRect(rubberBand_->pos(), rubberBand_->rect().size())));
            selectionArea.closeSubpath();
            scene()->clearSelection();
            scene()->setSelectionArea(selectionArea, Qt::IntersectsItemShape, viewportTransform());

            // Compare old and new selection lists and invert the selection state of elements contained in both

            QList<QGraphicsItem *> selectList = scene()->selectedItems();
            foreach (QGraphicsItem *item, oldSelection)
            {
                if (selectList.contains(item))
                {
                    item->setSelected(false);
                    selectList.removeOne(item);
                }
                else
                {
                    item->setSelected(true);
                }
            }

            // deselect elements which were not in the oldSelection

            if ((event->modifiers() & Qt::AltModifier) != 0)
            {
                foreach (QGraphicsItem *item, selectList)
                {
                    item->setSelected(false);
                }
            }

            rubberBand_->hide();
            doBoxSelect_ = BBOff;
        }
    }

    doPan_ = false;
    doKeyPan_ = false;
    additionalSelection_ = false;
    setInteractive(true);

    //	if(doBoxSelect_)
    //	{
    //	}
    //	else if(doKeyPan_)
    //	{
    //	}
    //	else if(doPan_)
    //	{
    //	}
    //	else
    //	{
    ////	if(event->button() == Qt::MidButton) // end panning anyway
    ////	{
    //	}
}

void
GraphView::wheelEvent(QWheelEvent *event)
{
    if (event->delta() > 0)
    {
        zoomIn();
    }
    else
    {
        zoomOut();
    }
}

void
GraphView::keyPressEvent(QKeyEvent *event)
{
    // TODO: This will not notice a key pressed, when the view is not active
    switch (event->key())
    {
    case Qt::Key_Space:
        doKeyPan_ = true;
        if (doBoxSelect_ == BBActive)
        {
            doBoxSelect_ = BBPressed;
        }
        break;

    case Qt::Key_Delete:
    {
        // Macro Command //
        //
        int numberSelectedItems = scene()->selectedItems().size();
        if (numberSelectedItems > 1)
        {
            topviewGraph_->getProjectData()->getUndoStack()->beginMacro(QObject::tr("Delete Elements"));
        }
        bool deletedSomething = false;
        do
        {
            deletedSomething = false;
            QList<QGraphicsItem *> selectList = scene()->selectedItems();

            foreach (QGraphicsItem *item, selectList)
            {
                GraphElement *graphElement = dynamic_cast<GraphElement *>(item);
                if (graphElement)
                {
                    if (graphElement->deleteRequest())
                    {
                        deletedSomething = true;
                        break;
                    }
                }
            }
        } while (deletedSomething);

        // Macro Command //
        //
        if (numberSelectedItems > 1)
        {
            topviewGraph_->getProjectData()->getUndoStack()->endMacro();
        }
        break;
    }

    default:
        QGraphicsView::keyPressEvent(event);
    }
}

void
GraphView::keyReleaseEvent(QKeyEvent *event)
{
    switch (event->key())
    {
    case Qt::Key_Space:
        doKeyPan_ = true;
        if (doBoxSelect_ == BBActive)
        {
            doBoxSelect_ = BBPressed;
        }
        break;

    default:
        QGraphicsView::keyReleaseEvent(event);
    }
}
