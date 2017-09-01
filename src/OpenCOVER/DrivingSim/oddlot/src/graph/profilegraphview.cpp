/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   08.06.2010
**
**************************************************************************/

#include "profilegraphview.hpp"

#include "profilegraphscene.hpp"

// GraphView //
//
#include "graphview.hpp"

// Items //
//
#include "items/view/ruler.hpp"

// Tools //
//
#include "src/gui/tools/toolaction.hpp"
#include "src/gui/tools/selectiontool.hpp"

// Qt //
//
#include <QWheelEvent>
#include <QDebug>
#include <QGraphicsRectItem>
#include <QApplication>

ProfileGraphView::ProfileGraphView(ProfileGraphScene *scene, QWidget *parent)
    : QGraphicsView(scene, parent)
    , doPan_(false)
    , doKeyPan_(false)
    , doBoxSelect_(GraphView::BBOff)
    , rubberBand_(NULL)
    , additionalSelection_(false)
{
    //	resetViewTransformation();

    horizontalRuler_ = new Ruler(Qt::Horizontal);
    scene->addItem(horizontalRuler_);

    verticalRuler_ = new Ruler(Qt::Vertical);
    scene->addItem(verticalRuler_);

	rubberBand_ = new QRubberBand(QRubberBand::Rectangle, this);

    // Zoom to mouse pos //
    //
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
}

ProfileGraphView::~ProfileGraphView()
{
    scene()->removeItem(horizontalRuler_);
    delete horizontalRuler_;

    scene()->removeItem(verticalRuler_);
    delete verticalRuler_;
}

/*! \brief Resets the transformation of the view.
*
* \note The default view matrix is rotated 180 degrees around the x-Axis,
* because up should be positive.
*/
void
ProfileGraphView::resetViewTransformation()
{
    QTransform trafo;
    trafo.rotate(180.0, Qt::XAxis);

    resetMatrix();
    setTransform(trafo);
}

//################//
// SLOTS          //
//################//

/**
*/
void
ProfileGraphView::rebuildRulers()
{
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
ProfileGraphView::zoomIn(Qt::Orientations orientation)
{
    if (orientation == Qt::Horizontal)
    {
        scale(1.25, 1.00);
    }
    else if (orientation == Qt::Vertical)
    {
        scale(1.00, 1.25);
    }
    else
    {
        scale(1.25, 1.25); // both
    }
    rebuildRulers();
    //	update();
}

///**
//*/
//void
//	ProfileGraphView
//	::zoomIn(double zoom)
//{
//	if(zoom < 1.0) zoom = 1.0;
//
//	#define MAX_ZOOM_IN 50.0f
//		if(getScale() * zoom >= MAX_ZOOM_IN)
//		{
//			zoom = MAX_ZOOM_IN/getScale();
//		}
//		scale(zoom, zoom);
//		update();
//	#undef MAX_ZOOM_IN
//}

/**
*/
void
ProfileGraphView::zoomOut(Qt::Orientations orientation)
{
    if (orientation == Qt::Horizontal)
    {
        scale(0.8, 1.00);
    }
    else if (orientation == Qt::Vertical)
    {
        scale(1.00, 0.8);
    }
    else
    {
        scale(0.8, 0.8); // both
    }
    rebuildRulers();
    //	update();
}

/*! \brief .
*
*/
void
ProfileGraphView::toolAction(ToolAction *toolAction)
{

}

//################//
// EVENTS         //
//################//

/*! \brief Mouse events for panning, etc.
*
*/
void
ProfileGraphView::resizeEvent(QResizeEvent *event)
{
    QGraphicsView::resizeEvent(event);
    rebuildRulers();
}

/*! \brief Mouse events for panning, etc.
*
*/
void
ProfileGraphView::scrollContentsBy(int dx, int dy)
{
    QGraphicsView::scrollContentsBy(dx, dy);
    rebuildRulers();
}

/*! \brief Mouse events for panning, etc.
*
*/
void
ProfileGraphView::mousePressEvent(QMouseEvent *event)
{
    if (doKeyPan_)
    {
        setDragMode(QGraphicsView::ScrollHandDrag);
        setInteractive(false); // this prevents the event from being passed to the scene
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

	else if (event->button() == Qt::LeftButton)
	{
		QGraphicsItem *item = NULL;
		if ((event->modifiers() & (Qt::AltModifier | Qt::ControlModifier)) == 0)
		{
			item = scene()->itemAt(mapToScene(event->pos()), QGraphicsView::transform());
		}

		if (item)
		{
			QGraphicsView::mousePressEvent(event); // pass to baseclass
		}
		else
		{
			doBoxSelect_ = GraphView::BBActive;

			if ((event->modifiers() & (Qt::ControlModifier | Qt::AltModifier)) != 0)
			{
				additionalSelection_ = true;
			}

			mp_ = event->pos();
		}
	}
}

/*! \brief Mouse events for panning, etc.
*
*/
void
ProfileGraphView::mouseMoveEvent(QMouseEvent *event)
{
    if (doBoxSelect_ == GraphView::BBActive)
    {

        // Check for enough drag distance
        if ((mp_ - event->pos()).manhattanLength() < QApplication::startDragDistance())
        {

            return;
        }
		else
		{
			if (!rubberBand_->isVisible())
			{
				rubberBand_->show();
			}
		}

        QPoint ep = event->pos();

        rubberBand_->setGeometry(QRect(qMin(mp_.x(), ep.x()), qMin(mp_.y(), ep.y()),
                                       qAbs(mp_.x() - ep.x()) + 1, qAbs(mp_.y() - ep.y()) + 1));

        //		QGraphicsView::mouseMoveEvent(event); // pass to baseclass
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

/*! \brief Mouse events for panning, etc.
*
*/
void
ProfileGraphView::mouseReleaseEvent(QMouseEvent *event)
{
    /*	QGraphicsView::mouseReleaseEvent(event); // pass to baseclass
	setDragMode(QGraphicsView::NoDrag);

	doPan_ = false;
	doKeyPan_ = false;

	setInteractive(true);*/

    if (doKeyPan_)
    {
        setDragMode(QGraphicsView::NoDrag);
        setInteractive(true);
        if (doBoxSelect_ == GraphView::BBPressed)
        {
            doBoxSelect_ = GraphView::BBActive;
        }
    }

#ifdef USE_MIDMOUSE_PAN
    if (doPan_)
    {
        setDragMode(QGraphicsView::NoDrag);
        setInteractive(true);
        if (doBoxSelect_ == GraphView::BBPressed)
        {
            doBoxSelect_ = GraphView::BBActive;
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

        if (doBoxSelect_ == GraphView::BBActive)
        {
			doBoxSelect_ = GraphView::BBOff;

			if ((mp_ - event->pos()).manhattanLength() < QApplication::startDragDistance())
			{
				if ((event->modifiers() & (Qt::AltModifier | Qt::ControlModifier)) != 0)
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
					QGraphicsView::mousePressEvent(event);
				}
				return;
			}

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
			doBoxSelect_ = GraphView::BBOff;
		}
	}

	doPan_ = false;
    doKeyPan_ = false;
    additionalSelection_ = false;
    setInteractive(true);
}

/*! \brief Key events for panning, etc.
*
*/
void
ProfileGraphView::keyPressEvent(QKeyEvent *event)
{
    // TODO: This will not notice a key pressed, when the view is not active
    switch (event->key())
    {
    case Qt::Key_Space:
        doKeyPan_ = true;
        break;

    default:
        QGraphicsView::keyPressEvent(event);
    }
}

/*! \brief Key events for panning, etc.
*
*/
void
ProfileGraphView::keyReleaseEvent(QKeyEvent *event)
{
    switch (event->key())
    {
    case Qt::Key_Space:
        doKeyPan_ = false;
        break;

    default:
        QGraphicsView::keyReleaseEvent(event);
    }
}

/*! \brief Wheel events for zooming.
*
*/
void
ProfileGraphView::wheelEvent(QWheelEvent *event)
{
    if (event->delta() > 0)
    {
        if (event->modifiers() & Qt::ControlModifier)
        {
            if (event->modifiers() & Qt::AltModifier)
            {
                zoomIn(Qt::Vertical | Qt::Horizontal); // ctrl+alt
            }
            else
            {
                zoomIn(Qt::Vertical); // ctrl
            }
        }
        else if (event->modifiers() & Qt::AltModifier)
        {
            zoomIn(Qt::Horizontal); // alt
        }
        else
        {
            zoomIn(Qt::Vertical | Qt::Horizontal); // none
        }
    }
    else
    {
        if (event->modifiers() & Qt::ControlModifier)
        {
            if (event->modifiers() & Qt::AltModifier)
            {
                zoomOut(Qt::Vertical | Qt::Horizontal); // ctrl+alt
            }
            else
            {
                zoomOut(Qt::Vertical); // ctrl
            }
        }
        else if (event->modifiers() & Qt::AltModifier)
        {
            zoomOut(Qt::Horizontal); // alt
        }
        else
        {
            zoomOut(Qt::Vertical | Qt::Horizontal); // none
        }
    }
}
