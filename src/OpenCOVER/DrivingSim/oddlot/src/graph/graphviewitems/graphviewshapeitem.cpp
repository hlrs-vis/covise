/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the tools applications of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:GPL-EXCEPT$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 3 as published by the Free Software
** Foundation with exceptions as appearing in the file LICENSE.GPL3-EXCEPT
** included in the packaging of this file. Please review the following
** information to ensure the GNU General Public License requirements will
** be met: https://www.gnu.org/licenses/gpl-3.0.html.
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "graphviewshapeitem.hpp"

#include "../graphview.hpp"

// Data //
//
// #include "src/data/oscsystem/oscelement.hpp"

// Graph //
//
#include "src/graph/editors/osceditor.hpp"

#include <QMouseEvent>
#include <QContextMenuEvent>
#include <QApplication>

#include <QPen>
#include <QColor>
#include <QMouseEvent>

#include <cmath>


GraphViewShapeItem::GraphViewShapeItem(GraphView *view, OpenScenarioEditor *editor, int x, int y, int width, int height) :
    QObject()
    , QGraphicsPathItem()
	, selected(true)
    , view_(view)
    , editor_(editor)
    , canvasWidth(width)
    , canvasHeight(height)
    , graphicItemGroup_(NULL)
{

 //   setFixedSize(960, 640);

	m_numberOfSegments = 0;
    m_activeControlPoint = -1;

    m_mouseDrag = false;

    m_pointContextMenu = new QMenu();
    m_deleteAction = new QAction(tr("Delete point"), m_pointContextMenu);
    m_smoothAction = new QAction(tr("Smooth point"), m_pointContextMenu);
    m_cornerAction = new QAction(tr("Corner point"), m_pointContextMenu);

    m_smoothAction->setCheckable(true);

    m_pointContextMenu->addAction(m_deleteAction);
    m_pointContextMenu->addAction(m_smoothAction);
    m_pointContextMenu->addAction(m_cornerAction);

    QApplication::setOverrideCursor(Qt::CrossCursor);

    m_controlPoints = view_->getSplineControlPoints();
    if (m_controlPoints.size() > 0)
    {
        m_numberOfSegments = m_controlPoints.size() / 3;
        startPoint = m_controlPoints.at(0);
        endPoint = m_controlPoints.at(m_controlPoints.size() - 1);

        invalidateSmoothList();
        invalidate();
        createPath();
    }

}

GraphViewShapeItem::~GraphViewShapeItem()
{
//    m_controlPoints.clear();
    finishEditing();
}

QPointF GraphViewShapeItem::mapToCanvas(const QPointF &point)
{
    return QPointF(startPoint.x() + point.x() * canvasWidth,
					startPoint.y() - point.y() * canvasHeight);
}

QPointF GraphViewShapeItem::mapFromCanvas(const QPointF &point)
{
	QPointF d = point - startPoint;
		return QPointF(d.x()/canvasWidth,
			-d.y()/canvasHeight);
}

void 
GraphViewShapeItem::paintControlPoint(const QPointF &point, bool edit,
                                     bool realPoint, bool active, bool smooth)
{
    QGraphicsPathItem *pathItem = new QGraphicsPathItem(this);
    graphicItemGroup_->addToGroup(pathItem);

    int pointSize = 4;

    if (active)
        pathItem->setBrush(QColor(140, 140, 240, 255));
    else
        pathItem->setBrush(QColor(120, 120, 220, 255));

    if (realPoint) {
        pointSize = 6;
        pathItem->setBrush(QColor(80, 80, 210, 150));
    }

    pathItem->setPen(QColor(50, 50, 50, 140));

    if (!edit)
        pathItem->setBrush(QColor(160, 80, 80, 250));

    QPainterPath *path = new QPainterPath();
    if (smooth) {
        path->addEllipse(QRectF(point.x() - pointSize + 0.5,
                                    point.y() - pointSize + 0.5,
                                    pointSize * 2, pointSize * 2));
    } else {
        path->addRect(QRectF(point.x() - pointSize + 0.5,
                                 point.y() - pointSize + 0.5,
                                 pointSize * 2, pointSize * 2));
    }

    pathItem->setPath(*path);
}

static inline bool indexIsRealPoint(int i)
{
    return  !(i % 3);
}

static inline int pointForControlPoint(int i)
{
    if ((i % 3) == 1)
        return i - 1;

    if ((i % 3) == 2)
        return i + 1;

    return i;
}

void 
GraphViewShapeItem::createPath()
{
    if (graphicItemGroup_)
	{
        view_->scene()->removeItem(graphicItemGroup_);
        delete graphicItemGroup_;
//		view_->scene()->destroyItemGroup(graphicItemGroup_);
	}

	graphicItemGroup_ = new QGraphicsItemGroup(this);

	if (!startPoint.isNull())
	{
		paintControlPoint(startPoint, false, true, false, false);

		if (!endPoint.isNull())
		{
			paintControlPoint(endPoint, false, true, false, false);
		}
	}

	
    QGraphicsPathItem *pathCubicItem = new QGraphicsPathItem(this);
    QPen penCubic(QBrush(Qt::black), 2);
    pathCubicItem->setPen(penCubic);
    graphicItemGroup_->addToGroup(pathCubicItem);

    QGraphicsPathItem *pathHandleItem = new QGraphicsPathItem(this);
    QPen penHandle(Qt::black);
    penHandle.setStyle(Qt::DashLine);
    penHandle.setWidth(1);
    pathHandleItem->setPen(penHandle);
    graphicItemGroup_->addToGroup(pathHandleItem);

	if (selected)
	{
        QPainterPath pathSpline;
        QPainterPath pathHandle;
		for (int i = 0; i < m_numberOfSegments; i++) {
			QPainterPath pathCubic;
			QPointF p0;

			p0 = m_controlPoints.at(i * 3);

			pathCubic.moveTo(p0);

			QPointF p1 = m_controlPoints.at(i * 3 + 1);
			QPointF p2 = m_controlPoints.at(i * 3 + 2);
			QPointF p3 = m_controlPoints.at(i * 3 + 3);
			pathCubic.cubicTo(p1, p2, p3);
			pathSpline.addPath(pathCubic);

            QPainterPath path;
            path.moveTo(p0);
            path.lineTo(p1);
            path.moveTo(p3);
            path.lineTo(p2);
            pathHandle.addPath(path);
		}
        pathCubicItem->setPath(pathSpline);
        pathHandleItem->setPath(pathHandle);

		for (int i = 1; i < m_controlPoints.count() - 1; ++i)
			paintControlPoint(m_controlPoints.at(i),
			true,
			indexIsRealPoint(i),
			i == m_activeControlPoint,
			isControlPointSmooth(i)); 
	}
	else
	{
        QPainterPath pathSpline;
		for (int i = 0; i < m_numberOfSegments; i++) {
			QPainterPath path;
			QPointF p0;

			p0 = m_controlPoints.at(i * 3);

			path.moveTo(p0);

			QPointF p3 = m_controlPoints.at(i * 3 + 3);
			path.lineTo(p3);
			pathSpline.addPath(path);
		}
        pathCubicItem->setPath(pathSpline);

		for (int i = 3; i < m_controlPoints.count() - 1; i+=3)
			paintControlPoint(m_controlPoints.at(i),
			true,
			indexIsRealPoint(i),
			i == m_activeControlPoint,
			isControlPointSmooth(i)); 
	}

    view_->setSplineControlPoints(m_controlPoints);

}

void 
GraphViewShapeItem::mousePressEvent(QMouseEvent *e)
{
    if (e->button() == Qt::LeftButton) {
        QPointF p = view_->mapToScene(e->pos());
        m_activeControlPoint = findControlPoint(p);

        if (m_activeControlPoint < 0)
        {
            if (startPoint.isNull() || endPoint.isNull())
            {
                appendPoint(p);
            }
            else
            {
                if (abs(m_activeControlPoint) >= m_controlPoints.size() - 2)
                {
                    int size = m_controlPoints.size();
                    QPointF vec1 = m_controlPoints.at(size - 1) - m_controlPoints.at(size - 3);
                    QPointF vec2 = p - m_controlPoints.at(size - 3);

                    if (vec2.manhattanLength() > vec1.manhattanLength())
                    {
                        appendPoint(p);
                    }
                    else
                    {
                        addPoint(p);
                    }
                }
                else
                {
                    addPoint(p);
                }
            }
        }
        else
        {
            mouseMoveEvent(e);

            m_mousePress = p;
            m_mouseDrag = true;
            e->accept();
        }
    }
}

void GraphViewShapeItem::mouseReleaseEvent(QMouseEvent *e)
{
    if (e->button() == Qt::LeftButton) {
        m_activeControlPoint = -1;

        m_mouseDrag = false;
        e->accept();
    }
}

void 
GraphViewShapeItem::contextMenu(QContextMenuEvent *e)
{
    QPointF p = view_->mapToScene(e->pos());
    int index = findControlPoint(p);

    if (index > 1 && (index < m_controlPoints.size() - 1) && indexIsRealPoint(index)) {
        m_smoothAction->setChecked(isControlPointSmooth(index));
        QAction* action = m_pointContextMenu->exec(e->globalPos());
        if (action == m_deleteAction)
            deletePoint(index);
        else if (action == m_smoothAction)
            smoothPoint(index);
        else if (action == m_cornerAction)
            cornerPoint(index);
    } else {
		bool showMenu = false;
		if (!startPoint.isNull())
		{
			qreal d = QLineF(e->pos(),startPoint).length();
			if (d >= 10) 
			{
				showMenu = true;
			}
		}
    }
}

void 
GraphViewShapeItem::invalidate()
{
	if (canvasHeight > 1.0e-03)
	{
    QEasingCurve easingCurve(QEasingCurve::BezierSpline);

    for (int i = 0; i < m_numberOfSegments; ++i) {
        easingCurve.addCubicBezierSegment(mapFromCanvas(m_controlPoints.at(i * 3 + 1)),
                                          mapFromCanvas(m_controlPoints.at(i * 3 + 2)),
                                          mapFromCanvas(m_controlPoints.at(i * 3 + 3)));
    }
    setEasingCurve(easingCurve);
	}
}

void 
GraphViewShapeItem::invalidateSmoothList()
{
    m_smoothList.clear();

    for (int i = 0; i < (m_numberOfSegments - 1); ++i)
        m_smoothList.append(isSmooth(i * 3 + 3));

}


int 
GraphViewShapeItem::findControlPoint(const QPointF &point)
{
    int pointIndex = -1;
    qreal distance = -1;
    for (int i = 1; i<m_controlPoints.size(); ++i) {
        qreal d = QLineF(point, m_controlPoints.at(i)).length();
        if (distance < 0 || d < distance) {
            distance = d;
            pointIndex = i;
        }
    }

    if (distance > 10)
    {
        return -pointIndex;
    }

    return pointIndex;
}

int 
GraphViewShapeItem::findRealPoint(const QPointF &point)
{
    int pointIndex = -1;
    qreal distance = -1;
    for (int i = 0; i<m_controlPoints.size(); i += 3) {
        qreal d = QLineF(point, m_controlPoints.at(i)).length();
        if (distance < 0 || d < distance) {
            distance = d;
            pointIndex = i;
        }
    }

    return pointIndex;
}

static inline bool veryFuzzyCompare(qreal r1, qreal r2)
{
    if (qFuzzyCompare(r1, 2))
        return true;

    int r1i = qRound(r1 * 20);
    int r2i = qRound(r2 * 20);

    if (qFuzzyCompare(qreal(r1i) / 20, qreal(r2i) / 20))
        return true;

    return false;
}

bool 
GraphViewShapeItem::isSmooth(int i) const
{
    if (i == 0)
        return false;

    QPointF p = m_controlPoints.at(i);
    QPointF p_before = m_controlPoints.at(i - 1);
    QPointF p_after = m_controlPoints.at(i + 1);

    QPointF v1 = p_after - p;
    v1 = v1 / v1.manhattanLength(); //normalize

    QPointF v2 = p - p_before;
    v2 = v2 / v2.manhattanLength(); //normalize

    return veryFuzzyCompare(v1.x(), v2.x()) && veryFuzzyCompare(v1.y(), v2.y());
}

void 
GraphViewShapeItem::smoothPoint(int index)
{
    if (m_smoothAction->isChecked()) {

        QPointF before = QPointF(0,0);
        if (index > 3)
            before = m_controlPoints.at(index - 3);

        QPointF after = QPointF(1.0, 1.0);
        if ((index + 3) < m_controlPoints.count())
            after = m_controlPoints.at(index + 3);

        QPointF tangent = (after - before) / 6;

        QPointF thisPoint =  m_controlPoints.at(index);

        if (index > 0)
            m_controlPoints[index - 1] = thisPoint - tangent;

        if (index + 1  < m_controlPoints.count())
            m_controlPoints[index + 1] = thisPoint + tangent;

        m_smoothList[(index - 1) / 3] = true;
    } else {
        m_smoothList[(index - 1) / 3] = false;
    }
    invalidate();
 //   update();
    createPath();
}

void 
GraphViewShapeItem::cornerPoint(int index)
{
    QPointF before = QPointF(0,0);
    if (index > 3)
        before = m_controlPoints.at(index - 3);

    QPointF after = QPointF(1.0, 1.0);
    if ((index + 3) < m_controlPoints.count())
        after = m_controlPoints.at(index + 3);

    QPointF thisPoint =  m_controlPoints.at(index);

    if (index > 0)
        m_controlPoints[index - 1] = (before - thisPoint) / 3 + thisPoint;

    if (index + 1  < m_controlPoints.count())
        m_controlPoints[index + 1] = (after - thisPoint) / 3 + thisPoint;

    m_smoothList[(index - 1) / 3] = false;
    invalidate();
    createPath();
}

void 
GraphViewShapeItem::deletePoint(int index)
{
    m_controlPoints.remove(index - 1, 3);
    m_numberOfSegments--;

    invalidateSmoothList();
    invalidate();
    createPath();
}

void 
GraphViewShapeItem::appendPoint(const QPointF point)
{
	if (startPoint.isNull() || endPoint.isNull())
	{
		addPoint(point);
		return;
	}


	QPointF d = (point - endPoint)/6;
	m_controlPoints.append(endPoint + d);
	m_controlPoints.append(point - d);
	m_controlPoints.append(point);

	endPoint = point;
	canvasWidth = std::abs(endPoint.x() - startPoint.x());
	canvasHeight = std::abs(endPoint.y() - startPoint.y());

	m_numberOfSegments++;

    invalidateSmoothList();
    invalidate();
    createPath();
}




void 
GraphViewShapeItem::addPoint(const QPointF point)
{
	
	if (startPoint.isNull())
	{
		startPoint = point;
        m_controlPoints.insert(0, point);
//		update();
        createPath();
		return;
	}
	else if (endPoint.isNull())
	{
		endPoint = point;
		canvasWidth = std::abs(endPoint.x() - startPoint.x());
		canvasHeight = std::abs(endPoint.y() - startPoint.y());

		QPointF d = (endPoint - startPoint)/6;
		m_controlPoints.insert(1, endPoint);
		m_controlPoints.insert(1, endPoint - d);
		m_controlPoints.insert(1, startPoint + d);
	}
	else
	{
        int splitIndex = findRealPoint(point);
/*		for (int i=1; i < m_controlPoints.size() - 1; ++i) {
			if (indexIsRealPoint(i) && m_controlPoints.at(i).x() > point.x()) {
				break;
			} else if (indexIsRealPoint(i))
				splitIndex = i;
		} */
        if (splitIndex == m_controlPoints.size() - 1)
        {
            splitIndex -= 3;
        }
        else if ((splitIndex > 0) && (m_controlPoints.size() > 4))
        {
            if (((point.x() < m_controlPoints.at(splitIndex).x()) && (m_controlPoints.at(splitIndex).x() > m_controlPoints.at(splitIndex - 3).x())) || ((point.x() > m_controlPoints.at(splitIndex).x()) && (m_controlPoints.at(splitIndex).x() < m_controlPoints.at(splitIndex - 3).x())))
            {
                    splitIndex -= 3;
            }
        }

		QPointF before = startPoint;
		if (splitIndex > 1)
			before = m_controlPoints.at(splitIndex);

		QPointF after = endPoint;
		if ((splitIndex + 3) < m_controlPoints.count())
			after = m_controlPoints.at(splitIndex + 3);

		if (splitIndex > 1) {
			m_controlPoints.insert(splitIndex + 2, (point + after) / 2);
			m_controlPoints.insert(splitIndex + 2, point);
			m_controlPoints.insert(splitIndex + 2, (point + before) / 2);
		} else {
			m_controlPoints.insert(splitIndex + 1, (point + after) / 2);
			m_controlPoints.insert(splitIndex + 1, point);
			m_controlPoints.insert(splitIndex + 1, (point + before) / 2);
		}

		selected = true;
	}
    m_numberOfSegments++;

    invalidateSmoothList();
    invalidate();
    createPath();
}


bool 
GraphViewShapeItem::isControlPointSmooth(int i) const
{
    if (i == 0)
        return false;

    if (i == m_controlPoints.count() - 1)
        return false;

    if (m_numberOfSegments == 1)
        return false;

    int index = pointForControlPoint(i);

    if (index == 0)
        return false;

    if (index == m_controlPoints.count() - 1)
        return false;

    return m_smoothList.at((index - 1) / 3);
}



void GraphViewShapeItem::mouseMoveEvent(QMouseEvent *e)
{
    QPointF p = view_->mapToScene(e->pos());
    // If we've moved more then 25 pixels, assume user is dragging
/*    if (!m_mouseDrag && QPointF(m_mousePress - p).manhattanLength() > qApp->startDragDistance())
	{
        m_mouseDrag = true;
		selected = true;
	} */


	if (m_mouseDrag && !startPoint.isNull() && m_activeControlPoint < 0)
	{
		qreal d = QLineF(p, startPoint).length();
        if (d < 10)
		{
            QPointF distance = p - startPoint;
			startPoint = p;

			m_controlPoints[1] += distance;

			invalidate();
            createPath();

			canvasWidth = std::abs(endPoint.x() - startPoint.x());
			canvasHeight = std::abs(endPoint.y() - startPoint.y());
		}
	}


	else  if (m_mouseDrag && m_activeControlPoint > 0 && m_activeControlPoint < m_controlPoints.size()) {
        if (indexIsRealPoint(m_activeControlPoint)) {
            //move also the tangents
            QPointF targetPoint = p;
            QPointF distance = targetPoint - m_controlPoints[m_activeControlPoint];
            m_controlPoints[m_activeControlPoint] = targetPoint;
			if (m_activeControlPoint > 0)
			{
				m_controlPoints[m_activeControlPoint - 1] += distance;
			}
			if (m_controlPoints.size() > (m_activeControlPoint + 1))
			{
				m_controlPoints[m_activeControlPoint + 1] += distance;
			}
        } else {
            if (!isControlPointSmooth(m_activeControlPoint)) {
                m_controlPoints[m_activeControlPoint] = p;
            } else {
                QPointF targetPoint = p;
                QPointF distance = targetPoint - m_controlPoints[m_activeControlPoint];
                m_controlPoints[m_activeControlPoint] = p;

                if ((m_activeControlPoint > 1) && (m_activeControlPoint % 3) == 1) { //right control point
                    m_controlPoints[m_activeControlPoint - 2] -= distance;
                } else if ((m_activeControlPoint < (m_controlPoints.count() - 2)) //left control point
                           && (m_activeControlPoint % 3) == 1) {
                    m_controlPoints[m_activeControlPoint + 2] -= distance;
                }
            }
        }
        invalidate();
        createPath();
	}
	
	if (!endPoint.isNull() && (m_activeControlPoint == m_controlPoints.size() - 1))
	{
		endPoint = m_controlPoints[m_activeControlPoint];
		canvasWidth = std::abs(endPoint.x() - startPoint.x());
		canvasHeight = std::abs(endPoint.y() - startPoint.y());
	}
}

void 
GraphViewShapeItem::setEasingCurve(const QEasingCurve &easingCurve)
{
    if (m_easingCurve == easingCurve)
        return;

    m_easingCurve = easingCurve;
	
	m_controlPoints.clear();
    m_controlPoints.append(startPoint);
	foreach (QPointF point, m_easingCurve.toCubicSpline())
	{
		m_controlPoints.append(mapToCanvas(point));
	}

    m_numberOfSegments = m_controlPoints.count() / 3;
    update();
    emit easingCurveChanged();

}

void
GraphViewShapeItem::finishEditing()
{
    QApplication::setOverrideCursor(Qt::ArrowCursor);
}



