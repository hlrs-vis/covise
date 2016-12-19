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

#ifndef GRAPHVIEWSHAPEITEM_HPP
#define GRAPHVIEWSHAPEITEM_HPP

#include <QWidget>
#include <QMenu>
#include <QAction>

#include <QEasingCurve>
#include <QMap>
#include <QGraphicsPathItem>

class GraphView;
class OpenScenarioEditor;

class GraphViewShapeItem :  public QObject, public QGraphicsPathItem
{
    Q_OBJECT

     Q_PROPERTY(QEasingCurve easingCurve READ easingCurve WRITE setEasingCurve NOTIFY easingCurveChanged);

public:
    explicit GraphViewShapeItem(GraphView *view, OpenScenarioEditor *editor, int x, int y, int width, int height);
    virtual ~GraphViewShapeItem();

    void createPath();
    void contextMenu(QContextMenuEvent *event);

	//################//
	// SIGNALS        //
	//################//

signals:
    void easingCurveChanged();

    //################//
    // SLOTS          //
    //################//

public slots:
    void setEasingCurve(const QEasingCurve &easingCurve);

    //################//
    // EVENTS         //
    //################//

public:
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);

protected:
 //   void contextMenuEvent(QContextMenuEvent *);


private:

	double s_;
	double t_;
    QPointF pos_;
	QGraphicsItemGroup *graphicItemGroup_;

    QPointF pressPos_;
	QPointF lastPos_;
	bool doPan_;
	bool copyPan_;

    QColor color_;

    // Spline //
    //
    GraphView *view_;
    OpenScenarioEditor *editor_;

    void invalidate();
    void invalidateSmoothList();

    QEasingCurve easingCurve() const
    { return m_easingCurve; }

	QPointF mapToCanvas(const QPointF &point);
	QPointF mapFromCanvas(const QPointF &point);
	void paintControlPoint(const QPointF &point, bool edit, bool realPoint, bool active, bool smooth);

    int findControlPoint(const QPointF &point, qreal &distance);
    int findRealPoint(const QPointF &point);
    bool isSmooth(int i) const;

    void smoothPoint( int index);
    void cornerPoint( int index);
    void deletePoint(int index);
    void addPoint(const QPointF point);
	void appendPoint(const QPointF point);
    void finishEditing();

    bool isControlPointSmooth(int i) const;

    QEasingCurve m_easingCurve;
    QVector<QPointF> m_controlPoints;
    QVector<bool> m_smoothList;
    int m_numberOfSegments;
    int m_activeControlPoint;
    bool m_mouseDrag;
    QPointF m_mousePress;

    QMenu *m_pointContextMenu;
    QMenu *m_curveContextMenu;
    QAction *m_deleteAction;
    QAction *m_smoothAction;
    QAction *m_cornerAction;
    QAction *m_addPoint;
	QAction *m_appendPoint;
    QAction *m_finishEdit;

	QPointF startPoint;
	QPointF endPoint;
	int canvasWidth;
	int canvasHeight;

	bool selected;
};

#endif // GRAPHVIEWSHAPEITEM_HPP
