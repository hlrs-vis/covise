/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   15.07.2010
**
**************************************************************************/

#ifndef SHAPESECTIONPOLYNOMIALITEM_HPP
#define SHAPESECTIONPOLYNOMIALITEM_HPP

#include "src/graph/items/roadsystem/sections/lateralsectionitem.hpp"

#include "src/graph/editors/shapeeditor.hpp"


class ShapeSectionPolynomialItems;
class ShapeSection;
class PolynomialLateralSection;
class SplineMoveHandle;
class ShapeEditor;

#include <QEasingCurve>
#include <QAction>

class ShapeSectionPolynomialItem : public LateralSectionItem
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ShapeSectionPolynomialItem(ShapeSectionPolynomialItems *parentShapeSectionPolynomialItems, PolynomialLateralSection *polynomialLateralSection, ShapeEditor *shapeEditor);
    virtual ~ShapeSectionPolynomialItem();


    // Section //
    //
	PolynomialLateralSection *getPolynomialLateralSection() const
    {
        return polynomialLateralSection_;
    }

	// Root Item //
	//
	ShapeSectionPolynomialItems *getParentPolynomialItems()
	{
		return parentShapeSectionPolynomialItems_;
	}

	void contextMenu(QGraphicsSceneContextMenuEvent *event);

	// set Handles //
	//
	void deleteControlPointHandles();
	void deleteRealPointHighHandle();

    // Graphics //
    //
    void updateColor();
    virtual void createPath();

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest()
    {
        return false;
    };

private:
    ShapeSectionPolynomialItem(); /* not allowed */
    ShapeSectionPolynomialItem(const ShapeSectionPolynomialItem &); /* not allowed */
    ShapeSectionPolynomialItem &operator=(const ShapeSectionPolynomialItem &); /* not allowed */

    void init();

public:

	//################//
	// SIGNALS        //
	//################//

signals:
	void easingCurveChanged();

    //################//
    // SLOTS          //
    //################//

public slots:
    virtual bool removeSection();
//	void setEasingCurve(const QEasingCurve &easingCurve);

    //################//
    // EVENTS         //
    //################//


public:
/*	virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
	virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
	virtual void mousePressEvent(QGraphicsSceneMouseEvent *event); */


    //################//
    // PROPERTIES     //
    //################//

private:
    // ShapeSectionPolynomialItems //
    //
	ShapeSectionPolynomialItems *parentShapeSectionPolynomialItems_;

    // Section //
    //
    PolynomialLateralSection *polynomialLateralSection_;
	ShapeSection *shapeSection_;

	// Editor //
	//
	ShapeEditor *shapeEditor_;

    // ContextMenu //
    //
    QAction *splitAction_;

	// SplineMoveHandles //
	//
	SplineMoveHandle *realPointLowHandle_, *controlPointLowHandle_, *controlPointHighHandle_, *realPointHighHandle_;

	// Spline Control Points
	//
	QVector<QPointF> controlPoints_;
	QVector<bool> smoothList_;

	int numberOfSegments_;
	int activeControlPoint_ ;
	bool mouseDrag_;
	int canvasWidth_, canvasHeight_;
	QPointF startPoint_, endPoint_;

	QAction *addPoint_;
	QAction *appendPoint_;

	QPointF mapToCanvas(const QPointF &point);
	QPointF mapFromCanvas(const QPointF &point);

	void invalidate();
	void invalidateSmoothList();

	QEasingCurve easingCurve() const
	{
		return easingCurve_;
	}

	QPainterPath  paintControlPoint(const QPointF &point, bool edit, bool realPoint, bool active, bool smooth);

	int findControlPoint(const QPointF &point, qreal &distance);
	int findRealPoint(const QPointF &point);
	bool isSmooth(int i) const;

	void smoothPoint(int index);
	void cornerPoint(int index);
	void deletePoint(int index);
	void addPoint(const QPointF point);
	void appendPoint(const QPointF point);

	bool isControlPointSmooth(int i) const;

	QEasingCurve easingCurve_;
	QPointF mousePress_;
};

#endif // SHAPESECTIONPOLYNOMIALITEM_HPP
