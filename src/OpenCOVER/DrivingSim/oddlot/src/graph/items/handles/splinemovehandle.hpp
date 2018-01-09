/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   20.04.2010
**
**************************************************************************/

#ifndef SPLINEMOVEHANDLE_HPP
#define SPLINEMOVEHANDLE_HPP

#include "src/graph/items/handles/movehandle.hpp"

// Data //
//
#include "src/data/roadsystem/lateralSections/polynomiallateralsection.hpp"

class ShapeEditor;
class ShapeSection;
class ShapeSectionPolynomialItem;
class ShapeSectionPolynomialItems;

class QGraphicsTextItem;

class SplineMoveHandle : public MoveHandle
{
	Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SplineMoveHandle(ShapeEditor *shapeEditor, ShapeSectionPolynomialItem *parent, SplineControlPoint *corner);
    virtual ~SplineMoveHandle();


    // Observer Pattern //
    //
    virtual void updateObserver();

protected:
private:
    SplineMoveHandle(); /* not allowed */
    SplineMoveHandle(const SplineMoveHandle &); /* not allowed */

	void init();
    void updateColor();

	//################//
	// SLOTS          //
	//################//

public slots:
	void deleteCorner();
/*	void smooth();
	void corner(); */

    //################//
    // EVENTS         //
    //################//

protected:
	virtual void mousePressEvent(QGraphicsSceneMouseEvent *event);
	virtual void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
	virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
	virtual void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
	virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);

protected:
	virtual void contextMenuEvent(QGraphicsSceneContextMenuEvent *event);

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    ShapeEditor *shapeEditor_;
	SplineControlPoint *corner_;
	ShapeSectionPolynomialItem *parentPolynomialItem_;
	ShapeSectionPolynomialItems *parentPolynomialItems_;

	PolynomialLateralSection *lateralSection_;
	ShapeSection *shapeSection_;

	QAction *deleteAction_;
/*	QAction *smoothAction_;
	QAction *cornerAction_; */

	bool firstShapeSectionPoint_;
	bool lastShapeSectionPoint_;

	QGraphicsTextItem *textHandle_;

};

#endif // SPLINEMOVEHANDLE_HPP
