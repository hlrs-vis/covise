/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   14.07.2010
**
**************************************************************************/

#ifndef SHAPESECTIONPOLYNOMIALITEMS_HPP
#define SHAPESECTIONPOLYNOMIALITEMS_HPP

#include "src/graph/items/graphelement.hpp"

class ShapeEditor;
class ShapeSection;
class SplineControlPoint;
class SplineMoveHandle;

class ShapeSectionPolynomialItems : public GraphElement
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ShapeSectionPolynomialItems(ProfileGraph *profileGraph, ShapeSection *shapeSection);
    virtual ~ShapeSectionPolynomialItems();

	virtual ProfileGraph *getProfileGraph() const
	{
		return profileGraph_;
	}

	// ShapeSectionPolynomialItems //
	//
	void createPolynomialItems();
	virtual QRectF boundingRect();
	QRectF *getCanvas()
	{
		return splineCanvas_;
	}

	void initNormalization();
	QPointF normalize(const QPointF &p);
	QPointF sceneCoords(const QPointF &start, const QPointF &p);

/*	void appendShapeSectionPolynomialItem(ShapeSectionPolynomialItem *polynomialItem);
	bool removeShapeSectionPolynomialItem(ShapeSectionPolynomialItem *polynomialItem);*/
/*	QMap<QString, RoadItem *> getRoadItems() const
	{
		return roadItems_;
	} */

	// RoadSystem //
	//
/*	RoadSystem *getRoadSystem() const
	{
		return roadSystem_;
	} */


    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest()
	{
		return false;
	};

private:
    ShapeSectionPolynomialItems(); /* not allowed */
    ShapeSectionPolynomialItems(const ShapeSectionPolynomialItems &); /* not allowed */
    ShapeSectionPolynomialItems &operator=(const ShapeSectionPolynomialItems &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
	ProfileGraph *profileGraph_;
	ShapeEditor *shapeEditor_;

    ShapeSection *shapeSection_;
	QTransform *transform_;  // transformation matrix to map points to values between (0,0) and (1,1) recommended by QEasingCurve
	QTransform *transformInverted_;

	QRectF *splineCanvas_;

	SplineMoveHandle *lastSplineMoveHandle_;
};

#endif // SHAPESECTIONPOLYNOMIALITEMS_HPP
