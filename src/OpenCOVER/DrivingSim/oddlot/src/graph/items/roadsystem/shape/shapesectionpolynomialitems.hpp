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

	virtual void createPath(); // draws lanes

	// ShapeSectionPolynomialItems //
	//
	void createPolynomialItems();

	virtual QRectF boundingRect();

	double getSectionWidth();

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
};

#endif // SHAPESECTIONPOLYNOMIALITEMS_HPP
