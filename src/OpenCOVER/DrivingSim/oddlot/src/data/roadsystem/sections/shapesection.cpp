/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.03.2010
**
**************************************************************************/

#include "shapesection.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"


//####################//
// Constructors       //
//####################//

ShapeSection::ShapeSection(double s)
    : RoadSection(s)
{
}


void
ShapeSection::addShape(double t, Polynomial *poly)
{
	shapes_.insert(t, poly);

	addShapeSectionChanges(ShapeSection::CSS_ShapeSectionChange);
}

bool
ShapeSection::delShape(double t)
{
	Polynomial *poly = getShape(t);
	if (!poly)
	{ 
		qDebug("WARNING 1003221756! Tried to delete a shape section that wasn't there.");
		return false;
	}

	shapes_.remove(t);

	addShapeSectionChanges(ShapeSection::CSS_ShapeSectionChange);
}

Polynomial *
ShapeSection::getShape(double t) const
{
	QMap<double, Polynomial *>::const_iterator i = shapes_.upperBound(t);
	if (i == shapes_.constBegin())
	{
		//		qDebug("WARNING 1003221755! Trying to get superelevationSection but coordinate is out of bounds!");
		return NULL;
	}
	else
	{
		--i;
		return i.value();
	}
}

double 
ShapeSection::getShapeElevationDegree(double t)
{
	Polynomial *poly = getShape(t);
	t = t - shapes_.key(poly);

	return poly->f(t);
}

//####################//
// Section Functions  //
//####################//

/*!
* Returns the end coordinate of this section.
* In road coordinates [m].
*
*/
double
ShapeSection::getSEnd() const
{
    return getParentRoad()->getShapeSectionEnd(getSStart());
}

/*!
* Returns the length coordinate of this section.
* In [m].
*
*/
double
ShapeSection::getLength() const
{
    return getParentRoad()->getShapeSectionEnd(getSStart()) - getSStart();
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
ShapeSection::notificationDone()
{
    shapeSectionChanges_ = 0x0;
    RoadSection::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
ShapeSection::addShapeSectionChanges(int changes)
{
    if (changes)
    {
		shapeSectionChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
ShapeSection *
ShapeSection::getClone()
{
    // ShapeSection //
    //
    ShapeSection *clone = new ShapeSection(getSStart());

    return clone;
}

//###################//
// Visitor Pattern   //
//###################//

/*!
* Accepts a visitor for this section.
*
* \param visitor The visitor that will be visited.
*/
void
ShapeSection::accept(Visitor *visitor)
{
    visitor->visit(this);
}
