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

ShapeSection::ShapeSection(double s, double t, PolynomialLateralSection *poly)
    : RoadSection(s)
{
	if (!poly)
	{
		poly = new PolynomialLateralSection(t);
	}

	addShape(poly->getTStart(), poly);

}

ShapeSection::~ShapeSection()
{
	foreach(PolynomialLateralSection *poly, shapes_)
	{
		delete poly;
	}
}

void
ShapeSection::addShape(double t, PolynomialLateralSection *poly)
{
	poly->setParentSection(this);

	if (!shapes_.isEmpty())
	{
		PolynomialLateralSection *lastSection = getShape(poly->getTStart()); // the section that is here before insertion
		if (lastSection)
		{
			lastSection->addLateralSectionChanges(LateralSection::CLS_LengthChange);
		}
	}
	shapes_.insert(t, poly);

	addShapeSectionChanges(ShapeSection::CSS_ShapeSectionChange);
}

bool
ShapeSection::delShape(double t)
{
	PolynomialLateralSection *poly = getShape(t);
	if (!poly)
	{ 
		qDebug("WARNING 1003221756! Tried to delete a shape section that wasn't there.");
		return false;
	}

	poly->setParentSection(NULL);
	shapes_.remove(t);

	addShapeSectionChanges(ShapeSection::CSS_ShapeSectionChange);

	PolynomialLateralSection *lastSection = getShape(t); // the section that is here before insertion
	if (lastSection)
	{
		lastSection->addLateralSectionChanges(LateralSection::CLS_LengthChange);
	}

	return true;
}

PolynomialLateralSection *
ShapeSection::getShape(double t) const
{
	QMap<double, PolynomialLateralSection *>::const_iterator i = shapes_.upperBound(t);
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

/*! \brief Moves the section to the new t coordinate.
*
* The t coordinate will be clamped to [0.0, roadWidth].
*/
bool
ShapeSection::moveLateralSection(LateralSection *section, double newT)
{
	// Clamp (just in case) //
	//
	/* if (newT < 0.0)
	{
		qDebug("WARNING 1007141735! Tried to move a lateral section but failed (t < 0.0).");
		newT = 0.0;
	} */

	if (!section)
	{
		return false;
	}
	double oldT = section->getTStart();

	// Previous section //
	//
	double previousT = getFirstPolynomialLateralSection()->getTStart();
	if (newT > oldT)
	{
		// Expand previous section //
		//
		previousT = section->getTStart() - 0.001;
	}
	else
	{
		// Shrink previous section //
		//
		previousT = newT;
	}
	LateralSection *previousSection = getShape(previousT);
	if (previousSection)
	{
		previousSection->addLateralSectionChanges(LateralSection::CLS_LengthChange);
	}

	// Set and insert //
	//
	PolynomialLateralSection *polySection = dynamic_cast<PolynomialLateralSection *>(section);  // Has to be implemented for different types
	if (polySection)
	{
		polySection->setTStart(newT);
		shapes_.remove(oldT);
		shapes_.insert(newT, polySection);
	}

	return true;
}

double 
ShapeSection::getShapeElevationDegree(double t)
{
	PolynomialLateralSection *poly = getShape(t);
	t = t - shapes_.key(poly);

	return poly->f(t);
}

int
ShapeSection::getShapesMaxDegree()
{
	int degree = -1;
	foreach(PolynomialLateralSection *poly, shapes_)
	{
		int d = poly->getDegree();
		if (d == 3)
		{
			return 3;
		}
		else if (d > degree)
		{
			degree = d;
		}
	}

	return degree;
}

double
ShapeSection::getWidth()
{
	return getParentRoad()->getMaxWidth(getSStart()) - getParentRoad()->getMinWidth(getSStart());
}

double 
ShapeSection::getLength(double tStart)
{
	PolynomialLateralSection *nextSection = getPolynomialLateralSectionNext(tStart);
	if (nextSection)
	{
		return nextSection->getTStart() - tStart;
	}
	
	return getParentRoad()->getMaxWidth(getSStart()) - tStart;

}

void 
ShapeSection::checkSmooth(PolynomialLateralSection *lateralSectionBefore, PolynomialLateralSection *lateralSection)
{
	double df1 = lateralSectionBefore->df(lateralSection->getTStart() - lateralSectionBefore->getTStart());
	double df2 = lateralSection->df(0.0);

	if (abs(df2 - df1) < NUMERICAL_ZERO6)
	{
		lateralSectionBefore->getRealPointHigh()->setSmooth(true);
		lateralSectionBefore->getSplineControlPointHigh()->setSmooth(true);
		lateralSection->getRealPointLow()->setSmooth(true);
		lateralSection->getSplineControlPointLow()->setSmooth(true);
	}
}

/*
QVector<QPointF> 
ShapeSection::getControlPoints()
{
	QVector<QPointF> controlPoints;
	QPointF p0, p1, p2, p3;
	QMap<double, PolynomialLateralSection *>::const_iterator it = shapes_.constBegin();
	while (it != shapes_.constEnd())
	{
		Polynomial *poly = it.value();
		double xlength;
		if (shapes_.size() < 2)
		{
			xlength = getParentRoad()->getMaxWidth(getSStart()) - getParentRoad()->getMinWidth(getSStart());
		}
		else if (it == shapes_.constEnd() - 1)
		{
			xlength = getParentRoad()->getMaxWidth(getSStart()) - getParentRoad()->getMinWidth(getSStart()) - it.key();
		}
		else
		{
			xlength = (it + 1).key() - it.key();
		}
		poly->getControlPoints(xlength, p0, p1, p2, p3);
		controlPoints.append(p0);
		controlPoints.append(p1);
		controlPoints.append(p2);

		it++;
	}
	controlPoints.append(p3);

	return controlPoints;
}

void 
ShapeSection::setPolynomialParameters(QVector<QPointF> controlPoints)
{
	int i = 0;
	QMap<double, Polynomial *>::const_iterator it = shapes_.constBegin();
	while ((it != shapes_.constEnd()) && (i + 3 < controlPoints.size()))
	{
		it.value()->setParametersFromControlPoints(controlPoints.at(i), controlPoints.at(i + 1), controlPoints.at(i + 2), controlPoints.at(i + 3));
		i += 3;
	}

	addShapeSectionChanges(ShapeSection::CSS_ParameterChange);
}
*/

PolynomialLateralSection *
ShapeSection::getFirstPolynomialLateralSection() const
{
	if (!shapes_.isEmpty())
	{
		return shapes_.first();
	}

	return NULL;
}

PolynomialLateralSection *
ShapeSection::getLastPolynomialLateralSection() const
{
	if (!shapes_.isEmpty())
	{
		return shapes_.last();
	}

	return NULL;
}

double 
ShapeSection::getPolynomialLateralSectionEnd(double t) const
{
	QMap<double, PolynomialLateralSection *>::const_iterator nextIt = shapes_.upperBound(t);
	if (nextIt == shapes_.constEnd())
	{
		return getParentRoad()->getMaxWidth(getSStart());
	}
	else
	{
		return (*nextIt)->getTStart();
	}
}

PolynomialLateralSection *
ShapeSection::getPolynomialLateralSection(double t) const
{
	QMap<double, PolynomialLateralSection *>::const_iterator i = shapes_.upperBound(t);
	if (i == shapes_.constBegin())
	{
		//		qDebug("WARNING 1003221757! Trying to get crossfallSection but coordinate is out of bounds!");
		return NULL;
	}
	else
	{
		--i;
		return i.value();
	}
}

PolynomialLateralSection *
ShapeSection::getPolynomialLateralSectionBefore(double t) const
{
	QMap<double, PolynomialLateralSection *>::const_iterator i = shapes_.upperBound(t); // the second one after the one we want
	if (i == shapes_.constBegin())
	{
		return NULL;
	}
	--i;

	if (i == shapes_.constBegin())
	{
		return NULL;
	}
	--i;

	return i.value();
}

PolynomialLateralSection *
ShapeSection::getPolynomialLateralSectionNext(double t) const
{
	QMap<double, PolynomialLateralSection *>::const_iterator i = shapes_.upperBound(t);
	if (i == shapes_.constEnd())
	{
		return NULL;
	}

	return i.value();
}

void 
ShapeSection::setPolynomialLateralSections(QMap<double, PolynomialLateralSection *> newShapes)
{
	foreach(PolynomialLateralSection *section, shapes_)
	{
		section->setParentSection(NULL);
	}

	foreach(PolynomialLateralSection *section, newShapes)
	{
		section->setParentSection(this);
	}
	shapes_ = newShapes;
	addShapeSectionChanges(ShapeSection::CSS_ParameterChange);
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
	ShapeSection *clone;
	if (getParentRoad()->getLaneSection(getSStart()))
	{
		clone = new ShapeSection(getSStart(), getParentRoad()->getMinWidth(getSStart()));
	}
	else
	{
		clone = new ShapeSection(getSStart(), 0.0);
	}

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
