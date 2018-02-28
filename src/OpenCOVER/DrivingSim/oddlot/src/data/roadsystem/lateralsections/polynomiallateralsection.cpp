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

#include "polynomiallateralsection.hpp"

// Data //
//
#include "src/data/roadsystem/sections/shapesection.hpp"
#include "src/data/projectdata.hpp"
#include "src/data/changemanager.hpp"




//####################//
// Constructors       //
//####################//


PolynomialLateralSection::PolynomialLateralSection(double t, double a, double b, double c, double d)
    : LateralSection(t)
	, Polynomial(a, b, c, d)
	, polynomialLateralSectionChanges_(0x0)
{
	realPointLow_ = new SplineControlPoint(this, QPointF(0, 0), true);
	realPointHigh_ = new SplineControlPoint(this, QPointF(0, 0), false);
}

PolynomialLateralSection::~PolynomialLateralSection()
{
}

void
PolynomialLateralSection::getRealPointsFromParameters()
{
	QPointF T(getTStart(), 0);
	double l = getLength();


	realPointLow_->getPoint() = QPointF(0, a_) + T;
	realPointHigh_->getPoint() = QPointF(l, f(l)) + T;
}

void
PolynomialLateralSection::getControlPointsFromParameters(bool markZeroLengthSection)
{

	QPointF T(getTStart(), 0);
	double l = getLength();

	if (markZeroLengthSection && (abs(l) < NUMERICAL_ZERO6))
	{
		realPointLow_ = realPointHigh_ = NULL;
	}
	else
	{
		realPointLow_->getPoint() = QPointF(0, a_) + T;
		realPointHigh_->getPoint() = QPointF(l, f(l)) + T;
	}

}


void
PolynomialLateralSection::setControlPoints(QPointF p0, QPointF p3)  //, QPointF p2, QPointF p3)
{
	realPointLow_->getPoint() = p0;
	realPointHigh_->getPoint() = p3;

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
PolynomialLateralSection::getTEnd() const
{
    return getParentSection()->getPolynomialLateralSectionEnd(getTStart());
}

/*!
* Returns the length coordinate of this section.
* In [m].
*
*/
double
PolynomialLateralSection::getLength() const
{
    return getParentSection()->getLength(getTStart());
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
PolynomialLateralSection::notificationDone()
{
    polynomialLateralSectionChanges_ = 0x0;
    LateralSection::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
PolynomialLateralSection::addPolynomialLateralSectionChanges(int changes)
{
    if (changes)
    {
		polynomialLateralSectionChanges_ |= changes;
        notifyObservers();
		getProjectData()->getChangeManager()->notifyObservers();  // to be deleted, done by commands
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
PolynomialLateralSection *
PolynomialLateralSection::getClone()
{
    // PolynomialLateralSection //
    //
    PolynomialLateralSection *clone = new PolynomialLateralSection(getTStart(), getA(), getB(), getC(), getD());
	clone->realPointLow_ = new SplineControlPoint(clone, realPointLow_->getPoint(), true);
	clone->realPointHigh_ = new SplineControlPoint(clone, realPointHigh_->getPoint(), false);

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
PolynomialLateralSection::accept(Visitor *visitor)
{
    visitor->visit(this);
}


//#######################//
// Spline Control Points //
//######################//

SplineControlPoint::SplineControlPoint(PolynomialLateralSection *parentLateralSection, QPointF p, bool low, bool smooth)
	: parentLateralSection_(parentLateralSection)
	, point_(p)
	, low_(low)
{
}


