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
	realPointLow_ = new SplineControlPoint(this, QPointF(0, 0), true, true);
	realPointHigh_ = new SplineControlPoint(this, QPointF(0, 0), true, false);
	controlPointLow_ = new SplineControlPoint(this, QPointF(0, 0), false, true);
	controlPointHigh_ = new SplineControlPoint(this, QPointF(0, 0), false, false);
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
PolynomialLateralSection::getControlPointsFromParameters()
{

	QPointF T(getTStart(), 0);
	double l = getLength();


	if (degree_ <= 0)
	{
		// if df == 0 
		realPointLow_->getPoint() = QPointF(0, a_) + T;
		realPointHigh_->getPoint() = QPointF(l, f(l)) + T;

		if (realPointLow_->isSmooth())
		{
			PolynomialLateralSection *lateralSectionBefore = getParentSection()->getPolynomialLateralSectionBefore(getTStart());
			controlPointLow_->getPoint() = realPointLow_->getPoint() + (realPointHigh_->getPoint() - lateralSectionBefore->getRealPointLow()->getPoint()) / 6;
		}
		else
		{
			controlPointLow_->getPoint() = QPointF(l / 3, b_ * l / 3 + a_) + T;
		}

		if (realPointHigh_->isSmooth())
		{
			PolynomialLateralSection *lateralSectionNext = getParentSection()->getPolynomialLateralSectionNext(getTStart());

			controlPointHigh_->getPoint() = realPointHigh_->getPoint() - (lateralSectionNext->getRealPointHigh()->getPoint() - realPointLow_->getPoint()) / 6;
		}
		else
		{
			double cl = 2 * l / 3;
			//		controlPointHigh_->getPoint() = QPointF(cl, l* (realPointHigh_->getPoint().y() - df(l)) / 3) + T;
			controlPointHigh_->getPoint() = QPointF(cl, f(l) - (l *df(l) / 3)) + T;
		}
	}
	else
	{
		getPolynomialControlPoints(getLength(), realPointLow_->getPoint(), controlPointLow_->getPoint(), controlPointHigh_->getPoint(), realPointHigh_->getPoint());


		realPointLow_->getPoint() += T;
		controlPointLow_->getPoint() += T;
		controlPointHigh_->getPoint() += T;
		realPointHigh_->getPoint() += T;
	}

}

void 
PolynomialLateralSection::setPolynomialParameters()
{
	QPointF T(getTStart(), 0);
	setParametersFromControlPoints(realPointLow_->getPoint() - T, controlPointLow_->getPoint() - T, controlPointHigh_->getPoint() - T, realPointHigh_->getPoint() - T);
}

void 
PolynomialLateralSection::setControlPoints(SplineControlPoint &p0, SplineControlPoint &p1, SplineControlPoint &p2, SplineControlPoint &p3)
{
	*realPointLow_ = p0;
	*controlPointLow_ = p1;
	*controlPointHigh_ = p2;
	*realPointHigh_ = p3;

	int degree = getDegree();
	setPolynomialParameters();

	ShapeSection *parentSection = getParentSection();
	if (parentSection)
	{
		if (getDegree() != degree)
		{
			getParentSection()->addShapeSectionChanges(ShapeSection::CSS_ParameterChange);
		}

		addPolynomialLateralSectionChanges(PolynomialLateralSection::CPL_ParameterChange);
	}
}

void
PolynomialLateralSection::setControlPoints(QPointF p0, QPointF p1, QPointF p2, QPointF p3)
{
	realPointLow_->getPoint() = p0;
	controlPointLow_->getPoint() = p1;
	controlPointHigh_->getPoint() = p2;
	realPointHigh_->getPoint() = p3;

	int degree = getDegree();
	setPolynomialParameters();

	ShapeSection *parentSection = getParentSection();
	if (parentSection)
	{
		if (getDegree() != degree)
		{
			getParentSection()->addShapeSectionChanges(ShapeSection::CSS_ParameterChange);
		}

		addPolynomialLateralSectionChanges(PolynomialLateralSection::CPL_ParameterChange);
	}

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
	clone->realPointLow_ = realPointLow_->getClone();
	clone->controlPointLow_ = controlPointLow_->getClone();
	clone->controlPointHigh_ = controlPointHigh_->getClone();
	clone->realPointHigh_ = realPointHigh_->getClone();

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

SplineControlPoint::SplineControlPoint(PolynomialLateralSection *parentLateralSection, QPointF p, bool real, bool low, bool smooth)
	: parentLateralSection_(parentLateralSection)
	, point_(p)
	, real_(real)
	, low_(low)
	, smooth_(smooth)
{
}

SplineControlPoint *
SplineControlPoint::getClone()
{
	return new SplineControlPoint(parentLateralSection_, point_, real_, low_, smooth_);
}

void 
SplineControlPoint::setSmooth(bool smooth)
{
	smooth_ = smooth;
	if (parentLateralSection_->getProjectData())
	{
		parentLateralSection_->addPolynomialLateralSectionChanges(PolynomialLateralSection::CPL_ParameterChange);
	}

}
