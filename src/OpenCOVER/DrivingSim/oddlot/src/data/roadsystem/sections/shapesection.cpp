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

// Eigen //
//
#include <Eigen/Sparse>

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
ShapeSection::calculateShapeParameters()
{
	int n = shapes_.size() - 1;
	Eigen::VectorXd b(n), c(n);
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	tripletList.reserve(n);


	if (n == 0)
	{
		PolynomialLateralSection *poly = getPolynomialLateralSection(0.0);
		QPointF low = poly->getRealPointLow()->getPoint();
		QPointF high = poly->getRealPointHigh()->getPoint();
		poly->setParameters(low.y(), (high.y() - low.y()) / (high.x() - low.x()), 0.0, 0.0);

		poly->addPolynomialLateralSectionChanges(PolynomialLateralSection::CPL_ParameterChange);
		addShapeSectionChanges(ShapeSection::CSS_ParameterChange);
		return;
	}

	QMap<double, PolynomialLateralSection *>::const_iterator it = shapes_.constBegin();
	it++;
	int i = 1;
	while (it != shapes_.constEnd())
	{
		PolynomialLateralSection *poly = it.value();
		PolynomialLateralSection *lateralSectionBefore = getPolynomialLateralSectionBefore(poly->getTStart());
		PolynomialLateralSection *nextLateralSection = getPolynomialLateralSectionNext(poly->getTStart());
		double a2 = poly->getRealPointHigh()->getPoint().y();
		double a1 = poly->getRealPointLow()->getPoint().y();
		double a0 = lateralSectionBefore->getRealPointLow()->getPoint().y();
		double x2 = poly->getRealPointHigh()->getPoint().x();

		double x1 = poly->getTStart();
		double x0 = lateralSectionBefore->getTStart();

		b(i - 1) = (3 * ((a2 - a1) / (x2 - x1) - (a1 - a0) / (x1 - x0)));

		if (i > 1)
		{
			tripletList.push_back(T(i - 1, i - 2, x1 - x0));
			tripletList.push_back(T(i - 2, i - 1, x1 - x0));

		}
		tripletList.push_back(T(i - 1, i - 1, 2 * (x2 - x0)));

		i++;
		it++;
	}
//	PolynomialLateralSection *p = (--it).value();
//	qDebug() << "Last Point " << p->getRealPointHigh()->getPoint().x() << "," << p->getRealPointHigh()->getPoint().y();


	Eigen::SparseMatrix<double> A(n, n);
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;

	solver.compute(A);
	if (solver.info() != Eigen::ComputationInfo::Success)
	{
		qDebug() << "Solver not successful!";
		return;
	} 

	c = solver.solve(b);

/*	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			qDebug() << i << ";" << j << " " << A.coeff(i, j);
		}
	}

	for (int j = 0; j < n; j++)
	{
		qDebug() << "c " << c(j) << "; b " << b(j);
	} */

	it = shapes_.constBegin();
		
	PolynomialLateralSection *poly = it.value();

	PolynomialLateralSection *nextLateralSection = getPolynomialLateralSectionNext(poly->getTStart());
	double a0 = poly->getRealPointLow()->getPoint().y();
	double a1 = nextLateralSection->getRealPointLow()->getPoint().y();
	double x = nextLateralSection->getTStart() - poly->getTStart();

	double b0 = (a1 - a0) / x - (c(0) * x / 3);
	double d0 = c(0) / (3 * x);

	poly->setParameters(a0, b0, 0.0, d0);
//	qDebug() << "point: " << poly->getRealPointLow()->getPoint().x() << "," << poly->getRealPointLow()->getPoint().y() << " parameter: " << a0 << "," << b0 << ",0.0," << d0;
	poly->addPolynomialLateralSectionChanges(PolynomialLateralSection::CPL_ParameterChange);
	it++;
	i = 1;
	while (it != shapes_.constEnd())
	{
		PolynomialLateralSection *poly = it.value();
		a1 = poly->getRealPointLow()->getPoint().y();
		PolynomialLateralSection *nextLateralSection = getPolynomialLateralSectionNext(poly->getTStart());
		double a2 = poly->getRealPointHigh()->getPoint().y();
		double x = poly->getRealPointHigh()->getPoint().x() - poly->getTStart();
		double b1, d1;

		if (nextLateralSection)
		{
			b1 = (a2 - a1) / x - (c(i) - c(i - 1)) * x / 3  - c(i - 1) * x;
			d1 = (c(i) - c(i - 1)) / (3 * x);
		}
		else
		{
			b1 = (a2 - a1) / x + c(i - 1) * x / 3 - c(i - 1) * x;
			d1 = -c(i - 1) / (3 * x);
		}

		poly->setParameters(a1, b1, c(i-1), d1);
//		qDebug() << "point: " << poly->getRealPointLow()->getPoint().x() << "," << poly->getRealPointLow()->getPoint().y() << " parameter: " << a1 << "," << b1 << "," << c(i-1) << "," << d1;
		poly->addPolynomialLateralSectionChanges(PolynomialLateralSection::CPL_ParameterChange);

		it++;
		i++;
	}

	addShapeSectionChanges(ShapeSection::CSS_ParameterChange);

}


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

	foreach (PolynomialLateralSection *poly, shapes_)
	{
		clone->addShape(poly->getTStart(), poly->getClone());
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
