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

#ifndef POLYNOMIALLATERALSECTION_HPP
#define POLYNOMIALLATERALSECTION_HPP

#include "lateralsection.hpp"
#include "src/util/math/polynomial.hpp"

class Polynomial;
class QPointF;
class PolynomialLateralSection;

class SplineControlPoint 
{
public:
	explicit SplineControlPoint(PolynomialLateralSection *parentLateralSection, QPointF p, bool low, bool smooth = false);
	~SplineControlPoint() {};

	PolynomialLateralSection *getParent()
	{
		return parentLateralSection_;
	}

	QPointF &getPoint()
	{
		return point_;
	}

	bool isLow()
	{
		return low_;
	}


private:
	QPointF point_;
	bool low_;

	PolynomialLateralSection *parentLateralSection_;
};

class PolynomialLateralSection : public LateralSection, public Polynomial
{
public:

    //################//
    // STATIC         //
    //################//

    enum PolynomialLateralSectionChange
    {
        CPL_ParameterChange = 0x1,
		CPL_PolynomialLateralSectionChange = 0x2
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit PolynomialLateralSection(double t, double a = 0.0, double b = 0.0, double c = 0.0, double d = 0.0);
	virtual ~PolynomialLateralSection();

	// LateralSection //
	//
	virtual double getTEnd() const;
	virtual double getLength() const;

    // PolynomialLateralSection //
    //
	void getRealPointsFromParameters();
	void getControlPointsFromParameters(bool markZeroLengthSection);

	SplineControlPoint *getRealPointLow()
	{
		return realPointLow_;
	}
	SplineControlPoint *getRealPointHigh()
	{
		return realPointHigh_;
	}

	void setControlPoints(QPointF p0, QPointF p1); 


    // Observer Pattern //
    //
    virtual void notificationDone();
    int getPolynomialLateralSectionChanges() const
    {
        return polynomialLateralSectionChanges_;
    }
    void addPolynomialLateralSectionChanges(int changes);

    // Prototype Pattern //
    //
    PolynomialLateralSection *getClone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    PolynomialLateralSection(); /* not allowed */
    PolynomialLateralSection(const PolynomialLateralSection &); /* not allowed */
    PolynomialLateralSection &operator=(const PolynomialLateralSection &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int polynomialLateralSectionChanges_;

	SplineControlPoint *realPointLow_, *realPointHigh_; 
};

#endif // POLYNOMIALLATERALSECTION_HPP
