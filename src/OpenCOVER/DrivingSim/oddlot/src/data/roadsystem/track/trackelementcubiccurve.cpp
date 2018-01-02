/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/21/2010
**
**************************************************************************/

#include "trackelementcubiccurve.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

#include <cmath>


TrackElementCubicCurve::TrackElementCubicCurve(double x, double y, double angleDegrees, double s, double length, Polynomial *polynomialU, Polynomial *polynomialV, const QString &pRange)
    : TrackElement(x, y, angleDegrees, s, length)
	, polynomialU_(polynomialU)
	, polynomialV_(polynomialV)
	, pRange_(pRange)
    , trackElementCubicCurveChanges_(0x0)
{
    setTrackType(TrackComponent::DTT_CUBICCURVE);

	Va_ = polynomialV_->getA();
	Vb_ = polynomialV_->getB();
	Vc_ = polynomialV_->getC();
	Vd_ = polynomialV_->getD();

	Ua_ = polynomialU_->getA();
	Ub_ = polynomialU_->getB();
	Uc_ = polynomialU_->getC();
	Ud_ = polynomialU_->getD();

	if (pRange_ != "arcLength")
	{
		pRange_ = "normalized";
	}
}

TrackElementCubicCurve::~TrackElementCubicCurve()
{
}


//#########################//
// Track Element Functions //
//#########################//

/** Returns point on the track at road coordinate s.
Natural coordinates relative to Geometry (see FEM).
(Natuerliche Koordinaten)
The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QPointF
TrackElementCubicCurve::getPoint(double s, double d)
{
	double t = getT(s);

	//	qDebug() << "p3: " << getHeadingRad(s) << " " << -sin(getHeadingRad(s)) << " " << cos(getHeadingRad(s)) << " " << f(t);

	double hdg = hdgRad(t);
	return QPointF(f(Ua_, Ub_, Uc_, Ud_, t), f(Va_, Vb_, Vc_, Vd_, t)) + QVector2D(-sin(hdg), cos(hdg)).toPointF() * d;
}

/** Returns heading on the track at road coordinate s.
Natural coordinates relative to Geometry (see FEM).
(Natuerliche Koordinaten)
The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementCubicCurve::getHeading(double s)
{
	return hdgRad(getT(s)) * 360 / (2 * M_PI);
}

/** Returns heading on the track at road coordinate s.
Natural coordinates relative to Geometry (see FEM).
(Natuerliche Koordinaten)
The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementCubicCurve::getHeadingRad(double s)
{
	return hdgRad(getT(s));
}

/** Returns local point on the track at road coordinate s.
Relative to the parent composite.
The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QPointF
TrackElementCubicCurve::getLocalPoint(double s, double d)
{
	return getLocalTransform().map(getPoint(s, d));
}

/** Returns local heading of the track at road coordinate s.
Relative to the parent composite.
The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementCubicCurve::getLocalHeading(double s)
{
	return heading() + getHeading(s);
}

/** Returns local heading of the track at road coordinate s.
Relative to the parent composite.
The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementCubicCurve::getLocalHeadingRad(double s)
{
	return heading() * 2 * M_PI / 360.0 + getHeadingRad(s);
}

/*!
* Returns local tangent to the track at road coordinate s.
* Relative to the parent composite.
* The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QVector2D
TrackElementCubicCurve::getLocalTangent(double s)
{
	double hdg = getLocalHeadingRad(s);
	return QVector2D(cos(hdg), sin(hdg)); // hypotenuse = 1.0
}

/*!
* Returns local normal to the track at road coordinate s.
* Relative to the parent composite.
* The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QVector2D
TrackElementCubicCurve::getLocalNormal(double s)
{
	double hdg = getLocalHeadingRad(s);
	return QVector2D(sin(hdg), -cos(hdg)); // hypotenuse = 1.0
}

/** Returns curvature of the track at road coordinate s.
Independent of coordinate system.
The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementCubicCurve::getCurvature(double s)
{
	double t = getT(s);
	double curvature = (df(Ub_, Uc_, Ud_, t) * ddf(Vc_, Vd_, t) - ddf(Uc_ , Ud_, t) * df(Vb_, Vc_, Vd_, t)) / pow((df(Ub_, Uc_, Ud_, t) * df(Ub_, Uc_, Ud_, t) + df(Vb_, Vc_, Vd_, t) * df(Vb_, Vc_, Vd_, t)), 1.5);

	return curvature;
//	return ddf(Vc_, Vd_, t) / pow((1 + df(Vb_, Vc_, Vd_, t) * df(Vb_, Vc_, Vd_, t)), 1.5);
}

/** Evaluates the polynomial for a given argument x.
*/
double
TrackElementCubicCurve::f(double a, double b, double c, double d, double t) const
{
	return (a + b * t + c * t * t + d * t * t * t);
}

/** Calculates the first derivative of the polynomial
* for a given argument x.
*/
double
TrackElementCubicCurve::df(double b, double c, double d, double t) const
{
	return (b + 2 * c * t + 3 * d * t * t);
}

/** Calculates the second derivative of the polynomial
* for a given argument x.
*/
double
TrackElementCubicCurve::ddf(double c, double d, double t) const
{
	return (2 * c + 6 * d * t);
}

double
TrackElementCubicCurve::getT(double s)
{
	s = s - getSStart();

	double t = 0.5; // first approximation

	if (pRange_ == "arcLength")
	{
		t = s;
	}

	for (int i = 0; i < 20; ++i)
	{

		// New code with integration //
		//
		
		double f = getParametricCurveLength(0.0, t) - s;
		double df = sqrt((Ub_ + 2.0 * Uc_ * t + 3.0 * Ud_ * t * t) * (Ub_ + 2.0 * Uc_ * t + 3.0 * Ud_ * t * t) + (Vb_ + 2.0 * Vc_ * t + 3.0 * Vd_ * t * t) * (Vb_ + 2.0 * Vc_ * t + 3.0 * Vd_ * t * t));
		t -= f / df;
	}

	return t;
}

double
TrackElementCubicCurve::getParametricCurveLength(double from, double to)
{
	double factor = (to - from) / 2.0; 

	double l = 0.568888888888889 * parametricF(0.0, factor) + 0.4786286704993665 * parametricF(-0.5384693101056831, factor) 
		+ 0.4786286704993665 * parametricF(0.5384693101056831, factor) + 0.2369268850561891 * parametricF(-0.906179845938664, factor) 
		+ 0.2369268850561891 * parametricF(0.9061798459386640, factor);


	l = l * factor;
//	qDebug() << "to: Length: " << to << "," << l;

	return l;
}

double 
TrackElementCubicCurve::parametricF(double t, double factor)
{
	t = t * factor + factor;
	double q = sqrt((Ub_ + 2.0 * Uc_ * t + 3.0 * Ud_ * t * t) * (Ub_ + 2.0 * Uc_ * t + 3.0 * Ud_ * t * t) + (Vb_ + 2.0 * Vc_ * t + 3.0 * Vd_ * t * t) * (Vb_ + 2.0 * Vc_ * t + 3.0 * Vd_ * t * t));
	
	return q;
}

double 
TrackElementCubicCurve::hdgRad(double t) const
{
	//double q = sqrt((Ub_ + 2.0 * Uc_ * t + 3.0 * Ud_ * t * t) * (Ub_ + 2.0 * Uc_ * t + 3.0 * Ud_ * t * t) + (Vb_ + 2.0 * Vc_ * t + 3.0 * Vd_ * t * t) * (Vb_ + 2.0 * Vc_ * t + 3.0 * Vd_ * t * t));
	//q = q / abs(f(Ua_, Ub_, Uc_, Ud_, t));
	//return atan2(q, 1);
		return atan2((3 * Vd_ * t * t + 2 * Vc_ * t + Vb_) , (3 * Ud_ * t * t + 2 * Uc_ * t + Ub_));
}

void
TrackElementCubicCurve::getControlPoints(double a, double b, double c, double d, double &p0, double &p1, double &p2, double &p3)
{

	p0 = a;
	p3 = a + b + c + d;
	p1 = b / 3 + p0;
	p2 = p3 - ((b + 2 * c + 3 * d) / 3);
}

void 
TrackElementCubicCurve::setParametersFromControlPoints(double &a, double &b, double &c, double &d, double p0, double p1, double p2, double p3)
{
	a = p0;
	b = 3 * (p1 - p0);
	c = 3 * (p0 + p2) - 6 * p1;
	d = -p0 + 3 * (p1 - p2) + p3;
}

/*! \brief Convenience function.
*  keep endPoint, edit startPoint, heading, length
*/
void
TrackElementCubicCurve::setLocalStartPoint(const QPointF &startPoint)
{
	// Local to internal (Parameters are given in internal coordinates) //
	//
	double p0, p1, p2, p3;

	getControlPoints(Ua_, Ub_, Uc_, Ud_, p0, p1, p2, p3);
	setParametersFromControlPoints(Ua_, Ub_, Uc_, Ud_, startPoint.x(), p1, p2, p3);

	getControlPoints(Va_, Vb_, Vc_, Vd_, p0, p1, p2, p3);
	setParametersFromControlPoints(Va_, Vb_, Vc_, Vd_, startPoint.y(), p1, p2, p3);

	setLocalTranslation(startPoint);
	setLength(getParametricCurveLength(0.0, 1.0));

	if (getParentRoad())
	{
		getParentRoad()->rebuildTrackComponentList();
	}

	addTrackElementCubicCurveChanges(TrackElementCubicCurve::CTCC_ParameterChange);
} 

/*! \brief Convenience function.
* keep startPoint, edit endPoint, heading, length
*/
void
TrackElementCubicCurve::setLocalEndPoint(const QPointF &endP)
{
	// Local to internal (Parameters are given in internal coordinates) //
	//

	double p0, p1, p2, p3;

	getControlPoints(Ua_, Ub_, Uc_, Ud_, p0, p1, p2, p3);
	setParametersFromControlPoints(Ua_, Ub_, Uc_, Ud_, p0, p1, p2, endP.x());

	getControlPoints(Va_, Vb_, Vc_, Vd_, p0, p1, p2, p3);
	setParametersFromControlPoints(Va_, Vb_, Vc_, Vd_, p0, p1, p2, endP.y());

	setLength(getParametricCurveLength(0.0, 1.0));

	if (getParentRoad())
	{
		getParentRoad()->rebuildTrackComponentList();
	}

	addTrackElementCubicCurveChanges(TrackElementCubicCurve::CTCC_ParameterChange);
} 

/*! \brief Convenience function.
* keep startPoint, endPoint, edit heading and length
*/
 void
TrackElementCubicCurve::setLocalStartHeading(double startHeading)
{

	while (startHeading <= -180.0)
	{
		startHeading += 360.0;
	}
	while (startHeading > 180.0)
	{
		startHeading -= 360.0;
	}

	// Local to internal (Parameters are given in internal coordinates) //
	//
//	double deltaHeading = startHeading - heading();
	double deltaHeading = getLocalHeading(getSStart()) - startHeading;

	QTransform trafo;
	trafo.rotate(deltaHeading);
	double p0, p1, p2, p3;
	getControlPoints(Ua_, Ub_, Uc_, Ud_, p0, p1, p2, p3);
	double q0, q1, q2, q3;
	getControlPoints(Va_, Vb_, Vc_, Vd_, q0, q1, q2, q3);
	QPointF controlPoint = trafo.inverted().map(QPointF(p1, q1));

	setParametersFromControlPoints(Ua_, Ub_, Uc_, Ud_, p0, controlPoint.x(), p2, p3);
	setParametersFromControlPoints(Va_, Vb_, Vc_, Vd_, q0, controlPoint.y(), q2, q3);

	setLength(getParametricCurveLength(0.0, 1));
	setLocalRotation(startHeading);

	if (getParentRoad())
	{
		getParentRoad()->rebuildTrackComponentList();
	}

	addTrackElementCubicCurveChanges(TrackElementCubicCurve::CTCC_ParameterChange);
} 

/*! \brief Convenience function.
*  Keep endPoint and startPoint. Edit heading and length
*/
void
TrackElementCubicCurve::setLocalEndHeading(double endHeading)
{
	while (endHeading <= -180.0)
	{
		endHeading += 360.0;
	}
	while (endHeading > 180.0)
	{
		endHeading -= 360.0;
	}

	// Local to internal (Parameters are given in internal coordinates) //
	//
//	endHeading = endHeading - getLocalHeading(getSStart());
	double deltaHeading = getLocalHeading(getSEnd()) - endHeading;
	QTransform trafo;
	trafo.rotate(deltaHeading);
	double p0, p1, p2, p3;
	getControlPoints(Ua_, Ub_, Uc_, Ud_, p0, p1, p2, p3);
	double q0, q1, q2, q3;
	getControlPoints(Va_, Vb_, Vc_, Vd_, q0, q1, q2, q3);
	QPointF controlPoint = trafo.inverted().map(QPointF(p2, q2));

	setParametersFromControlPoints(Ua_, Ub_, Uc_, Ud_, p0, p1, controlPoint.x(), p3);
	setParametersFromControlPoints(Va_, Vb_, Vc_, Vd_, q0, q1, controlPoint.y(), q3);

	setLength(getParametricCurveLength(0.0, 1));

	if (getParentRoad())
	{
		getParentRoad()->rebuildTrackComponentList();
	}

	addTrackElementCubicCurveChanges(TrackElementCubicCurve::CTCC_ParameterChange);
}


//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
TrackElementCubicCurve::notificationDone()
{
    trackElementCubicCurveChanges_ = 0x0;
    TrackElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
TrackElementCubicCurve::addTrackElementCubicCurveChanges(int changes)
{
    if (changes)
    {
        trackElementCubicCurveChanges_ |= changes;
        notifyObservers();
        addTrackComponentChanges(TrackComponent::CTC_ShapeChange);
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
TrackComponent *
TrackElementCubicCurve::getClone() const
{
    return getClonedCubicCurve();
}

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
TrackElementCubicCurve *
TrackElementCubicCurve::getClonedCubicCurve() const
{
    return new TrackElementCubicCurve(pos().x(), pos().y(), heading(), getSStart(), getLength(), polynomialU_, polynomialV_, pRange_);
}

//#################//
// VISITOR         //
//#################//

/** Accepts a visitor.
*
*/
void
TrackElementCubicCurve::accept(Visitor *visitor)
{
    visitor->visit(this);
}
