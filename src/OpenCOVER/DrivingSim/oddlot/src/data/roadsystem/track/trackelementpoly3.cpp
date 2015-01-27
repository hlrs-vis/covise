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

#include "trackelementpoly3.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

// Utils //
//
#include "src/util/odd.hpp"
#include "math.h"

// Qt //
//
#include <QPointF>
#include <QVector2D>

TrackElementPoly3::TrackElementPoly3(double x, double y, double angleDegrees, double s, double length, double a, double b, double c, double d)
    : TrackElement(x, y, angleDegrees, s, length)
    , Polynomial(a, b, c, d)
    , trackElementPoly3Changes_(0x0)
{
    setTrackType(TrackComponent::DTT_POLY3);
}

TrackElementPoly3::TrackElementPoly3(double x, double y, double angleDegrees, double s, double length, const Polynomial &polynomial)
    : TrackElement(x, y, angleDegrees, s, length)
    , Polynomial(polynomial)
    , trackElementPoly3Changes_(0x0)
{
    setTrackType(TrackComponent::DTT_POLY3);
}

TrackElementPoly3::~TrackElementPoly3()
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
TrackElementPoly3::getPoint(double s, double d)
{
    double t = getT(s);

    //	qDebug() << "p3: " << getHeadingRad(s) << " " << -sin(getHeadingRad(s)) << " " << cos(getHeadingRad(s)) << " " << f(t);

    return QPointF(t, f(t)) + QVector2D(-sin(getHeadingRad(s)), cos(getHeadingRad(s))).toPointF() * d;
}

/** Returns heading on the track at road coordinate s.
	Natural coordinates relative to Geometry (see FEM).
	(Natuerliche Koordinaten)
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementPoly3::getHeading(double s)
{
    return hdg(getT(s));
}

/** Returns heading on the track at road coordinate s.
	Natural coordinates relative to Geometry (see FEM).
	(Natuerliche Koordinaten)
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementPoly3::getHeadingRad(double s)
{
    return hdgRad(getT(s));
}

/** Returns local point on the track at road coordinate s.
	Relative to the parent composite.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QPointF
TrackElementPoly3::getLocalPoint(double s, double d)
{
    return getLocalTransform().map(getPoint(s, d));
}

/** Returns local heading of the track at road coordinate s.
	Relative to the parent composite.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementPoly3::getLocalHeading(double s)
{
    return heading() + getHeading(s);
}

/** Returns local heading of the track at road coordinate s.
	Relative to the parent composite.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementPoly3::getLocalHeadingRad(double s)
{
    return heading() * 2 * M_PI / 360.0 + getHeadingRad(s);
}

/*!
* Returns local tangent to the track at road coordinate s.
* Relative to the parent composite.
* The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QVector2D
TrackElementPoly3::getLocalTangent(double s)
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
TrackElementPoly3::getLocalNormal(double s)
{
    double hdg = getLocalHeadingRad(s);
    return QVector2D(sin(hdg), -cos(hdg)); // hypotenuse = 1.0
}

/** Returns curvature of the track at road coordinate s.
	Independent of coordinate system.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementPoly3::getCurvature(double s)
{
    double t = getT(s);
    return ddf(t) / pow((1 + df(t) * df(t)), 1.5);
}

double
TrackElementPoly3::getT(double s)
{
    s = s - getSStart();

    double t = s * 0.5; // first approximation

    for (int i = 0; i < 20; ++i)
    {

        // Flo's code //
        //
        // Cut taylor series approximation (1-degree) of arc length integral, solved with Newton-Raphson for t with respect to s
        //		double b = getB();
        //		double c = getC();
        //		double d = getD();
        //		double f = t*sqrt(pow(((3*d*pow(t,2))/4+c*t+b),2)+1)-s;
        //		double df = sqrt(pow(((3*d*pow(t,2))/4+c*t+b),2)+1)+(t*((3*d*t)/2+c)*((3*d*pow(t,2))/4+c*t+b))/sqrt(pow(((3*d*pow(t,2))/4+c*t+b),2)+1);

        // New code with integration //
        //
        double f = getCurveLength(0.0, t) - s;
        double df = sqrt(1.0 + (b_ + 2.0 * c_ * t + 3.0 * d_ * t * t) * (b_ + 2.0 * c_ * t + 3.0 * d_ * t * t));
        t -= f / df;
    }
    return t;
}

/*! \brief Convenience function.
*
*/
void
TrackElementPoly3::setLocalStartPoint(const QPointF &startPoint)
{
    // Local to internal (Parameters are given in internal coordinates) //
    //
    QPointF deltaPos(getLocalTransform().inverted().map(startPoint) /* - getPoint(getSStart())*/); // getPoint(s_) == 0 by definition
    QPointF endPoint = getPoint(getSEnd()) - deltaPos;

    double l = endPoint.x();
    double h0 = getA();
    double dh0 = getB();
    double h1 = endPoint.y();
    double dh1 = df(getT(getSEnd()));

    double d = (dh1 + dh0 - 2.0 * h1 / l + 2.0 * h0 / l) / (l * l);
    double c = (h1 - d * l * l * l - dh0 * l - h0) / (l * l);
    d_ = d;
    c_ = c;

    setLocalTranslation(startPoint);
    setLength(getCurveLength(0.0, l));

    if (getParentRoad())
    {
        getParentRoad()->rebuildTrackComponentList();
    }

    addTrackElementPoly3Changes(TrackElementPoly3::CTP_ParameterChange);
}

/*! \brief Convenience function.
*
*/
void
TrackElementPoly3::setLocalEndPoint(const QPointF &endP)
{
    // Local to internal (Parameters are given in internal coordinates) //
    //
    QPointF endPoint = getLocalTransform().inverted().map(endP);

    double l = endPoint.x();
    double h0 = getA();
    double dh0 = getB();
    double h1 = endPoint.y();
    double dh1 = df(getT(getSEnd()));

    double d = (dh1 + dh0 - 2.0 * h1 / l + 2.0 * h0 / l) / (l * l);
    double c = (h1 - d * l * l * l - dh0 * l - h0) / (l * l);
    d_ = d;
    c_ = c;

    setLength(getCurveLength(0.0, l));

    if (getParentRoad())
    {
        getParentRoad()->rebuildTrackComponentList();
    }

    addTrackElementPoly3Changes(TrackElementPoly3::CTP_ParameterChange);
}

/*! \brief Convenience function.
*
*/
void
TrackElementPoly3::setLocalStartHeading(double startHeading)
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
    double deltaHeading = startHeading - heading();

    double endHeading = getHeading(getSEnd()) - deltaHeading;

    QTransform trafo;
    trafo.rotate(deltaHeading);
    QPointF endPoint = trafo.inverted().map(getPoint(getSEnd()));

    double l = endPoint.x();
    double h0 = getA();
    double dh0 = getB();
    double h1 = endPoint.y();
    double dh1 = tan(endHeading * 2.0 * M_PI / 360.0);

    double d = (dh1 + dh0 - 2.0 * h1 / l + 2.0 * h0 / l) / (l * l);
    double c = (h1 - d * l * l * l - dh0 * l - h0) / (l * l);
    d_ = d;
    c_ = c;

    setLength(getCurveLength(0.0, l));
    setLocalRotation(startHeading);

    if (getParentRoad())
    {
        getParentRoad()->rebuildTrackComponentList();
    }

    addTrackElementPoly3Changes(TrackElementPoly3::CTP_ParameterChange);
}

/*! \brief Convenience function.
*
*/
void
TrackElementPoly3::setLocalEndHeading(double endHeading)
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
    endHeading = endHeading - getLocalHeading(getSStart());

    double l = getT(getSEnd());
    double h0 = getA();
    double dh0 = getB();
    double h1 = f(l);

    double dh1 = tan(endHeading * 2.0 * M_PI / 360.0);

    double d = (dh1 + dh0 - 2.0 * h1 / l + 2.0 * h0 / l) / (l * l);
    double c = (h1 - d * l * l * l - dh0 * l - h0) / (l * l);
    d_ = d;
    c_ = c;

    setLength(getCurveLength(0.0, l));

    if (getParentRoad())
    {
        getParentRoad()->rebuildTrackComponentList();
    }

    addTrackElementPoly3Changes(TrackElementPoly3::CTP_ParameterChange);
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
TrackElementPoly3::notificationDone()
{
    trackElementPoly3Changes_ = 0x0;
    TrackElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
TrackElementPoly3::addTrackElementPoly3Changes(int changes)
{
    if (changes)
    {
        trackElementPoly3Changes_ |= changes;
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
TrackElementPoly3::getClone() const
{
    return getClonedPoly3();
}

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
TrackElementPoly3 *
TrackElementPoly3::getClonedPoly3() const
{
    return new TrackElementPoly3(pos().x(), pos().y(), heading(), getSStart(), getLength(), getA(), getB(), getC(), getD());
}

//#################//
// VISITOR         //
//#################//

/** Accepts a visitor.
*
*/
void
TrackElementPoly3::accept(Visitor *visitor)
{
    visitor->visit(this);
}
