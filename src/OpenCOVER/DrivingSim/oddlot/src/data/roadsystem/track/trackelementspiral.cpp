/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   08.02.2010
**
**************************************************************************/

#include "trackelementspiral.hpp"

// Utils //
//
#include "src/util/odd.hpp"
#include "math.h"

// Qt //
//
#include <QPointF>
#include <QVector2D>

#include <QDebug>

//#################//
// CONSTRUCTOR     //
//#################//

TrackElementSpiral::TrackElementSpiral(double x, double y, double angleDegrees, double s, double length, double curvStart, double curvEnd)
    : TrackElement(x, y, angleDegrees, s, length)
    , curvStart_(curvStart)
    , curvEnd_(curvEnd)
    , trackElementSpiralChanges_(0x0)
{
    init();
}

TrackElementSpiral::~TrackElementSpiral()
{
}

void
TrackElementSpiral::init()
{
    // Type //
    //
    setTrackType(TrackComponent::DTT_SPIRAL);

    // Clothoid Parameters Ax, Ay //
    //
    double curveDelta = curvEnd_ - curvStart_;
    if (curveDelta > 0)
    {
        ax_ = ay_ = sqrt(getLength() / curveDelta);
    }
    else if (curveDelta < 0)
    {
        ax_ = sqrt(-getLength() / curveDelta);
        ay_ = -ax_;
    }
    else
    {
        ax_ = ay_ = 0.0;
    }

    //	qDebug() << "a: " << ax_ << " " << ay_;

    // Cache Values //
    //
    lStart_ = curvStart_ * ax_ * ay_;

    //	qDebug() << "lStart: " << lStart_;

    headingStart_ = 0.0;
    headingStart_ = getHeading(getSStart());

    //	qDebug() << "heading: " << headingStart_;

    // this transformation moves the clothoid so the start point is in the origin
    clothoidTrafo_.reset();
    clothoidTrafo_.rotate(-headingStart_);
    clothoidTrafo_.translate(-clothoidApproximation(lStart_).x(), -clothoidApproximation(lStart_).y());
}

//###############//
// SPIRAL        //
//###############//

void
TrackElementSpiral::setCurvStartAndLength(double curvStart, double length)
{
    setLength(length);
    curvStart_ = curvStart;

    init();

    addTrackElementSpiralChanges(TrackElementSpiral::CTS_CurvStartChange);
}

void
TrackElementSpiral::setCurvEndAndLength(double curvEnd, double length)
{
    setLength(length);
    curvEnd_ = curvEnd;

    init();

    addTrackElementSpiralChanges(TrackElementSpiral::CTS_CurvEndChange);
}

//#########################//
// Track Element Functions //
//#########################//

/** Returns the point on the track at road coordinate s.
	Natural coordinates relative to Geometry.
	(Natuerliche Koordinaten)
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QPointF
TrackElementSpiral::getPoint(double s, double d)
{
    double l = lStart_ + (s - getSStart());

    if (d == 0.0)
        return clothoidTrafo_.map(clothoidApproximation(l));
    else
    {
        //		double n = getHeadingRad(s)/* * 2.0*M_PI/360.0*/;
        //		return trafo.map(clothoidApproximation(l)+ QPointF(-sin(n)*d, cos(n)*d));
        return clothoidTrafo_.map(clothoidApproximation(l)) + QVector2D(-sin(getHeadingRad(s)), cos(getHeadingRad(s))).toPointF() * d;
        //		return trafo.map(clothoidApproximation(l))+ getNormal(s).toPointF()*d;
    }
}

QPointF
TrackElementSpiral::clothoidApproximation(double l)
{
    double x = l / ax_;

    // xpow: x, x^3, x^5,...
    double xpow[9];
    double x2 = x * x;
    xpow[0] = x;
    for (int i = 1; i < 9; ++i)
    {
        xpow[i] = xpow[i - 1] * x2;
    }

    return QPointF(
        ax_ * (f8 * xpow[8] - f6 * xpow[6] + f4 * xpow[4] - f2 * xpow[2] + f0 * xpow[0]),
        ay_ * (-f7 * xpow[7] + f5 * xpow[5] - f3 * xpow[3] + f1 * xpow[1]));

    // DELETE SOMETIME IN FUTURE:
    //	return QPointF(
    //		ax_ * (l - pow(l,5.0)/40.0 + pow(l,9.0)/3456.0 - pow(l,13.0)/599040.0 + pow(l,17.0)/175472640.0),
    //		ay_ * (pow(l,3.0)/6.0 - pow(l,7.0)/336.0 + pow(l,11.0)/42240.0 - pow(l,15.0)/9676800.0)
    //	);
}

double
TrackElementSpiral::x(double x)
{
    // xpow: x, x^3, x^5,...
    double xpow[9];
    double x2 = x * x;
    xpow[0] = x;
    for (int i = 1; i < 9; ++i)
    {
        xpow[i] = xpow[i - 1] * x2;
    }

    return f8 * xpow[8] - f6 * xpow[6] + f4 * xpow[4] - f2 * xpow[2] + f0 * xpow[0];
}

double
TrackElementSpiral::y(double x)
{
    // xpow: x, x^3, x^5,...
    double xpow[9];
    double x2 = x * x;
    xpow[0] = x;
    for (int i = 1; i < 9; ++i)
    {
        xpow[i] = xpow[i - 1] * x2;
    }

    return -f7 * xpow[7] + f5 * xpow[5] - f3 * xpow[3] + f1 * xpow[1];
}

/** Returns the heading of the track at road coordinate s.
	Natural coordinates relative to Geometry.
	(Natuerliche Koordinaten)
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementSpiral::getHeading(double s)
{
    double l = lStart_ + (s - getSStart());
    //	return l*l / (2.0*ax_*ay_) * 360.0/(2.0*M_PI);
    return l * l / (2.0 * ax_ * ay_) * 360.0 / (2.0 * M_PI) - headingStart_;
}

/** Returns the heading of the track at road coordinate s.
	Natural coordinates relative to Geometry.
	(Natuerliche Koordinaten)
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementSpiral::getHeadingRad(double s)
{
    double l = lStart_ + (s - getSStart());
    return l * l / (2.0 * ax_ * ay_) - headingStart_ * 2.0 * M_PI / 360.0;
    //	return l*getCurvature(s)/2.0;
}

/** Returns the local point on the track at road coordinate s.
	Relative to the parent composite.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QPointF
TrackElementSpiral::getLocalPoint(double s, double d)
{
    return getLocalTransform().map(getPoint(s, d));
}

/** Returns the local heading of the track at road coordinate s.
	Relative to the parent composite.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementSpiral::getLocalHeading(double s)
{
    return heading() + getHeading(s);
}

/** Returns the local heading of the track at road coordinate s.
	Relative to the parent composite.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementSpiral::getLocalHeadingRad(double s)
{
    return heading() * 2.0 * M_PI / 360.0 + getHeadingRad(s);
}

/** Returns the curvature of the track at road coordinate s.
	Independent of coordinate system.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementSpiral::getCurvature(double s)
{
    double l = lStart_ + (s - getSStart());
    return l / (ax_ * ay_);
}

/** Returns the radius of the track at road coordinate s.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementSpiral::getRadius(double s)
{
    return 1.0 / getCurvature(s); // can't be safe!
    // TODO infinity
}

/** Returns the center of the circle that is tangent to track
	at the given s coordinate.
	Natural coordinates relative to Geometry.
	(Natuerliche Koordinaten)
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QPointF
TrackElementSpiral::getRadiusCenter(double s)
{
    return QPointF(0, getRadius(s)); // TODO
}

/*!
* Returns tangent to the track at road coordinate s.
*
* The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QVector2D
TrackElementSpiral::getTangent(double s)
{
    double heading = getHeadingRad(s);
    return QVector2D(cos(heading), sin(heading)); // hypotenuse = 1.0
}

/*!
* Returns normal to the track at road coordinate s.
*
* The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QVector2D
TrackElementSpiral::getNormal(double s)
{
    double heading = getHeadingRad(s);
    if (getCurvature(s) < 0.0) // flip
    {
        return QVector2D(-sin(heading), cos(heading)); // hypotenuse = 1.0
    }
    else
    {
        return QVector2D(sin(heading), -cos(heading)); // hypotenuse = 1.0
    }
}

/*!
* Returns local tangent to the track at road coordinate s.
* Relative to the parent composite.
* The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QVector2D
TrackElementSpiral::getLocalTangent(double s)
{
    double heading = getLocalHeadingRad(s);
    return QVector2D(cos(heading), sin(heading)); // hypotenuse = 1.0
}

/*!
* Returns local normal to the track at road coordinate s.
* Relative to the parent composite.
* The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QVector2D
TrackElementSpiral::getLocalNormal(double s)
{
    double heading = getLocalHeadingRad(s);
    return QVector2D(sin(heading), -cos(heading)); // hypotenuse = 1.0

    // Do not know why the normal was flipped. Problems with driving direction of the road.
    //
    if (getCurvature(s) < 0.0) // flip
    {
        return QVector2D(-sin(heading), cos(heading)); // hypotenuse = 1.0
    }
    else
    {
        return QVector2D(sin(heading), -cos(heading)); // hypotenuse = 1.0
    }
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
TrackElementSpiral::notificationDone()
{
    trackElementSpiralChanges_ = 0x0;
    TrackElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
TrackElementSpiral::addTrackElementSpiralChanges(int changes)
{
    if (changes)
    {
        trackElementSpiralChanges_ |= changes;
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
TrackElementSpiral::getClone() const
{
    return getClonedSpiral();
}

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
TrackElementSpiral *
TrackElementSpiral::getClonedSpiral() const
{
    return new TrackElementSpiral(pos().x(), pos().y(), heading(), getSStart(), getLength(), curvStart_, curvEnd_);
}

//#################//
// VISITOR         //
//#################//

/** Accepts a visitor.
*
*/
void
TrackElementSpiral::accept(Visitor *visitor)
{
    visitor->visit(this);
}

//###################//
// Series Expansion  //
//###################//

double TrackElementSpiral::f0 = 1.0;
double TrackElementSpiral::f1 = 1.0 / 6.0;
double TrackElementSpiral::f2 = 1.0 / 40.0;
double TrackElementSpiral::f3 = 1.0 / 336.0;
double TrackElementSpiral::f4 = 1.0 / 3456.0;
double TrackElementSpiral::f5 = 1.0 / 42240.0;
double TrackElementSpiral::f6 = 1.0 / 599040.0;
double TrackElementSpiral::f7 = 1.0 / 9676800.0;
double TrackElementSpiral::f8 = 1.0 / 175472640.0;
