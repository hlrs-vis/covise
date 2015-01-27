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

#include "trackelementarc.hpp"

// Utils //
//
#include "src/util/odd.hpp"
#include "math.h"

// Qt //
//
#include <QPointF>
#include <QVector2D>

//#################//
// CONSTRUCTOR     //
//#################//

TrackElementArc::TrackElementArc(double x, double y, double angleDegrees, double s, double length, double curvature)
    : TrackElement(x, y, angleDegrees, s, length)
    , curvature_(curvature)
    , trackElementArcChanges_(0x0)
{
    setTrackType(TrackComponent::DTT_ARC);
}

TrackElementArc::~TrackElementArc()
{
}

//###############//
// ARC           //
//###############//

void
TrackElementArc::setCurvature(double curvature)
{
    curvature_ = curvature;
    addTrackElementArcChanges(TrackElementArc::CTA_CurvChange);
}

//#################//
// TRACK COMPONENT //
//#################//

/** Returns the point on track at road coordinate s.
	Natural coordinates relative to Geometry (see FEM).
	(Natuerliche Koordinaten)
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QPointF
TrackElementArc::getPoint(double s, double d)
{
    //	double radius = getRadius(s);
    double radius = 1.0 / curvature_; // shortcut

    //	double heading = getHeadingRad(s);
    double heading = (s - getSStart()) * curvature_; // shortcut

    return QPointF((radius - d) * sin(heading), radius - (radius - d) * cos(heading));
}

/** Returns the heading on the track at road coordinate s.
	Natural coordinates relative to Geometry (see FEM).
	(Natuerliche Koordinaten) in []
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementArc::getHeading(double s)
{
    //	return (s-getSStart())/getRadius(s) * 360.0/(2*M_PI); // const. //
    return (s - getSStart()) * curvature_ * 360.0 / (2 * M_PI); // shortcut
}

/** Returns the heading on the track at road coordinate s.
	Natural coordinates relative to Geometry (see FEM).
	(Natuerliche Koordinaten) in []
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementArc::getHeadingRad(double s)
{
    //	return (s-getSStart())/getRadius(s); // const. //
    return (s - getSStart()) * curvature_; // shortcut
}

/** Returns the local point on the track at road coordinate s.
	Relative to the parent composite.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QPointF
TrackElementArc::getLocalPoint(double s, double d)
{
    return getLocalTransform().map(getPoint(s, d));
}

/** Returns the local heading of the track at road coordinate s.
	Relative to the parent composite.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementArc::getLocalHeading(double s)
{
    //	return heading() + getHeading(s);
    return heading() + (s - getSStart()) * curvature_ * 360.0 / (2 * M_PI); // shortcut
}

/** Returns the local heading of the track at road coordinate s.
	Relative to the parent composite.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementArc::getLocalHeadingRad(double s)
{
    //	return heading() * 2*M_PI/360.0 + getHeadingRad(s);
    return heading() * 2 * M_PI / 360.0 + (s - getSStart()) * curvature_; // shortcut
}

/** Returns the curvature of the track at road coordinate s.
	Independent of coordinate system.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementArc::getCurvature(double /*s*/)
{
    return curvature_; // const.
}

/** Returns the radius of the track at road coordinate s.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementArc::getRadius(double /*s*/)
{
    return 1.0 / curvature_; // checked earlier so should be safe! // const.
}

/** Returns the center of the circle that is tangent to track
	at the given s coordinate.
	Natural coordinates relative to Geometry (see FEM).
	(Natuerliche Koordinaten)
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
//QPointF
//	TrackElementArc
//	::getRadiusCenter(double s)
//{
//	return QPointF(0, getRadius(s)); // const.
//}

/*!
* Returns local tangent to the track at road coordinate s.
* Relative to the parent composite.
* The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QVector2D
TrackElementArc::getLocalTangent(double s)
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
TrackElementArc::getLocalNormal(double s)
{
    double heading = getLocalHeadingRad(s);

    return QVector2D(sin(heading), -cos(heading)); // hypotenuse = 1.0

    // Do not know why the normal was flipped. Problems with driving direction of the road.
    //
    if (curvature_ < 0.0) // flip
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
TrackElementArc::notificationDone()
{
    trackElementArcChanges_ = 0x0;
    TrackElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
TrackElementArc::addTrackElementArcChanges(int changes)
{
    if (changes)
    {
        trackElementArcChanges_ |= changes;
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
TrackElementArc::getClone() const
{
    return getClonedArc();
}

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
TrackElementArc *
TrackElementArc::getClonedArc() const
{
    return new TrackElementArc(pos().x(), pos().y(), heading(), getSStart(), getLength(), curvature_);
}

//#################//
// VISITOR         //
//#################//

/** Accepts a visitor.
*
*/
void
TrackElementArc::accept(Visitor *visitor)
{
    visitor->visit(this);
}
