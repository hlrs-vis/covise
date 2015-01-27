/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   1/22/2010
**
**************************************************************************/

#include "trackelementline.hpp"

// Utils //
//
#include "src/util/odd.hpp"
#include "math.h"

// Qt //
//
#include <QPointF>
#include <QVector2D>

TrackElementLine::TrackElementLine(double x, double y, double angleDegrees, double s, double length)
    : TrackElement(x, y, angleDegrees, s, length)
    , trackElementLineChanges_(0x0)
{
    setTrackType(TrackComponent::DTT_LINE);
}

TrackElementLine::~TrackElementLine()
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
TrackElementLine::getPoint(double s, double d)
{
    return QPointF(s - getSStart(), d);
}

/** Returns heading on the track at road coordinate s.
	Natural coordinates relative to Geometry (see FEM).
	(Natuerliche Koordinaten)
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementLine::getHeading(double /*s*/)
{
    return 0.0; // const. //
}

/** Returns heading on the track at road coordinate s.
	Natural coordinates relative to Geometry (see FEM).
	(Natuerliche Koordinaten)
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementLine::getHeadingRad(double /*s*/)
{
    return 0.0; // const. //
}

/** Returns local point on the track at road coordinate s.
	Relative to the parent composite.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QPointF
TrackElementLine::getLocalPoint(double s, double d)
{
    return getLocalTransform().map(getPoint(s, d));
}

/** Returns local heading of the track at road coordinate s.
	Relative to the parent composite.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementLine::getLocalHeading(double /*s*/)
{
    return heading(); // const. //
}

/** Returns local heading of the track at road coordinate s.
	Relative to the parent composite.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementLine::getLocalHeadingRad(double /*s*/)
{
    return heading() * 2 * M_PI / 360.0; // const. //
}

/*!
* Returns local tangent to the track at road coordinate s.
* Relative to the parent composite.
* The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QVector2D
TrackElementLine::getLocalTangent(double /*s*/)
{
    //	double heading = getLocalHeadingRad(s);
    double hdg = heading() * 2.0 * M_PI / 360.0; // shortcut
    return QVector2D(cos(hdg), sin(hdg)); // hypotenuse = 1.0
}

/*!
* Returns local normal to the track at road coordinate s.
* Relative to the parent composite.
* The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QVector2D
TrackElementLine::getLocalNormal(double /*s*/)
{
    //	double heading = getLocalHeadingRad(s);
    double hdg = heading() * 2.0 * M_PI / 360.0; // shortcut
    return QVector2D(sin(hdg), -cos(hdg)); // hypotenuse = 1.0
}

/** Returns curvature of the track at road coordinate s.
	Independent of coordinate system.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackElementLine::getCurvature(double /*s*/)
{
    return 0.0;
}

/*! \brief Convenience function.
*
* Keep endPoint. Edit heading, startPoint and length.
*/
void
TrackElementLine::setLocalStartPoint(const QPointF &startPoint)
{
    QPointF endPoint = getLocalPoint(getSEnd());
    QVector2D line = QVector2D(endPoint - startPoint);

    setLocalTranslation(startPoint);
    setLocalRotation(atan2(line.y(), line.x()) * 360.0 / (2.0 * M_PI));
    setLength(line.length());
}

/*! \brief Convenience function.
*
* Keep startPoint. Edit heading, endPoint and length.
*/
void
TrackElementLine::setLocalEndPoint(const QPointF &endPoint)
{
    QVector2D line = QVector2D(endPoint - getLocalPoint(getSStart()));

    setLocalRotation(atan2(line.y(), line.x()) * 360.0 / (2.0 * M_PI));
    setLength(line.length());
}

/*! \brief Convenience function.
*
* Keep startPoint and length. Edit heading and endPoint.
*/
void
TrackElementLine::setLocalStartHeading(double startHeadingDegrees)
{
    qDebug("TrackElementLine setLocalStartHeading not yet tested");

    setLocalRotation(startHeadingDegrees);
}

/*! \brief Convenience function.
*
* Keep endPoint and length. Edit heading and startPoint.
*/
void
TrackElementLine::setLocalEndHeading(double endHeadingDegrees)
{
    qDebug("TrackElementLine setLocalEndHeading not yet tested");

    QPointF endPoint = getLocalPoint(getSEnd());
    double length = QVector2D(endPoint - getLocalPoint(getSStart())).length();
    QPointF line = length * QPointF(cos(endHeadingDegrees * 2.0 * M_PI / 360.0), sin(endHeadingDegrees * 2.0 * M_PI / 360.0));

    setLocalTranslation(endPoint - line);
    setLocalRotation(endHeadingDegrees);
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
TrackElementLine::notificationDone()
{
    trackElementLineChanges_ = 0x0;
    TrackElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
TrackElementLine::addTrackElementLineChanges(int changes)
{
    if (changes)
    {
        trackElementLineChanges_ |= changes;
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
TrackElementLine::getClone() const
{
    return getClonedLine();
}

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
TrackElementLine *
TrackElementLine::getClonedLine() const
{
    return new TrackElementLine(pos().x(), pos().y(), heading(), getSStart(), getLength());
}

//#################//
// VISITOR         //
//#################//

/** Accepts a visitor.
*
*/
void
TrackElementLine::accept(Visitor *visitor)
{
    visitor->visit(this);
}
