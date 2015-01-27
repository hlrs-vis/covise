/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   1/18/2010
**
**************************************************************************/

#include "trackcomposite.hpp"

// Qt //
//
#include <QVector2D>

// Utils //
//
#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

TrackComposite::TrackComposite()
    : TrackComponent(0.0, 0.0, 0.0)
    , cachedLength_(0.0)
{
    // TrackType //
    //
    setTrackType(TrackComponent::DTT_COMPOSITE);
}

TrackComposite::~TrackComposite()
{
    foreach (TrackComponent *child, trackComponents_)
    {
        //		delete child;
    }
}

//#################//
// TRACK COMPOSITE //
//#################//

bool
TrackComposite::addTrackComponent(TrackComponent *trackComponent)
{
    trackComponents_.insert(trackComponent->getSStart(), trackComponent);

    trackComponent->setParentComponent(this);

    updateLength();

    return true;
}

bool
TrackComposite::delTrackComponent(TrackComponent *trackComponent)
{
    updateLength();

    return trackComponents_.remove(trackComponent->getSStart());
}

TrackComponent *
TrackComposite::getChild(double s) const
{
    QMap<double, TrackComponent *>::const_iterator i = trackComponents_.upperBound(s);
    if (!(i == trackComponents_.begin()))
        --i;
    return i.value();
}

//#################//
// TRACK COMPONENT //
//#################//

/*! \brief Returns the start coordinate of the first child element.
*/
double
TrackComposite::getSStart() const
{
    return trackComponents_.constBegin().value()->getSStart();
}

/*! \brief Returns the end coordinate of the last child element.
*/
double
TrackComposite::getSEnd() const
{
    return (--trackComponents_.constEnd()).value()->getSEnd();
}

/*! \brief \todo caching.
*
*
*/
double
TrackComposite::updateLength()
{
    cachedLength_ = 0.0;
    foreach (TrackComponent *child, trackComponents_)
    {
        cachedLength_ += child->getLength();
    }
    return cachedLength_;
}

/*! \brief Returns the curvature at a specific road coordinate s.
*
*
*/
double
TrackComposite::getCurvature(double s)
{
    return getChild(s)->getCurvature(s);
}

/*! \brief Returns the point at a specific road coordinate s.
*/
QPointF
TrackComposite::getPoint(double s, double d)
{
    return getChild(s)->getLocalPoint(s, d);
}

/*! \brief Returns the heading [deg] at a specific road coordinate s.
*/
double
TrackComposite::getHeading(double s)
{
    return getChild(s)->getLocalHeading(s);
}

/*! \brief Returns the heading [rad] at a specific road coordinate s.
*/
double
TrackComposite::getHeadingRad(double s)
{
    return getChild(s)->getLocalHeadingRad(s); // child's local is relative/local to composite
}

/*! \brief Returns the tangent at a specific road coordinate s.
*/
QVector2D
TrackComposite::getTangent(double s)
{
    return getChild(s)->getLocalTangent(s);
}

/*! \brief Returns the normal at a specific road coordinate s.
*/
QVector2D
TrackComposite::getNormal(double s)
{
    return getChild(s)->getLocalNormal(s);
}

/*! \brief Returns the point at a specific road coordinate s.
*
* Relative to the composite's parent.
*/
QPointF
TrackComposite::getLocalPoint(double s, double d)
{
    return getLocalTransform().map(getPoint(s, d)); // composite's local is relative to composite's parent
}

/*! \brief Returns the heading [deg] at a specific road coordinate s.
*
* Relative to the composite's parent.
*/
double
TrackComposite::getLocalHeading(double s)
{
    return heading() + getHeading(s);
}

/*! \brief Returns the heading [rad] at a specific road coordinate s.
*
* Relative to the composite's parent.
*/
double
TrackComposite::getLocalHeadingRad(double s)
{
    return heading() * 2.0 * M_PI / 360.0 + getHeadingRad(s);
}

/*! \brief Returns the tangent at a specific road coordinate s.
*
* Relative to the composite's parent.
*/
QVector2D
TrackComposite::getLocalTangent(double s)
{
    QTransform trafo;
    trafo.rotate(getLocalHeading(getSStart()));
    return QVector2D(trafo.map(getTangent(s).toPointF()));
}

/*! \brief Returns the normal at a specific road coordinate s.
*
* Relative to the composite's parent.
*/
QVector2D
TrackComposite::getLocalNormal(double s)
{
    QTransform trafo;
    trafo.rotate(getLocalHeading(getSStart()));
    return QVector2D(trafo.map(getNormal(s).toPointF()));
}

void
TrackComposite::setSStart(double s)
{
    double deltaS = s - getSStart();

    // Fill new list //
    //
    QMap<double, TrackComponent *> newTrackComponents;
    double length = 0.0;
    QMap<double, TrackComponent *>::const_iterator i = trackComponents_.begin();
    while (i != trackComponents_.end())
    {
        TrackComponent *track = i.value();
        track->setSStart(track->getSStart() + deltaS);
        newTrackComponents.insert(track->getSStart(), track);
        length += track->getLength();
        ++i;
    }
    trackComponents_ = newTrackComponents;

    // Road Length //
    //
    cachedLength_ = length;
    addTrackComponentChanges(TrackComponent::CTC_SChange);
    addTrackComponentChanges(TrackComponent::CTC_LengthChange);
}

//################//
// VISITOR        //
//################//

/*! \brief * Accepts a visitor and passes it to all child
* nodes if autoTraverse is true.
*/
void
TrackComposite::accept(Visitor *visitor)
{
    visitor->visit(this);
}

/*! \brief Passes a visitor to all child nodes.
*/
void
TrackComposite::acceptForChildNodes(Visitor *visitor)
{
    foreach (TrackComponent *child, trackComponents_)
        child->accept(visitor);
}
