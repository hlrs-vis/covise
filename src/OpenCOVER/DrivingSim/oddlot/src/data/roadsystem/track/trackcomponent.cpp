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

#include "trackcomponent.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

#include <QVector2D>
#include <QTransform>
#include <math.h>

//################//
// CONSTRUCTOR    //
//################//

TrackComponent::TrackComponent(double x, double y, double angleDegrees)
    : DataElement()
    , trackType_(TrackComponent::DTT_NONE)
    , parentRoad_(NULL)
    , parentComponent_(NULL)
    , trackComponentChanges_(0x0)
{
    pos_.setX(x);
    pos_.setY(y);
    setLocalRotation(angleDegrees);
}

TrackComponent::~TrackComponent()
{
}

void
TrackComponent::setTrackType(TrackComponent::DTrackType type)
{
    trackType_ = type;
}

//#################//
// PARENTS         //
//#################//

TrackComponent *
TrackComponent::getParentComponent() const
{
    return parentComponent_;
}

RSystemElementRoad *
TrackComponent::getParentRoad() const
{
    if (parentComponent_)
    {
        return parentComponent_->getParentRoad();
    }
    else
    {
        return parentRoad_;
    }
}

void
TrackComponent::setParentRoad(RSystemElementRoad *parentRoad)
{
    parentRoad_ = parentRoad;
    parentComponent_ = NULL;
    setParentElement(parentRoad);
    addTrackComponentChanges(TrackComponent::CTC_ParentChanged);
}

void
TrackComponent::setParentComponent(TrackComponent *parentComponent)
{
    parentComponent_ = parentComponent;
    parentRoad_ = NULL;
    setParentElement(parentComponent);
    addTrackComponentChanges(TrackComponent::CTC_ParentChanged);
}

//#################//
// TRACK COMPONENT //
//#################//

/** Returns the local transformation matrix.
* This matrix is relative to the parent node.
*/
const QTransform
TrackComponent::getLocalTransform()
{
    QTransform trafo;
    trafo.translate(pos_.x(), pos_.y());
    trafo.rotate(heading_);
    return trafo;
}

/** Calculates and returns the global transformation matrix.
* This matrix is relative to the root node.
* Since the hierarchy is relatively flat in most cases
* (shouldn't be more than 1-3) this is not cached!
*/
const QTransform
TrackComponent::getGlobalTransform()
{
    QTransform trafo;
    trafo.translate(pos_.x(), pos_.y());
    trafo.rotate(heading_);

    if (parentComponent_)
        trafo *= parentComponent_->getGlobalTransform();

    return trafo;
}

/** Calculates and returns the global point on the track
	at road coordinate s. Relative to the root node.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QPointF
TrackComponent::getGlobalPoint(double s, double d)
{
    QPointF point = getLocalPoint(s, d);
    if (parentComponent_)
        point = parentComponent_->getGlobalTransform().map(point);
    return point;
}

/** Calculates and returns the global heading of the track at road coordinate s.
	Relative to the root node.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackComponent::getGlobalHeading(double s)
{
    double heading = getLocalHeading(s);
    if (parentComponent_)
        heading += parentComponent_->getGlobalHeading(parentComponent_->getSStart());
    while (heading >= 360.0)
    {
        heading = heading - 360.0;
    }
    return heading;
}

/** Calculates and returns the global heading of the track at road coordinate s.
	Relative to the root node.
	The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
double
TrackComponent::getGlobalHeadingRad(double s)
{
    double heading = getLocalHeadingRad(s);
    if (parentComponent_)
    {
        heading += parentComponent_->getGlobalHeadingRad(parentComponent_->getSStart());
    }
    while (heading >= 2.0 * M_PI)
    {
        heading = heading - 2.0 * M_PI;
    }
    return heading;
}

/*!
* Calculates and returns the global heading as a normed vector at road coordinate s.
* The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QVector2D
TrackComponent::getGlobalTangent(double s)
{
    QVector2D tangent = getLocalTangent(s);
    if (parentComponent_)
    {
        QTransform trafo;
        trafo.rotate(parentComponent_->getGlobalHeading(parentComponent_->getSStart()));
        tangent = QVector2D(trafo.map(tangent.toPointF()));
    }
    return tangent;
}

/*!
* Calculates and returns the global normal to the track as a normed vector at road coordinate s.
* The s-Coordinate is NOT clamped to [s_, s_ + length_].
*/
QVector2D
TrackComponent::getGlobalNormal(double s)
{
    QVector2D normal = getLocalNormal(s);
    if (parentComponent_)
    {
        QTransform trafo;
        trafo.rotate(parentComponent_->getGlobalHeading(parentComponent_->getSStart()));
        normal = QVector2D(trafo.map(normal.toPointF()));
    }
    return normal;
}

void
TrackComponent::setGlobalTranslation(const QPointF &pos)
{
    QPointF startPoint(pos);
    if (parentComponent_)
    {
        startPoint = parentComponent_->getGlobalTransform().inverted().map(startPoint);
    }
    setLocalTranslation(startPoint);
}

void
TrackComponent::setGlobalRotation(double angleDegrees)
{
    if (parentComponent_)
    {
        angleDegrees = angleDegrees - parentComponent_->getGlobalHeading(parentComponent_->getSStart());
    }
    setLocalRotation(angleDegrees);
}

/** Set the matrix of the component.
	*/
void
TrackComponent::setLocalTransform(double x, double y, double angleDegrees)
{
    pos_.setX(x);
    pos_.setY(y);
    setLocalRotation(angleDegrees);
    addTrackComponentChanges(TrackComponent::CTC_TransformChange);
}

/** Set the matrix of the component.
	*/
void
TrackComponent::setLocalTransform(const QPointF &pos, double angleDegrees)
{
    pos_ = pos;
    setLocalRotation(angleDegrees);
    addTrackComponentChanges(TrackComponent::CTC_TransformChange);
}

/** Set the length of the track.
	*/
void
TrackComponent::setLength(double len)
{
}
    void setLength(float length);
/** Set the matrix of the component.
	*/
//void
//	TrackComponent
//	::setLocalTransform(const QTransform & matrix)
//{

//}

/** Sets the x and y coordinate of the matrix.
	*/
void
TrackComponent::setLocalTranslation(double x, double y)
{
    pos_.setX(x);
    pos_.setY(y);
    addTrackComponentChanges(TrackComponent::CTC_TransformChange);
}

/** Sets the x and y coordinate of the matrix with a QPointF.
	*/
void
TrackComponent::setLocalTranslation(const QPointF &pos)
{
    pos_ = pos;
    addTrackComponentChanges(TrackComponent::CTC_TransformChange);
}

/** Translates the matrix by x and y.
	*/
void
TrackComponent::translateLocal(double x, double y)
{
    pos_ += QPointF(x, y);
    addTrackComponentChanges(TrackComponent::CTC_TransformChange);
}

/** Translates the matrix by QPointF.
	*/
void
TrackComponent::translateLocal(const QPointF &pos)
{
    pos_ += pos;
    addTrackComponentChanges(TrackComponent::CTC_TransformChange);
}

/** Sets the rotation.
*/
void
TrackComponent::setLocalRotation(double angleDegrees)
{
    while (angleDegrees < 0.0)
    {
        angleDegrees = angleDegrees + 360.0;
    }
    while (angleDegrees >= 360.0)
    {
        angleDegrees = angleDegrees - 360.0;
    }
    heading_ = angleDegrees;
    addTrackComponentChanges(TrackComponent::CTC_TransformChange);
}

/** Rotates the matrix.
*/
void
TrackComponent::rotateLocal(double angleDegrees)
{
    setLocalRotation(heading_ + angleDegrees);
}

/*! \brief Convenience function.
*
* This function may set position, heading and length. Transformations
* that violate the constraints are not caught! If the component is a
* composite, it is likely that the child nodes are also modified.
*/
void
TrackComponent::setGlobalStartPoint(const QPointF &startPoint)
{
    QPointF point(startPoint);
    if (parentComponent_)
    {
        point = parentComponent_->getGlobalTransform().inverted().map(point);
    }
    setLocalStartPoint(point);
}

/*! \brief Convenience function.
*
* This function may set position, heading and length. Transformations
* that violate the constraints are not caught! If the component is a
* composite, it is likely that the child nodes are also modified.
*/
void
TrackComponent::setGlobalEndPoint(const QPointF &endPoint)
{
    QPointF point(endPoint);
    if (parentComponent_)
    {
        point = parentComponent_->getGlobalTransform().inverted().map(point);
    }
    setLocalEndPoint(point);
}

/*! \brief Convenience function.
*
* This function may set position, heading and length. Transformations
* that violate the constraints are not caught! If the component is a
* composite, it is likely that the child nodes are also modified.
*/
void
TrackComponent::setGlobalStartHeading(double startHeading)
{
    if (parentComponent_)
    {
        startHeading = startHeading - parentComponent_->getGlobalHeading(getSStart());
    }
    setLocalStartHeading(startHeading);
}

/*! \brief Convenience function.
*
* This function may set position, heading and length. Transformations
* that violate the constraints are not caught! If the component is a
* composite, it is likely that the child nodes are also modified.
*/
void
TrackComponent::setGlobalEndHeading(double endHeading)
{
    if (parentComponent_)
    {
        endHeading = endHeading - parentComponent_->getGlobalHeading(getSStart()); // It's really sStart! (=>heading_)
    }
    setLocalEndHeading(endHeading);
}

/*! \brief Convenience function.
*
* This function may set position, heading and length. Transformations
* that violate the constraints are not caught! If the component is a
* composite, it is likely that the child nodes are also modified.
*/
void
TrackComponent::setGlobalPointAndHeading(const QPointF &newPoint, double newHeading, bool isStart)
{
    QPointF point(newPoint);
    if (parentComponent_)
    {
        point = parentComponent_->getGlobalTransform().inverted().map(point);
        newHeading = newHeading - parentComponent_->getGlobalHeading(getSStart()); // It's really sStart! (=>heading_)
    }

    setLocalPointAndHeading(point, newHeading, isStart);
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
TrackComponent::notificationDone()
{
    trackComponentChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
TrackComponent::addTrackComponentChanges(int changes)
{
    if (changes)
    {
        trackComponentChanges_ |= changes;

        if (((changes & TrackComponent::CTC_ShapeChange) || (changes & TrackComponent::CTC_LengthChange) || (changes & TrackComponent::CTC_TransformChange)) && (parentRoad_))
        {
            parentRoad_->addRoadChanges(RSystemElementRoad::CRD_ShapeChange);
        }

        notifyObservers();
    }
}
