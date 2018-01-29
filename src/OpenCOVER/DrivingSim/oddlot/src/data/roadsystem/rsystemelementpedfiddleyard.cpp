/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.02.2010
**
**************************************************************************/

#include "rsystemelementpedfiddleyard.hpp"

//#############################//
//                             //
// RSystemElementPedFiddleyard //
//                             //
//#############################//

RSystemElementPedFiddleyard::RSystemElementPedFiddleyard(const odrID &id, const QString &name, const odrID &roadId)
    : RSystemElement(name, id, RSystemElement::DRE_PedFiddleyard)
    , roadId_(roadId)
{
}

RSystemElementPedFiddleyard::~RSystemElementPedFiddleyard()
{
    foreach (PedFiddleyardSink *child, sinks_)
        delete child;

    foreach (PedFiddleyardSource *child, sources_)
        delete child;
}

/** Adds a source to this fiddleyard.
*/
void
RSystemElementPedFiddleyard::addSource(PedFiddleyardSource *source)
{
    sources_.insert(source->getId(), source);
}

/** Adds a sink to this fiddleyard.
*/
void
RSystemElementPedFiddleyard::addSink(PedFiddleyardSink *sink)
{
    sinks_.insert(sink->getId(), sink);
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
RSystemElementPedFiddleyard *
RSystemElementPedFiddleyard::getClone()
{
    // New RSystemElementPedFiddleyard //
    //
    RSystemElementPedFiddleyard *clonedRSystemElementPedFiddleyard = new RSystemElementPedFiddleyard(getID(), getName(), roadId_);

    // Sources/Sinks //
    //
    foreach (PedFiddleyardSource *source, sources_)
    {
        clonedRSystemElementPedFiddleyard->addSource(source->getClone());
    }
    foreach (PedFiddleyardSink *sink, sinks_)
    {
        clonedRSystemElementPedFiddleyard->addSink(sink->getClone());
    }

    return clonedRSystemElementPedFiddleyard;
}

//###################//
// Visitor Pattern //
//###################//

/** Accepts a visitor.
*/
void
RSystemElementPedFiddleyard::accept(Visitor *visitor)
{
    visitor->visit(this);
}

/** Accepts a visitor for the sources.
*/
void
RSystemElementPedFiddleyard::acceptForSources(Visitor *visitor)
{
    foreach (PedFiddleyardSource *child, sources_)
        child->accept(visitor);
}

/** Accepts a visitor for the sinks.
*/
void
RSystemElementPedFiddleyard::acceptForSinks(Visitor *visitor)
{
    foreach (PedFiddleyardSink *child, sinks_)
        child->accept(visitor);
}

/** Accepts a visitor for the sources and the sinks.
*/
void
RSystemElementPedFiddleyard::acceptForChildNodes(Visitor *visitor)
{
    acceptForSources(visitor);
    acceptForSinks(visitor);
}

//########################//
//                        //
// FiddleyardSource       //
//                        //
//########################//

PedFiddleyardSource::PedFiddleyardSource(const odrID &id, int lane, double velocity)
    : id_(id)
    , lane_(lane)
    , velocity_(velocity)
    , startTimeSet_(false)
    , repeatTimeSet_(false)
    , timeDevianceSet_(false)
    , directionSet_(false)
    , sOffsetSet_(false)
    , vOffsetSet_(false)
    , velocityDevianceSet_(false)
    , accelerationSet_(false)
    , accelerationDevianceSet_(false)
{
}

void
PedFiddleyardSource::addPedestrian(const QString &id, double numerator)
{
    if (peds_.remove(id))
    {
        qDebug("WARNING 1011180940! Pedestrian added twice to source. Ignoring first one.");
    }
    peds_.insert(id, numerator);
}

/** Accepts a visitor.
*/
void
PedFiddleyardSource::accept(Visitor *visitor)
{
    visitor->visit(this);
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
PedFiddleyardSource *
PedFiddleyardSource::getClone()
{
    // New JunctionConnection //
    //
    PedFiddleyardSource *clonedPedFiddleyardSource = new PedFiddleyardSource(id_, lane_, velocity_);

    // Add optional attributes
    if (startTimeSet_)
        clonedPedFiddleyardSource->setStartTime(startTime_);
    if (repeatTimeSet_)
        clonedPedFiddleyardSource->setRepeatTime(repeatTime_);
    if (timeDevianceSet_)
        clonedPedFiddleyardSource->setTimeDeviance(timeDeviance_);
    if (directionSet_)
        clonedPedFiddleyardSource->setDirection(direction_);
    if (sOffsetSet_)
        clonedPedFiddleyardSource->setSOffset(sOffset_);
    if (vOffsetSet_)
        clonedPedFiddleyardSource->setVOffset(vOffset_);
    if (velocityDevianceSet_)
        clonedPedFiddleyardSource->setVelocityDeviance(velocityDeviance_);
    if (accelerationSet_)
        clonedPedFiddleyardSource->setAcceleration(acceleration_);
    if (accelerationDevianceSet_)
        clonedPedFiddleyardSource->setAccelerationDeviance(accelerationDeviance_);

    return clonedPedFiddleyardSource;
}

//########################//
//                        //
// FiddleyardSink         //
//                        //
//########################//

PedFiddleyardSink::PedFiddleyardSink(const odrID &id, int lane)
    : id_(id)
    , lane_(lane)
    , sinkProbSet_(false)
    , directionSet_(false)
    , sOffsetSet_(false)
    , vOffsetSet_(false)
{
}

/** Accepts a visitor.
*/
void
PedFiddleyardSink::accept(Visitor *visitor)
{
    visitor->visit(this);
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
PedFiddleyardSink *
PedFiddleyardSink::getClone()
{
    PedFiddleyardSink *clonedPedFiddleyardSink = new PedFiddleyardSink(id_, lane_);

    // Add optional attributes
    if (sinkProbSet_)
        clonedPedFiddleyardSink->setSinkProb(sinkProb_);
    if (directionSet_)
        clonedPedFiddleyardSink->setDirection(direction_);
    if (sOffsetSet_)
        clonedPedFiddleyardSink->setSOffset(sOffset_);
    if (vOffsetSet_)
        clonedPedFiddleyardSink->setVOffset(vOffset_);

    return clonedPedFiddleyardSink;
}
