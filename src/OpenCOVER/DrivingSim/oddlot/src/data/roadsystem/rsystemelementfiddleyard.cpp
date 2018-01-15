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

#include "rsystemelementfiddleyard.hpp"


//##########################//
//                          //
// RSystemElementFiddleyard //
//                          //
//##########################//

RSystemElementFiddleyard::RSystemElementFiddleyard(const QString &name, const odrID &id, const QString &elementType, const QString &elementId, const QString &contactPoint)
    : RSystemElement(name, id, RSystemElement::DRE_Fiddleyard)
    , elementType_(elementType)
    , elementId_(elementId)
    , contactPoint_(contactPoint)
{
}

RSystemElementFiddleyard::~RSystemElementFiddleyard()
{
    foreach (FiddleyardSink *child, sinks_)
        delete child;

    foreach (FiddleyardSource *child, sources_)
        delete child;
}

/** Adds a source to this fiddleyard.
*/
void
RSystemElementFiddleyard::addSource(FiddleyardSource *source)
{
    sources_.insert(source->getId(), source);
}

/** Adds a sink to this fiddleyard.
*/
void
RSystemElementFiddleyard::addSink(FiddleyardSink *sink)
{
    sinks_.insert(sink->getId(), sink);
}

void
RSystemElementFiddleyard::setElementType(const QString &elementType)
{
    elementType_ = elementType;
}

void
RSystemElementFiddleyard::setElementId(const QString &elementId)
{
    elementId_ = elementId;
}

void
RSystemElementFiddleyard::setContactPoint(const QString &contactPoint)
{
    contactPoint_ = contactPoint;
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
RSystemElementFiddleyard *
RSystemElementFiddleyard::getClone()
{
    // New RSystemElementFiddleyard //
    //
    RSystemElementFiddleyard *clonedRSystemElementFiddleyard = new RSystemElementFiddleyard(getName(), getID(), elementType_, elementId_, contactPoint_);

    // Sources/Sinks //
    //
    foreach (FiddleyardSource *source, sources_)
    {
        clonedRSystemElementFiddleyard->addSource(source->getClone());
    }
    foreach (FiddleyardSink *sink, sinks_)
    {
        clonedRSystemElementFiddleyard->addSink(sink->getClone());
    }

    return clonedRSystemElementFiddleyard;
}

void
RSystemElementFiddleyard::updateIds(const QMultiMap<QString, RoadSystem::IdType> &roadIds)
{
	RoadSystem *roadSystem = getRoadSystem();

	elementId_ = roadSystem->getNewId(roadIds, elementId_, "road");

    foreach (FiddleyardSource *source, sources_)
    {
		QString id = source->getId();
		QString newId = roadSystem->getNewId(roadIds, id, "fiddleyard");
        if (id != newId)
        {
            source->setId(newId);
        }
    }
    foreach (FiddleyardSink *sink, sinks_)
    {
		QString id = sink->getId();
		QString newId = roadSystem->getNewId(roadIds, id, "fiddleyard");
         if (id != newId)
        {
            sink->setId(newId);
        }
    }
}

//###################//
// Visitor Pattern //
//###################//

/** Accepts a visitor.
*/
void
RSystemElementFiddleyard::accept(Visitor *visitor)
{
    visitor->visit(this);
}

/** Accepts a visitor for the sources.
*/
void
RSystemElementFiddleyard::acceptForSources(Visitor *visitor)
{
    foreach (FiddleyardSource *child, sources_)
        child->accept(visitor);
}

/** Accepts a visitor for the sinks.
*/
void
RSystemElementFiddleyard::acceptForSinks(Visitor *visitor)
{
    foreach (FiddleyardSink *child, sinks_)
        child->accept(visitor);
}

/** Accepts a visitor for the sources and the sinks.
*/
void
RSystemElementFiddleyard::acceptForChildNodes(Visitor *visitor)
{
    acceptForSources(visitor);
    acceptForSinks(visitor);
}

//########################//
//                        //
// FiddleyardSource       //
//                        //
//########################//

FiddleyardSource::FiddleyardSource(QString &id, int lane, double startTime, double repeatTime, double velocity, double velocityDeviance)
    : id_(id)
    , lane_(lane)
    , startTime_(startTime)
    , repeatTime_(repeatTime)
    , velocity_(velocity)
    , velocityDeviance_(velocityDeviance)
{
}

void
FiddleyardSource::addVehicle(const QString &id, double numerator)
{
    if (vehicles_.remove(id))
    {
        qDebug("WARNING 1011180939! Vehicle added twice to source. Ignoring first one.");
    }
    vehicles_.insert(id, numerator);
}

/** Accepts a visitor.
*/
void
FiddleyardSource::accept(Visitor *visitor)
{
    visitor->visit(this);
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
FiddleyardSource *
FiddleyardSource::getClone()
{
    // New JunctionConnection //
    //
    FiddleyardSource *clonedFiddleyardSource = new FiddleyardSource(id_, lane_, startTime_, repeatTime_, velocity_, velocityDeviance_);

    //  //
    //
    //	QMap<int, int>::const_iterator i = laneLinks_.constBegin();
    //	while (i != laneLinks_.constEnd())
    //	{
    //		clonedJunctionConnection->addLaneLink(i.key(), i.value());
    //		++i;
    //	}

    return clonedFiddleyardSource;
}

//########################//
//                        //
// FiddleyardSink         //
//                        //
//########################//

FiddleyardSink::FiddleyardSink(const QString &id, int lane)
    : id_(id)
    , lane_(lane)
{
}

/** Accepts a visitor.
*/
void
FiddleyardSink::accept(Visitor *visitor)
{
    visitor->visit(this);
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
FiddleyardSink *
FiddleyardSink::getClone()
{
    return new FiddleyardSink(id_, lane_);
}
