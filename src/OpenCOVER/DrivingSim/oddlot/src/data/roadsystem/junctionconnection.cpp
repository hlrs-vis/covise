/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/5/2010
**
**************************************************************************/

#include "junctionconnection.hpp"

#include "rsystemelementjunction.hpp"
#include "src/data/projectdata.hpp"

#include "qmap.h"

JunctionConnection::ContactPointValue 
JunctionConnection::parseContactPoint(const QString &value)
{
	if (value == "start")
	{
		return JunctionConnection::JCP_START;
	}
	else if (value == "end")
	{
		return JunctionConnection::JCP_END;
	}
	else
	{
		qDebug("WARNING: unknown connection contact point: %s", value.toUtf8().constData());
		return JunctionConnection::JCP_NONE;
	}
}

QString 
JunctionConnection::parseContactPointBack(ContactPointValue value)
{
	if (value == JunctionConnection::JCP_START)
	{
		return QString("start");
	}
	else if (value == JunctionConnection::JCP_END)
	{
		return QString("end");
	}
	else
	{
		qDebug("WARNING: unknown connection contact point");
		return "";
	}
}

JunctionConnection::JunctionConnection(const QString &id, const odrID &incomingRoad, const odrID &connectingRoad, JunctionConnection::ContactPointValue contactPoint, double numerator)
    : DataElement()
    , junctionConnectionChanges_(0x0)
    , parentJunction_(NULL)
    , id_(id)
    , incomingRoad_(incomingRoad)
    , connectingRoad_(connectingRoad)
    , contactPoint_(contactPoint)
{
    userData_.numerator = numerator;
}

void
JunctionConnection::setParentJunction(RSystemElementJunction *parentJunction)
{
    parentJunction_ = parentJunction;
    setParentElement(parentJunction_);
    addJunctionConnectionChanges(CJC_ParentJunctionChanged);
}

void
JunctionConnection::setId(const QString &id)
{
    id_ = id;
    addJunctionConnectionChanges(CJC_IdChanged);
}

void
JunctionConnection::setIncomingRoad(const odrID &id)
{
    incomingRoad_ = id;
    addJunctionConnectionChanges(CJC_IncomingRoadChanged);
}

void
JunctionConnection::setConnectingRoad(const odrID &id)
{
    connectingRoad_ = id;
    addJunctionConnectionChanges(CJC_ConnectingRoadChanged);
}

void
JunctionConnection::setContactPoint(JunctionConnection::ContactPointValue contactPoint)
{
    contactPoint_ = contactPoint;
    addJunctionConnectionChanges(CJC_ContactPointChanged);
}

void
JunctionConnection::setNumerator(double numerator)
{
    userData_.numerator = numerator;
    addJunctionConnectionChanges(CJC_NumeratorChanged);
}

void
JunctionConnection::addLaneLink(int from, int to)
{
    laneLinks_.insert(from, to);
    addJunctionConnectionChanges(CJC_LaneLinkChanged);
}

void 
JunctionConnection::removeLaneLink(int from)
{
    if (laneLinks_.contains(from))
    {
        laneLinks_.remove(from);
    }
    addJunctionConnectionChanges(CJC_LaneLinkChanged);
}

void 
JunctionConnection::removeLaneLinks()
{
    laneLinks_.clear();
    addJunctionConnectionChanges(CJC_LaneLinkChanged);
}


//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
JunctionConnection::notificationDone()
{
    junctionConnectionChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
JunctionConnection::addJunctionConnectionChanges(int changes)
{
    if (changes)
    {
        junctionConnectionChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
JunctionConnection *
JunctionConnection::getClone()
{
    // New JunctionConnection //
    //
    JunctionConnection *clonedJunctionConnection = new JunctionConnection("clonedJunction", incomingRoad_, connectingRoad_, contactPoint_, userData_.numerator);

    // LaneLinks //
    //
    QMap<int, int>::const_iterator i = laneLinks_.constBegin();
    while (i != laneLinks_.constEnd())
    {
        clonedJunctionConnection->addLaneLink(i.key(), i.value());
        ++i;
    }

    return clonedJunctionConnection;
}

//###################//
// Visitor Pattern   //
//###################//

/*!
* Accepts a visitor and passes it to all child
* nodes if autoTraverse is true.
*/
void
JunctionConnection::accept(Visitor *visitor)
{
    visitor->visit(this);
}
