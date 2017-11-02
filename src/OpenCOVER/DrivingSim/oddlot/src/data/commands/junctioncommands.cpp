/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   21.05.2010
**
**************************************************************************/

#include "junctioncommands.hpp"

#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/roadlink.hpp"

//#########################//
// RemoveJunctionCommand //
//#########################//

RemoveJunctionCommand::RemoveJunctionCommand(RSystemElementJunction *junction, DataCommand *parent)
    : DataCommand(parent)
    , junction_(junction)
{
    // Check for validity //
    //
    if (!junction_ || !junction->getRoadSystem())
    {
        setInvalid(); // Invalid
        setText(QObject::tr("RemoveJunctionCommand: Internal error! No junction specified."));
        return;
    }

    roadSystem_ = junction->getRoadSystem();
    connections_ = junction->getConnections();

    foreach (JunctionConnection *conn, connections_)
    {
        RSystemElementRoad *incomingRoad = roadSystem_->getRoad(conn->getIncomingRoad());
        RSystemElementRoad *connectingRoad = roadSystem_->getRoad(conn->getConnectingRoad());
        JunctionConnection::ContactPointValue contactPoint = conn->getContactPoint();

        if (connectingRoad)
        {
            if ((contactPoint == JunctionConnection::JCP_START) && (connectingRoad->getPredecessor()))
            {
                predecessors_.insert(connectingRoad, connectingRoad->getPredecessor());
            }
            else if ((contactPoint == JunctionConnection::JCP_END) && (connectingRoad->getSuccessor()))
            {
                successors_.insert(connectingRoad, connectingRoad->getSuccessor());
            }
        }

        if (incomingRoad)
        {
            if (incomingRoad->getPredecessor() && (incomingRoad->getPredecessor()->getElementId() == junction->getID()))
            {
                predecessors_.insert(incomingRoad, incomingRoad->getPredecessor());
            }
            if (incomingRoad->getSuccessor() && (incomingRoad->getSuccessor()->getElementId() == junction->getID()))
            {
                successors_.insert(incomingRoad, incomingRoad->getSuccessor());
            }
        }
    }
    setValid();
    setText(QObject::tr("Remove Junction"));
}

/*! \brief .
*
*/
RemoveJunctionCommand::~RemoveJunctionCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (road is still owned by the roadsystem)
    }
    else
    {
        delete junction_;
    }
}

/*! \brief .
*
*/
void
RemoveJunctionCommand::redo()
{
    QMap<RSystemElementRoad *, RoadLink *>::const_iterator predIt = predecessors_.constBegin();
    while (predIt != predecessors_.constEnd())
    {
        predIt.key()->delPredecessor();
        predIt++;
    }

    QMap<RSystemElementRoad *, RoadLink *>::const_iterator succIt = successors_.constBegin();
    while (succIt != successors_.constEnd())
    {
        succIt.key()->delSuccessor();
        succIt++;
    }

    foreach (JunctionConnection *conn, connections_)
    {
        RSystemElementRoad *road = roadSystem_->getRoad(conn->getConnectingRoad());
        if (road)
        {
            road->setJunction("-1");
        }
    }

    roadSystem_->delJunction(junction_);

    setRedone();
}

/*! \brief
*
*/
void
RemoveJunctionCommand::undo()
{
    roadSystem_->addJunction(junction_);

    foreach (JunctionConnection *conn, connections_)
    {
        junction_->addConnection(conn);
        RSystemElementRoad *road = roadSystem_->getRoad(conn->getConnectingRoad());
        if (road)
        {
            road->setJunction(junction_->getID());
        }
    }

    QMap<RSystemElementRoad *, RoadLink *>::const_iterator predIt = predecessors_.constBegin();
    while (predIt != predecessors_.constEnd())
    {
        predIt.key()->setPredecessor(predIt.value());
        predIt++;
    }

    QMap<RSystemElementRoad *, RoadLink *>::const_iterator succIt = successors_.constBegin();
    while (succIt != successors_.constEnd())
    {
        succIt.key()->setSuccessor(succIt.value());
        succIt++;
    }

    setUndone();
}

//#########################//
// NewJunctionCommand //
//#########################//

NewJunctionCommand::NewJunctionCommand(RSystemElementJunction *newJunction, RoadSystem *roadSystem, DataCommand *parent)
    : DataCommand(parent)
    , newJunction_(newJunction)
    , roadSystem_(roadSystem)
{
    // Check for validity //
    //
    if (!newJunction || !roadSystem_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("NewJunctionCommand: Internal error! No new road specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("New Road"));
    }
}

/*! \brief .
*
*/
NewJunctionCommand::~NewJunctionCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        delete newJunction_;
    }
    else
    {
        // nothing to be done (road is now owned by the roadsystem)
    }
}

/*! \brief .
*
*/
void
NewJunctionCommand::redo()
{
    roadSystem_->addJunction(newJunction_);

    setRedone();
}

/*! \brief
*
*/
void
NewJunctionCommand::undo()
{
    roadSystem_->delJunction(newJunction_);

    setUndone();
}

//#########################//
// AddConnectionCommand //
//#########################//

AddConnectionCommand::AddConnectionCommand(RSystemElementJunction *junction, JunctionConnection *connection, DataCommand *parent)
    : DataCommand(parent)
    , junction_(junction)
    , connection_(connection)
{
    // Check for validity //
    //
    if (!junction || !connection_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("AddConnectionCommand: Internal error! No connection specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Add connection"));
    }
}

/*! \brief .
*
*/
AddConnectionCommand::~AddConnectionCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        delete connection_;
    }
    else
    {
        // nothing to be done (road is now owned by the roadsystem)
    }
}

/*! \brief .
*
*/
void
AddConnectionCommand::redo()
{
    junction_->addConnection(connection_);

    setRedone();
}

/*! \brief
*
*/
void
AddConnectionCommand::undo()
{
    junction_->delConnection(connection_);

    setUndone();
}

//###############################//
// SetConnectionLaneLinkCommand //
//##############################//

SetConnectionLaneLinkCommand::SetConnectionLaneLinkCommand(JunctionConnection *connection, int from, int to, DataCommand *parent)
    : DataCommand(parent)
    , connection_(connection)
    , newFrom_(from)
    , to_(to)
{
    oldFrom_ = connection->getFromLane(to);
    // Check for validity //
    //
    if (!connection || (newFrom_ == oldFrom_))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("SetConnectionLaneLinkCommand: Internal error! No new from lane specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Set connection lane link"));
    }
}

/*! \brief .
*
*/
SetConnectionLaneLinkCommand::~SetConnectionLaneLinkCommand()
{
    // Clean up //
    //
    if (isUndone())
    {

    }
    else
    {
        // nothing to be done 
    }
}

/*! \brief .
*
*/
void
SetConnectionLaneLinkCommand::redo()
{
    connection_->removeLaneLink(oldFrom_);
    connection_->addLaneLink(newFrom_, to_);

    setRedone();
}

/*! \brief
*
*/
void
SetConnectionLaneLinkCommand::undo()
{
    connection_->removeLaneLink(newFrom_);
    connection_->addLaneLink(oldFrom_, to_);

    setUndone();
}


//###################################//
// RemoveConnectionLaneLinksCommand //
//##################################//

RemoveConnectionLaneLinksCommand::RemoveConnectionLaneLinksCommand(JunctionConnection *connection, DataCommand *parent)
    : DataCommand(parent)
    , connection_(connection)
{
    oldLaneLinks_ = connection_->getLaneLinks();
    // Check for validity //
    //
    if (!connection || (oldLaneLinks_.size() == 0))
    {
        setInvalid(); // Invalid
        setText(QObject::tr("RemoveConnectionLaneLinksCommand: Internal error! No connection or lane links."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("Remove connection lane links"));
    }
}

/*! \brief .
*
*/
RemoveConnectionLaneLinksCommand::~RemoveConnectionLaneLinksCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        oldLaneLinks_.clear();
    }
    else
    {
        // nothing to be done 
    }
}

/*! \brief .
*
*/
void
RemoveConnectionLaneLinksCommand::redo()
{
    connection_->removeLaneLinks();

    setRedone();
}

/*! \brief
*
*/
void
RemoveConnectionLaneLinksCommand::undo()
{
    connection_->setLaneLinks(oldLaneLinks_);

    setUndone();
}

