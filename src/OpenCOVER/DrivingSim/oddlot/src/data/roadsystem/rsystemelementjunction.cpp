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

#include "rsystemelementjunction.hpp"

#include "junctionconnection.hpp"

RSystemElementJunction::RSystemElementJunction(const QString &name, const QString &id)
    : RSystemElement(name, id, RSystemElement::DRE_Junction)
    , junctionChanges_(0x0)
{
}

RSystemElementJunction::~RSystemElementJunction()
{
    /*	foreach(JunctionConnection * child, connections_)
	{
		delete child;
	}*/
}

QList<JunctionConnection *>
RSystemElementJunction::getConnections(const QString &incomingRoad) const
{

    return connections_.values(incomingRoad);
}

QList<JunctionConnection *>
RSystemElementJunction::getConnectingRoadConnections(const QString &connectingRoad) const
{
    QList<JunctionConnection *> junctionConnections;

    foreach (JunctionConnection *conn, connections_)
    {
        if ((conn->getConnectingRoad() == connectingRoad))
        {
            junctionConnections.append(conn);
        }
    }
    return junctionConnections;
}

JunctionConnection *RSystemElementJunction::getConnection(const QString &incomingRoad, const QString &connectingRoad) const
{
    foreach (JunctionConnection *conn, connections_)
    {
        if ((conn->getIncomingRoad() == incomingRoad) && (conn->getConnectingRoad() == connectingRoad))
            return conn;
    }
    return NULL;
}

void
RSystemElementJunction::addConnection(JunctionConnection *connection)
{
    // Check for duplicate connection
    foreach (JunctionConnection *conn, connections_)
    {
        if ((conn->getConnectingRoad() == connection->getConnectingRoad())
            && (conn->getIncomingRoad() == connection->getIncomingRoad())
            && (conn->getContactPoint() == connection->getContactPoint()))
        {
            qDebug("WARNING! Connection already present");
            return;
        }
    }

    int connectionIdCount = 0;
    QString id = connection->getId();
    QMap<QString, JunctionConnection *>::const_iterator it = connections_.constBegin();
    while (it != connections_.constEnd())
    {
        if (it.value()->getId() == id)
        {
            id = QString("jc%1").arg(connectionIdCount);
            connectionIdCount++;
            it = connections_.constBegin();

            continue;
        }

        it++;
    }

    connection->setId(id);
    connections_.insert(connection->getIncomingRoad(), connection);
    connection->setParentJunction(this);
    addJunctionChanges(CJN_ConnectionChanged);
}

void
RSystemElementJunction::delConnection(JunctionConnection *connection)
{
    connections_.remove(connection->getIncomingRoad(), connection);
    connection->setParentJunction(NULL);
    addJunctionChanges(CJN_ConnectionChanged);
}

void
RSystemElementJunction::delConnections()
{
    connections_.clear();
    addJunctionChanges(CJN_ConnectionChanged);
}

void
RSystemElementJunction::checkConnectionIds(const QMap<QString, QString> &roadIds)
{
    QList<JunctionConnection *> newConnections;
    QMultiMap<QString, JunctionConnection *>::const_iterator connectionIt = connections_.constBegin();

    while (connectionIt != connections_.constEnd())
    {
        JunctionConnection *connection = connectionIt.value();
        JunctionConnection *newConnection = connection->getClone();
        if (roadIds.find(connection->getIncomingRoad()) != roadIds.end())
        {
            newConnection->setIncomingRoad(roadIds.find(connection->getIncomingRoad()).value());
        }

        if (roadIds.find(connection->getConnectingRoad()) != roadIds.end())
        {
            newConnection->setConnectingRoad(roadIds.find(connection->getConnectingRoad()).value());
        }

        newConnections.append(newConnection);

        connectionIt++;
    }

    delConnections();
    for (int i = 0; i < newConnections.size(); i++)
    {
        addConnection(newConnections.at(i));
    }
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
RSystemElementJunction::notificationDone()
{
    junctionChanges_ = 0x0;
    RSystemElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
RSystemElementJunction::addJunctionChanges(int changes)
{
    if (changes)
    {
        junctionChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
RSystemElementJunction *
RSystemElementJunction::getClone()
{
    // New Junction //
    //
    RSystemElementJunction *clonedJunction = new RSystemElementJunction(getName(), getID());

    // JunctionConnections //
    //
    foreach (JunctionConnection *connection, connections_)
    {
        clonedJunction->addConnection(connection->getClone());
    }

    return clonedJunction;
}

//###################//
// Visitor Pattern   //
//###################//

/*!
* Accepts a visitor and passes it to all child
* nodes if autoTraverse is true.
*/
void
RSystemElementJunction::accept(Visitor *visitor)
{
    visitor->visit(this);
}

/*!
* Accepts a visitor and passes it to all child nodes.
*/
void
RSystemElementJunction::acceptForChildNodes(Visitor *visitor)
{
    acceptForConnections(visitor);
}

/*!
* Accepts a visitor and passes it to the type sections.
*/
void
RSystemElementJunction::acceptForConnections(Visitor *visitor)
{
    foreach (JunctionConnection *child, connections_)
        child->accept(visitor);
}
