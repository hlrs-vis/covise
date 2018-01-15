/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.07.2010
**
**************************************************************************/

#include "idchangevisitor.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementcontroller.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/rsystemelementfiddleyard.hpp"

#include "src/data/roadsystem/roadlink.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"

/*! \brief \note Use this visitor only on not yet added DataElements since
* there is no undo support. So far this is only used by the AddRoadSystemPrototypeCommand
* so all is fine so far!
*
*/
IdChangeVisitor::IdChangeVisitor(const QMap<QString, QString> &roadIds, const QMap<QString, QString> &controllerIds, const QMap<QString, QString> &junctionIds, const QMap<QString, QString> &fiddleyardIds)
    : roadIds_(roadIds)
    , controllerIds_(controllerIds)
    , junctionIds_(junctionIds)
    , fiddleyardIds_(fiddleyardIds)
{
}

/*!
*
*/
//void
//	IdChangeVisitor
//	::visit(RoadSystem * roadSystem)
//{
//	// Run //
//	//
//	roadSystem->acceptFor (this, false);
//}

/*!
*
*/
void
IdChangeVisitor::visit(RSystemElementRoad *road)
{
    // Road Id //
    //
    //if(roadIds_.contains(road->getID()))
    //{
    //	the id of the road should be set already
    //}

    RoadLink *roadLink = NULL;

    // Predecessor //
    //
    roadLink = road->getPredecessor();
    if (roadLink)
    {
        if (roadLink->getElementType() == "road")
        {
            if (roadIds_.contains(roadLink->getElementId()))
            {
                roadLink->setElementId(roadIds_.value(roadLink->getElementId()));
            }
        }
        else if (roadLink->getElementType() == "junction")
        {
            if (junctionIds_.contains(roadLink->getElementId()))
            {
                roadLink->setElementId(junctionIds_.value(roadLink->getElementId()));
            }
        }
        else if (roadLink->getElementType() == "fiddleyard")
        {
            if (fiddleyardIds_.contains(roadLink->getElementId()))
            {
                roadLink->setElementId(fiddleyardIds_.value(roadLink->getElementId()));
            }
        }
    }

    // Successor //
    //
    roadLink = road->getSuccessor();
    if (roadLink)
    {
        if (roadLink->getElementType() == "road")
        {
            if (roadIds_.contains(roadLink->getElementId()))
            {
                roadLink->setElementId(roadIds_.value(roadLink->getElementId()));
            }
        }
        else if (roadLink->getElementType() == "junction")
        {
            if (junctionIds_.contains(roadLink->getElementId()))
            {
                roadLink->setElementId(junctionIds_.value(roadLink->getElementId()));
            }
        }
        else if (roadLink->getElementType() == "fiddleyard")
        {
            if (fiddleyardIds_.contains(roadLink->getElementId()))
            {
                roadLink->setElementId(fiddleyardIds_.value(roadLink->getElementId()));
            }
        }
    }

    // Junction (if road is a path) //
    //
    if (road->getJunction() != "-1" && road->getJunction() != "")
    {
        if (junctionIds_.contains(road->getJunction()))
        {
            road->setJunction(junctionIds_.value(road->getJunction()));
        }
    }
}

/*!
*
*/
void
IdChangeVisitor::visit(RSystemElementController *controller)
{
    // Controller Id //
    //
    //if(controllerIds_.contains(controller->getID()))
    //{
    //	the id of the controller should be set already
    //}
}

/*!
*
*/
void
IdChangeVisitor::visit(RSystemElementJunction *junction)
{
    // Junction Id //
    //
    //if(junctionIds_.contains(junction->getID()))
    //{
    //	the id of the junction should be set already
    //}

    // JunctionConnections //
    //
    junction->acceptForConnections(this);
}

/*!
*
*/
void
IdChangeVisitor::visit(JunctionConnection *connection)
{
    // IncomingRoad //
    //
    if (roadIds_.contains(connection->getIncomingRoad()))
    {
        connection->setIncomingRoad(roadIds_.value(connection->getIncomingRoad()));
    }

    // ConnectingRoad //
    //
    if (roadIds_.contains(connection->getConnectingRoad()))
    {
 //       connection->setContactPoint(roadIds_.value(connection->getConnectingRoad()));
		connection->setConnectingRoad(roadIds_.value(connection->getConnectingRoad()));
    }
}

/*!
*
*/
void
IdChangeVisitor::visit(RSystemElementFiddleyard *fiddleyard)
{
    // Fiddleyard Id //
    //
    //if(fiddleyardIds_.contains(fiddleyard->getID()))
    //{
    //	the id of the fiddleyard should be set already
    //}

    // Link //
    //
    if (fiddleyard->getElementType() == "road")
    {
        if (roadIds_.contains(fiddleyard->getElementId()))
        {
            fiddleyard->setElementId(roadIds_.value(fiddleyard->getElementId()));
        }
    }
    else if (fiddleyard->getElementType() == "junction")
    {
        if (junctionIds_.contains(fiddleyard->getElementId()))
        {
            fiddleyard->setElementId(junctionIds_.value(fiddleyard->getElementId()));
        }
    }
}
