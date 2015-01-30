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

#include "roadsystem.hpp"

// Data //
//
#include "src/data/projectdata.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"

#include "rsystemelementroad.hpp"
#include "rsystemelementcontroller.hpp"
#include "rsystemelementjunction.hpp"
#include "rsystemelementfiddleyard.hpp"
#include "rsystemelementpedfiddleyard.hpp"
#include "roadlink.hpp"

#include "src/data/tilesystem/tilesystem.hpp"

/*! \brief CONSTRUCTOR.
*
*/
RoadSystem::RoadSystem()
    : DataElement()
    , roadSystemChanges_(0x0)
{
}

RoadSystem::~RoadSystem()
{
    // Delete child nodes //
    //
    foreach (RSystemElementRoad *child, roads_)
        delete child;

    foreach (RSystemElementController *child, controllers_)
        delete child;

    foreach (RSystemElementJunction *child, junctions_)
        delete child;

    foreach (RSystemElementFiddleyard *child, fiddleyards_)
        delete child;

    foreach (RSystemElementPedFiddleyard *child, pedFiddleyards_)
        delete child;
}

//##################//
// RSystemElements  //
//##################//

RSystemElementRoad *
RoadSystem::getRoad(const QString &id) const
{
    return roads_.value(id, NULL);
}

QList<RSystemElementRoad *>
RoadSystem::getTileRoads(const QString &tileId) const
{
    QList<RSystemElementRoad *> tileRoads;
    QMap<QString, RSystemElementRoad *>::const_iterator it = roads_.constBegin();
    while (it != roads_.constEnd())
    {
        if (it.key().startsWith(tileId + "_"))
        {
            tileRoads.append(it.value());
        }

        it++;
    }

    return tileRoads;
}

void
RoadSystem::addRoad(RSystemElementRoad *road)
{
    if (getProjectData())
    {
        // Id //
        //
        QString name = road->getName();
        QString id = getUniqueId(road->getID(), name);
        if (id != road->getID())
        {
            road->setID(id);
            if (name != road->getName())
            {
                road->setName(name);
            }
        }
    }

    // Insert //
    //
    road->setRoadSystem(this);

    roads_.insert(road->getID(), road);
    addRoadSystemChanges(RoadSystem::CRS_RoadChange);
}

bool
RoadSystem::delRoad(RSystemElementRoad *road)
{
    QStringList parts = road->getID().split("_");

    if (roads_.remove(road->getID()) && elementIds_.remove(parts.at(0), parts.at(1).toInt()))
    {
        addRoadSystemChanges(RoadSystem::CRS_RoadChange);

        road->setRoadSystem(NULL);
        return true;
    }
    else
    {
        qDebug("WARNING 1005311350! Delete road not successful!");
        return false;
    }
}

RSystemElementController *
RoadSystem::getController(const QString &id) const
{
    return controllers_.value(id, NULL);
}

QList<RSystemElementController *>
RoadSystem::getTileControllers(const QString &tileId) const
{
    QList<RSystemElementController *> tileControllers;
    QMap<QString, RSystemElementController *>::const_iterator it = controllers_.constBegin();
    while (it != controllers_.constEnd())
    {
        if (it.key().startsWith(tileId + "_"))
        {
            tileControllers.append(it.value());
        }

        it++;
    }

    return tileControllers;
}

void
RoadSystem::addController(RSystemElementController *controller)
{
    addRoadSystemChanges(RoadSystem::CRS_ControllerChange);
    QString name = controller->getName();
    controllers_.insert(getUniqueId(controller->getID(), name), controller);
    if (name != controller->getName())
    {
        controller->setName(name);
    }

    controller->setRoadSystem(this);
}

bool
RoadSystem::delController(RSystemElementController *controller)
{
    QStringList parts = controller->getID().split("_");

    if (controllers_.remove(controller->getID()) && elementIds_.remove(parts.at(0), parts.at(1).toInt()))
    {
        addRoadSystemChanges(RoadSystem::CRS_ControllerChange);

        controller->setRoadSystem(NULL);
        return true;
    }
    else
    {
        qDebug("WARNING 1007081033! Delete controller not successful!");
        return false;
    }
}

RSystemElementJunction *
RoadSystem::getJunction(const QString &id) const
{
    return junctions_.value(id, NULL);
}

QList<RSystemElementJunction *>
RoadSystem::getTileJunctions(const QString &tileId) const
{
    QList<RSystemElementJunction *> tileJunctions;
    QMap<QString, RSystemElementJunction *>::const_iterator it = junctions_.constBegin();
    while (it != junctions_.constEnd())
    {
        if (it.key().startsWith(tileId + "_"))
        {
            tileJunctions.append(it.value());
        }

        it++;
    }

    return tileJunctions;
}

void
RoadSystem::addJunction(RSystemElementJunction *junction)
{
    if (getProjectData())
    {
        // Id //
        //
        QString name = junction->getName();

        QString id = getUniqueId(junction->getID(), name);
        if (id != junction->getID())
        {
            junction->setID(id);
            if (name != junction->getName())
            {
                junction->setName(name);
            }
        }
    }

    // Insert //
    //
    junctions_.insert(junction->getID(), junction);
    addRoadSystemChanges(RoadSystem::CRS_JunctionChange);

    junction->setRoadSystem(this);
}

bool
RoadSystem::delJunction(RSystemElementJunction *junction)
{
    QStringList parts = junction->getID().split("_");

    if (junctions_.remove(junction->getID()) && elementIds_.remove(parts.at(0), parts.at(1).toInt()))
    {
        addRoadSystemChanges(RoadSystem::CRS_JunctionChange);

        junction->delConnections();
        junction->setRoadSystem(NULL);
        return true;
    }
    else
    {
        qDebug("WARNING 1007081034! Delete junction not successful!");
        return false;
    }
}

RSystemElementFiddleyard *
RoadSystem::getFiddleyard(const QString &id) const
{
    return fiddleyards_.value(id, NULL);
}

QList<RSystemElementFiddleyard *>
RoadSystem::getTileFiddleyards(const QString &tileId) const
{
    QList<RSystemElementFiddleyard *> tileFiddleyards;
    QMap<QString, RSystemElementFiddleyard *>::const_iterator it = fiddleyards_.constBegin();
    while (it != fiddleyards_.constEnd())
    {
        if (it.key().startsWith(tileId + "_"))
        {
            tileFiddleyards.append(it.value());
        }

        it++;
    }

    return tileFiddleyards;
}

void
RoadSystem::addFiddleyard(RSystemElementFiddleyard *fiddleyard)
{
    addRoadSystemChanges(RoadSystem::CRS_FiddleyardChange);
    QString name = fiddleyard->getName();
    fiddleyards_.insert(getUniqueId(fiddleyard->getID(), name), fiddleyard);

    if (name != fiddleyard->getName())
    {
        fiddleyard->setName(name);
    }

    fiddleyard->setRoadSystem(this);
}

bool
RoadSystem::delFiddleyard(RSystemElementFiddleyard *fiddleyard)
{
    QStringList parts = fiddleyard->getID().split("_");

    if (fiddleyards_.remove(fiddleyard->getID()) && elementIds_.remove(parts.at(0), parts.at(1).toInt()))
    {
        addRoadSystemChanges(RoadSystem::CRS_FiddleyardChange);

        fiddleyard->setRoadSystem(NULL);
        return true;
    }
    else
    {
        qDebug("WARNING 1007081035! Delete fiddleyard not successful!");
        return false;
    }
}

RSystemElementPedFiddleyard *
RoadSystem::getPedFiddleyard(const QString &id) const
{
    return pedFiddleyards_.value(id, NULL);
}

QList<RSystemElementPedFiddleyard *>
RoadSystem::getTilePedFiddleyards(const QString &tileId) const
{
    QList<RSystemElementPedFiddleyard *> tilePedFiddleyards;
    QMap<QString, RSystemElementPedFiddleyard *>::const_iterator it = pedFiddleyards_.constBegin();
    while (it != pedFiddleyards_.constEnd())
    {
        if (it.key().startsWith(tileId + "_"))
        {
            tilePedFiddleyards.append(it.value());
        }

        it++;
    }

    return tilePedFiddleyards;
}

void
RoadSystem::addPedFiddleyard(RSystemElementPedFiddleyard *fiddleyard)
{
    addRoadSystemChanges(RoadSystem::CRS_PedFiddleyardChange);
    QString name = fiddleyard->getName();
    pedFiddleyards_.insert(getUniqueId(fiddleyard->getID(), name), fiddleyard);

    if (name != fiddleyard->getName())
    {
        fiddleyard->setName(name);
    }

    fiddleyard->setRoadSystem(this);
}

bool
RoadSystem::delPedFiddleyard(RSystemElementPedFiddleyard *fiddleyard)
{
    QStringList parts = fiddleyard->getID().split("_");

    if (pedFiddleyards_.remove(fiddleyard->getID()) && elementIds_.remove(parts.at(0), parts.at(1).toInt()))
    {
        addRoadSystemChanges(RoadSystem::CRS_PedFiddleyardChange);

        fiddleyard->setRoadSystem(NULL);
        return true;
    }
    else
    {
        qDebug("WARNING 1007081036! Delete pedFiddleyard not successful!");
        return false;
    }
}

//##################//
// IDs              //
//##################//

const QString
RoadSystem::getUniqueId(const QString &suggestion, QString &name)
{
    QString tileId = getProjectData()->getTileSystem()->getCurrentTile()->getID();
    QList<int> currentTileElementIds_ = elementIds_.values(tileId);

    // Try suggestion //
    //
    if (!suggestion.isNull() && !suggestion.isEmpty() && !name.isEmpty())
    {
        bool number = false;
        QStringList parts = suggestion.split("_");

        if (parts.size() > 2)
        {
            parts.at(0).toInt(&number);
            if (tileId == parts.at(0))
            {
                int nr = parts.at(1).toInt(&number);

                if (number && !currentTileElementIds_.contains(nr))
                {
                    elementIds_.insert(tileId, nr);
                    return suggestion;
                }
            }
        }
    }

    // Create new one //
    //

    if (name.isEmpty())
    {
        name = "unnamed";
    }
    /*	else if (name.contains("_"))       // get rid of old name concatention
	{
		int index = name.indexOf("_");
		name = name.left(index-1);
	}*/

    QString id;

    int index = 0;
    while ((index < currentTileElementIds_.size()) && currentTileElementIds_.contains(index))
    {
        index++;
    }

    id = QString("%1_%2_%3").arg(tileId).arg(index).arg(name);
    elementIds_.insert(tileId, index);
    return id;
}

// The ID is already unique
//
void
RoadSystem::changeUniqueId(RSystemElement *element, QString newId)
{

    if (roads_.contains(element->getID()))
    {
        roads_.remove(element->getID());
        RSystemElementRoad *road = static_cast<RSystemElementRoad *>(element);

        if (road->getJunction() == "-1")
        {
            RoadLink *predecessor = road->getPredecessor();
            if (predecessor)
            {
                if (predecessor->getElementType() == "road")
                {
                    RoadLink *predSuccessor = getRoad(predecessor->getElementId())->getSuccessor();
                    RoadLink *roadLink = new RoadLink("road", newId, predSuccessor->getContactPoint());
                    getRoad(predecessor->getElementId())->setSuccessor(roadLink);
                }
                else if (predecessor->getElementType() == "junction") // incoming road
                {
                    RSystemElementJunction *junction = getJunction(predecessor->getElementId());

                    // Check all connecting roads
                    QList<JunctionConnection *> connectionList = junction->getConnections(road->getID());
                    for (int i = 0; i < connectionList.size(); i++)
                    {
                        RSystemElementRoad *connectingRoad = getRoad(connectionList.at(i)->getConnectingRoad());
                        RoadLink *connectingRoadSuccessor = connectingRoad->getSuccessor();
                        if (connectingRoadSuccessor->getElementId() == road->getID())
                        {
                            RoadLink *roadLink = new RoadLink("road", newId, connectingRoadSuccessor->getContactPoint());
                            connectingRoad->setSuccessor(roadLink);
                        }
                        else
                        {
                            RoadLink *connectingRoadPredecessor = connectingRoad->getPredecessor();

                            RoadLink *roadLink = new RoadLink("road", newId, connectingRoadPredecessor->getContactPoint());
                            connectingRoad->setPredecessor(roadLink);
                        }
                    }

                    QMap<QString, QString> idsChanged;
                    idsChanged.insert(element->getID(), newId);
                    junction->checkConnectionIds(idsChanged);
                }
            }

            RoadLink *successor = road->getSuccessor();
            if (successor)
            {
                if (successor->getElementType() == "road")
                {
                    RoadLink *succPredecessor = getRoad(successor->getElementId())->getPredecessor();
                    RoadLink *roadLink = new RoadLink("road", newId, succPredecessor->getContactPoint());
                    getRoad(successor->getElementId())->setPredecessor(roadLink);
                }
                else if (successor->getElementType() == "junction") // incoming road
                {
                    RSystemElementJunction *junction = getJunction(successor->getElementId());

                    // Check all connecting roads
                    QList<JunctionConnection *> connectionList = junction->getConnections(road->getID());
                    for (int i = 0; i < connectionList.size(); i++)
                    {
                        RSystemElementRoad *connectingRoad = getRoad(connectionList.at(i)->getConnectingRoad());

                        RoadLink *connectingRoadSuccessor = connectingRoad->getSuccessor();
                        if (connectingRoadSuccessor->getElementId() == road->getID())
                        {
                            RoadLink *roadLink = new RoadLink("road", newId, connectingRoadSuccessor->getContactPoint());
                            connectingRoad->setSuccessor(roadLink);
                        }
                        else
                        {
                            RoadLink *connectingRoadPredecessor = connectingRoad->getPredecessor();

                            RoadLink *roadLink = new RoadLink("road", newId, connectingRoadPredecessor->getContactPoint());
                            connectingRoad->setPredecessor(roadLink);
                        }
                    }

                    QMap<QString, QString> idsChanged;
                    idsChanged.insert(element->getID(), newId);
                    junction->checkConnectionIds(idsChanged);
                }
            }
        }

        else // Connecting road
        {
            QMap<QString, QString> idsChanged;
            idsChanged.insert(element->getID(), newId);
            getJunction(road->getJunction())->checkConnectionIds(idsChanged);
        }

        element->setID(newId);
        roads_.insert(newId, static_cast<RSystemElementRoad *>(element));

        addRoadSystemChanges(RoadSystem::CRS_RoadChange);
    }
    else if (junctions_.contains(element->getID()))
    {
        junctions_.remove(element->getID());
        foreach (RSystemElementRoad *road, roads_)
        {
            if (road->getJunction() == element->getID())
            {
                road->setJunction(newId);
            }
            else if (road->getPredecessor() && (road->getPredecessor()->getElementId() == element->getID()))
            {
                RoadLink *predecessor = road->getPredecessor();
                RoadLink *roadLink = new RoadLink("junction", newId, predecessor->getContactPoint());
                road->setPredecessor(roadLink);
            }
            else if (road->getSuccessor() && (road->getSuccessor()->getElementId() == element->getID()))
            {
                RoadLink *successor = road->getSuccessor();
                RoadLink *roadLink = new RoadLink("junction", newId, successor->getContactPoint());
                road->setSuccessor(roadLink);
            }
        }
        element->setID(newId);
        junctions_.insert(newId, static_cast<RSystemElementJunction *>(element));

        addRoadSystemChanges(RoadSystem::CRS_JunctionChange);
    }
    else if (controllers_.contains(element->getID()))
    {
        controllers_.remove(element->getID());
        element->setID(newId);
        controllers_.insert(newId, static_cast<RSystemElementController *>(element));

        addRoadSystemChanges(RoadSystem::CRS_ControllerChange);
    }
    else if (fiddleyards_.contains(element->getID()))
    {
        fiddleyards_.remove(element->getID());
        element->setID(newId);
        fiddleyards_.insert(newId, static_cast<RSystemElementFiddleyard *>(element));

        addRoadSystemChanges(RoadSystem::CRS_FiddleyardChange);
    }
    else if (pedFiddleyards_.contains(element->getID()))
    {
        pedFiddleyards_.remove(element->getID());
        element->setID(newId);
        pedFiddleyards_.insert(newId, static_cast<RSystemElementPedFiddleyard *>(element));

        addRoadSystemChanges(RoadSystem::CRS_PedFiddleyardChange);
    }
}

void
RoadSystem::checkIDs(const QMap<QString, QString> &idMap)
{
    //Roads //
    //
    QString tileId = getProjectData()->getTileSystem()->getCurrentTile()->getID();
    QMap<QString, RSystemElementRoad *>::const_iterator it = roads_.constBegin();

    while (it != roads_.constEnd())
    {
        RSystemElementRoad *road = it.value();
        QStringList parts = road->getID().split("_");

        if (tileId == parts.at(0))
        {

            RoadLink *predecessor = road->getPredecessor();

            if (predecessor != NULL)
            {
                if (idMap.find(predecessor->getElementId()) != idMap.end())
                {
                    RoadLink *roadLink = new RoadLink(predecessor->getElementType(), idMap.find(predecessor->getElementId()).value(), predecessor->getContactPoint());
                    road->setPredecessor(roadLink);
                }
                else
                {
//                    qDebug() << "Road " << road->getID() << " Predecessor " << predecessor->getElementId() << " has the old ID!";
                }
            }

            RoadLink *successor = road->getSuccessor();

            if (successor != NULL)
            {
                if (idMap.find(successor->getElementId()) != idMap.end())
                {
                    RoadLink *roadLink = new RoadLink(successor->getElementType(), idMap.find(successor->getElementId()).value(), successor->getContactPoint());
                    road->setSuccessor(roadLink);
                }
                else
                {
//                    qDebug() << "Road " << road->getID() << " Successor " << successor->getElementId() << " has the old ID!";
                }
            }

            QString junction = road->getJunction();

            if (junction != "-1")
            {
                if (idMap.find(junction) != idMap.end())
                {
                    road->setJunction(idMap.find(junction).value());
                }
                else
                {
//                    qDebug() << "Road " << road->getID() << " Junction " << junction << " has the old ID!";
                }
            }
        }

        it++;
    }

    //Junctions //
    //

    QMap<QString, RSystemElementJunction *>::const_iterator iter = junctions_.constBegin();

    while (iter != junctions_.constEnd())
    {
        RSystemElementJunction *junction = iter.value();
        QStringList parts = junction->getID().split("_");

        if (tileId == parts.at(0))
        {

            junction->checkConnectionIds(idMap);
        }

        iter++;
    }

    // Controllers //
    //
    QMap<QString, RSystemElementController *>::const_iterator controlIt = controllers_.constBegin();

    while (controlIt != controllers_.constEnd())
    {
        QList<ControlEntry *> entryList = controlIt.value()->getControlEntries();
        for (int i = 0; i < entryList.size(); i++)
        {
            ControlEntry *entry = entryList.at(i);

            if (idMap.find(entry->getSignalId()) != idMap.end())
            {
                entry->setSignalId(idMap.find(entry->getSignalId()).value());
            }
        }

        controlIt++;
    }
    //FiddleJards //
    //

    QMap<QString, RSystemElementFiddleyard *>::const_iterator fIter = fiddleyards_.constBegin();

    while (fIter != fiddleyards_.constEnd())
    {
        RSystemElementFiddleyard *fj = fIter.value();
        QStringList parts = fj->getID().split("_");

        if (tileId == parts.at(0))
        {

            fj->updateIds(idMap);
        }

        fIter++;
    }
}

void
RoadSystem::checkLinking()
{

    QMap<QString, RSystemElementRoad *>::const_iterator it = roads_.constBegin();

    while (it != roads_.constEnd())
    {
        RSystemElementRoad *road = it.value();
        RSystemElementRoad *predRoad = NULL;
        if (road->getPredecessor() && (road->getPredecessor()->getElementType() == "road"))
        {
            RSystemElementRoad *predRoad = getRoad(road->getPredecessor()->getElementId());
        }
        RSystemElementRoad *succRoad = NULL;
        if (road->getSuccessor() && (road->getSuccessor()->getElementType() == "road"))
        {
            RSystemElementRoad *succRoad = getRoad(road->getSuccessor()->getElementId());
        }

        QList<LaneSection *> laneSectionList;
        laneSectionList.append(road->getLaneSection(0.0));
        laneSectionList.append(road->getLaneSection(road->getLength()));

        for (int i = 0; i < laneSectionList.count(); i++)
        {
            LaneSection *laneSection = laneSectionList.at(i);
            QMap<int, Lane *>::const_iterator laneIt = laneSection->getLanes().constBegin();
            while (laneIt != laneSection->getLanes().constEnd())
            {
                Lane *lane = laneIt.value();
                if (lane->getPredecessor() != Lane::NOLANE)
                {
                    if (!road->getPredecessor())
                    {
                        lane->setPredecessor(Lane::NOLANE);
                    }
                    else
                    {
                        if (road->getPredecessor()->getContactPoint() == "end") // check orientation of lanes
                        {

                            if (((lane->getId() < 0) && (lane->getPredecessor() > 0)) || ((lane->getId() > 0) && (lane->getPredecessor() < 0)))
                            {
                                lane->setPredecessor(-lane->getPredecessor());
                            }

                            if (predRoad != NULL) // check type of lanes
                            {
                                LaneSection *predLaneSection = predRoad->getLaneSection(predRoad->getLength()); // compare lane types
                                if (predLaneSection->getLane(lane->getPredecessor())->getLaneType() != lane->getLaneType())
                                {
                                    lane->setPredecessor(Lane::NOLANE);
                                }
                            }
                        }
                        else if (road->getPredecessor()->getContactPoint() == "start")
                        {
                            if (((lane->getId() < 0) && (lane->getPredecessor() < 0)) || ((lane->getId() > 0) && (lane->getPredecessor() > 0)))
                            {
                                lane->setPredecessor(-lane->getPredecessor());
                            }

                            if (predRoad != NULL) // check type of lanes
                            {
                                LaneSection *predLaneSection = predRoad->getLaneSection(0.0); // compare lane types
                                if (predLaneSection->getLane(lane->getPredecessor())->getLaneType() != lane->getLaneType())
                                {
                                    lane->setPredecessor(Lane::NOLANE);
                                }
                            }
                        }
                    }
                }

                if (lane->getSuccessor() != Lane::NOLANE)
                {
                    if (!road->getSuccessor())
                    {
                        lane->setSuccessor(Lane::NOLANE);
                    }
                    else
                    {

                        if (road->getSuccessor()->getContactPoint() == "start")
                        {
                            if (((lane->getId() < 0) && (lane->getSuccessor() > 0)) || ((lane->getId() > 0) && (lane->getSuccessor() < 0)))
                            {
                                lane->setSuccessor(-lane->getSuccessor());
                            }

                            if (succRoad != NULL)
                            {
                                LaneSection *succLaneSection = predRoad->getLaneSection(0.0); // compare lane types
                                if (succLaneSection->getLane(lane->getSuccessor())->getLaneType() != lane->getLaneType())
                                {
                                    lane->setSuccessor(Lane::NOLANE);
                                }
                            }
                        }
                        else if (road->getSuccessor()->getContactPoint() == "end")
                        {
                            if (((lane->getId() < 0) && (lane->getSuccessor() < 0)) || ((lane->getId() > 0) && (lane->getSuccessor() > 0)))
                            {
                                lane->setSuccessor(-lane->getSuccessor());
                            }

                            if (succRoad != NULL)
                            {
                                LaneSection *succLaneSection = predRoad->getLaneSection(predRoad->getLength()); // compare lane types
                                if (succLaneSection->getLane(lane->getSuccessor())->getLaneType() != lane->getLaneType())
                                {
                                    lane->setSuccessor(Lane::NOLANE);
                                }
                            }
                        }
                    }
                }
//                qDebug() << "Lane Predecessor: " << lane->getPredecessor() << "Lane Successor: " << lane->getSuccessor();
                laneIt++;
            }
        }
        it++;
    }
}

//##################//
// ProjectData      //
//##################//

void
RoadSystem::setParentProjectData(ProjectData *projectData)
{
    parentProjectData_ = projectData;
    setParentElement(projectData);
    addRoadSystemChanges(RoadSystem::CRS_ProjectDataChanged);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
RoadSystem::notificationDone()
{
    roadSystemChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
RoadSystem::addRoadSystemChanges(int changes)
{
    if (changes)
    {
        roadSystemChanges_ |= changes;
        notifyObservers();
    }
}

//##################//
// Visitor Pattern  //
//##################//

/*! \brief Accepts a visitor.
*
* With autotraverse: visitor will be send to roads, fiddleyards, etc.
* Without: accepts visitor as 'this'.
*/
void
RoadSystem::accept(Visitor *visitor)
{
    visitor->visit(this);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
RoadSystem::acceptForChildNodes(Visitor *visitor)
{
    foreach (RSystemElementRoad *child, roads_)
        child->accept(visitor);

    foreach (RSystemElementController *child, controllers_)
        child->accept(visitor);

    foreach (RSystemElementJunction *child, junctions_)
        child->accept(visitor);

    foreach (RSystemElementFiddleyard *child, fiddleyards_)
        child->accept(visitor);

    foreach (RSystemElementPedFiddleyard *child, pedFiddleyards_)
        child->accept(visitor);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
RoadSystem::acceptForRoads(Visitor *visitor)
{
    foreach (RSystemElementRoad *child, roads_)
        child->accept(visitor);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
RoadSystem::acceptForControllers(Visitor *visitor)
{
    foreach (RSystemElementController *child, controllers_)
        child->accept(visitor);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
RoadSystem::acceptForJunctions(Visitor *visitor)
{
    foreach (RSystemElementJunction *child, junctions_)
        child->accept(visitor);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
RoadSystem::acceptForFiddleyards(Visitor *visitor)
{
    foreach (RSystemElementFiddleyard *child, fiddleyards_)
        child->accept(visitor);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
RoadSystem::acceptForPedFiddleyards(Visitor *visitor)
{
    foreach (RSystemElementPedFiddleyard *child, pedFiddleyards_)
        child->accept(visitor);
}
