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
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/roadsystem/sections/signalreference.hpp"
#include "src/data/roadsystem/sections/objectreference.hpp"

#include "rsystemelementroad.hpp"
#include "rsystemelementcontroller.hpp"
#include "rsystemelementjunction.hpp"
#include "rsystemelementfiddleyard.hpp"
#include "rsystemelementpedfiddleyard.hpp"
#include "rsystemelementjunctiongroup.hpp"
#include "roadlink.hpp"


#include "src/data/tilesystem/tilesystem.hpp"

#include <cmath>

// Qt //
//
#include <QVector2D>

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

	foreach(RSystemElementJunctionGroup *child, junctionGroups_)
		delete child;
}

//##################//
// RSystemElements  //
//##################//

RSystemElementRoad *
RoadSystem::getRoad(const odrID &id) const
{
    return roads_.value(id.getID(), NULL);
}

QList<RSystemElementRoad *> 
RoadSystem::getRoads(const odrID &junction) const
{
    QList<RSystemElementRoad *> roadList;

    auto it = roads_.constBegin();
    while (it != roads_.constEnd())
    {
        if (it.value()->getJunction() == junction)
        {
            roadList.append(it.value());
        }
        it++;
    }

    return roadList;
}

QList<RSystemElementRoad *>
RoadSystem::getTileRoads(const odrID &tileId) const
{
    QList<RSystemElementRoad *> tileRoads;
    auto it = roads_.constBegin();
    while (it != roads_.constEnd())
    {
        if (it.value()->getID().getTileID() == tileId.getID())
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
	if (road->getID().isInvalid())
	{
		road->setID(getID(road->getName(), odrID::ID_Road));
	}

    // Insert //
    //
    road->setRoadSystem(this);

    roads_.insert(road->getID().getID(), road);
    addRoadSystemChanges(RoadSystem::CRS_RoadChange);
}

bool
RoadSystem::delRoad(RSystemElementRoad *road)
{
    if (roads_.remove(road->getID().getID()))
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
RoadSystem::getController(const odrID &id) const
{
    return controllers_.value(id.getID(), NULL);
}

QList<RSystemElementController *>
RoadSystem::getTileControllers(const odrID &tileId) const
{
    QList<RSystemElementController *> tileControllers;
    auto it = controllers_.constBegin();
    while (it != controllers_.constEnd())
    {
        if (it.value()->getID().getTileID() == tileId.getID())
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
    QString name = controller->getName();
	if (controller->getID().isInvalid())
	{
		controller->setID(getID(name, odrID::ID_Controller));
	}
    controller->setRoadSystem(this);
    controllers_.insert(controller->getID().getID(), controller);

    addRoadSystemChanges(RoadSystem::CRS_ControllerChange);
}

bool
RoadSystem::delController(RSystemElementController *controller)
{

    if (controllers_.remove(controller->getID().getID()))
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
RoadSystem::getJunction(const odrID &id) const
{
    return junctions_.value(id.getID(), NULL);
}

QList<RSystemElementJunction *>
RoadSystem::getTileJunctions(const odrID &tileId) const
{
    QList<RSystemElementJunction *> tileJunctions;
    auto it = junctions_.constBegin();
    while (it != junctions_.constEnd())
    {
        if (it.value()->getID().getTileID() == tileId.getID())
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
	if (junction->getID().isInvalid())
	{
		if (getProjectData())
		{
			// Id //
			//
			QString name = junction->getName();
			junction->setID(getID(name, odrID::ID_Junction));
		}
	}

    // Insert //
    //
    junctions_.insert(junction->getID().getID(), junction);
    addRoadSystemChanges(RoadSystem::CRS_JunctionChange);

    junction->setRoadSystem(this);
}

bool
RoadSystem::delJunction(RSystemElementJunction *junction)
{
    
    if (junctions_.remove(junction->getID().getID()))
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

QList<RSystemElementJunctionGroup *>
RoadSystem::getTileJunctionGroups(const odrID &tileId) const
{
	QList<RSystemElementJunctionGroup *> tileJunctionGroups;
	auto it = junctionGroups_.constBegin();
	while (it != junctionGroups_.constEnd())
	{
		if (it.value()->getID().getTileID() == tileId.getID())
		{
			tileJunctionGroups.append(it.value());
		}

		it++;
	}

	return tileJunctionGroups;
}

void
RoadSystem::addJunctionGroup(RSystemElementJunctionGroup *junctionGroup)
{
    if (junctionGroup->getID().isInvalid() && getProjectData())
    {
        // Id //
        //
        QString name = junctionGroup->getName();
		junctionGroup->setID(getID(name, odrID::ID_Junction));
    }

    // Insert //
    //
    junctionGroups_.insert(junctionGroup->getID().getID(), junctionGroup);
    addRoadSystemChanges(RoadSystem::CRS_JunctionGroupChange);

    junctionGroup->setRoadSystem(this);
}

bool
RoadSystem::delJunctionGroup(RSystemElementJunctionGroup *junctionGroup)
{

    if (junctionGroups_.remove(junctionGroup->getID().getID()))
    {
        addRoadSystemChanges(RoadSystem::CRS_JunctionGroupChange);

		junctionGroup->clearReferences();
        junctionGroup->setRoadSystem(NULL);
        return true;
    }
    else
    {
        qDebug("WARNING 1007081034! Delete junctionGroup not successful!");
        return false;
    }
}

RSystemElementFiddleyard *
RoadSystem::getFiddleyard(const odrID &id) const
{
    return fiddleyards_.value(id.getID(), NULL);
}

QList<RSystemElementFiddleyard *>
RoadSystem::getTileFiddleyards(const odrID &tileId) const
{
    QList<RSystemElementFiddleyard *> tileFiddleyards;
    auto it = fiddleyards_.constBegin();
    while (it != fiddleyards_.constEnd())
    {

		if (it.value()->getID().getTileID() == tileId.getID())
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
	if (fiddleyard->getID().isInvalid())
	{
		QString name = fiddleyard->getName();
		fiddleyard->setID(getID(name, odrID::ID_Fiddleyard));
	}
    fiddleyard->setRoadSystem(this);
}

bool
RoadSystem::delFiddleyard(RSystemElementFiddleyard *fiddleyard)
{

    if (fiddleyards_.remove(fiddleyard->getID().getID()))
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
RoadSystem::getPedFiddleyard(const odrID &id) const
{
    return pedFiddleyards_.value(id.getID(), NULL);
}

QList<RSystemElementPedFiddleyard *>
RoadSystem::getTilePedFiddleyards(const odrID &tileId) const
{
    QList<RSystemElementPedFiddleyard *> tilePedFiddleyards;
    auto it = pedFiddleyards_.constBegin();
    while (it != pedFiddleyards_.constEnd())
    {
        if (it.value()->getID().getTileID() == tileId.getID())
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
	if (fiddleyard->getID().isInvalid())
	{
		QString name = fiddleyard->getName();
		fiddleyard->setID(getID(name, odrID::ID_PedFiddleyard));
	}
	fiddleyard->setRoadSystem(this);
}

bool
RoadSystem::delPedFiddleyard(RSystemElementPedFiddleyard *fiddleyard)
{

    if (pedFiddleyards_.remove(fiddleyard->getID().getID()))
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



// The ID is already unique
//
void
RoadSystem::changeUniqueId(RSystemElement *element, const odrID &newId)
{
	if( (dynamic_cast<RSystemElementRoad *>(element)!=NULL) &&  (roads_.contains(element->getID().getID())) )
    {
        roads_.remove(element->getID().getID());
        RSystemElementRoad *road = static_cast<RSystemElementRoad *>(element);

        if (road->getJunction().isInvalid())
        {
            RoadLink *predecessor = road->getPredecessor();

            if (predecessor)
            {
                if (predecessor->getElementType() == "road")
                {
					RSystemElementRoad *predecessorRoad = getRoad(predecessor->getElementId());
					if (predecessor->getContactPoint() == JunctionConnection::JCP_END)
					{
						RoadLink *predSuccessor = predecessorRoad->getSuccessor();
						RoadLink *roadLink = new RoadLink("road", newId, predSuccessor->getContactPoint());
						predecessorRoad->setSuccessor(roadLink);
					}
					else
					{
						RoadLink *predPredecessor = predecessorRoad->getPredecessor();
						RoadLink *roadLink = new RoadLink("road", newId, predPredecessor->getContactPoint());
						predecessorRoad->setPredecessor(roadLink);
					}
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

                    QMap<odrID, odrID> idChanged;
                    idChanged.insert(element->getID(), newId); 
                    junction->checkConnectionIds(idChanged);
                }
            }

            RoadLink *successor = road->getSuccessor();
            if (successor)
            {
                if (successor->getElementType() == "road")
                {
					RSystemElementRoad *successorRoad = getRoad(successor->getElementId());
					if (successor->getContactPoint() == JunctionConnection::JCP_START)
					{
                    RoadLink *succPredecessor = successorRoad->getPredecessor();
                    RoadLink *roadLink = new RoadLink("road", newId, succPredecessor->getContactPoint());
                    successorRoad->setPredecessor(roadLink);
					}
					else
					{
						RoadLink *succSuccessor = successorRoad->getSuccessor();
						RoadLink *roadLink = new RoadLink("road", newId, succSuccessor->getContactPoint());
						successorRoad->setSuccessor(roadLink);
					}
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

                    QMap<odrID, odrID> idChanged;
                    idChanged.insert(element->getID(), newId);
                    junction->checkConnectionIds(idChanged);
                }
            }
        }

        else // Connecting road
        {
			QMap<odrID, odrID> idChanged;
			idChanged.insert(element->getID(), newId);
            getJunction(road->getJunction())->checkConnectionIds(idChanged);
        }

        element->setID(newId);
        roads_.insert(newId.getID(), static_cast<RSystemElementRoad *>(element));

        addRoadSystemChanges(RoadSystem::CRS_RoadChange);
	}
    else if (junctions_.contains(element->getID().getID()))
    {
        junctions_.remove(element->getID().getID());
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
        junctions_.insert(newId.getID(), static_cast<RSystemElementJunction *>(element));

        addRoadSystemChanges(RoadSystem::CRS_JunctionChange);
    }
    else if (controllers_.contains(element->getID().getID()))
    {
        controllers_.remove(element->getID().getID());
        element->setID(newId);
        controllers_.insert(newId.getID(), static_cast<RSystemElementController *>(element));

        addRoadSystemChanges(RoadSystem::CRS_ControllerChange);
    }
    else if (fiddleyards_.contains(element->getID().getID()))
    {
        fiddleyards_.remove(element->getID().getID());
        element->setID(newId);
        fiddleyards_.insert(newId.getID(), static_cast<RSystemElementFiddleyard *>(element));

        addRoadSystemChanges(RoadSystem::CRS_FiddleyardChange);
    }
    else if (pedFiddleyards_.contains(element->getID().getID()))
    {
        pedFiddleyards_.remove(element->getID().getID());
        element->setID(newId);
        pedFiddleyards_.insert(newId.getID(), static_cast<RSystemElementPedFiddleyard *>(element));

        addRoadSystemChanges(RoadSystem::CRS_PedFiddleyardChange);
    }
}

odrID RoadSystem::getID(const QString &name, odrID::IDType t)
{
	if (getProjectData() && getProjectData()->getTileSystem()->getCurrentTile())
	{
		return odrID(getProjectData()->getTileSystem()->getCurrentTile()->uniqueID(t), getProjectData()->getTileSystem()->getCurrentTile()->getID().getID(), name, t);
	}
	return odrID(uniqueID(), 0, name, t);
}
odrID RoadSystem::getID(int32_t tileID, odrID::IDType t)
{
	return odrID(uniqueID(), tileID, "unknown",t);
}

odrID RoadSystem::getID(odrID::IDType t)// creates a unique ID with name unknown in current Tile
{
	
	return getID("unknown",t);
}

odrID RoadSystem::getID(int32_t ID, int32_t tileID, QString &name, odrID::IDType t)
{
	return odrID(ID,tileID,name,t);
}

int RoadSystem::uniqueID()
{
	static int lastID = 0;
	while (allIDs.contains(lastID))
		lastID++;
	allIDs.insert(lastID);
	return lastID;
}

void
RoadSystem::StringToNumericalIDs(const QMap<odrID, odrID> &idMap)
{
    //Roads //
    //
/*    odrID tileId = getProjectData()->getTileSystem()->getCurrentTile()->getID();
    auto it = roads_.constBegin();

    while (it != roads_.constEnd())
    {
        RSystemElementRoad *road = it.value();

		if (road->getID().getTileID() == tileId.getID())
		{

			RoadLink *predecessor = road->getPredecessor();

			if (predecessor != NULL)
			{
				odrID id = predecessor->getElementId();
				QString type = predecessor->getElementType();
				odrID newId = getID(idMap, id, type);

				if (newId != id)
				{
					RoadLink *roadLink = new RoadLink(type, newId, predecessor->getContactPoint());
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
				QString id = successor->getElementId();
				QString type = successor->getElementType();
				QString newId = getNewId(idMap, id, type);

				if (newId != id)
				{
					RoadLink *roadLink = new RoadLink(type, newId, successor->getContactPoint());
					road->setSuccessor(roadLink);
				}
				else
				{
					//                    qDebug() << "Road " << road->getID() << " Successor " << successor->getElementId() << " has the old ID!";
				}
			}

			odrID junction = road->getJunction();

			if (junction.isValid())
			{
				QString newId = getNewId(idMap, junction, "junction");
				if (junction != newId)
				{
					road->setJunction(newId);
				}
				else
				{
					//                    qDebug() << "Road " << road->getID() << " Junction " << junction << " has the old ID!";
				}
			}
		}

		// SignalReferences //
		//
		QMap<double, SignalReference *>::const_iterator signalRefIt = road->getSignalReferences().constBegin();

		while (signalRefIt != road->getSignalReferences().constEnd())
		{
			SignalReference *entry = signalRefIt.value();

			QString signalId = entry->getReferenceId();

			QString newId = getNewId(idMap, signalId, "signal");
			if (signalId != newId)
			{
				entry->setReferenceId(newId);
			}

			signalRefIt++;
		}

		// ObjectReferences //
		//
		QMap<double, ObjectReference *>::const_iterator objectRefIt = road->getObjectReferences().constBegin();

		while (objectRefIt != road->getObjectReferences().constEnd())
		{
			ObjectReference *entry = objectRefIt.value();

			QString objectId = entry->getReferenceId();

			QString newId = getNewId(idMap, objectId, "object");
			if (objectId != newId)
			{
				entry->setReferenceId(newId);
			}

			objectRefIt++;
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

	// JunctionGroups //
	//

	QMap<QString, RSystemElementJunctionGroup *>::const_iterator juncIt = junctionGroups_.constBegin();
	
	while (juncIt != junctionGroups_.constEnd())
	{
		RSystemElementJunctionGroup *junctionGroup = juncIt.value();
		QList<QString> references = junctionGroup->getJunctionReferences();

		for (int i = 0; i < references.size(); i++)
		{
			QString referenceId = references.at(i);
			QString newId = getNewId(idMap, referenceId, "junction");
			if (newId != referenceId)
			{
				junctionGroup->delJunction(referenceId);
				junctionGroup->addJunction(newId);
			}
		}

		juncIt++;
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
			QString signalId = entry->getSignalId();

			QString newId = getNewId(idMap, signalId, "signal");
			if (signalId != newId)
			{
				entry->setSignalId(newId);
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
    }*/
}

void 
RoadSystem::updateControllers()
{
     auto controlIt = controllers_.constBegin();

     while (controlIt != controllers_.constEnd())
     {
         RSystemElementController *controller = controlIt.value();
         for (int i = 0; i < controller->getControlEntries().size(); i++)
         {
             ControlEntry *control = controller->getControlEntries().at(i);
             auto iter = roads_.constBegin();
             while (iter != roads_.constEnd())
             {
                 Signal * signal = iter.value()->getSignal(control->getSignalId());
                 if (signal)
                 {
                     controller->addControlEntry(control, signal);
                     break;
                 }

                 iter++;
             }
         }
         controlIt++;
     }
}

RSystemElementRoad *
RoadSystem::findClosestRoad(const QPointF &to, double &s, double &t, QVector2D &vec)
{

	if (roads_.count() < 1)
	{
		return NULL;
	}

	auto it = roads_.constBegin();
	RSystemElementRoad *road = it.value();
	s = road->getSFromGlobalPoint(to, 0.0, road->getLength());
	vec = QVector2D(road->getGlobalPoint(s) - to);
	t = vec.length();

	while (++it != roads_.constEnd())
	{
		RSystemElementRoad *newRoad = it.value();
		double newS = newRoad->getSFromGlobalPoint(to, 0.0, newRoad->getLength());
		QVector2D newVec = QVector2D(newRoad->getGlobalPoint(newS) - to);
		double dist = newVec.length();

		if (dist < t)
		{
			road = newRoad;
			t = dist;
			s = newS;
			vec = newVec;
		}
	}

	QVector2D normal = road->getGlobalNormal(s);

	double skalar = QVector2D::dotProduct(normal.normalized(), vec.normalized());
	if (std::abs(skalar) < 1.0 - NUMERICAL_ZERO3) 
	{
		t = 0;
	}
	else if (skalar < 0)
	{
		t = -t;
	}

	return road;
}

//##################//
// OpenDRIVEData      //
//##################//
void
RoadSystem::verify()
{
    foreach (RSystemElementRoad * road, roads_)
    {
        road->verifyLaneLinkage();
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

	foreach(RSystemElementJunctionGroup *child, junctionGroups_)
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

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
RoadSystem::acceptForJunctionGroups(Visitor *visitor)
{
	foreach(RSystemElementJunctionGroup *child, junctionGroups_)
		child->accept(visitor);
}
