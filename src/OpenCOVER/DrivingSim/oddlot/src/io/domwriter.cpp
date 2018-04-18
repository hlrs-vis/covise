/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   05.03.2010
**
**************************************************************************/

#include "domwriter.hpp"

#include "QDomDocument"
#include "QFile"
#include "QFileInfo"

#include "src/data/projectdata.hpp"
#include "src/data/tilesystem/tilesystem.hpp"
#include "src/data/tilesystem/tile.hpp"

// RoadSystem //
//
#include "src/data/roadsystem/roadsystem.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementcontroller.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/rsystemelementfiddleyard.hpp"
#include "src/data/roadsystem/rsystemelementpedfiddleyard.hpp"
#include "src/data/roadsystem/rsystemelementjunctiongroup.hpp"

#include "src/data/roadsystem/roadlink.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"

#include "src/data/roadsystem/sections/typesection.hpp"
#include "src/data/roadsystem/sections/surfacesection.hpp"

#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementspiral.hpp"
#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"
#include "src/data/roadsystem/track/trackelementpoly3.hpp"
#include "src/data/roadsystem/track/trackelementcubiccurve.hpp"

#include "src/data/roadsystem/sections/elevationsection.hpp"
#include "src/data/roadsystem/sections/superelevationsection.hpp"
#include "src/data/roadsystem/sections/crossfallsection.hpp"
#include "src/data/roadsystem//sections/shapesection.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"
#include "src/data/roadsystem/sections/laneoffset.hpp"
#include "src/data/roadsystem/sections/laneborder.hpp"
#include "src/data/roadsystem/sections/laneroadmark.hpp"
#include "src/data/roadsystem/sections/lanespeed.hpp"
#include "src/data/roadsystem/sections/laneheight.hpp"
#include "src/data/roadsystem/sections/lanerule.hpp"
#include "src/data/roadsystem/sections/laneaccess.hpp"

#include "src/data/roadsystem/sections/crosswalkobject.hpp"
#include "src/data/roadsystem/sections/objectobject.hpp"
#include "src/data/roadsystem/sections/objectreference.hpp"
#include "src/data/roadsystem/sections/parkingspaceobject.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/roadsystem/sections/signalreference.hpp"
#include "src/data/roadsystem/sections/sensorobject.hpp"
#include "src/data/roadsystem/sections/bridgeobject.hpp"
#include "src/data/roadsystem/sections/tunnelobject.hpp"

#include "src/data/georeference.hpp"

// VehicleSystem //
//
#include "src/data/vehiclesystem/vehiclesystem.hpp"
#include "src/data/vehiclesystem/vehiclegroup.hpp"
#include "src/data/vehiclesystem/roadvehicle.hpp"
#include "src/data/vehiclesystem/poolvehicle.hpp"
#include "src/data/vehiclesystem/pool.hpp"
#include "src/data/vehiclesystem/carpool.hpp"

// PedestrianSystem //
//
#include "src/data/pedestriansystem/pedestriansystem.hpp"
#include "src/data/pedestriansystem/pedestriangroup.hpp"
#include "src/data/pedestriansystem/pedestrian.hpp"

// ScenerySystem //
//
#include "src/data/scenerysystem/scenerysystem.hpp"
#include "src/data/scenerysystem/scenerymap.hpp"
#include "src/data/scenerysystem/heightmap.hpp"
#include "src/data/scenerysystem/scenerytesselation.hpp"

// SignalManager //
//
#include "src/data/signalmanager.hpp"

// ProjectWidget //
//
#include "src/gui/projectwidget.hpp"

#include "src/gui/exportsettings.hpp"

// MainWindow //
//
#include "src/mainwindow.hpp"

// Utils //
//
#include "math.h"

DomWriter::DomWriter(ProjectData *projectData)
    : projectData_(projectData)
{

	tileSystem_ = projectData_->getTileSystem();
    doc_ = new QDomDocument();
}

/** Run this visitor through the roadsystem.
*/
void
DomWriter::runToTheHills()
{
    // <?xml?> //
    //
	exportIDvar = ExportSettings::instance()->ExportIDVariant();
    setlocale(LC_NUMERIC, "C");
    QDomNode xml = doc_->createProcessingInstruction("xml", "version=\"1.0\"");
    doc_->appendChild(xml);

    // <OpenDRIVE> //
    //
    root_ = doc_->createElement("OpenDRIVE");
    doc_->appendChild(root_);

    // <header> //
    //
    header_ = doc_->createElement("header");
	header_.setAttribute("revMajor", projectData_->getRevMajor());
	header_.setAttribute("revMinor", projectData_->getRevMinor());
	header_.setAttribute("name", projectData_->getName());
	header_.setAttribute("version", projectData_->getVersion());
	header_.setAttribute("date", projectData_->getDate());
	header_.setAttribute("north", projectData_->getNorth());
	header_.setAttribute("south", projectData_->getSouth());
	header_.setAttribute("east", projectData_->getEast());
	header_.setAttribute("west", projectData_->getWest());

	root_.appendChild(header_);
	tileSystem_->write(doc_,root_);

	GeoReference *geoReferenceParams = projectData_->getGeoReference();
	if (geoReferenceParams)
	{
		geoReferenceParams->accept(this);
	}


    // Run for your life //
    //
    projectData_->getRoadSystem()->accept(this);
    projectData_->getVehicleSystem()->accept(this);
    projectData_->getPedestrianSystem()->accept(this);
    projectData_->getScenerySystem()->accept(this);
}

void
DomWriter::visit(RoadSystem *roadSystem)
{
    // Child Nodes //
    //
    roadSystem->acceptForChildNodes(this);
}

QString DomWriter::getIDString(const odrID &ID, const QString &name)
{
	// check if this ID has already been written
	QString IDS = writtenIDs[ID.getType()].value(ID, "");
	if (IDS != "")
	{
		return IDS;
	}
	if (ID.getID() == -1)
		return("-1");
	
	if(exportIDvar == ExportSettings::EXPORT_ORIGINAL)
	{
		//write original ID if possible, otherwise create a unique ID based on the original one
		IDS = ID.getName();
		if (IDS == "")
		{
			IDS = name;
		}
		QString TileID;
		if (ID.getTileID() > 0)
		{
			TileID = "_" + QString::number(ID.getTileID());
		}
		TileID += "_" + QString::number(ID.getID());
		int counter = 0;
		while (writtenIDStrings[ID.getType()].contains(IDS))
		{
			if (counter == 0)
				IDS = ID.getName() + TileID;
			else
				IDS = ID.getName() + TileID + "_" + QString::number(counter);
			counter++;
		}
	}
	else if (exportIDvar == ExportSettings::EXPORT_NUMERICAL)
	{
		//write original ID if possible, otherwise create a unique ID based on the original one
		IDS = QString::number(ID.getID());

		if (writtenIDStrings[ID.getType()].contains(IDS))
		{
			IDS = QString::number(ID.getTileID()*1000.0 + ID.getID());
			int counter = 0;
			while (writtenIDStrings[ID.getType()].contains(IDS))
			{

				IDS = QString::number(ID.getTileID()*counter*(1000.0) + ID.getID());
				counter++;
			}
		}
	}
	else if (exportIDvar == ExportSettings::EXPORT_TILE_ID)
	{
		//write original ID if possible, otherwise create a unique ID based on the original one
		IDS = ID.getName();
		if (IDS == "")
		{
			IDS = name;
		}
		QString TileID;
		if (ID.getTileID() > 0)
		{
			TileID = "_" + QString::number(ID.getTileID());
		}
		TileID += "_" + QString::number(ID.getID());
		int counter = 0;
		while (writtenIDStrings[ID.getType()].contains(IDS))
		{
			if (counter == 0)
				IDS = ID.getName() + TileID;
			else
				IDS = ID.getName() + TileID + "_" + QString::number(counter);
			counter++;
		}
	}
	writtenIDs[ID.getType()].insert(ID,IDS);
	writtenIDStrings[ID.getType()].insert(IDS);
	return IDS;
}

void
DomWriter::visit(RSystemElementRoad *road)
{
    // <road> //
    //
    currentRoad_ = doc_->createElement("road");
    currentRoad_.setAttribute("name", road->getName());
    currentRoad_.setAttribute("length", road->getLength());
    currentRoad_.setAttribute("id", getIDString(road->getID(),road->getName()));
    currentRoad_.setAttribute("junction", getIDString(road->getJunction(), ""));
	addTileInfo(currentRoad_, road->getID().getTileID());
    root_.appendChild(currentRoad_);

    // <road><link><predecessor/successor> //
    //
    RoadLink *pred = road->getPredecessor();
    RoadLink *succ = road->getSuccessor();
    if (pred || succ)
    {
        QDomElement linkElement = doc_->createElement("link");
        currentRoad_.appendChild(linkElement);

        if (pred)
        {
            QDomElement predElement = doc_->createElement("predecessor");
            predElement.setAttribute("elementType", pred->getElementType());
            predElement.setAttribute("elementId", getIDString(pred->getElementId(), ""));
			if (pred->getElementType() != "junction")
			{
				predElement.setAttribute("contactPoint", JunctionConnection::parseContactPointBack(pred->getContactPoint()));
			}
            linkElement.appendChild(predElement);
        }

        if (succ)
        {
            QDomElement succElement = doc_->createElement("successor");
            succElement.setAttribute("elementType", succ->getElementType());
            succElement.setAttribute("elementId", getIDString(succ->getElementId(), ""));
			if(succ->getElementType()!="junction")
			{
				succElement.setAttribute("contactPoint", JunctionConnection::parseContactPointBack(succ->getContactPoint()));
			}
            linkElement.appendChild(succElement);
        }
    }

    // <road><planView> //
    //
    currentPVElement_ = doc_->createElement("planView");
    currentRoad_.appendChild(currentPVElement_);

    // <road><elevationProfile> //
    //
    if (!road->getElevationSections().isEmpty())
    {
        currentElevationProfileElement_ = doc_->createElement("elevationProfile");
        currentRoad_.appendChild(currentElevationProfileElement_);
    }

    // <road><lateralProfile> //
    //
    if (!road->getSuperelevationSections().isEmpty() || !road->getCrossfallSections().isEmpty() || !road->getShapeSections().isEmpty())
    {
        currentLateralProfileElement_ = doc_->createElement("lateralProfile");
        currentRoad_.appendChild(currentLateralProfileElement_);
    }

    // <road><lanes> //
    //
    currentLanesElement_ = doc_->createElement("lanes");
    currentRoad_.appendChild(currentLanesElement_);

    // <road><objects><crosswalk> //
    //TODO: other objects??
    if (!road->getCrosswalks().isEmpty() || !road->getObjects().isEmpty() || !road->getBridges().isEmpty())
    {
        currentObjectsElement_ = doc_->createElement("objects");
        currentRoad_.appendChild(currentObjectsElement_);

	signalManager_ = projectData_->getProjectWidget()->getMainWindow()->getSignalManager();
    }
    if (!road->getSignals().isEmpty())
    {
        currentSignalsElement_ = doc_->createElement("signals");
        currentRoad_.appendChild(currentSignalsElement_);
    }

    if (!road->getSensors().isEmpty())
    {
        currentSensorsElement_ = doc_->createElement("sensors");
        currentRoad_.appendChild(currentSensorsElement_);
    }

    // Child Nodes //
    //
    road->acceptForChildNodes(this);
}

//################//
// OBJECT         //
//################//

void
DomWriter::visit(Object *object)
{
    if (object->getType().contains("625")) // conversion to OpenDRIVE 1.4
	{
        QStringList parts = object->getType().split("-");

        bool number = false;
		QString type = parts.at(0);
        QString subclass = "";
        QString subtype = "-1";
       
        if (number && (parts.size() > 1))
        {
            subtype = parts.at(1);
            if (!number)
            {
                subtype = "-1";
            }
            if (parts.size() > 2)
            {
                subclass = parts.at(2);
            }
        }
        
        double length = 0.0;
        double s = object->getSStart();
        RoadSystem * roadSystem = projectData_->getRoadSystem();
        bool loadObject = false;
        if (object->getType() == "625-10-20")
        {
            loadObject = true;
        }

        double radHeading = object->getHeading() / 180.0 * (M_PI);

        do
        {
            Signal::OrientationType orientation = object->getOrientation();
            

            int fromLane = 0;
            int toLane = 0;
            if (orientation == Signal::OrientationType::NEGATIVE_TRACK_DIRECTION)
            {
                fromLane = object->getParentRoad()->getLaneSection(object->getSStart())->getRightmostLaneId();
            }
            else if (orientation == Signal::OrientationType::POSITIVE_TRACK_DIRECTION)
            {
                toLane = object->getParentRoad()->getLaneSection(object->getSStart())->getLeftmostLaneId();
            }
            else
            {
                fromLane = object->getParentRoad()->getLaneSection(object->getSStart())->getRightmostLaneId();
                toLane = object->getParentRoad()->getLaneSection(object->getSStart())->getLeftmostLaneId();

            }

            QString name = object->getName();
            odrID id = roadSystem->getID(name,odrID::ID_Object);

            Signal * signal = new Signal(id, "", s, object->getT(), false, (Signal::OrientationType)orientation, object->getzOffset(), "Germany", type, subclass, subtype, 0.0, object->getHeading(), object->getPitch(), object->getRoll(), "km/h", "", object->getWidth(), object->getHeight(), object->getPole(), 2, fromLane, toLane, 0, 0);
            

            if (!currentSignalsElement_.isElement())
            {
                currentSignalsElement_ = doc_->createElement("signals");
                currentRoad_.appendChild(currentSignalsElement_);
            }
            visit(signal);


            s += object->getRepeatDistance();
            length += object->getRepeatDistance();
        }while (length < object->getRepeatLength());

    }   
    else if (object->getType() != "")
    {
        QDomElement objectElement = doc_->createElement("object");

		addTileInfo(objectElement, object->getId().getTileID());

        if (object->getRepeatLength() > NUMERICAL_ZERO3)
		{
            QDomElement repeatElement = doc_->createElement("repeat");

            repeatElement.setAttribute("s", object->getRepeatS());
            if (object->getRepeatS() + object->getRepeatLength() > object->getParentRoad()->getLength()) // TODO: check this in the settings
            {
                object->setRepeatLength(object->getParentRoad()->getLength() - object->getRepeatS() - NUMERICAL_ZERO8);
            }
            repeatElement.setAttribute("length", object->getRepeatLength());
            repeatElement.setAttribute("distance", object->getRepeatDistance());
			repeatElement.setAttribute("tStart", object->getRepeatTStart());
			repeatElement.setAttribute("tEnd", object->getRepeatTEnd());
			repeatElement.setAttribute("widthStart", object->getRepeatWidthStart());
			repeatElement.setAttribute("widthEnd", object->getRepeatWidthEnd());
			repeatElement.setAttribute("heightStart", object->getRepeatHeightStart());
			repeatElement.setAttribute("heightEnd", object->getRepeatHeightEnd());
			repeatElement.setAttribute("zOffsetStart", object->getRepeatZOffsetStart());
			repeatElement.setAttribute("zOffsetEnd", object->getRepeatZOffsetEnd());

            objectElement.appendChild(repeatElement);
        }

        ObjectContainer * objectContainer = signalManager_->getObjectContainer(object->getType());
        if (objectContainer)
        {
            QList<ObjectCorner *> objectCorners =  objectContainer->getObjectCorners();
            if (objectCorners.size() > 0)
            {
                QDomElement outlineElement = doc_->createElement("outline");

                for (int i = 0; i < objectCorners.size(); i++)
                {
                    QDomElement cornerElement = doc_->createElement("cornerLocal");
                    ObjectCorner *corner = objectCorners.at(i);

                    cornerElement.setAttribute("height", corner->getHeight());
                    cornerElement.setAttribute("z", corner->getZ());
                    cornerElement.setAttribute("v", corner->getV());
                    cornerElement.setAttribute("u", corner->getU());
                    outlineElement.appendChild(cornerElement);
                }
                objectElement.appendChild(outlineElement);
            }
        }
        else
        {
            qDebug() << "Domwriter: Error! Prototype of Object with type " << object->getType() << " not in signs.xml";
        }

        // Set mandatory attributes
        objectElement.setAttribute("id", getIDString(object->getId(), object->getName()));
        //Texture and model file are ancillary data
        //
        QDomElement userData;

        if (object->getTextureFileName() != "")
        {
            userData = doc_->createElement("userData");
            userData.setAttribute("code", "textureFile");
            if (object->getTextureFileName().contains("/"))
            {
                userData.setAttribute("value", object->getTextureFileName());
            }
            else
            {
                userData.setAttribute("value", "maps/" + object->getTextureFileName());
            }

            objectElement.appendChild(userData);
        }


        if (objectContainer && (objectContainer->getObjectFile() != "")) 
        {
            userData = doc_->createElement("userData");

            userData.setAttribute("code", "modelFile");
            if (objectContainer->getObjectFile().contains("/"))
            {
                userData.setAttribute("value", objectContainer->getObjectFile());
            }
            else
            {
                userData.setAttribute("value", "objects/" + objectContainer->getObjectFile());
            }


            objectElement.appendChild(userData);
            objectElement.setAttribute("name", object->getName());
        }
        else 
        {
            QString objectName = object->getType();

            if ((object->getName() != "") && (object->getName() != "unnamed"))
            {
                objectName += "_" + object->getName();
            }

            if (object->getPole())
            {
                objectElement.setAttribute("name", objectName + "_p");
            }
            else
            {
                objectElement.setAttribute("name", objectName);
            }
        }

        objectElement.setAttribute("type", object->getType());
        objectElement.setAttribute("s", object->getSStart());
        objectElement.setAttribute("t", object->getT());
        objectElement.setAttribute("zOffset", object->getzOffset());
        objectElement.setAttribute("validLength", object->getValidLength());

		objectElement.setAttribute("orientation", Signal::parseOrientationTypeBack(object->getOrientation()));
        objectElement.setAttribute("length", object->getLength());
        objectElement.setAttribute("width", object->getWidth());
        objectElement.setAttribute("radius", object->getRadius());
        objectElement.setAttribute("height", object->getHeight());
        objectElement.setAttribute("hdg", object->getHeading() / 180.0 * (M_PI));
        objectElement.setAttribute("pitch", object->getPitch() / 180.0 * (M_PI));
        objectElement.setAttribute("roll", object->getRoll() / 180.0 * (M_PI));

		ParkingSpace *parkingSpace = object->getParkingSpace();
		if (parkingSpace)
		{
			QDomElement parking = doc_->createElement("parkingSpace");
			parking.setAttribute("access", ParkingSpace::parseParkingSpaceAccessBack(parkingSpace->getAccess()));
			parking.setAttribute("restrictions", parkingSpace->getRestrictions());

			int entries = parkingSpace->getMarkingsSize();
			if (entries > 0)
			{
				QString side, type, color;
				double width;
				for (int i = 0; i < entries; i++)
				{
					QDomElement parkingMarking = doc_->createElement("marking");

					parkingSpace->getMarking(i, side, type, width, color);
					parkingMarking.setAttribute("side", side);
					parkingMarking.setAttribute("type", type);
					parkingMarking.setAttribute("width", width);
					parkingMarking.setAttribute("color", color);
					parking.appendChild(parkingMarking);
				}
			}

			objectElement.appendChild(parking);
		}

		Outline *outline = object->getOutline();
		if (outline)
		{
			QList<ObjectCorner *> corners = outline->getCorners();
			if (!corners.isEmpty())
			{
				QDomElement outlineElement = doc_->createElement("outline");
				foreach (ObjectCorner *corner, corners)
				{
					QDomElement cornerElement;
					if (corner->getLocal())
					{
						cornerElement = doc_->createElement("cornerLocal");
						cornerElement.setAttribute("u", corner->getU());
						cornerElement.setAttribute("v", corner->getV());
						cornerElement.setAttribute("z", corner->getZ());
						cornerElement.setAttribute("height", corner->getHeight());
					}
					else
					{
						cornerElement = doc_->createElement("cornerRoad");
						cornerElement.setAttribute("u", corner->getU());
						cornerElement.setAttribute("v", corner->getV());
						cornerElement.setAttribute("dz", corner->getZ());
						cornerElement.setAttribute("height", corner->getHeight());
					}
					outlineElement.appendChild(cornerElement);
				}
				objectElement.appendChild(outlineElement);
			}

		}

        currentObjectsElement_.appendChild(objectElement);
    }
}

//################//
// SIGNALREFERENCE     //
//################//

void
DomWriter::visit(ObjectReference *objectReference)
{

	QDomElement objectReferenceElement = doc_->createElement("objectReference");

	objectReferenceElement.setAttribute("s", objectReference->getSStart());
	objectReferenceElement.setAttribute("t", objectReference->getReferenceT());
	objectReferenceElement.setAttribute("id", getIDString(objectReference->getReferenceId(), ""));
	objectReferenceElement.setAttribute("zOffset", objectReference->getReferenceZOffset());
	objectReferenceElement.setAttribute("validLength", objectReference->getReferenceValidLength());
	objectReferenceElement.setAttribute("orientation", Signal::parseOrientationTypeBack(objectReference->getReferenceOrientation()));

	addTileInfo(objectReferenceElement, objectReference->getId().getTileID());

	foreach(Signal::Validity validity, objectReference->getValidityList())
	{
		QDomElement validityElement = doc_->createElement("validity");

		validityElement.setAttribute("fromLane", validity.fromLane);
		validityElement.setAttribute("toLane", validity.toLane);

		objectReferenceElement.appendChild(validityElement);
	}

	currentObjectsElement_.appendChild(objectReferenceElement);
}


//################//
// BRIDGE       //
//################//

void
DomWriter::visit(Bridge *bridge)
{

    QDomElement bridgeElement;

    bridgeElement = doc_->createElement("bridge");

    // Set mandatory attributes
    bridgeElement.setAttribute("id", getIDString(bridge->getId(), bridge->getName()));

	addTileInfo(bridgeElement, bridge->getId().getTileID());

    QString bridgeName;

    if (bridge->getName() != "unnamed")
    {
        bridgeName = bridge->getName();
    }

    bridgeElement.setAttribute("name", bridgeName);
    bridgeElement.setAttribute("s", bridge->getSStart());
    bridgeElement.setAttribute("length", bridge->getLength());

    QString typeName = "concrete";
    if (bridge->getType() == 1)
    {
        typeName = "steel";
    }
    else if (bridge->getType() == 2)
    {
        typeName = "brick";
    }
    else if (bridge->getType() == 3)
    {
        typeName = "wood";
    }

    bridgeElement.setAttribute("type", typeName);
    // model file are ancillary data
    //
    if (bridge->getFileName() != "")
    {
        QDomElement userData = doc_->createElement("userData");

        userData.setAttribute("code", "filename");
        if (bridge->getFileName().contains("/"))
        {
            userData.setAttribute("value", bridge->getFileName());
        }
        else
        {
            userData.setAttribute("value", "objects/" + bridge->getFileName());
        }

        bridgeElement.appendChild(userData);
    }

    currentObjectsElement_.appendChild(bridgeElement);
}


//################//
// TUNNEL       //
//################//

void
DomWriter::visit(Tunnel *tunnel)
{

    QDomElement tunnelElement;
        tunnelElement = doc_->createElement("tunnel");

        // Set mandatory attributes
        tunnelElement.setAttribute("id", getIDString(tunnel->getId(), tunnel->getName()));

        tunnelElement.setAttribute("name", tunnel->getName());

        tunnelElement.setAttribute("s", tunnel->getSStart());
        tunnelElement.setAttribute("length", tunnel->getLength());

        tunnelElement.setAttribute("lighting", tunnel->getLighting());
		tunnelElement.setAttribute("daylight", tunnel->getDaylight());
        tunnelElement.setAttribute("type", Tunnel::parseTunnelTypeBack(tunnel->getType()));

		addTileInfo(tunnelElement, tunnel->getId().getTileID());
    // model file are ancillary data
    //
    if (tunnel->getFileName() != "")
    {
        QDomElement userData = doc_->createElement("userData");

        userData.setAttribute("code", "filename");
        if (tunnel->getFileName().contains("/"))
        {
            userData.setAttribute("value", tunnel->getFileName());
        }
        else
        {
            userData.setAttribute("value", "objects/" + tunnel->getFileName());
        }

        tunnelElement.appendChild(userData);
    }

    currentObjectsElement_.appendChild(tunnelElement);
}

//################//
// CROSSWALK      //
//################//

void
DomWriter::visit(Crosswalk *crosswalk)
{

    QDomElement crosswalkElement = doc_->createElement("crosswalk");

    // Set mandatory attributes
    crosswalkElement.setAttribute("id", getIDString(crosswalk->getId(), crosswalk->getName()));
    crosswalkElement.setAttribute("name", crosswalk->getName());
    crosswalkElement.setAttribute("s", crosswalk->getS());
    crosswalkElement.setAttribute("length", crosswalk->getLength());

	addTileInfo(crosswalkElement, crosswalk->getId().getTileID());

    // Set optional attributes, if they were provided
    if (crosswalk->hasCrossProb())
        crosswalkElement.setAttribute("crossProb", crosswalk->getCrossProb());
    if (crosswalk->hasResetTime())
        crosswalkElement.setAttribute("resetTime", crosswalk->getResetTime());
    if (crosswalk->hasType())
        crosswalkElement.setAttribute("type", crosswalk->getType());
    if (crosswalk->hasDebugLvl())
        crosswalkElement.setAttribute("debugLvl", crosswalk->getDebugLvl());

    // Set validity element, if it was provided
    if (crosswalk->hasFromLane() || crosswalk->hasToLane())
    {
        QDomElement validityElement = doc_->createElement("validity");
        if (crosswalk->hasFromLane())
        {
            validityElement.setAttribute("fromLane", crosswalk->getFromLane());
        }
        if (crosswalk->hasToLane())
        {
            validityElement.setAttribute("toLane", crosswalk->getToLane());
        }

        crosswalkElement.appendChild(validityElement);
    }

    currentObjectsElement_.appendChild(crosswalkElement);
}
//################//
// SIGNAL         //
//################//

void
DomWriter::visit(Signal *signal)
{

    QDomElement signalElement = doc_->createElement("signal");

    QDomElement validityElement = doc_->createElement("validity");

    validityElement.setAttribute("fromLane", signal->getValidFromLane());
    validityElement.setAttribute("toLane", signal->getValidToLane());
    signalElement.appendChild(validityElement);

    QDomElement userData = doc_->createElement("userData");

    userData.setAttribute("code", "subclass");
    userData.setAttribute("value", signal->getTypeSubclass());

    signalElement.appendChild(userData);

    userData = doc_->createElement("userData");

    userData.setAttribute("code", "size");
    userData.setAttribute("value", signal->getSize());

    signalElement.appendChild(userData);

    //Pedestrian Crossing has ancillary data
    //
    if (signal->getType() == "293")
    {
        userData = doc_->createElement("userData");

        userData.setAttribute("code", "crossprob");
        userData.setAttribute("value", signal->getCrossingProbability());

        signalElement.appendChild(userData);

        userData = doc_->createElement("userData");

        userData.setAttribute("code", "resetTime");
        userData.setAttribute("value", signal->getResetTime());

        signalElement.appendChild(userData);

    }

    // Set mandatory attributes
    signalElement.setAttribute("id", getIDString(signal->getId(), signal->getName()));

	addTileInfo(signalElement, signal->getId().getTileID());
    QString signalName; // The name has the format: type.typeSubclass-subtype_name_p

    QString type = signal->getType();
    QString subtype = signal->getSubtype();
    if ((type != "-1") && (type != "none"))
    {
        if (!signal->getTypeSubclass().isEmpty())
        {
            if ((subtype != "-1") && (subtype != "none"))
            {
                signalName = type + "." + signal->getTypeSubclass() + "-" + subtype;
            }
            else
            {
                signalName = type + "." + signal->getTypeSubclass();
            }
        }
        else if ((subtype != "-1") && (subtype != "none"))
        {
            signalName = type + "-" + subtype;
        }
        else
        {
            signalName = type;
        }
    }

    double hOffset = signal->getHeading();
    if ((type == "625") && (subtype == "10") && (signal->getTypeSubclass() == "20"))
    {
        signalName += "_" + QString("%1").arg(qRound(signal->getHeading()));
        hOffset = 0.0;
    }
    else if ((signal->getName() != "") && (signal->getName() != "unnamed"))
    {
        signalName += "_" + signal->getName();

        if (signal->getPole())
        {
            signalName += "_p";
        }
    }
    else if (signal->getPole())
    {
        signalName += "_p";
    }

    signalElement.setAttribute("name", signalName);

    QFile file;
    bool textureFile = false;
    QString textureFilename;
    QString dir = projectData_->getProjectWidget()->getMainWindow()->getCovisedir() + "/share/covise/signals/";
    if (file.exists(dir + signal->getCountry() + "/" + signalName + ".png"))
    {
        textureFile = true;
        if (signalName.contains("/"))
        {
            textureFilename += ".png";
        }
        else
        { 
            textureFilename = "signals/" +  signal->getCountry() + "/" + signalName + ".png";
        }
    }
    else if (file.exists(dir + signal->getCountry() + "/" + signalName + ".tif"))
    {
        textureFile = true;
        if (signalName.contains("/"))
        {
           textureFilename += ".tif";
        }
        else
        { 
            textureFilename = "signals/" +  signal->getCountry() + "/" + signalName + ".tif";
        }
    }

    if (textureFile)
    {
        userData = doc_->createElement("userData");

        userData.setAttribute("code", "textureFile");
        userData.setAttribute("value", textureFilename);

        signalElement.appendChild(userData);
    }

    bool modelFile = false;
    QString modelFilename;
    if (file.exists(dir + signal->getCountry() + "/" + signalName + ".osg"))
    {
        modelFile = true;
        if (signalName.contains("/"))
        {
            modelFilename += ".osg";
        }
        else
        { 
            modelFilename = "signals/" +  signal->getCountry() + "/" + signalName + ".osg";
        }
    }

    if (modelFile)
    {
        userData = doc_->createElement("userData");

        userData.setAttribute("code", "modelFile");
        userData.setAttribute("value", modelFilename);

        signalElement.appendChild(userData);
    }


    signalElement.setAttribute("s", signal->getSStart());

    // Calculate new t - temporary
    //
    /*	RSystemElementRoad * road = signal->getParentRoad();
	LaneSection * laneSection =road->getLaneSection(signal->getS());
	double t = signal->getT();
	Lane * nextLane;
	double sSection = signal->getS() - laneSection->getSStart();

	int i = 0;

	if (signal->getT() > 0)
	{
		while (nextLane = laneSection->getNextUpper(i))
		{
			t += nextLane->getWidth(sSection);
			i = nextLane->getId();
		}
	}
	else
	{
		while (nextLane = laneSection->getNextLower(i))
		{
			t -= nextLane->getWidth(sSection);
			i = nextLane->getId();
		}
	}

	signalElement.setAttribute("t", t);
*/ signalElement.setAttribute("t", signal->getT());

    if (signal->getDynamic())
        signalElement.setAttribute("dynamic", "yes");
    else
        signalElement.setAttribute("dynamic", "no");

	signalElement.setAttribute("orientation", Signal::parseOrientationTypeBack(signal->getOrientation()));
    signalElement.setAttribute("zOffset", signal->getZOffset());
    signalElement.setAttribute("country", signal->getCountry());
    signalElement.setAttribute("type", signal->getType());
    signalElement.setAttribute("subtype", signal->getSubtype());
    signalElement.setAttribute("value", signal->getValue());
    signalElement.setAttribute("hOffset", hOffset / 180.0 * (M_PI));
    signalElement.setAttribute("pitch", signal->getPitch() / 180.0 * (M_PI));
    signalElement.setAttribute("roll", signal->getRoll() / 180.0 * (M_PI));

	signalElement.setAttribute("unit", signal->getUnit());
	signalElement.setAttribute("text", signal->getText());
	signalElement.setAttribute("width", signal->getWidth());
	signalElement.setAttribute("height", signal->getHeight());

    currentSignalsElement_.appendChild(signalElement);

/*    if (signal->getPole())
    {
        // Create a simple pole object which is not used in ODDlot to stay conform with the standard //
        //
        QDomElement objectElement = doc_->createElement("object");
        QString name = "simplePole";
        QString id = projectData_->getRoadSystem()->getUniqueId("", name);
        objectElement.setAttribute("id", id);

        userData = doc_->createElement("userData");

        userData.setAttribute("code", "modelFile");
        if (signal->getZOffset() < 3.1)
        {
            userData.setAttribute("value", "objects/simplePole3.0.WRL");
        }
        else
        {
            userData.setAttribute("value", "objects/simplePole5.5.WRL");
        }

        objectElement.appendChild(userData);

        objectElement.setAttribute("type", "simplePole");
        objectElement.setAttribute("name", name);
        objectElement.setAttribute("s", signal->getSStart());
        objectElement.setAttribute("t", signal->getT());
        objectElement.setAttribute("zOffset", 0.0);
        objectElement.setAttribute("validLength", 0.0);
        if (signal->getOrientation() == Signal::NEGATIVE_TRACK_DIRECTION)
            objectElement.setAttribute("orientation", "-");
        else
            objectElement.setAttribute("orientation", "+");
        objectElement.setAttribute("length", 0.0);
        objectElement.setAttribute("width", 0.0);
        objectElement.setAttribute("radius", 0.0);
        objectElement.setAttribute("height", 0.0);
        objectElement.setAttribute("hdg", 0.0);
        objectElement.setAttribute("pitch", 0.0);
        objectElement.setAttribute("roll", 0.0);

        currentObjectsElement_.appendChild(objectElement);
    }
*/
}

//################//
// SIGNALREFERENCE     //
//################//

void
DomWriter::visit(SignalReference *signalReference)
{

	QDomElement signalReferenceElement = doc_->createElement("signalReference");

	signalReferenceElement.setAttribute("s", signalReference->getSStart());
	signalReferenceElement.setAttribute("t", signalReference->getReferenceT());
	signalReferenceElement.setAttribute("id", getIDString(signalReference->getReferenceId(), ""));
	signalReferenceElement.setAttribute("orientation", Signal::parseOrientationTypeBack(signalReference->getReferenceOrientation()));

	addTileInfo(signalReferenceElement, signalReference->getReferenceId().getTileID());
	foreach (Signal::Validity validity, signalReference->getValidityList())
	{
		QDomElement validityElement = doc_->createElement("validity");

		validityElement.setAttribute("fromLane", validity.fromLane);
		validityElement.setAttribute("toLane", validity.toLane);

		signalReferenceElement.appendChild(validityElement);
	}

	currentSignalsElement_.appendChild(signalReferenceElement);
}


//################//
// SENSOR         //
//################//

void
DomWriter::visit(Sensor *sensor)
{

    QDomElement sensorElement = doc_->createElement("sensor");

    // Set mandatory attributes
    sensorElement.setAttribute("id", sensor->getId());
    sensorElement.setAttribute("s", sensor->getS());

    currentSensorsElement_.appendChild(sensorElement);
}

//################//
// TYPESECTIONS   //
//################//

void
DomWriter::visit(TypeSection *section)
{
    QDomElement element = doc_->createElement("type");
    element.setAttribute("s", section->getSStart());
    element.setAttribute("type", TypeSection::parseRoadTypeBack(section->getRoadType()));
    if(section->getSpeedRecord()!=NULL)
    {
        QDomElement speedElement = doc_->createElement("speed");
        double speed = section->getSpeedRecord()->maxSpeed;
        if(speed == 0)
        {
            speedElement.setAttribute("max", "no limit");
        }
        else if(speed < 0)
        {
            speedElement.setAttribute("max", "undefined");
        }
        else
        {
            speedElement.setAttribute("max", QString::number(speed));
        }
		speedElement.setAttribute("unit", section->getSpeedRecord()->maxUnit);
        element.appendChild(speedElement);
	}
	currentRoad_.appendChild(element);
}

//#################//
// SURFACESECTIONS //
//#################//
void
DomWriter::visit(SurfaceSection *section)
{
	QDomElement element = doc_->createElement("surface");
	for (int i = 0; i < section->getNumCRG(); i++)
	{
		QDomElement crgElement = doc_->createElement("CRG");

		if (section->getFile(i).length() > 0)
			crgElement.setAttribute("file", section->getFile(i));
		if (section->getSStart(i).length() > 0)
			crgElement.setAttribute("sStart", section->getSStart(i));
		if (section->getSEnd(i).length() > 0)
			crgElement.setAttribute("sEnd", section->getSEnd(i));
		if (section->getOrientation(i).length() > 0)
			crgElement.setAttribute("orientation", section->getOrientation(i));
		if (section->getMode(i).length() > 0)
			crgElement.setAttribute("mode", section->getMode(i));
		if (section->getPurpose(i).length() > 0)
			crgElement.setAttribute("purpose", section->getPurpose(i));
		if (section->getSOffset(i).length() > 0)
			crgElement.setAttribute("sOffset", section->getSOffset(i));
		if (section->getTOffset(i).length() > 0)
			crgElement.setAttribute("tOffset", section->getTOffset(i));
		if (section->getZOffset(i).length() > 0)
			crgElement.setAttribute("zOffset", section->getZOffset(i));
		if (section->getZScale(i).length() > 0)
			crgElement.setAttribute("zScale", section->getZScale(i));
		if (section->getHOffset(i).length() > 0)
			crgElement.setAttribute("hOffset", section->getHOffset(i));

		element.appendChild(crgElement);
	}
	currentRoad_.appendChild(element);
}

//###################//
// ELEVATIONSECTIONS //
//###################//

void
DomWriter::visit(ElevationSection *section)
{
	QDomElement element = doc_->createElement("elevation");
	element.setAttribute("s", section->getSStart());
	element.setAttribute("a", section->getA());
	element.setAttribute("b", section->getB());
	element.setAttribute("c", section->getC());
	element.setAttribute("d", section->getD());
	currentElevationProfileElement_.appendChild(element);
}

//################//
// SUPERELVATIONSECTIONS   //
//################//

void
DomWriter::visit(SuperelevationSection *section)
{
	QDomElement element = doc_->createElement("superelevation");
	element.setAttribute("s", section->getSStart());
	element.setAttribute("a", section->getA() * M_PI / 180.0);
	element.setAttribute("b", section->getB() * M_PI / 180.0);
	element.setAttribute("c", section->getC() * M_PI / 180.0);
	element.setAttribute("d", section->getD() * M_PI / 180.0);
	currentLateralProfileElement_.appendChild(element);
}

//################//
// CROSSFALLSECTIONS   //
//################//

void
DomWriter::visit(CrossfallSection *section)
{
	QDomElement element = doc_->createElement("crossfall");
	element.setAttribute("side", CrossfallSection::parseCrossfallSideBack(section->getSide()));
	element.setAttribute("s", section->getSStart());
	element.setAttribute("a", section->getA() * M_PI / 180.0);
	element.setAttribute("b", section->getB() * M_PI / 180.0);
	element.setAttribute("c", section->getC() * M_PI / 180.0);
	element.setAttribute("d", section->getD() * M_PI / 180.0);
	currentLateralProfileElement_.appendChild(element);
}

//################//
// SHAPESECTIONS   //
//################//

void
DomWriter::visit(ShapeSection *section)
{
	if (section->getShapes().size() == 1)		// Do not write section, if there was no editing
	{
		if (section->getFirstPolynomialLateralSection()->getDegree() == -1)
		{
			return;
		}
	}

	QMap<double, PolynomialLateralSection *>::const_iterator iter = section->getShapes().constBegin();
	while (iter != section->getShapes().constEnd())
	{
		QDomElement element = doc_->createElement("shape");
		element.setAttribute("s", section->getSStart());
		element.setAttribute("t", iter.key());
		PolynomialLateralSection *poly = iter.value();
		element.setAttribute("a", poly->getA());
		element.setAttribute("b", poly->getB());
		element.setAttribute("c", poly->getC());
		element.setAttribute("d", poly->getD());
		currentLateralProfileElement_.appendChild(element);

		iter++;
	}
}

//################//
// TRACKS         //
//################//

void
DomWriter::visit(TrackSpiralArcSpiral *track)
{
    track->acceptForChildNodes(this);
}

void
DomWriter::visit(TrackElement *track)
{
    double s = track->getSStart();
    currentTrackElement_.setAttribute("s", track->getSStart());
    currentTrackElement_.setAttribute("x", track->getGlobalPoint(s).x());
    currentTrackElement_.setAttribute("y", track->getGlobalPoint(s).y());
    currentTrackElement_.setAttribute("hdg", track->getGlobalHeadingRad(s));
    currentTrackElement_.setAttribute("length", track->getLength());
}

void
DomWriter::visit(TrackElementLine *track)
{
    currentTrackElement_ = doc_->createElement("geometry");
    currentPVElement_.appendChild(currentTrackElement_);
    visit((TrackElement *)track);

    QDomElement element = doc_->createElement("line");
    currentTrackElement_.appendChild(element);
}

void
DomWriter::visit(TrackElementArc *track)
{
    currentTrackElement_ = doc_->createElement("geometry");
    currentPVElement_.appendChild(currentTrackElement_);
    visit((TrackElement *)track);

    QDomElement element = doc_->createElement("arc");
    element.setAttribute("curvature", track->getCurvature(track->getSStart()));
    currentTrackElement_.appendChild(element);
}

void
DomWriter::visit(TrackElementSpiral *track)
{
    currentTrackElement_ = doc_->createElement("geometry");
    currentPVElement_.appendChild(currentTrackElement_);
    visit((TrackElement *)track);

    QDomElement element = doc_->createElement("spiral");
    element.setAttribute("curvStart", track->getCurvature(track->getSStart()));
    element.setAttribute("curvEnd", track->getCurvature(track->getSEnd()));
    currentTrackElement_.appendChild(element);
}

void
DomWriter::visit(TrackElementPoly3 *track)
{
    currentTrackElement_ = doc_->createElement("geometry");
    currentPVElement_.appendChild(currentTrackElement_);
    visit((TrackElement *)track);

    QDomElement element = doc_->createElement("poly3");
    element.setAttribute("a", track->getA());
    element.setAttribute("b", track->getB());
    element.setAttribute("c", track->getC());
    element.setAttribute("d", track->getD());
    currentTrackElement_.appendChild(element);
}

void
DomWriter::visit(TrackElementCubicCurve *track)
{
	currentTrackElement_ = doc_->createElement("geometry");
	currentPVElement_.appendChild(currentTrackElement_);
	visit((TrackElement *)track);

	QDomElement element = doc_->createElement("paramPoly3");
	Polynomial *polynomialU = track->getPolynomialU();
	Polynomial *polynomialV = track->getPolynomialV();
	element.setAttribute("aU", polynomialU->getA());
	element.setAttribute("bU", polynomialU->getB());
	element.setAttribute("cU", polynomialU->getC());
	element.setAttribute("dU", polynomialU->getD());
	element.setAttribute("aV", polynomialV->getA());
	element.setAttribute("bV", polynomialV->getB());
	element.setAttribute("cV", polynomialV->getC());
	element.setAttribute("dV", polynomialV->getD());
	element.setAttribute("pRange", track->getPRange());

	currentTrackElement_.appendChild(element);
}

//################//
// LANES          //
//################//

void
DomWriter::visit(LaneSection *laneSection)
{
    currentLaneSectionElement_ = doc_->createElement("laneSection");
    currentLaneSectionElement_.setAttribute("s", laneSection->getSStart());
	currentLaneSectionElement_.setAttribute("singleSide", laneSection->getSide() ? "true" : "false");
    currentLanesElement_.appendChild(currentLaneSectionElement_);
    QDomElement NullNode;
    currentRightLaneElement_ = NullNode;
    currentCenterLaneElement_ = NullNode;
    currentLeftLaneElement_ = NullNode;

    laneSection->acceptForLanes(this);
}

void
DomWriter::visit(Lane *lane)
{

    // <laneSection><right/left/center><lane> //
    //
    if (lane->getId() < 0)
    {
        if (currentRightLaneElement_.isNull())
        {
            currentRightLaneElement_ = doc_->createElement("right");
            currentLaneSectionElement_.appendChild(currentRightLaneElement_);
        }
        currentLaneElement_ = doc_->createElement("lane");
        currentRightLaneElement_.appendChild(currentLaneElement_);
    }
    else if (lane->getId() == 0)
    {
        if (currentCenterLaneElement_.isNull())
        {
            currentCenterLaneElement_ = doc_->createElement("center");
            currentLaneSectionElement_.appendChild(currentCenterLaneElement_);
        }
        currentLaneElement_ = doc_->createElement("lane");
        currentCenterLaneElement_.appendChild(currentLaneElement_);
    }
    else
    {
        if (currentLeftLaneElement_.isNull())
        {
            currentLeftLaneElement_ = doc_->createElement("left");
            currentLaneSectionElement_.appendChild(currentLeftLaneElement_);
        }
        currentLaneElement_ = doc_->createElement("lane");
        currentLeftLaneElement_.appendChild(currentLaneElement_);
    }
    currentLaneElement_.setAttribute("id", lane->getId());
    currentLaneElement_.setAttribute("type", Lane::parseLaneTypeBack(lane->getLaneType()));
    currentLaneElement_.setAttribute("level", lane->getLevel());

    // <lane><link><predecessor/successor> //
    //
    if (lane->getPredecessor() != Lane::NOLANE || lane->getSuccessor() != Lane::NOLANE)
    {
        QDomElement linkElement = doc_->createElement("link");
        currentLaneElement_.appendChild(linkElement);
        if (lane->getPredecessor() != Lane::NOLANE)
        {
            QDomElement pElement = doc_->createElement("predecessor");
            pElement.setAttribute("id", lane->getPredecessor());
            linkElement.appendChild(pElement);
        }
        if (lane->getSuccessor() != Lane::NOLANE)
        {
            QDomElement sElement = doc_->createElement("successor");
            sElement.setAttribute("id", lane->getSuccessor());
            linkElement.appendChild(sElement);
        }
    }

    // <lane><width/roadMark/speed/height> //
    //
    lane->acceptForChildNodes(this);
}

void
DomWriter::visit(LaneWidth *laneWidth)
{
    QDomElement element = doc_->createElement("width");
    element.setAttribute("sOffset", laneWidth->getSOffset());
    element.setAttribute("a", laneWidth->getA());
    element.setAttribute("b", laneWidth->getB());
    element.setAttribute("c", laneWidth->getC());
    element.setAttribute("d", laneWidth->getD());
    currentLaneElement_.appendChild(element);
}

void
DomWriter::visit(LaneOffset *laneOffset)
{
	QDomElement element = doc_->createElement("laneOffset");
	element.setAttribute("s", laneOffset->getSOffset());
	element.setAttribute("a", laneOffset->getA());
	element.setAttribute("b", laneOffset->getB());
	element.setAttribute("c", laneOffset->getC());
	element.setAttribute("d", laneOffset->getD());
	currentLanesElement_.appendChild(element);
}

void
DomWriter::visit(LaneBorder *laneBorder)
{
	QDomElement element = doc_->createElement("border");
	element.setAttribute("sOffset", laneBorder->getSOffset());
	element.setAttribute("a", laneBorder->getA());
	element.setAttribute("b", laneBorder->getB());
	element.setAttribute("c", laneBorder->getC());
	element.setAttribute("d", laneBorder->getD());
	currentLaneElement_.appendChild(element);
}

void
DomWriter::visit(LaneRoadMark *laneRoadMark)
{
    QDomElement element = doc_->createElement("roadMark");
    element.setAttribute("sOffset", laneRoadMark->getSOffset());
    element.setAttribute("type", LaneRoadMark::parseRoadMarkTypeBack(laneRoadMark->getRoadMarkType()));
    element.setAttribute("weight", LaneRoadMark::parseRoadMarkWeightBack(laneRoadMark->getRoadMarkWeight()));
    element.setAttribute("color", LaneRoadMark::parseRoadMarkColorBack(laneRoadMark->getRoadMarkColor()));
    element.setAttribute("width", laneRoadMark->getRoadMarkWidth());
    element.setAttribute("laneChange", LaneRoadMark::parseRoadMarkLaneChangeBack(laneRoadMark->getRoadMarkLaneChange()));
	element.setAttribute("material", laneRoadMark->getRoadMarkMaterial());
	element.setAttribute("height", laneRoadMark->getRoadMarkHeight());

	LaneRoadMarkType *userType = laneRoadMark->getUserType();
	if (userType)
	{
		int size = userType->sizeOfRoadMarkTypeLines();
		if (size > 0)
		{
			QDomElement typeElement = doc_->createElement("type");
			typeElement.setAttribute("name", userType->getLaneRoadMarkTypeName());
			typeElement.setAttribute("width", userType->getLaneRoadMarkTypeWidth());

			double length, space, tOffset, sOffset, width;
			QString rule;
			for (int i = 0; i < size; i++)
			{
				QDomElement lineElement = doc_->createElement("line");
				userType->getRoadMarkTypeLine(i, length, space, tOffset, sOffset, rule, width);
				lineElement.setAttribute("length", length);
				lineElement.setAttribute("space", space);
				lineElement.setAttribute("tOffset", tOffset);
				lineElement.setAttribute("sOffset", sOffset);
				lineElement.setAttribute("rule", rule);
				lineElement.setAttribute("width", width);

				typeElement.appendChild(lineElement);
			}
			element.appendChild(typeElement);
		}
	}
    currentLaneElement_.appendChild(element);
}

void
DomWriter::visit(LaneSpeed *laneSpeed)
{
    QDomElement element = doc_->createElement("speed");
    element.setAttribute("sOffset", laneSpeed->getSOffset());
    element.setAttribute("max", laneSpeed->getMaxSpeed());
	element.setAttribute("unit", laneSpeed->getMaxSpeedUnit());
    currentLaneElement_.appendChild(element);
}

void
DomWriter::visit(LaneHeight *laneHeight)
{
    QDomElement element = doc_->createElement("height");
    element.setAttribute("sOffset", laneHeight->getSOffset());
    element.setAttribute("inner", laneHeight->getInnerHeight());
    element.setAttribute("outer", laneHeight->getOuterHeight());
    currentLaneElement_.appendChild(element);
}

void
DomWriter::visit(LaneRule *laneRule)
{
	QDomElement element = doc_->createElement("rule");
	element.setAttribute("sOffset", laneRule->getSOffset());
	element.setAttribute("value", laneRule->getValue());
	currentLaneElement_.appendChild(element);
}

void
DomWriter::visit(LaneAccess *laneAccess)
{
	QDomElement element = doc_->createElement("access");
	element.setAttribute("sOffset", laneAccess->getSOffset());
	element.setAttribute("restriction", LaneAccess::parseLaneRestrictionBack(laneAccess->getRestriction()));
	currentLaneElement_.appendChild(element);
}

void DomWriter::addTileInfo(QDomElement element, uint32_t tileID)
{
	if (tileID > 0)
	{
		QDomElement userData = doc_->createElement("userData");

		userData.setAttribute("code", "tile");
		userData.setAttribute("value", QString::number(tileID));
	}
}
//################//
// CONTROLLER     //
//################//

void
DomWriter::visit(RSystemElementController *controller)
{
    QDomElement controllerElement = doc_->createElement("controller");
	addTileInfo(controllerElement,controller->getID().getTileID());

    QDomElement userData = doc_->createElement("userData");

    QString script = controller->getScript();
    double cycleTime = controller->getCycleTime();
    if (script == "")
    {
        script = QString("%1_%2.qs").arg("lights").arg(controller->getID().writeString());
        cycleTime = 40.0;
    }
    userData.setAttribute("code", "script");
    userData.setAttribute("value", script);

    controllerElement.appendChild(userData);

    userData = doc_->createElement("userData");

    userData.setAttribute("code", "cycleTime");
    userData.setAttribute("value", cycleTime);

    controllerElement.appendChild(userData);

    if (!controller->getControlEntries().isEmpty())
    {
        for (int i = 0; i < controller->getControlEntries().size(); i++)
        {
            QDomElement controlElement = doc_->createElement("control");
            ControlEntry *control = controller->getControlEntries().at(i);

            controlElement.setAttribute("signalId", control->getSignalId().writeString());
            controlElement.setAttribute("type", control->getType());

            controllerElement.appendChild(controlElement);
        }
    }

    // Set mandatory attributes
    controllerElement.setAttribute("id", getIDString(controller->getID(), controller->getName()));
    controllerElement.setAttribute("name", controller->getName());
    controllerElement.setAttribute("sequence", controller->getSequence());

    root_.appendChild(controllerElement);

    // Write script file
    if (!controller->getControlEntries().isEmpty())
    {
        QList<QString> signalsType;
        for (int i = 0; i < controller->getControlEntries().size(); i++)
        {
            ControlEntry *control = controller->getControlEntries().at(i);
            Signal * signal = controller->getSignal(control->getSignalId());

            if (signal)
            {
                signalsType.append(signal->getType());
            }
        }

        QString name = QString("%1/%2").arg(QFileInfo(projectData_->getProjectWidget()->getFileName()).path()).arg(script);
        QFile file(name);
        file.open(QIODevice::WriteOnly | QIODevice::Text);
        QTextStream out(&file);
        out << "function update(time) {\n";
        out << "   if(time==0) {\n";
        
        for (int i = 0; i < controller->getControlEntries().size(); i++)
        {
            ControlEntry *control = controller->getControlEntries().at(i);
            if (signalsType.at(i) == "3")
            {
                out << "      signal_" << control->getSignalId().writeString() << ".yellow=1;\n";
            }
            else if (signalsType.at(i) == "2")
            {
                out << "      signal_" << control->getSignalId().writeString() << ".green=1;\n";
            }
            out << "      signal_" << control->getSignalId().writeString() << ".red=0;\n";
        }
        out << "   }\n";

        out << "   else if(time==1) {\n";
        for (int i = 0; i < controller->getControlEntries().size(); i++)
        {
            ControlEntry *control = controller->getControlEntries().at(i);
            if (signalsType.at(i) == "3")
            {
                out << "      signal_" << control->getSignalId().writeString() << ".yellow=0;\n";
                out << "      signal_" << control->getSignalId().writeString() << ".green=1;\n";
            }
        }
        out << "   }\n";

        out << "   else if(time==18) {\n";
        for (int i = 0; i < controller->getControlEntries().size(); i++)
        {
            ControlEntry *control = controller->getControlEntries().at(i);
            if (signalsType.at(i) == "3")
            {
                out << "      signal_" << control->getSignalId().writeString() << ".yellow=1;\n";
                out << "      signal_" << control->getSignalId().writeString() << ".green=0;\n";
            }
        }
        out << "   }\n";

        out << "   else if(time==20) {\n";
        for (int i = 0; i < controller->getControlEntries().size(); i++)
        {
            ControlEntry *control = controller->getControlEntries().at(i);
            if (signalsType.at(i) == "3")
            {
                out << "      signal_" << control->getSignalId().writeString() << ".yellow=0;\n";
            }
            else if (signalsType.at(i) == "2")
            {
                out << "      signal_" << control->getSignalId().writeString() << ".green=0;\n";
            }
            out << "      signal_" << control->getSignalId().writeString() << ".red=1;\n";
        }
        out << "   }\n";
        out << "}\n";

        file.close();
    }
}

//################//
// JUNCTION       //
//################//

void
DomWriter::visit(RSystemElementJunction *junction)
{
    // <junction> //
    //
    currentJunctionElement_ = doc_->createElement("junction");
    currentJunctionElement_.setAttribute("name", junction->getName());
    currentJunctionElement_.setAttribute("id", getIDString(junction->getID(), junction->getName()));

	addTileInfo(currentJunctionElement_, junction->getID().getTileID());
    root_.appendChild(currentJunctionElement_);


    junction->acceptForChildNodes(this);
}

void
DomWriter::visit(JunctionConnection *connection)
{
    QDomElement element = doc_->createElement("connection");
    element.setAttribute("id", connection->getId());
    element.setAttribute("incomingRoad", getIDString(connection->getIncomingRoad(), ""));
    element.setAttribute("connectingRoad", getIDString(connection->getConnectingRoad(), ""));
    element.setAttribute("contactPoint", JunctionConnection::parseContactPointBack(connection->getContactPoint()));

    QDomElement userData = doc_->createElement("userData");

    userData.setAttribute("code", "numerator");
    userData.setAttribute("value", connection->getNumerator());

    element.appendChild(userData);
    currentJunctionElement_.appendChild(element);

    QMap<int, int> links = connection->getLaneLinks();
    QMap<int, int>::const_iterator i = links.constBegin();
    while (i != links.constEnd())
    {
        QDomElement child = doc_->createElement("laneLink");
        child.setAttribute("from", i.key());
        child.setAttribute("to", i.value());
        ++i;
        element.appendChild(child);
    }
}

//################//
// JUNCTIONGROUP       //
//################//

void
DomWriter::visit(RSystemElementJunctionGroup *junctionGroup)
{
	// <junctionGroup> //
	//
	QDomElement currentJunctionGroupElement = doc_->createElement("junctionGroup");
	currentJunctionGroupElement.setAttribute("name", junctionGroup->getName());
	currentJunctionGroupElement.setAttribute("id", getIDString(junctionGroup->getID(), junctionGroup->getName()));
	currentJunctionGroupElement.setAttribute("type", junctionGroup->getType());

	addTileInfo(currentJunctionGroupElement, junctionGroup->getID().getTileID());

	foreach(QString reference, junctionGroup->getJunctionReferences())
	{
		QDomElement junctionReference = doc_->createElement("junctionReference");
		junctionReference.setAttribute("junction", reference);
		currentJunctionGroupElement.appendChild(junctionReference);
	}
	root_.appendChild(currentJunctionGroupElement);
}


//################//
// FIDDLEYARD     //
//################//

void
DomWriter::visit(RSystemElementFiddleyard *fiddleyard)
{
    // <fiddleyard> //
    //
    currentFiddleyardElement_ = doc_->createElement("fiddleyard");
    currentFiddleyardElement_.setAttribute("id", getIDString(fiddleyard->getID(), fiddleyard->getName()));
    currentFiddleyardElement_.setAttribute("name", fiddleyard->getName());

	addTileInfo(currentFiddleyardElement_, fiddleyard->getID().getTileID());
    root_.appendChild(currentFiddleyardElement_);

    // <fiddleyard><link> //
    //
    QDomElement linkElement = doc_->createElement("link");
    linkElement.setAttribute("elementType", fiddleyard->getElementType());
    linkElement.setAttribute("elementId", fiddleyard->getElementId().getID());
    linkElement.setAttribute("contactPoint", fiddleyard->getContactPoint());
    currentFiddleyardElement_.appendChild(linkElement);

    // <fiddleyard><source> //
    //
    fiddleyard->acceptForSources(this);

    // <fiddleyard><sink> //
    //
    fiddleyard->acceptForSinks(this);
}

void
DomWriter::visit(FiddleyardSource *source)
{
    // <fiddleyard><source> //
    //
    QDomElement sourceElement = doc_->createElement("source");
    sourceElement.setAttribute("id", getIDString(source->getId(), ""));
    sourceElement.setAttribute("lane", source->getLane());
    sourceElement.setAttribute("startTime", source->getStartTime());
    sourceElement.setAttribute("repeatTime", source->getRepeatTime());
    sourceElement.setAttribute("velocity", source->getVelocity());
    sourceElement.setAttribute("velocityDeviance", source->getVelocityDeviance());
	addTileInfo(sourceElement, source->getId().getTileID());
    currentFiddleyardElement_.appendChild(sourceElement);

    // <fiddleyard><source><vehicle> //
    //
    QMap<odrID, double> vehicles = source->getVehicles();
    QMap<odrID, double>::const_iterator i = vehicles.constBegin();
    while (i != vehicles.constEnd())
    {
        QDomElement child = doc_->createElement("vehicle");
        child.setAttribute("id", getIDString(i.key(), ""));
        child.setAttribute("numerator", i.value());
        ++i;

		//addTileInfo(child, vehicle->getID().getTileID());
        sourceElement.appendChild(child);
    }
}

void
DomWriter::visit(FiddleyardSink *sink)
{
    // <fiddleyard><sink> //
    //
    QDomElement sinkElement = doc_->createElement("sink");
    sinkElement.setAttribute("id", getIDString(sink->getId(), ""));
    sinkElement.setAttribute("lane", sink->getLane());
    currentFiddleyardElement_.appendChild(sinkElement);
}

//################//
// PEDFIDDLEYARD  //
//################//

void
DomWriter::visit(RSystemElementPedFiddleyard *fiddleyard)
{
    // <pedFiddleyard> //
    //
    currentPedFiddleyardElement_ = doc_->createElement("pedFiddleyard");
    currentPedFiddleyardElement_.setAttribute("id", getIDString(fiddleyard->getID(), fiddleyard->getName()));
    currentPedFiddleyardElement_.setAttribute("name", fiddleyard->getName());
    currentPedFiddleyardElement_.setAttribute("roadId", fiddleyard->getRoadId().getID());
	addTileInfo(currentPedFiddleyardElement_, fiddleyard->getID().getTileID());
    root_.appendChild(currentPedFiddleyardElement_);

    // <pedFiddleyard><source> //
    //
    fiddleyard->acceptForSources(this);

    // <pedFiddleyard><sink> //
    //
    fiddleyard->acceptForSinks(this);
}

void
DomWriter::visit(PedFiddleyardSource *source)
{
    // <pedFiddleyard><source> //
    //
    QDomElement sourceElement = doc_->createElement("source");

    // Set mandatory attributes
    sourceElement.setAttribute("id", getIDString(source->getId(), ""));
    sourceElement.setAttribute("lane", source->getLane());
    sourceElement.setAttribute("velocity", source->getVelocity());
	addTileInfo(sourceElement, source->getId().getTileID());

    // Set optional attributes
    if (source->hasStartTime())
        sourceElement.setAttribute("startTime", source->getStartTime());
    if (source->hasRepeatTime())
        sourceElement.setAttribute("repeatTime", source->getRepeatTime());
    if (source->hasTimeDeviance())
        sourceElement.setAttribute("timeDeviance", source->getTimeDeviance());
    if (source->hasDirection())
        sourceElement.setAttribute("direction", source->getDirection());
    if (source->hasSOffset())
        sourceElement.setAttribute("sOffset", source->getSOffset());
    if (source->hasVOffset())
        sourceElement.setAttribute("vOffset", source->getVOffset());
    if (source->hasVelocityDeviance())
        sourceElement.setAttribute("velocityDeviance", source->getVelocityDeviance());
    if (source->hasAcceleration())
        sourceElement.setAttribute("acceleration", source->getAcceleration());
    if (source->hasAccelerationDeviance())
        sourceElement.setAttribute("accelerationDeviance", source->getAccelerationDeviance());
    currentPedFiddleyardElement_.appendChild(sourceElement);

    // <pedFiddleyard><source><ped> //
    //
    QMap<QString, double> peds = source->getPedestrians();
    QMap<QString, double>::const_iterator i = peds.constBegin();
    while (i != peds.constEnd())
    {
        QDomElement child = doc_->createElement("ped");
        if (i.key().length() > 0)
            child.setAttribute("templateId", i.key());
        if (i.value() > 0.0)
            child.setAttribute("numerator", i.value());
        ++i;
        sourceElement.appendChild(child);
    }
}

void
DomWriter::visit(PedFiddleyardSink *sink)
{
    // <pedFiddleyard><sink> //
    //
    QDomElement sinkElement = doc_->createElement("sink");

    // Set mandatory attributes
    sinkElement.setAttribute("id", getIDString(sink->getId(), ""));
    sinkElement.setAttribute("lane", sink->getLane());

	addTileInfo(sinkElement, sink->getId().getTileID());

    // Set optional attributes
    if (sink->hasSinkProb())
        sinkElement.setAttribute("sinkProb", sink->getSinkProb());
    if (sink->hasDirection())
        sinkElement.setAttribute("direction", sink->getDirection());
    if (sink->hasSOffset())
        sinkElement.setAttribute("sOffset", sink->getSOffset());
    if (sink->hasVOffset())
        sinkElement.setAttribute("vOffset", sink->getVOffset());
    currentPedFiddleyardElement_.appendChild(sinkElement);
}

//################//
// VEHICLESYSTEM  //
//################//

void
DomWriter::visit(VehicleSystem *vehicleSystem)
{
    vehicleSystem->acceptForChildNodes(this);
}

void
DomWriter::visit(CarPool *carPool)
{
    currentCarPoolElement_ = doc_->createElement("carpool");
    root_.appendChild(currentCarPoolElement_);


    carPool->acceptForChildNodes(this);
}

void
DomWriter::visit(Pool *pool)
{
    QDomElement poolElement = doc_->createElement("pool");
    currentCarPoolElement_.appendChild(poolElement);

    poolElement.setAttribute("id", getIDString(pool->getID(), pool->getName()));
    poolElement.setAttribute("name", pool->getName());
    poolElement.setAttribute("velocity", pool->getVelocity());
    poolElement.setAttribute("velocityDeviance", pool->getVelocityDeviance());
    poolElement.setAttribute("numerator", pool->getNumerator());
	addTileInfo(poolElement, pool->getID().getTileID());

    QList<PoolVehicle *> vehicles = pool->getVehicles();
    QList<PoolVehicle *>::const_iterator i = vehicles.constBegin();
    while (i != vehicles.constEnd())
    {
        QDomElement child = doc_->createElement("vehicle");
        child.setAttribute("id", (*i)->getID());
        child.setAttribute("numerator", (*i)->getNumerator());
        ++i;
        poolElement.appendChild(child);
    }
}

void
DomWriter::visit(VehicleGroup *vehicleGroup)
{
    currentVehicleGroupElement_ = doc_->createElement("vehicles");
    root_.appendChild(currentVehicleGroupElement_);

    if (vehicleGroup->getRangeLOD() != VehicleGroup::defaultRangeLOD)
    {
        currentVehicleGroupElement_.setAttribute("rangeLOD", vehicleGroup->getRangeLOD());
    }

    if (vehicleGroup->hasPassThreshold())
        currentVehicleGroupElement_.setAttribute("passThreshold", vehicleGroup->getPassThreshold());


    vehicleGroup->acceptForChildNodes(this);
}

void
DomWriter::visit(RoadVehicle *roadVehicle)
{
    QDomElement vehicleElement = doc_->createElement("roadVehicle");
    vehicleElement.setAttribute("id", roadVehicle->getId());
    vehicleElement.setAttribute("name", roadVehicle->getName());
    vehicleElement.setAttribute("type", RoadVehicle::parseRoadVehicleTypeBack(roadVehicle->getType()));
    currentVehicleGroupElement_.appendChild(vehicleElement);

    // intelligence //
    //
    QDomElement element = doc_->createElement("intelligence");
    element.setAttribute("type", RoadVehicle::parseRoadVehicleIntelligenceTypeBack(roadVehicle->getIntelligenceType()));
    vehicleElement.appendChild(element);

    // geometry //
    //
    if (roadVehicle->getIntelligenceType() != RoadVehicle::DRVI_HUMAN)
    {
        element = doc_->createElement("geometry");
        element.setAttribute("modelFile", roadVehicle->getModelFile());
        vehicleElement.appendChild(element);
    }

    // dynamics //
    //
    if (roadVehicle->getMaxAcceleration() != RoadVehicle::defaultMaxAcceleration
        || roadVehicle->getIndicatoryVelocity() != RoadVehicle::defaultIndicatoryVelocity
        || roadVehicle->getMaxCrossAcceleration() != RoadVehicle::defaultMaxCrossAcceleration)
    {
        element = doc_->createElement("dynamics");
        if (roadVehicle->getMaxAcceleration() != RoadVehicle::defaultMaxAcceleration)
        {
            QDomElement subelement = doc_->createElement("maximumAcceleration");
            subelement.setAttribute("value", roadVehicle->getMaxAcceleration());
            element.appendChild(subelement);
        }
        if (roadVehicle->getIndicatoryVelocity() != RoadVehicle::defaultIndicatoryVelocity)
        {
            QDomElement subelement = doc_->createElement("indicatoryVelocity");
            subelement.setAttribute("value", roadVehicle->getIndicatoryVelocity());
            element.appendChild(subelement);
        }
        if (roadVehicle->getMaxCrossAcceleration() != RoadVehicle::defaultMaxCrossAcceleration)
        {
            QDomElement subelement = doc_->createElement("maximumCrossAcceleration");
            subelement.setAttribute("value", roadVehicle->getMaxCrossAcceleration());
            element.appendChild(subelement);
        }
        vehicleElement.appendChild(element);
    }

    // behaviour //
    //
    if (roadVehicle->getMinimumGap() != RoadVehicle::defaultMinimumGap
        || roadVehicle->getPursueTime() != RoadVehicle::defaultPursueTime
        || roadVehicle->getComfortableDeceleration() != RoadVehicle::defaultComfortableDecelaration
        || roadVehicle->getSaveDeceleration() != RoadVehicle::defaultSaveDeceleration
        || roadVehicle->getApproachFactor() != RoadVehicle::defaultApproachFactor
        || roadVehicle->getLaneChangeThreshold() != RoadVehicle::defaultLaneChangeThreshold
        || roadVehicle->getPolitenessFactor() != RoadVehicle::defaultPolitenessFactor)
    {
        element = doc_->createElement("behaviour");
        if (roadVehicle->getMinimumGap() != RoadVehicle::defaultMinimumGap)
        {
            QDomElement subelement = doc_->createElement("minimumGap");
            subelement.setAttribute("value", roadVehicle->getMinimumGap());
            element.appendChild(subelement);
        }
        if (roadVehicle->getPursueTime() != RoadVehicle::defaultPursueTime)
        {
            QDomElement subelement = doc_->createElement("pursueTime");
            subelement.setAttribute("value", roadVehicle->getPursueTime());
            element.appendChild(subelement);
        }
        if (roadVehicle->getComfortableDeceleration() != RoadVehicle::defaultComfortableDecelaration)
        {
            QDomElement subelement = doc_->createElement("comfortableDeceleration");
            subelement.setAttribute("value", roadVehicle->getComfortableDeceleration());
            element.appendChild(subelement);
        }
        if (roadVehicle->getSaveDeceleration() != RoadVehicle::defaultSaveDeceleration)
        {
            QDomElement subelement = doc_->createElement("saveDeceleration");
            subelement.setAttribute("value", roadVehicle->getSaveDeceleration());
            element.appendChild(subelement);
        }
        if (roadVehicle->getApproachFactor() != RoadVehicle::defaultApproachFactor)
        {
            QDomElement subelement = doc_->createElement("approachFactor");
            subelement.setAttribute("value", roadVehicle->getApproachFactor());
            element.appendChild(subelement);
        }
        if (roadVehicle->getLaneChangeThreshold() != RoadVehicle::defaultLaneChangeThreshold)
        {
            QDomElement subelement = doc_->createElement("laneChangeTreshold");
            subelement.setAttribute("value", roadVehicle->getLaneChangeThreshold()); // type in specification!
            element.appendChild(subelement);
        }
        if (roadVehicle->getPolitenessFactor() != RoadVehicle::defaultPolitenessFactor)
        {
            QDomElement subelement = doc_->createElement("politenessFactor");
            subelement.setAttribute("value", roadVehicle->getPolitenessFactor());
            element.appendChild(subelement);
        }
        vehicleElement.appendChild(element);
    }
}

//##################//
// PEDESTRIANSYSTEM //
//##################//

void
DomWriter::visit(PedestrianSystem *pedestrianSystem)
{
    pedestrianSystem->acceptForChildNodes(this);
}

void
DomWriter::visit(PedestrianGroup *pedestrianGroup)
{
    currentPedestrianGroupElement_ = doc_->createElement("pedestrians");
    root_.appendChild(currentPedestrianGroupElement_);

    if (pedestrianGroup->hasSpawnRange())
        currentPedestrianGroupElement_.setAttribute("spawnRange", pedestrianGroup->getSpawnRange());
    if (pedestrianGroup->hasMaxPeds())
        currentPedestrianGroupElement_.setAttribute("maxPeds", pedestrianGroup->getMaxPeds());
    if (pedestrianGroup->hasReportInterval())
        currentPedestrianGroupElement_.setAttribute("reportInterval", pedestrianGroup->getReportInterval());
    if (pedestrianGroup->hasAvoidCount())
        currentPedestrianGroupElement_.setAttribute("avoidCount", pedestrianGroup->getAvoidCount());
    if (pedestrianGroup->hasAvoidTime())
        currentPedestrianGroupElement_.setAttribute("avoidTime", pedestrianGroup->getAvoidTime());
    if (pedestrianGroup->hasAutoFiddle())
        currentPedestrianGroupElement_.setAttribute("autoFiddle", pedestrianGroup->getAutoFiddle() ? "true" : "false");
    if (pedestrianGroup->hasMovingFiddle())
        currentPedestrianGroupElement_.setAttribute("movingFiddle", pedestrianGroup->getMovingFiddle() ? "true" : "false");


    pedestrianGroup->acceptForChildNodes(this);
}

void
DomWriter::visit(Pedestrian *ped)
{
    QDomElement pedElement;

    if (ped->isDefault())
    {
        pedElement = doc_->createElement("default");
    }
    else if (ped->isTemplate())
    {
        pedElement = doc_->createElement("template");
        pedElement.setAttribute("id", ped->getId());
    }
    else
    {
        pedElement = doc_->createElement("ped");
        pedElement.setAttribute("id", ped->getId());
        pedElement.setAttribute("name", ped->getName());
        if (ped->getTemplateId().length() > 0)
            pedElement.setAttribute("templateId", ped->getTemplateId());
    }
    if (ped->getRangeLOD().length() > 0)
        pedElement.setAttribute("rangeLOD", ped->getRangeLOD());
    if (ped->getDebugLvl().length() > 0)
        pedElement.setAttribute("debugLvl", ped->getDebugLvl());
    currentPedestrianGroupElement_.appendChild(pedElement);

    QDomElement element;
    bool valueSet;

    // geometry //
    //
    valueSet = false;
    element = doc_->createElement("geometry");
    if (ped->getModelFile().length() > 0)
    {
        element.setAttribute("modelFile", ped->getModelFile());
        valueSet = true;
    }
    if (ped->getScale().length() > 0)
    {
        element.setAttribute("scale", ped->getScale());
        valueSet = true;
    }
    if (ped->getHeading().length() > 0)
    {
        element.setAttribute("heading", ped->getHeading());
        valueSet = true;
    }
    if (valueSet)
        pedElement.appendChild(element);

    // start //
    //
    valueSet = false;
    element = doc_->createElement("start");
    if (ped->getStartRoadId().length() > 0)
    {
        element.setAttribute("roadId", ped->getStartRoadId());
        valueSet = true;
    }
    if (ped->getStartLane().length() > 0)
    {
        element.setAttribute("lane", ped->getStartLane());
        valueSet = true;
    }
    if (ped->getStartDir().length() > 0)
    {
        element.setAttribute("direction", ped->getStartDir());
        valueSet = true;
    }
    if (ped->getStartSOffset().length() > 0)
    {
        element.setAttribute("sOffset", ped->getStartSOffset());
        valueSet = true;
    }
    if (ped->getStartVOffset().length() > 0)
    {
        element.setAttribute("vOffset", ped->getStartVOffset());
        valueSet = true;
    }
    if (ped->getStartVel().length() > 0)
    {
        element.setAttribute("velocity", ped->getStartVel());
        valueSet = true;
    }
    if (ped->getStartAcc().length() > 0)
    {
        element.setAttribute("acceleration", ped->getStartAcc());
        valueSet = true;
    }
    if (valueSet)
        pedElement.appendChild(element);

    // animations //
    //
    valueSet = false;
    QDomElement animElement = doc_->createElement("animations");
    element = doc_->createElement("idle");
    if (ped->getIdleIdx().length() > 0 && ped->getIdleVel().length() > 0)
    {
        element.setAttribute("index", ped->getIdleIdx());
        element.setAttribute("velocity", ped->getIdleVel());
        animElement.appendChild(element);
        valueSet = true;
    }
    element = doc_->createElement("slow");
    if (ped->getSlowIdx().length() > 0 && ped->getSlowVel().length() > 0)
    {
        element.setAttribute("index", ped->getSlowIdx());
        element.setAttribute("velocity", ped->getSlowVel());
        animElement.appendChild(element);
        valueSet = true;
    }
    element = doc_->createElement("walk");
    if (ped->getWalkIdx().length() > 0 && ped->getWalkVel().length() > 0)
    {
        element.setAttribute("index", ped->getWalkIdx());
        element.setAttribute("velocity", ped->getWalkVel());
        animElement.appendChild(element);
        valueSet = true;
    }
    element = doc_->createElement("jog");
    if (ped->getJogIdx().length() > 0 && ped->getJogVel().length() > 0)
    {
        element.setAttribute("index", ped->getJogIdx());
        element.setAttribute("velocity", ped->getJogVel());
        animElement.appendChild(element);
        valueSet = true;
    }
    element = doc_->createElement("look");
    if (ped->getLookIdx().length() > 0)
    {
        element.setAttribute("index", ped->getLookIdx());
        animElement.appendChild(element);
        valueSet = true;
    }
    element = doc_->createElement("wave");
    if (ped->getWaveIdx().length() > 0)
    {
        element.setAttribute("index", ped->getWaveIdx());
        animElement.appendChild(element);
        valueSet = true;
    }
    if (valueSet)
        pedElement.appendChild(animElement);
}

//################//
// SCENERYSYSTEM  //
//################//

void
DomWriter::visit(ScenerySystem *system)
{
    foreach (QString file, projectData_->getScenerySystem()->getSceneryFiles())
    {
        QDomElement element = doc_->createElement("scenery");
        element.setAttribute("file", file);
        root_.appendChild(element);
    }

    system->acceptForChildNodes(this);
}

void
DomWriter::visit(SceneryMap *map)
{
    QDomElement mapElement = doc_->createElement("map");
    mapElement.setAttribute("filename", map->getFilename());
    mapElement.setAttribute("id", map->getId());

    if (map->getX() != 0.0)
    {
        mapElement.setAttribute("x", map->getX());
    }
    if (map->getY() != 0.0)
    {
        mapElement.setAttribute("y", map->getY());
    }

    mapElement.setAttribute("width", map->getWidth());
    mapElement.setAttribute("height", map->getHeight());

    if (map->getOpacity() != 1.0)
    {
        mapElement.setAttribute("opacity", map->getOpacity());
    }

    QDomElement element = doc_->createElement("scenery");
    element.appendChild(mapElement);
    root_.appendChild(element);
}

void
DomWriter::visit(Heightmap *map)
{
    QDomElement mapElement = doc_->createElement("heightmap");
    mapElement.setAttribute("filename", map->getFilename());
    mapElement.setAttribute("data", map->getHeightmapDataFilename());
    mapElement.setAttribute("id", map->getId());

    if (map->getX() != 0.0)
    {
        mapElement.setAttribute("x", map->getX());
    }
    if (map->getY() != 0.0)
    {
        mapElement.setAttribute("y", map->getY());
    }

    mapElement.setAttribute("width", map->getWidth());
    mapElement.setAttribute("height", map->getHeight());

    if (map->getOpacity() != 1.0)
    {
        mapElement.setAttribute("opacity", map->getOpacity());
    }

    QDomElement element = doc_->createElement("scenery");
    element.appendChild(mapElement);
    root_.appendChild(element);
}

void
DomWriter::visit(SceneryTesselation *tesselation)
{
    QDomElement envElement = doc_->createElement("environment");
    if (tesselation->getTesselateRoads())
    {
        envElement.setAttribute("tessellateRoads", "true");
    }
    else
    {
        envElement.setAttribute("tessellateRoads", "false");
    }
    if (tesselation->getTesselatePaths())
    {
        envElement.setAttribute("tessellatePaths", "true");
    }
    else
    {
        envElement.setAttribute("tessellatePaths", "false");
    }
    root_.appendChild(envElement);
}

//################//
// GEOREFERENCE      //
//################//

void
DomWriter::visit(GeoReference *geoReferenceParams)
{

	QDomElement geoReferenceElement = doc_->createElement("geoReference");

	QDomCDATASection CDATA = doc_->createCDATASection(geoReferenceParams->getParams());

	geoReferenceElement.appendChild(CDATA);

	header_.appendChild(geoReferenceElement);
}

