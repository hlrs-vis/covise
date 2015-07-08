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

// RoadSystem //
//
#include "src/data/roadsystem/roadsystem.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementcontroller.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/rsystemelementfiddleyard.hpp"
#include "src/data/roadsystem/rsystemelementpedfiddleyard.hpp"

#include "src/data/roadsystem/roadlink.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"

#include "src/data/roadsystem/sections/typesection.hpp"
#include "src/data/roadsystem/sections/surfacesection.hpp"

#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementspiral.hpp"
#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"
#include "src/data/roadsystem/track/trackelementpoly3.hpp"

#include "src/data/roadsystem/sections/elevationsection.hpp"
#include "src/data/roadsystem/sections/superelevationsection.hpp"
#include "src/data/roadsystem/sections/crossfallsection.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"
#include "src/data/roadsystem/sections/laneroadmark.hpp"
#include "src/data/roadsystem/sections/lanespeed.hpp"
#include "src/data/roadsystem/sections/laneheight.hpp"

#include "src/data/roadsystem/sections/crosswalkobject.hpp"
#include "src/data/roadsystem/sections/objectobject.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/roadsystem/sections/sensorobject.hpp"
#include "src/data/roadsystem/sections/bridgeobject.hpp"

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

// MainWindow //
//
#include "src/mainwindow.hpp"

// Utils //
//
#include "math.h"

DomWriter::DomWriter(ProjectData *projectData)
    : projectData_(projectData)
{
    doc_ = new QDomDocument();
}

/** Run this visitor through the roadsystem.
*/
void
DomWriter::runToTheHills()
{
    // <?xml?> //
    //
    QDomNode xml = doc_->createProcessingInstruction("xml", "version=\"1.0\"");
    doc_->appendChild(xml);

    // <OpenDRIVE> //
    //
    root_ = doc_->createElement("OpenDRIVE");
    doc_->appendChild(root_);

    // <header> //
    //
    QDomElement header = doc_->createElement("header");
    header.setAttribute("revMajor", projectData_->getRevMajor());
    header.setAttribute("revMinor", projectData_->getRevMinor());
    header.setAttribute("name", projectData_->getName());
    header.setAttribute("version", projectData_->getVersion());
    header.setAttribute("date", projectData_->getDate());
    header.setAttribute("north", projectData_->getNorth());
    header.setAttribute("south", projectData_->getSouth());
    header.setAttribute("east", projectData_->getEast());
    header.setAttribute("west", projectData_->getWest());
    root_.appendChild(header);

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

void
DomWriter::visit(RSystemElementRoad *road)
{
    // <road> //
    //
    currentRoad_ = doc_->createElement("road");
    currentRoad_.setAttribute("name", road->getName());
    currentRoad_.setAttribute("length", road->getLength());
    currentRoad_.setAttribute("id", road->getID());
    currentRoad_.setAttribute("junction", road->getJunction());
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
            predElement.setAttribute("elementId", pred->getElementId());
            predElement.setAttribute("contactPoint", pred->getContactPoint());
            linkElement.appendChild(predElement);
        }

        if (succ)
        {
            QDomElement succElement = doc_->createElement("successor");
            succElement.setAttribute("elementType", succ->getElementType());
            succElement.setAttribute("elementId", succ->getElementId());
            succElement.setAttribute("contactPoint", succ->getContactPoint());
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
    if (!road->getSuperelevationSections().isEmpty() || !road->getCrossfallSections().isEmpty())
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
        int type = parts.at(0).toInt(&number);
        QString subclass = "";
        int subtype = -1;
       
        if (number && (parts.size() > 1))
        {
            subtype = parts.at(1).toInt(&number);
            if (!number)
            {
                subtype = -1;
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

        double radHeading = object->getHeading() / 360.0 * (2.0 * M_PI);

        do
        {
            Object::ObjectOrientation orientation = object->getOrientation();
            

            int fromLane = 0;
            int toLane = 0;
            if (orientation == Object::ObjectOrientation::NEGATIVE_TRACK_DIRECTION)
            {
                fromLane = object->getParentRoad()->getLaneSection(object->getSStart())->getRightmostLaneId();
            }
            else if (orientation == Object::ObjectOrientation::POSITIVE_TRACK_DIRECTION)
            {
                toLane = object->getParentRoad()->getLaneSection(object->getSStart())->getLeftmostLaneId();
            }
            else
            {
                fromLane = object->getParentRoad()->getLaneSection(object->getSStart())->getRightmostLaneId();
                toLane = object->getParentRoad()->getLaneSection(object->getSStart())->getLeftmostLaneId();

            }

            QString name = object->getName();
            QString id = roadSystem->getUniqueId(object->getId(), name);

            Signal * signal = new Signal(id, "", s, object->getT(), false, (Signal::OrientationType)orientation, object->getzOffset(), "Germany", type, subclass, subtype, 0.0, object->getHeading(), object->getPitch(), object->getRoll(), object->getPole(), 2, fromLane, toLane, 0, 0);
            

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
        if (object->getRepeatS() != -1)
        {
            QDomElement repeatElement = doc_->createElement("repeat");

            repeatElement.setAttribute("s", object->getRepeatS());
            if (object->getRepeatS() + object->getRepeatLength() > object->getParentRoad()->getLength()) // TODO: check this in the settings
            {
                object->setRepeatLength(object->getParentRoad()->getLength() - object->getRepeatS() - NUMERICAL_ZERO8);
            }
            repeatElement.setAttribute("length", object->getRepeatLength());
            repeatElement.setAttribute("distance", object->getRepeatDistance());
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
        objectElement.setAttribute("id", object->getId());
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
        if (object->getOrientation() == Object::NEGATIVE_TRACK_DIRECTION)
            objectElement.setAttribute("orientation", "-");
        else
            objectElement.setAttribute("orientation", "+");
        objectElement.setAttribute("length", object->getLength());
        objectElement.setAttribute("width", object->getWidth());
        objectElement.setAttribute("radius", object->getRadius());
        objectElement.setAttribute("height", object->getHeight());
        objectElement.setAttribute("hdg", object->getHeading() / 360.0 * (2.0 * M_PI));
        objectElement.setAttribute("pitch", object->getPitch() / 360.0 * (2.0 * M_PI));
        objectElement.setAttribute("roll", object->getRoll() / 360.0 * (2.0 * M_PI));

        currentObjectsElement_.appendChild(objectElement);
    }
}

//################//
// BRIDGE       //
//################//

void
DomWriter::visit(Bridge *bridge)
{

    QDomElement bridgeElement;
    if (bridge->getName() == "Tunnel") // TODO: Add selection in the bridge settings
    {
        bridgeElement = doc_->createElement("tunnel");

        // Set mandatory attributes
        bridgeElement.setAttribute("id", bridge->getId());

        bridgeElement.setAttribute("name", bridge->getName());

        bridgeElement.setAttribute("s", bridge->getSStart());
        bridgeElement.setAttribute("length", bridge->getLength());

        bridgeElement.setAttribute("lighting", 0.5);
        bridgeElement.setAttribute("daylight", 0.5);
        bridgeElement.setAttribute("type", "standard");
    }
    else
    {
        bridgeElement = doc_->createElement("bridge");

        // Set mandatory attributes
        bridgeElement.setAttribute("id", bridge->getId());

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

        bridgeElement.setAttribute("type", typeName);
    }

    // model file are ancillary data
    //
    if ((bridge->getFileName() != "") && (bridge->getFileName() != "Tunnel"))
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
// CROSSWALK      //
//################//

void
DomWriter::visit(Crosswalk *crosswalk)
{

    QDomElement crosswalkElement = doc_->createElement("crosswalk");

    // Set mandatory attributes
    crosswalkElement.setAttribute("id", crosswalk->getId());
    crosswalkElement.setAttribute("name", crosswalk->getName());
    crosswalkElement.setAttribute("s", crosswalk->getS());
    crosswalkElement.setAttribute("length", crosswalk->getLength());

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
    if (signal->getType() == 293)
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
    signalElement.setAttribute("id", signal->getId());
    QString signalName; // The name has the format: type.typeSubclass-subtype_name_p

    if (signal->getType() >= 0)
    {
        if (!signal->getTypeSubclass().isEmpty())
        {
            if (signal->getSubtype() >= 0)
            {
                signalName = QString::number(signal->getType()) + "." + signal->getTypeSubclass() + "-" + QString::number(signal->getSubtype());
            }
            else
            {
                signalName = QString::number(signal->getType()) + "." + signal->getTypeSubclass();
            }
        }
        else if (signal->getSubtype() >= 0)
        {
            signalName = QString::number(signal->getType()) + "-" + QString::number(signal->getSubtype());
        }
        else
        {
            signalName = QString::number(signal->getType());
        }
    }

    double hOffset = signal->getHeading();
    if ((signal->getType() == 625) && (signal->getSubtype() == 10) && (signal->getTypeSubclass() == "20"))
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
    bool textureFile;
    QString textureFilename;
    QString dir = projectData_->getProjectWidget()->getMainWindow()->getCovisedir() + "/share/covise/signals/";
    if (textureFile = file.exists(dir + signal->getCountry() + "/" + signalName + ".png"))
    {
        if (signalName.contains("/"))
        {
            textureFilename += ".png";
        }
        else
        { 
            textureFilename = "signals/" +  signal->getCountry() + "/" + signalName + ".png";
        }
    }
    else if (textureFile = file.exists(dir + signal->getCountry() + "/" + signalName + ".tif"))
    {
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

    bool modelFile;
    QString modelFilename;
    if (modelFile = file.exists(dir + signal->getCountry() + "/" + signalName + ".osg"))
    {
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
    if (signal->getOrientation() == Signal::BOTH_DIRECTIONS)
        signalElement.setAttribute("orientation", "both");
    else if (signal->getOrientation() == Signal::NEGATIVE_TRACK_DIRECTION)
        signalElement.setAttribute("orientation", "-");
    else
        signalElement.setAttribute("orientation", "+");
    signalElement.setAttribute("zOffset", signal->getZOffset());
    signalElement.setAttribute("country", signal->getCountry());
    signalElement.setAttribute("type", signal->getType());
    signalElement.setAttribute("subtype", signal->getSubtype());
    signalElement.setAttribute("value", signal->getValue());
    signalElement.setAttribute("hOffset", hOffset / 360.0 * (2.0 * M_PI));
    signalElement.setAttribute("pitch", signal->getPitch() / 360.0 * (2.0 * M_PI));
    signalElement.setAttribute("roll", signal->getRoll() / 360.0 * (2.0 * M_PI));

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
    element.setAttribute("a", section->getA() * 2.0 * M_PI / 360.0);
    element.setAttribute("b", section->getB() * 2.0 * M_PI / 360.0);
    element.setAttribute("c", section->getC() * 2.0 * M_PI / 360.0);
    element.setAttribute("d", section->getD() * 2.0 * M_PI / 360.0);
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
    element.setAttribute("a", section->getA() * 2.0 * M_PI / 360.0);
    element.setAttribute("b", section->getB() * 2.0 * M_PI / 360.0);
    element.setAttribute("c", section->getC() * 2.0 * M_PI / 360.0);
    element.setAttribute("d", section->getD() * 2.0 * M_PI / 360.0);
    currentLateralProfileElement_.appendChild(element);
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

//################//
// LANES          //
//################//

void
DomWriter::visit(LaneSection *laneSection)
{
    currentLaneSectionElement_ = doc_->createElement("laneSection");
    currentLaneSectionElement_.setAttribute("s", laneSection->getSStart());
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
DomWriter::visit(LaneRoadMark *laneRoadMark)
{
    QDomElement element = doc_->createElement("roadMark");
    element.setAttribute("sOffset", laneRoadMark->getSOffset());
    element.setAttribute("type", LaneRoadMark::parseRoadMarkTypeBack(laneRoadMark->getRoadMarkType()));
    element.setAttribute("weight", LaneRoadMark::parseRoadMarkWeightBack(laneRoadMark->getRoadMarkWeight()));
    element.setAttribute("color", LaneRoadMark::parseRoadMarkColorBack(laneRoadMark->getRoadMarkColor()));
    element.setAttribute("width", laneRoadMark->getRoadMarkWidth());
    element.setAttribute("laneChange", LaneRoadMark::parseRoadMarkLaneChangeBack(laneRoadMark->getRoadMarkLaneChange()));
    currentLaneElement_.appendChild(element);
}

void
DomWriter::visit(LaneSpeed *laneSpeed)
{
    QDomElement element = doc_->createElement("speed");
    element.setAttribute("sOffset", laneSpeed->getSOffset());
    element.setAttribute("max", laneSpeed->getMaxSpeed());
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

//################//
// CONTROLLER     //
//################//

void
DomWriter::visit(RSystemElementController *controller)
{
    QDomElement controllerElement = doc_->createElement("controller");

    QDomElement userData = doc_->createElement("userData");

    QString script = controller->getScript();
    double cycleTime = controller->getCycleTime();
    if (script == "")
    {
        QStringList parts = controller->getID().split("_");
        script = QString("%1_%2.qs").arg("lights").arg(parts[1]);
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

            controlElement.setAttribute("signalId", control->getSignalId());
            controlElement.setAttribute("type", control->getType());

            controllerElement.appendChild(controlElement);
        }
    }

    // Set mandatory attributes
    controllerElement.setAttribute("id", controller->getID());
    controllerElement.setAttribute("name", controller->getName());
    controllerElement.setAttribute("sequence", controller->getSequence());

    root_.appendChild(controllerElement);

    // Write script file
    if (!controller->getControlEntries().isEmpty())
    {
        QList<int> signalsType;
        for (int i = 0; i < controller->getControlEntries().size(); i++)
        {
            ControlEntry *control = controller->getControlEntries().at(i);
            const QString id = control->getSignalId();
            Signal * signal = controller->getSignal(id);

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
            if (signalsType.at(i) == 3)
            {
                out << "      signal_" << control->getSignalId() << ".yellow=1;\n";
            }
            else if (signalsType.at(i) == 2)
            {
                out << "      signal_" << control->getSignalId() << ".green=1;\n";
            }
            out << "      signal_" << control->getSignalId() << ".red=0;\n";
        }
        out << "   }\n";

        out << "   else if(time==1) {\n";
        for (int i = 0; i < controller->getControlEntries().size(); i++)
        {
            ControlEntry *control = controller->getControlEntries().at(i);
            if (signalsType.at(i) == 3)
            {
                out << "      signal_" << control->getSignalId() << ".yellow=0;\n";
                out << "      signal_" << control->getSignalId() << ".green=1;\n";
            }
        }
        out << "   }\n";

        out << "   else if(time==18) {\n";
        for (int i = 0; i < controller->getControlEntries().size(); i++)
        {
            ControlEntry *control = controller->getControlEntries().at(i);
            if (signalsType.at(i) == 3)
            {
                out << "      signal_" << control->getSignalId() << ".yellow=1;\n";
                out << "      signal_" << control->getSignalId() << ".green=0;\n";
            }
        }
        out << "   }\n";

        out << "   else if(time==20) {\n";
        for (int i = 0; i < controller->getControlEntries().size(); i++)
        {
            ControlEntry *control = controller->getControlEntries().at(i);
            if (signalsType.at(i) == 3)
            {
                out << "      signal_" << control->getSignalId() << ".yellow=0;\n";
            }
            else if (signalsType.at(i) == 2)
            {
                out << "      signal_" << control->getSignalId() << ".green=0;\n";
            }
            out << "      signal_" << control->getSignalId() << ".red=1;\n";
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
    currentJunctionElement_.setAttribute("id", junction->getID());
    root_.appendChild(currentJunctionElement_);


    junction->acceptForChildNodes(this);
}

void
DomWriter::visit(JunctionConnection *connection)
{
    QDomElement element = doc_->createElement("connection");
    element.setAttribute("id", connection->getId());
    element.setAttribute("incomingRoad", connection->getIncomingRoad());
    element.setAttribute("connectingRoad", connection->getConnectingRoad());
    element.setAttribute("contactPoint", connection->getContactPoint());

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
// FIDDLEYARD     //
//################//

void
DomWriter::visit(RSystemElementFiddleyard *fiddleyard)
{
    // <fiddleyard> //
    //
    currentFiddleyardElement_ = doc_->createElement("fiddleyard");
    currentFiddleyardElement_.setAttribute("id", fiddleyard->getID());
    currentFiddleyardElement_.setAttribute("name", fiddleyard->getName());
    root_.appendChild(currentFiddleyardElement_);

    // <fiddleyard><link> //
    //
    QDomElement linkElement = doc_->createElement("link");
    linkElement.setAttribute("elementType", fiddleyard->getElementType());
    linkElement.setAttribute("elementId", fiddleyard->getElementId());
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
    sourceElement.setAttribute("id", source->getId());
    sourceElement.setAttribute("lane", source->getLane());
    sourceElement.setAttribute("startTime", source->getStartTime());
    sourceElement.setAttribute("repeatTime", source->getRepeatTime());
    sourceElement.setAttribute("velocity", source->getVelocity());
    sourceElement.setAttribute("velocityDeviance", source->getVelocityDeviance());
    currentFiddleyardElement_.appendChild(sourceElement);

    // <fiddleyard><source><vehicle> //
    //
    QMap<QString, double> vehicles = source->getVehicles();
    QMap<QString, double>::const_iterator i = vehicles.constBegin();
    while (i != vehicles.constEnd())
    {
        QDomElement child = doc_->createElement("vehicle");
        child.setAttribute("id", i.key());
        child.setAttribute("numerator", i.value());
        ++i;
        sourceElement.appendChild(child);
    }
}

void
DomWriter::visit(FiddleyardSink *sink)
{
    // <fiddleyard><sink> //
    //
    QDomElement sinkElement = doc_->createElement("sink");
    sinkElement.setAttribute("id", sink->getId());
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
    currentPedFiddleyardElement_.setAttribute("id", fiddleyard->getID());
    currentPedFiddleyardElement_.setAttribute("name", fiddleyard->getName());
    currentPedFiddleyardElement_.setAttribute("roadId", fiddleyard->getRoadId());
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
    sourceElement.setAttribute("id", source->getId());
    sourceElement.setAttribute("lane", source->getLane());
    sourceElement.setAttribute("velocity", source->getVelocity());

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
    sinkElement.setAttribute("id", sink->getId());
    sinkElement.setAttribute("lane", sink->getLane());

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

    poolElement.setAttribute("id", pool->getID());
    poolElement.setAttribute("name", pool->getName());
    poolElement.setAttribute("velocity", pool->getVelocity());
    poolElement.setAttribute("velocityDeviance", pool->getVelocityDeviance());
    poolElement.setAttribute("numerator", pool->getNumerator());

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
