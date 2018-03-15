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

#include "domparser.hpp"

#include "src/mainwindow.hpp"

// Data Model //
//
#include "src/data/projectdata.hpp"
#include "src/data/georeference.hpp"

#include "src/data/changemanager.hpp"
#include "src/data/prototypemanager.hpp"
#include "src/data/signalmanager.hpp"

#include "src/data/visitors/sparcsmergevisitor.hpp"

#include "src/data/tilesystem/tilesystem.hpp"
#include "src/data/tilesystem/tile.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementcontroller.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/rsystemelementfiddleyard.hpp"
#include "src/data/roadsystem/rsystemelementpedfiddleyard.hpp"
#include "src/data/roadsystem/rsystemelementjunctiongroup.hpp"

#include "src/data/roadsystem/sections/objectobject.hpp"
#include "src/data/roadsystem/sections/objectreference.hpp"
#include "src/data/roadsystem/sections/crosswalkobject.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/roadsystem/sections/signalreference.hpp"
#include "src/data/roadsystem/sections/sensorobject.hpp"
#include "src/data/roadsystem/sections/surfaceobject.hpp"
#include "src/data/roadsystem/sections/bridgeobject.hpp"
#include "src/data/roadsystem/sections/tunnelobject.hpp"
#include "src/data/roadsystem/sections/laneoffset.hpp"

#include "src/data/roadsystem/roadlink.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"

#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementspiral.hpp"
#include "src/data/roadsystem/track/trackelementpoly3.hpp"
#include "src/data/roadsystem/track/trackelementcubiccurve.hpp"

#include "src/data/roadsystem/sections/typesection.hpp"
#include "src/data/roadsystem/sections/surfacesection.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"
#include "src/data/roadsystem/sections/laneborder.hpp"
#include "src/data/roadsystem/sections/laneroadmark.hpp"
#include "src/data/roadsystem/sections/lanespeed.hpp"
#include "src/data/roadsystem/sections/laneheight.hpp"
#include "src/data/roadsystem/sections/lanerule.hpp"
#include "src/data/roadsystem/sections/laneaccess.hpp"

#include "src/data/roadsystem/sections/elevationsection.hpp"
#include "src/data/roadsystem/sections/superelevationsection.hpp"
#include "src/data/roadsystem/sections/crossfallsection.hpp"
#include "src/data/roadsystem/sections/shapesection.hpp"

#include "src/data/roadsystem/sections/objectobject.hpp"
#include "src/data/roadsystem/sections/parkingspaceobject.hpp"

#include "src/data/vehiclesystem/vehiclesystem.hpp"
#include "src/data/vehiclesystem/vehiclegroup.hpp"
#include "src/data/vehiclesystem/roadvehicle.hpp"
#include "src/data/vehiclesystem/poolvehicle.hpp"
#include "src/data/vehiclesystem/pool.hpp"
#include "src/data/vehiclesystem/carpool.hpp"

#include "src/data/pedestriansystem/pedestriansystem.hpp"
#include "src/data/pedestriansystem/pedestriangroup.hpp"
#include "src/data/pedestriansystem/pedestrian.hpp"

#include "src/data/scenerysystem/scenerysystem.hpp"
#include "src/data/scenerysystem/scenerymap.hpp"
#include "src/data/scenerysystem/heightmap.hpp"
#include "src/data/scenerysystem/scenerytesselation.hpp"

// Qt //
//
#include <QtGui>
#include <QDomDocument>
#include <QMessageBox>


// Utils //
//
#include "math.h"
#include "src/util/odd.hpp"

/** CONSTRUCTOR.
*
*/
DomParser::DomParser(ProjectData *projectData, QObject *parent)
    : QObject(parent)
    , projectData_(projectData)
    , roadSystem_(NULL)
    , vehicleSystem_(NULL)
    , pedestrianSystem_(NULL)
    , scenerySystem_(NULL)
    , mode_(MODE_NONE)
    , prototypeElementCount_(0x0)
    , elementCount_(0x0)
    , tileCount_(0x0)
	, tileSystem_(NULL)
{
    doc_ = new QDomDocument();
	disableWarnings = false;
}

/** DESTRUCTOR.
*
*/
DomParser::~DomParser()
{
    delete doc_;
}

//################//
// XODR           //
//################//

void DomParser::warning(QString title, QString text)
{
	if (!disableWarnings)
	{
		QMessageBox msgBox;
		msgBox.setIcon(QMessageBox::Warning);
		msgBox.setWindowTitle(title);
		msgBox.setText(text);
		QPushButton* pButtonYes = msgBox.addButton(tr("Ok"), QMessageBox::YesRole);
		QPushButton* pButtonDisable = msgBox.addButton(tr("Disable Warnings"), QMessageBox::NoRole);

		msgBox.exec();

		if (msgBox.clickedButton() == (QAbstractButton *)pButtonDisable)
		{
			disableWarnings = true;
		}
	}
}

/*! \brief Opens a .xodr file, creates a DOM tree and reads in the first level.
*
*/
bool
DomParser::parseXODR(QIODevice *source)
{
    roadSystem_ = projectData_->getRoadSystem();
    tileSystem_ = projectData_->getTileSystem();
    vehicleSystem_ = projectData_->getVehicleSystem();
    pedestrianSystem_ = projectData_->getPedestrianSystem();
    scenerySystem_ = projectData_->getScenerySystem();

    // Mode //
    //
    mode_ = DomParser::MODE_XODR;

    // Open file and parse tree //
    //
    QString errorStr = "";
    int errorLine = 0;
    int errorColumn = 0;

    if (!doc_->setContent(source, true, &errorStr, &errorLine, &errorColumn))
    {
        warning(tr("ODD: XML Parser Error"),
                             tr("Parse error at line %1, column %2:\n%3")
                                 .arg(errorLine)
                                 .arg(errorColumn)
                                 .arg(errorStr));
		
        return false;
    }

    // <OpenDRIVE> //
    //
    QDomElement root = doc_->documentElement();
    if (root.tagName() != "OpenDRIVE")
    {
        warning( tr("ODD: XML Parser Error"),
                             tr("Root element is not <OpenDRIVE>!"));
        return false;
    }

    // <OpenDRIVE><header> //
    //
    QDomElement child;
    child = root.firstChildElement("header");
    if (child.isNull())
    {
        warning( tr("ODD: XML Parser Error"),
                             tr("Missing <header> element!"));
        return false;
    }
    else
    {
        if (!parseHeaderElement(child))
		{
			return false;
		}
    }
	
    // RoadSystem / Tile //
    //

    parseTile(root);
    parseRoadSystem(root);

    
    roadSystem_->verify();
    roadSystem_->updateControllers();

    // VehicleSystem //
    //
    parseVehicleSystem(root);

    // PedestrianSystem //
    //
    parsePedestrianSystem(root);

    // ScenerySystem //
    //
    parseScenerySystem(root);

    // Postprocessing //
    //
    postprocess();

    return true;
}

//################//
// PROTOTYPES     //
//################//

/*! \brief Opens a prototype file/device and parses the prototypes.
*
*/
bool
DomParser::parsePrototypes(QIODevice *source)
{
	// Mode //
	//
	mode_ = DomParser::MODE_PROTOTYPES;
	for (int i = 0; i < odrID::NUM_IDs; i++)
	{
		if (!elementIDs_[i].empty())
		{
			elementIDs_[i].clear();
		}
	}

    // Open file and parse tree //
    //
    QString errorStr = "";
    int errorLine = 0;
    int errorColumn = 0;

    if (!doc_->setContent(source, true, &errorStr, &errorLine, &errorColumn))
    {
        warning( tr("ODD: XML Parser Error"),
                             tr("Parse error at line %1, column %2:\n%3")
                                 .arg(errorLine)
                                 .arg(errorColumn)
                                 .arg(errorStr));
        return false;
    }

    // Root Element //
    //
    QDomElement root = doc_->documentElement();
    if (root.tagName() != "ODDLot")
    {
        warning( tr("ODD: XML Parser Error"),
                             tr("Root element is not <ODDLot>!"));
        return false;
    }

	QDomElement header = root.firstChildElement("header");
	if (header.isNull())
	{
		opendriveVersion_ = 1.0;
	}
	else
	{
		if (!parseHeaderElement(header))
		{
			return false;
		}
	}


    // Prototypes //
    //
    QDomElement prototypesRoot;
    prototypesRoot = root.firstChildElement("prototypes");
    if (prototypesRoot.isNull())
    {
        warning( tr("ODD: XML Parser Error"),
                             tr("Missing <prototypes> element!"));
        return false;
    }

    // RoadSystem //
    //
    roadSystem_ = NULL; // No RoadSystem for prototypes

    // Merge Spiral-Arc-Spiral Tracks //
    //
    SpArcSMergeVisitor *spArcSMergeVisitor = new SpArcSMergeVisitor();

    // RoadType Prototypes //
    //
    QDomElement prototypes;
    prototypes = prototypesRoot.firstChildElement("roadTypePrototypes");
    if (!prototypes.isNull())
    {
        QDomElement prototype = prototypes.firstChildElement("roadTypePrototype");
        while (!prototype.isNull())
        {
            // Name and Icon //
            //
            QString name = parseToQString(prototype, "name", "noname", false); // mandatory
            QString icon = parseToQString(prototype, "icon", "", true); // optional
            QString system = parseToQString(prototype, "system", "", true); // optional
            QString type = parseToQString(prototype, "type", "", true); // optional
            QString lanes = parseToQString(prototype, "lanes", "", true); // optional

            // Road //
            //
            QDomElement child = prototype.firstChildElement("road");
            RSystemElementRoad *road = parseRoadElement(child);
            road->accept(spArcSMergeVisitor);

            ODD::mainWindow()->getPrototypeManager()->addRoadPrototype(name, QIcon(icon), road, PrototypeManager::PTP_RoadTypePrototype,system,type,lanes);

            prototype = prototype.nextSiblingElement("roadTypePrototype");
        }
    }

    // TrackPrototypes //
    //
    prototypes = prototypesRoot.firstChildElement("trackPrototypes");
    if (!prototypes.isNull())
    {
        QDomElement prototype = prototypes.firstChildElement("trackPrototype");
        while (!prototype.isNull())
        {
            // Name and Icon //
            //
            QString name = parseToQString(prototype, "name", "noname", false); // mandatory
            QString icon = parseToQString(prototype, "icon", "", true); // optional
            QString system = parseToQString(prototype, "system", "", true); // optional
            QString type = parseToQString(prototype, "type", "", true); // optional
            QString lanes = parseToQString(prototype, "lanes", "", true); // optional

            // Road //
            //
            QDomElement child = prototype.firstChildElement("road");
            RSystemElementRoad *road = parseRoadElement(child);
            road->accept(spArcSMergeVisitor);

            ODD::mainWindow()->getPrototypeManager()->addRoadPrototype(name, QIcon(icon), road, PrototypeManager::PTP_TrackPrototype,system,type,lanes);

            prototype = prototype.nextSiblingElement("trackPrototype");
        }
    }

    // Elevation Prototypes //
    //
    prototypes = prototypesRoot.firstChildElement("elevationPrototypes");
    if (!prototypes.isNull())
    {
        QDomElement prototype = prototypes.firstChildElement("elevationPrototype");
        while (!prototype.isNull())
        {
            // Name and Icon //
            //
            QString name = parseToQString(prototype, "name", "noname", false); // mandatory
            QString icon = parseToQString(prototype, "icon", "", true); // optional
            QString system = parseToQString(prototype, "system", "", true); // optional
            QString type = parseToQString(prototype, "type", "", true); // optional
            QString lanes = parseToQString(prototype, "lanes", "", true); // optional

            // Road //
            //
            QDomElement child = prototype.firstChildElement("road");
            RSystemElementRoad *road = parseRoadElement(child);
            road->accept(spArcSMergeVisitor);

            ODD::mainWindow()->getPrototypeManager()->addRoadPrototype(name, QIcon(icon), road, PrototypeManager::PTP_ElevationPrototype,system,type,lanes);

            prototype = prototype.nextSiblingElement("elevationPrototype");
        }
    }

    // Superelevation Prototypes //
    //
    prototypes = prototypesRoot.firstChildElement("superelevationPrototypes");
    if (!prototypes.isNull())
    {
        QDomElement prototype = prototypes.firstChildElement("superelevationPrototype");
        while (!prototype.isNull())
        {
            // Name and Icon //
            //
            QString name = parseToQString(prototype, "name", "noname", false); // mandatory
            QString icon = parseToQString(prototype, "icon", "", true); // optional
            QString system = parseToQString(prototype, "system", "", true); // optional
            QString type = parseToQString(prototype, "type", "", true); // optional
            QString lanes = parseToQString(prototype, "lanes", "", true); // optional

            // Road //
            //
            QDomElement child = prototype.firstChildElement("road");
            RSystemElementRoad *road = parseRoadElement(child);
            road->accept(spArcSMergeVisitor);

            ODD::mainWindow()->getPrototypeManager()->addRoadPrototype(name, QIcon(icon), road, PrototypeManager::PTP_SuperelevationPrototype,system,type,lanes);

            prototype = prototype.nextSiblingElement("superelevationPrototype");
        }
    }

    // Crossfall Prototypes //
    //
    prototypes = prototypesRoot.firstChildElement("crossfallPrototypes");
    if (!prototypes.isNull())
    {
        QDomElement prototype = prototypes.firstChildElement("crossfallPrototype");
        while (!prototype.isNull())
        {
            // Name and Icon //
            //
            QString name = parseToQString(prototype, "name", "noname", false); // mandatory
            QString icon = parseToQString(prototype, "icon", "", true); // optional
            QString system = parseToQString(prototype, "system", "", true); // optional
            QString type = parseToQString(prototype, "type", "", true); // optional
            QString lanes = parseToQString(prototype, "lanes", "", true); // optional

            // Road //
            //
            QDomElement child = prototype.firstChildElement("road");
            RSystemElementRoad *road = parseRoadElement(child);
            road->accept(spArcSMergeVisitor);

            ODD::mainWindow()->getPrototypeManager()->addRoadPrototype(name, QIcon(icon), road, PrototypeManager::PTP_CrossfallPrototype,system,type,lanes);

            prototype = prototype.nextSiblingElement("crossfallPrototype");
        }
    }

	// RoadShape Prototypes //
	//
	prototypes = prototypesRoot.firstChildElement("shapePrototypes");
	if (!prototypes.isNull())
	{
		QDomElement prototype = prototypes.firstChildElement("shapePrototype");
		while (!prototype.isNull())
		{
			// Name and Icon //
			//
			QString name = parseToQString(prototype, "name", "noname", false); // mandatory
			QString icon = parseToQString(prototype, "icon", "", true); // optional
			QString system = parseToQString(prototype, "system", "", true); // optional
			QString type = parseToQString(prototype, "type", "", true); // optional
			QString lanes = parseToQString(prototype, "lanes", "", true); // optional

																		  // Road //
																		  //
			QDomElement child = prototype.firstChildElement("road");
			RSystemElementRoad *road = parseRoadElement(child);
			road->accept(spArcSMergeVisitor);

			ODD::mainWindow()->getPrototypeManager()->addRoadPrototype(name, QIcon(icon), road, PrototypeManager::PTP_RoadShapePrototype, system, type, lanes);

			prototype = prototype.nextSiblingElement("shapePrototype");
		}
	}

    // LaneSection Prototypes //
    //
    prototypes = prototypesRoot.firstChildElement("laneSectionPrototypes");
    if (!prototypes.isNull())
    {
        QDomElement prototype = prototypes.firstChildElement("laneSectionPrototype");
        while (!prototype.isNull())
        {
            // Name and Icon //
            //
            QString name = parseToQString(prototype, "name", "noname", false); // mandatory
            QString icon = parseToQString(prototype, "icon", "", true); // optional
            QString system = parseToQString(prototype, "system", "", true); // optional
            QString type = parseToQString(prototype, "type", "", true); // optional
            QString lanes = parseToQString(prototype, "lanes", "", true); // optional

            // Road //
            //
            QDomElement child = prototype.firstChildElement("road");
            RSystemElementRoad *road = parseRoadElement(child);
            road->accept(spArcSMergeVisitor);

            ODD::mainWindow()->getPrototypeManager()->addRoadPrototype(name, QIcon(icon), road, PrototypeManager::PTP_LaneSectionPrototype,system,type,lanes);

            prototype = prototype.nextSiblingElement("laneSectionPrototype");
        }
    }

    // RoadSystem Prototypes //
    //
    prototypes = prototypesRoot.firstChildElement("roadSystemPrototypes");
    if (!prototypes.isNull())
    {
        QDomElement prototype = prototypes.firstChildElement("roadSystemPrototype");
        while (!prototype.isNull())
        {
            // Name and Icon //
            //
            QString name = parseToQString(prototype, "name", "noname", false); // mandatory
            QString icon = parseToQString(prototype, "icon", "", true); // optional

            // RoadSystem //
            //
            roadSystem_ = new RoadSystem();
            parseRoadSystem(prototype);

            foreach (RSystemElementRoad *road, roadSystem_->getRoads())
            {
                if (road->getTrackSections().isEmpty())
                {
                    warning( tr("ODD: XML Parser Error"),
                                         tr("Prototype has a road without Tracks (PlanView)!"));
                    return false;
                }

                if (road->getLaneSections().isEmpty())
                {
                    warning( tr("ODD: XML Parser Error"),
                                         tr("Prototype has a road without LaneSections!"));
                    return false;
                }

                road->accept(spArcSMergeVisitor);
            }

            ODD::mainWindow()->getPrototypeManager()->addRoadSystemPrototype(name, QIcon(icon), roadSystem_);

            prototype = prototype.nextSiblingElement("roadSystemPrototype");
        }
    }

    // Postprocessing //
    //
    //	postprocess();

    return true;
}

//################//
// POSTPROCESSING //
//################//

void
DomParser::postprocess()
{
    // Reset change //
    //
    projectData_->getChangeManager()->notifyObservers(); // why here? second one should suffice
    //	ClearChangesVisitor * clearVisitor = new ClearChangesVisitor();
    //	projectData_->accept(clearVisitor, true);

    // Merge Spiral-Arc-Spiral Tracks //
    //
    SpArcSMergeVisitor *spArcSMergeVisitor = new SpArcSMergeVisitor();
    roadSystem_->accept(spArcSMergeVisitor);

    // Reset change //
    //
    projectData_->getChangeManager()->notifyObservers();
}

//###################//
// SUPPORT FUNCTIONS //
//###################//

/*! \brief Support function that prints out a warning message.
*
*/
bool
DomParser::check(bool success, const QDomElement &element, const QString &attributeName, const QString &type)
{
    if (!success)
    {
        warning( tr("ODD: XML Parser Error"),
                             tr("Error parsing attribute \"%2\" of element <%1> in line %4 to type %3. A default value will be used. This can lead to major problems.")
                                 .arg(element.tagName())
                                 .arg(attributeName)
                                 .arg(type)
                                 .arg(element.lineNumber()));
    }
    return success;
}

/*! \brief Parses an attribute to int.
*/
int
DomParser::parseToInt(const QDomElement &element, const QString &attributeName, int defaultValue, bool optional)
{
    if (element.hasAttribute(attributeName))
    {
        // Attribute found //
        //
        // If an attribute is found, it must be valid (even if it is optional)
        bool parsed = false;
        int parsedValue = element.attribute(attributeName, "").toInt(&parsed);
        if (parsed)
        {
            // parsing successfull
            return parsedValue;
        }
        else
        {
            // parsing error -> warning message
            check(parsed, element, attributeName, "int");
            return defaultValue;
        }
    }
    else
    {
        // Attribute not found //
        //
        // if mandatory -> warning message
        if (!optional)
            check(false, element, attributeName, "int");
        return defaultValue;
    }
}

/*! \brief Parses an attribute to float.
*/
float
DomParser::parseToFloat(const QDomElement &element, const QString &attributeName, float defaultValue, bool optional)
{
    if (element.hasAttribute(attributeName))
    {
        // Attribute found //
        //
        // If an attribute is found, it must be valid (even if it is optional)
        bool parsed = false;
        float parsedValue = element.attribute(attributeName, "").toFloat(&parsed);

        if (parsed)
        {
            // parsing successfull
            return parsedValue;
        }
        else
        {
            // parsing error -> warning message
            check(parsed, element, attributeName, "float");
            return defaultValue;
        }
    }
    else
    {
        // Attribute not found //
        //
        // if mandatory -> warning message
        if (!optional)
            check(false, element, attributeName, "float");
        return defaultValue;
    }
}

/*! \brief Parses an attribute to double.
*/
double
DomParser::parseToDouble(const QDomElement &element, const QString &attributeName, double defaultValue, bool optional)
{
    if (element.hasAttribute(attributeName))
    {
        // Attribute found //
        //
        // If an attribute is found, it must be valid (even if it is optional)
        bool parsed = false;
        double parsedValue = element.attribute(attributeName, "").toDouble(&parsed);
        if (parsed)
        {
            // parsing successfull
            return parsedValue;
        }
        else
        {
            // parsing error -> warning message
            check(parsed, element, attributeName, "double");
            return defaultValue;
        }
    }
    else
    {
        // Attribute not found //
        //
        // if mandatory -> warning message
        if (!optional)
            check(false, element, attributeName, "double");
        return defaultValue;
    }
}

/*! \brief Parses an attribute to QString.
*/
QString
DomParser::parseToQString(const QDomElement &element, const QString &attributeName, const QString &defaultValue, bool optional)
{
    if (element.hasAttribute(attributeName))
    {
        // Attribute found //
        //
        QString parsedValue = element.attribute(attributeName, "");
        return parsedValue;
    }
    else
    {
        // Attribute not found //
        //
        // if mandatory -> warning message
        if (!optional)
            check(false, element, attributeName, "String");
        return defaultValue;
    }
}


//################//
// HEADER         //
//################//

/** Parses a <header> element.
*
*/
bool
DomParser::parseHeaderElement(QDomElement &element)
{
    int revMajor = parseToInt(element, "revMajor", 1, false); // mandatory
    int revMinor = parseToInt(element, "revMinor", 2, false); // mandatory

    QString name = parseToQString(element, "name", "Untitled", true); // optional
    opendriveVersion_ = parseToFloat(element, "version", 1.0f, true); // optional
	
	if ((opendriveVersion_ > ODD::getVersion()) || (revMajor > ODD::getRevMajor()) || (revMinor > ODD::getRevMinor()))
	{
		qDebug() << "Oddlot only supports OpenDrive versions up to 1.4";
		return false;
	}

    QString date = parseToQString(element, "date", "", true); // optional

    double north = parseToDouble(element, "north", 10000.0, true); // optional
    double south = parseToDouble(element, "south", -10000.0, true); // optional
    double east = parseToDouble(element, "east", 10000.0, true); // optional
    double west = parseToDouble(element, "west", -10000.0, true); // optional

    projectData_->setRevMajor(revMajor);
    projectData_->setRevMinor(revMinor);

    projectData_->setName(name);
    projectData_->setVersion(opendriveVersion_);
    projectData_->setDate(date);

    if (north > projectData_->getNorth())
    {
        projectData_->setNorth(north);
    }
    if (south < projectData_->getSouth())
    {
        projectData_->setSouth(south);
    }
    if (east > projectData_->getEast())
    {
        projectData_->setEast(east);
    }
    if (west < projectData_->getWest())
    {
        projectData_->setWest(west);
    }


	// <OpenDRIVE><georeference> //
	//

	QDomElement child = element.firstChildElement("geoReference");
	if (!child.isNull())
	{
		QString params = child.text();
		GeoReference *georeference = new GeoReference(params);
		projectData_->setGeoReference(georeference);

		qDebug() << "Georefernce: " << params;
	}

    return true;
}

//################//
// TILE           //
//################//

bool
DomParser::parseTile(QDomElement &root)
{
	QDomElement ancillary = root.firstChildElement("userData");
	while (!ancillary.isNull())
	{
		QString code = parseToQString(ancillary, "code", "", true);
		QString value = parseToQString(ancillary, "value", "", true);

		if (code == "tile")
		{
			QStringList param = value.split(" ");
			if (param.length() > 1)
			{
				int intID = param[0].toInt();
				odrID id;
				id.setID(intID);
				id.setName(param[1]);
				tileSystem_->addTile(new Tile(id));
			}
		}
		else
		{
		}
		ancillary = ancillary.nextSiblingElement("userData");
	}
    
    return true;
}


//################//
// ROAD           //
//################//

/*! \brief Parses the RoadSystem.
*
*/
bool
DomParser::parseRoadSystem(QDomElement &root)
{

    // <OpenDRIVE><road> //
    //

    QDomElement child = root.firstChildElement("road");
    if (child.isNull())
    {
        warning( tr("ODD: XML Parser Error"),
                             tr("Missing <road> element!"));
        return false;
    }
    else
    {
        while (!child.isNull())
        {
            RSystemElementRoad *road = parseRoadElement(child);

            child = child.nextSiblingElement("road");
        }
    }

    // Optional Elements //
    //
    child = root.firstChildElement("controller");
    while (!child.isNull())
    {
        parseControllerElement(child);
        child = child.nextSiblingElement("controller");
    }

    child = root.firstChildElement("junction");
    while (!child.isNull())
    {
        parseJunctionElement(child);
        child = child.nextSiblingElement("junction");
    }

    child = root.firstChildElement("fiddleyard");
    while (!child.isNull())
    {
        parseFiddleyardElement(child);
        child = child.nextSiblingElement("fiddleyard");
    }

    child = root.firstChildElement("pedFiddleyard");
    while (!child.isNull())
    {
        parsePedFiddleyardElement(child);
        child = child.nextSiblingElement("pedFiddleyard");
    }

	child = root.firstChildElement("junctionGroup");
	while (!child.isNull())
	{
		parseJunctionGroupElement(child);
		child = child.nextSiblingElement("junctionGroup");
	}

    return true;
}

void DomParser::StringToID(QString id, odrID &ID, odrID::IDType t, int TileID)
{
	if (id == "-1" || id == "" ) // for example no junction
	{
		ID = odrID::invalidID();
		return;
	}
	ID = elementIDs_[t].value(id,odrID::invalidID());
	if (!ID.isInvalid()) // we already created an ID for this string, thus use it.
	{
		return;
	}
	ID.setName(id);
	ID.setType(t);
	ID.setTileID(TileID);
	if (tileSystem_ != NULL)
	{
		Tile *tile = tileSystem_->getTile(TileID);
		ID.setID(tile->uniqueID(t));
	}
	else
	{
		ID.setID(elementIDs_[t].size());
	}
	elementIDs_[t].insert(id, ID);
}

int DomParser::parseTileID(const QDomElement &element)
{
	int tileID = 0;
	if (tileSystem_ != NULL)
	{
		QDomElement ancillary = element.firstChildElement("userData");
		while (!ancillary.isNull())
		{
			QString code = parseToQString(ancillary, "code", "", true);
			QString value = parseToQString(ancillary, "value", "", true);
			if (code == "tile")
			{
				tileID = value.toInt();
			}
			ancillary = ancillary.nextSiblingElement("userData");
		}
		if (tileSystem_->getTile(tileID) == NULL)
		{
			if(tileID>0)
			{
				warning("tile not found", "tileID=" + QString::number(tileID));
			}
			tileSystem_->addTile(new Tile(tileID));
		}
	}
	return tileID;
}
/** Parses a <road> element.
*
*/
RSystemElementRoad *
DomParser::parseRoadElement(QDomElement &element)
{
	int tileID = parseTileID(element);

    // 1.) Parse Attributes //
    //
    QString name = parseToQString(element, "name", "Untitled", true); // optional
    //	double length = parseToDouble(element, "length", 0.0, true);			// optional
    QString id = parseToQString(element, "id", "", true); // "id" is optional...
    if (id.isEmpty())
    {
        id = parseToQString(element, "ID", "", false); // ...but at least "ID" should be there
    }
	odrID roadID;
	StringToID(id, roadID, odrID::ID_Road, tileID);
    QString junction = parseToQString(element, "junction", "-1", true); // optional

	odrID junctionID;
	StringToID(junction, junctionID, odrID::ID_Junction,tileID);

    RSystemElementRoad *road = new RSystemElementRoad(name, roadID, junctionID);

    if (projectData_) // Change ids not for Prototypes
    {
        // Check if the ids have the format [Tilenumber]_[Elementnumber]_[Name]
        //
        odrID id = road->getID();
		tileSystem_->addTileIfNecessary(id);

        roadSystem_->addRoad(road); // This may change the ID!
       
    }
    else if (roadSystem_) // load as prototype --> IDs are only unique within prototypes but not globally
		// they well be changed once they are added to a read roadSystem
    {
		odrID id;
		id.setID(prototypeElementCount_++);
		id.setName(road->getName());
		id.setType(odrID::ID_Road);
        road->setID(id);
        roadSystem_->addRoad(road); // This may change the ID!
    }

    // 2.) Parse children //
    //
	QDomElement child;
	QDomElement lanes;

    // <link> <predecessor> (optional, max count: 1) //
    child = element.firstChildElement("link").firstChildElement("predecessor");
    if (!child.isNull())
    {
        QString elementType = parseToQString(child, "elementType", "", false); // mandatory
        QString elementId = parseToQString(child, "elementId", "", false); // mandatory
		
        bool isOptional = (elementType == "junction");
        QString contactPoint = parseToQString(child, "contactPoint", "", isOptional); // optional if junction
        if (!elementId.isEmpty())
        {
			odrID ID;
			RoadLink *roadLink = NULL;
			if (elementType == "junction")
			{
				StringToID(elementId, ID, odrID::ID_Junction, tileID);
				roadLink = new RoadLink(elementType, ID, JunctionConnection::JCP_NONE);
			}
			else
			{
				StringToID(elementId, ID, odrID::ID_Road,tileID);
				roadLink = new RoadLink(elementType, ID, JunctionConnection::parseContactPoint(contactPoint));
			}

            road->setPredecessor(roadLink);
        }
    }

    // <link> <successor> (optional, max count: 1) //
    child = element.firstChildElement("link").firstChildElement("successor");
    if (!child.isNull())
    {
        QString elementType = parseToQString(child, "elementType", "", false); // mandatory
        QString elementId = parseToQString(child, "elementId", "", false); // mandatory
        bool isOptional = (elementType == "junction");
        QString contactPoint = parseToQString(child, "contactPoint", "", isOptional); // optional if junction
        if (!elementId.isEmpty())
        {
			odrID ID;
			RoadLink *roadLink = NULL;
			if (elementType == "junction")
			{
				StringToID(elementId, ID, odrID::ID_Junction, tileID);
				roadLink = new RoadLink(elementType, ID, JunctionConnection::JCP_NONE);
			}
			else
			{
				StringToID(elementId, ID, odrID::ID_Road, tileID);
				roadLink = new RoadLink(elementType, ID, JunctionConnection::parseContactPoint(contactPoint));
			}
            road->setSuccessor(roadLink);
        }
    }

    // road:type (optional, max count: unlimited) //
    child = element.firstChildElement("type");
    if (!child.isNull())
    {
        while (!child.isNull())
        {
            parseTypeElement(child, road);
            child = child.nextSiblingElement("type");
        }
    }
    else
    {
        if (mode_ != DomParser::MODE_PROTOTYPES)
        {
            // default entry (do not use this with prototypes) //
            //
            TypeSection *typeSection = new TypeSection(0.0, TypeSection::RTP_UNKNOWN);
            road->addTypeSection(typeSection);
        }
    }

    // road:surface (optional, max count: 1)
    //     road:surface:CRG (optional, max count: unlimited)
    child = element.firstChildElement("surface");
    if (!child.isNull())
    {
        parseSurfaceElement(child, road);
    }

    // <planView> (mandatory, max count: 1)
    //		<geometry> (mandatory, max count: unlimited) //
    child = element.firstChildElement("planView");
    if (!child.isNull())
    {
        child = child.firstChildElement("geometry");
        if (child.isNull())
        {

            // TODO: NOT OPTIONAL
            qDebug("NOT OPTIONAL: <planView><geometry>");
        }
        while (!child.isNull())
        {
            parseGeometryElement(child, road);
            child = child.nextSiblingElement("geometry");
        }
    }
    else
    {
        // TODO: NOT OPTIONAL
        //		qDebug("NOT OPTIONAL: <planView>");
    }

    // <elevationProfile> //
    child = element.firstChildElement("elevationProfile");
    bool foundElevation = false;
    if (!child.isNull())
    {
        child = child.firstChildElement("elevation");
        while (!child.isNull())
        {
            foundElevation = true;
            parseElevationElement(child, road);
            child = child.nextSiblingElement("elevation");
        }
    }

    if (!foundElevation && (mode_ != DomParser::MODE_PROTOTYPES))
    {
        // default entry (do not use this with prototypes) //
        //
        ElevationSection *section = new ElevationSection(0.0, 0.0, 0.0, 0.0, 0.0);
        road->addElevationSection(section);
    }

    // <lateralProfile> //
    bool foundLateral = false;

    child = element.firstChildElement("lateralProfile");
    if (!child.isNull())
    {
        child = child.firstChildElement("superelevation");
        while (!child.isNull())
        {
            foundLateral = true;
            parseSuperelevationElement(child, road);
            child = child.nextSiblingElement("superelevation");
        }
    }

    child = element.firstChildElement("lateralProfile");
    if (!child.isNull())
    {
        child = child.firstChildElement("crossfall");
        while (!child.isNull())
        {
            foundLateral = true;
            parseCrossfallElement(child, road);
            child = child.nextSiblingElement("crossfall");
        }
    }


    // <lanes> //
    lanes = element.firstChildElement("lanes");
    if (!lanes.isNull())
    {
        child = lanes.firstChildElement("laneSection");
        if (child.isNull())
        {

            // TODO: NOT OPTIONAL
            qDebug("NOT OPTIONAL: <lanes><laneSection>");
        }
        while (!child.isNull())
        {
            parseLaneSectionElement(child, road);
            child = child.nextSiblingElement("laneSection");
        }
		child = lanes.firstChildElement("laneOffset"); // optional for id=0 (center)
		while (!child.isNull())
		{
			double sOffset = parseToDouble(child, "s", 0.0, false); // mandatory
			double a = parseToDouble(child, "a", 0.0, false); // mandatory
			double b = parseToDouble(child, "b", 0.0, false); // mandatory
			double c = parseToDouble(child, "c", 0.0, false); // mandatory
			double d = parseToDouble(child, "d", 0.0, false); // mandatory

			LaneOffset *offsetEntry = new LaneOffset(sOffset, a, b, c, d);
			road->addLaneOffset(offsetEntry);

			child = child.nextSiblingElement("laneOffset");
		}
    }
    else
    {
        // TODO: NOT OPTIONAL
        //		qDebug("NOT OPTIONAL: <lanes>");
    }

	bool foundShape = false;
	child = element.firstChildElement("lateralProfile");
	if (!child.isNull())
	{
		child = child.firstChildElement("shape");
		while (!child.isNull())
		{
			foundShape = true;
			parseShapeElement(child, road);
			child = child.nextSiblingElement("shape");
		}
	}

	if (!foundLateral)
	{
		if (mode_ != DomParser::MODE_PROTOTYPES)
		{
			SuperelevationSection *sESection = new SuperelevationSection(0.0, 0.0, 0.0, 0.0, 0.0);
			road->addSuperelevationSection(sESection);

			CrossfallSection *cSection = new CrossfallSection(CrossfallSection::DCF_SIDE_BOTH, 0.0, 0.0, 0.0, 0.0, 0.0);
			road->addCrossfallSection(cSection);
		}
	}

	if (!foundShape)
	{
		if (mode_ != DomParser::MODE_PROTOTYPES)
		{
			ShapeSection *sSection = new ShapeSection(0.0, road->getMinWidth(0.0));
			road->addShapeSection(sSection);
		}
	}
	else
	{
		foreach(ShapeSection *section, road->getShapeSections())
		{
			foreach(PolynomialLateralSection *poly, section->getShapes())
			{
				poly->getControlPointsFromParameters(mode_ != DomParser::MODE_PROTOTYPES);
			}
		}
	}

    // <objects>                //
    // (optional, max count: 1) //
    child = element.firstChildElement("objects");
    if (!child.isNull())
    {
        parseObjectsElement(child, road);

        // Check that the maximum (1) is not exceeded
        child = child.nextSiblingElement("objects");
        if (!child.isNull())
            qDebug("WARNING: maximum of one <objects> element, ignoring any subsequent ones");
    }

    // <signals> //
    child = element.firstChildElement("signals");
    if (!child.isNull())
    {
        parseSignalsElement(child, road);

        // Check that the maximum (1) is not exceeded
        child = child.nextSiblingElement("signals");
        if (!child.isNull())
            qDebug("WARNING: maximum of one <signals> element, ignoring any subsequent ones");
    }
    // <sensor> //
    child = element.firstChildElement("sensors");
    if (!child.isNull())
    {
        parseSensorsElement(child, road);

        // Check that the maximum (1) is not exceeded
        child = child.nextSiblingElement("sensors");
        if (!child.isNull())
            qDebug("WARNING: maximum of one <sensors> element, ignoring any subsequent ones");
    }

    // 3.) Clean up //
    //
    // TODO: check if all s coordinates are OK...

    return road;
}

/*! \brief Parses a road:type element.
*
*/
bool
DomParser::parseTypeElement(QDomElement &element, RSystemElementRoad *road)
{
    double s = parseToDouble(element, "s", 0.0, false); // mandatory
    QString type = parseToQString(element, "type", "unknown", false); // mandatory


    TypeSection *typeSection = new TypeSection(s, TypeSection::parseRoadType(type));

    
    QDomElement speedRecord;
    speedRecord = element.firstChildElement("speed");
    if (!speedRecord.isNull())
    {
        parseSpeedElement(speedRecord, typeSection);
    }

    road->addTypeSection(typeSection);

    return true;
}

/*! \brief Parses a road:type:speed element.
*
*/
bool
DomParser::parseSpeedElement(QDomElement &element, TypeSection *type)
{
    QString max = parseToQString(element, "max", "undefined", false); // mandatory
    QString unit = parseToQString(element, "unit", "m/s", true); // otional
	qDebug() << "Unit: " << unit;

    SpeedRecord *sr = new SpeedRecord(max, unit);
    type->setSpeedRecord(sr);

    return true;
}

/*! \brief Parses a road:objects element.
*
*/
bool
DomParser::parseObjectsElement(QDomElement &element, RSystemElementRoad *road)
{
    // Find all objects (unlimited)
    QDomElement child = element.firstChildElement("object");
    while (!child.isNull())
	{
		int tileID = parseTileID(child);
		Object::ObjectProperties objectProps;

        // Get mandatory attributes
       objectProps.type = parseToQString(child, "type", "", false); // mandatory

        // Don't create objects for the simple poles of signs and the traffic lights

		if (objectProps.type != "simplePole")
		{
			QString name = parseToQString(child, "name", "", false); // mandatory

			bool pole = false;

			QStringList nameParts = name.split("_"); // return only name in type.typeSubclass-subtype_name_p
			if (nameParts.size() > 1)
			{
				name = nameParts.at(1);
				if (name == nameParts.at(0)) // only to remove the bugs
				{
					name = "";
				}

				if (name == "p")
				{
					name = "";
					pole = true;
				}
				else if (nameParts.size() > 2)
				{
					pole = true;
				}
			}
			else
			{
				name = "";
			}

			QString id = parseToQString(child, "id", "", false); // mandatory
			QDomElement ancillary = child.firstChildElement("userData");
			QString modelFile = parseToQString(child, "modelFile", name, true); // optional
			QString textureFile = parseToQString(child, "textureFile", name, true); // optional
			while (!ancillary.isNull())
			{
				QString code = parseToQString(ancillary, "code", "", true);
				QString value = parseToQString(ancillary, "value", "", true);

				if (code == "textureFile")
				{
					textureFile = value;
				}
				else
				{
					modelFile = value;
				}

				ancillary = ancillary.nextSiblingElement("userData");
			}

			double s = parseToDouble(child, "s", 0.0, false); // mandatory
			objectProps.t = parseToDouble(child, "t", 0.0, false); // mandatory
			objectProps.zOffset = parseToDouble(child, "zOffset", 0.0, true); // optional
			objectProps.validLength = parseToDouble(child, "validLength", 0.0, true); // optional
			QString orientationString = parseToQString(child, "orientation", "+", true); // optional
			objectProps.orientation = Signal::parseOrientationType(orientationString);

			objectProps.length = parseToDouble(child, "length", 0.0, true); // optional
			objectProps.width = parseToDouble(child, "width", 0.0, true); // optional
			objectProps.radius = parseToDouble(child, "radius", 0.0, true); // optional
			objectProps.height = parseToDouble(child, "height", 0.0, true); // optional
			objectProps.hdg = parseToDouble(child, "hdg", 0.0, true) * 180.0 / (M_PI); // optional
			objectProps.pitch = parseToDouble(child, "pitch", 0.0, true) * 180.0 / (M_PI); // optional
			objectProps.roll = parseToDouble(child, "roll", 0.0, true) * 180.0 / (M_PI); // optional

			Object::ObjectRepeatRecord repeatProps{ -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0 };

			// Get <repeat> record
			QDomElement objectChild = child.firstChildElement("repeat");
			if (!objectChild.isNull())
			{
				repeatProps.s = parseToDouble(objectChild, "s", 0.0, false); // mandatory
				repeatProps.length = parseToDouble(objectChild, "length", 0.0, false); // mandatory
				repeatProps.distance = parseToDouble(objectChild, "distance", 0.0, false); // mandatory

				if (opendriveVersion_ >= 1.4)
				{
					repeatProps.tStart = parseToDouble(objectChild, "tStart", 0.0, false);
					repeatProps.tEnd = parseToDouble(objectChild, "tEnd", 0.0, false);
					repeatProps.widthStart = parseToDouble(objectChild, "widthStart", 0.0, false);
					repeatProps.widthEnd = parseToDouble(objectChild, "widthEnd", 0.0, false);
					repeatProps.heightStart = parseToDouble(objectChild, "heightStart", 0.0, false);
					repeatProps.heightEnd = parseToDouble(objectChild, "heightEnd", 0.0, false);
					repeatProps.zOffsetStart = parseToDouble(objectChild, "zOffsetStart", 0.0, false);
					repeatProps.zOffsetEnd = parseToDouble(objectChild, "zOffsetEnd", 0.0, false);
				}
			}


			// Convert old files
			/*	if ((type == "Tree") && (height < 2.0))
			{
			type = "Bush";
			}
			else if ((type == "PoleTraverse") && (width > 1.0))
			{
			type = type + QString("_%1").arg(int(width * 10));
			}
			*/

			// Construct object object;

			if (objectProps.type != "")
			{

				odrID ID;
				StringToID(id, ID, odrID::ID_Object,tileID);
				Object *object = new Object(ID, name, s, objectProps, repeatProps, textureFile);

				tileSystem_->addTileIfNecessary(ID);

				// Add to road
				road->addObject(object);

				// Get <parkingSpace> record
				QDomElement objectChild = child.firstChildElement("parkingSpace");
				if (!objectChild.isNull())
				{
					QString access = parseToQString(objectChild, "access", "all", false); // mandatory
					QString restrictions = parseToQString(objectChild, "restrictions", "", false); // mandatory

					ParkingSpace *parking = new ParkingSpace(object, ParkingSpace::parseParkingSpaceAccess(access), restrictions);

					QDomElement markingChild = objectChild.firstChildElement("marking");
					while (!markingChild.isNull())
					{
						QString side = parseToQString(markingChild, "side", "", false); // mandatory
						QString type = parseToQString(markingChild, "type", "none", false);
						double width = parseToDouble(markingChild, "width", 0.0, false);
						QString color = parseToQString(markingChild, "color", "standard", false);

						if (!parking->addMarking(side, type, width, color))
						{
							warning( tr("ODD: XML Parser Error"),
								tr("Error parsing attribute \"%2\" of element <%1> in line %3. This value is not defined. This can lead to major problems.")
								.arg(markingChild.tagName())
								.arg("side")
								.arg(markingChild.lineNumber()));
						}

						markingChild = markingChild.nextSiblingElement("marking");
					}
					object->setParkingSpace(parking);
				}

				// Get <parkingSpace> record
				objectChild = child.firstChildElement("outline");
				if (!objectChild.isNull())
				{
					QDomElement cornerChild = objectChild.firstChildElement("cornerRoad");

					// Corners of the outline of the object //
					//
					QList<ObjectCorner *> corners;

					// Road coordinates
					//
					while (!cornerChild.isNull())
					{
						double u = parseToDouble(cornerChild, "u", 0.0, false);
						double v = parseToDouble(cornerChild, "v", 0.0, false);
						double dz = parseToDouble(cornerChild, "dz", 0.0, false);
						double height = parseToDouble(cornerChild, "height", 0.0, false);

						ObjectCorner *objectCorner = new ObjectCorner(u, v, dz, height, false);
						corners.append(objectCorner);

						cornerChild = cornerChild.nextSiblingElement("cornerRoad");
					}

					cornerChild = objectChild.firstChildElement("cornerLocal");

					// Local coordinates
					//
					while (!cornerChild.isNull())
					{
						double u = parseToDouble(cornerChild, "u", 0.0, false);
						double v = parseToDouble(cornerChild, "v", 0.0, false);
						double z = parseToDouble(cornerChild, "z", 0.0, false);
						double height = parseToDouble(cornerChild, "height", 0.0, false);

						ObjectCorner *objectCorner = new ObjectCorner(u, v, z, height, true);
						corners.append(objectCorner);

						cornerChild = cornerChild.nextSiblingElement("cornerLocal");
					}

					if (!corners.isEmpty())
					{
						Outline *outline = new Outline(corners);
						object->setOutline(outline);
					}
				}
			}
		}

		// Attempt to locate another object
		child = child.nextSiblingElement("object");
	}

	child = element.firstChildElement("objectReference");

	while (!child.isNull())
	{
		int tileID = parseTileID(element);
		// Get mandatory attributes
		QString id = parseToQString(child, "id", "", false); // mandatory
		double s = parseToDouble(child, "s", 0.0, false);
		double t = parseToDouble(child, "t", 0.0, false);
		double zOffset = parseToDouble(child, "zOffset", 0.0, false);
		double validLength = parseToDouble(child, "validLength", 0.0, false);
		QString orientation = parseToQString(child, "orientation", "none", false);

		// Get validity record
		QList<Signal::Validity> validities;
		QDomElement objectChild = child.firstChildElement("validity");
		while (!objectChild.isNull())
		{
			int fromLane = parseToInt(objectChild, "fromLane", 0, false); // mandatory
			int toLane = parseToInt(objectChild, "toLane", 0, false); // mandatory

			validities.append(Signal::Validity{ fromLane, toLane });

			objectChild = objectChild.nextSiblingElement("validity");
		}


		odrID refID;
		StringToID(id, refID, odrID::ID_Bridge,tileID);
		odrID ID;
		
		ObjectReference *objectReference = new ObjectReference(ID, NULL, refID, s, t, zOffset, validLength, Signal::parseOrientationType(orientation), validities);
		road->addObjectReference(objectReference);

		// Attempt to locate another signal
		child = child.nextSiblingElement("objectReference");
	}

	// Find all bridges (unlimited)
	child = element.firstChildElement("bridge");
	while (!child.isNull())
	{
		int tileID = parseTileID(element);
		// Get mandatory attributes
		QString type = parseToQString(child, "type", "", false); // mandatory
		QString name = parseToQString(child, "name", "", false); // mandatory
		QString modelFile = parseToQString(child, "modelFile", name, true); // optional
		QDomElement ancillary = child.firstChildElement("userData");
		while (!ancillary.isNull())
		{
			modelFile = parseToQString(ancillary, "value", name, true);

			ancillary = ancillary.nextSiblingElement("userData");
		}

		QString id = parseToQString(child, "id", "", false); // mandatory
		double s = parseToDouble(child, "s", 0.0, false); // mandatory
		double length = parseToDouble(child, "length", 0.0, false); // mandatory

		odrID ID;
		StringToID(id, ID, odrID::ID_Bridge,tileID);
		// Construct bridge object
		Bridge *bridge = new Bridge(ID, modelFile, name, Bridge::parseBridgeType(type), s, length);

		tileSystem_->addTileIfNecessary(ID);

		// Add to road
		road->addBridge(bridge);

		// Attempt to locate another object
		child = child.nextSiblingElement("bridge");
	}

	// Find all tunnels (unlimited)
	child = element.firstChildElement("tunnel");
	while (!child.isNull())
	{
		int tileID = parseTileID(element);
		// Get mandatory attributes
		QString type = parseToQString(child, "type", "", false); // mandatory
		QString name = parseToQString(child, "name", "", false); // mandatory
		QString modelFile = parseToQString(child, "modelFile", name, true); // optional

		QString id = parseToQString(child, "id", "", false); // mandatory
		double s = parseToDouble(child, "s", 0.0, false); // mandatory
		double length = parseToDouble(child, "length", 0.0, false); // mandatory

		double lighting = parseToDouble(child, "lighting", 0.0, false);
		double daylight = parseToDouble(child, "daylight", 0.0, false);

		odrID ID;
		StringToID(id, ID, odrID::ID_Object,tileID);
		// Construct tunnel object
		Tunnel *tunnel = new Tunnel(ID, modelFile, name, Tunnel::parseTunnelType(type), s, length, lighting, daylight);

		tileSystem_->addTileIfNecessary(ID);

		// Add to road
		road->addBridge(tunnel);

		// Attempt to locate another object
		child = child.nextSiblingElement("tunnel");
	}

	// Find all crosswalks (unlimited) and create a signal. Conversion for old files.
	//
	child = element.firstChildElement("crosswalk");
	while (!child.isNull())
	{
		int tileID = parseTileID(element);
		// Get mandatory attributes
		QString id = parseToQString(child, "id", "", false); // mandatory
		QString name = parseToQString(child, "name", "", false); // mandatory
		double s = parseToDouble(child, "s", 0.0, false); // mandatory
		double length = parseToDouble(child, "length", 0.0, false); // mandatory

		odrID ID;
		StringToID(id, ID, odrID::ID_Object, tileID);
		// Construct crosswalk object
		Crosswalk *crosswalk = new Crosswalk(ID, name, s, length);

		// Get optional attributes
		if (parseToQString(child, "crossProb", "", true).length() > 0)
			crosswalk->setCrossProb(parseToDouble(child, "crossProb", 0.5, true)); // optional
		if (parseToQString(child, "resetTime", "", true).length() > 0)
			crosswalk->setResetTime(parseToDouble(child, "resetTime", 20.0, true)); // optional
		if (parseToQString(child, "type", "", true).length() > 0)
			crosswalk->setType(parseToQString(child, "type", "crosswalk", true)); // optional
		if (parseToQString(child, "debugLvl", "", true).length() > 0)
			crosswalk->setDebugLvl(parseToInt(child, "debugLvl", 0, true)); // optional

		// Get <validity> record
		QDomElement crosswalkChild = child.firstChildElement("validity");
		if (!crosswalkChild.isNull())
		{
			if (parseToQString(crosswalkChild, "fromLane", "", true).length() > 0)
				crosswalk->setFromLane(parseToInt(crosswalkChild, "fromLane", 0, true)); // optional
			if (parseToQString(crosswalkChild, "toLane", "", true).length() > 0)
				crosswalk->setToLane(parseToInt(crosswalkChild, "toLane", 0, true)); // optional
		}

		ID;
		StringToID(id, ID, odrID::ID_Object,tileID);
		// Construct signal object
		Signal *signal = new Signal(ID, name, s, 0.0, "no", Signal::POSITIVE_TRACK_DIRECTION, 0.0, "Germany", "293", "", "-1", length, 0.0, 0.0, 0.0, "km/h", "", 0.0, 0.0, false, 2, crosswalk->getFromLane(), crosswalk->getToLane(), crosswalk->getCrossProb(), crosswalk->getResetTime());
		// Add to road
		road->addSignal(signal);

		// Add to road
		road->addCrosswalk(crosswalk);

		// Attempt to locate another crosswalk
		child = child.nextSiblingElement("crosswalk");
	} // Find all crosswalks (unlimited)

	// Return successfully
	return true;
}
/*! \brief Parses a road:objects element.
*
*/
bool
DomParser::parseSignalsElement(QDomElement &element, RSystemElementRoad *road)
{
    QDomElement child = element.firstChildElement("signal");
    while (!child.isNull())
	{
		int tileID = parseTileID(element);
        // Get mandatory attributes
        QString id = parseToQString(child, "id", "", false); // mandatory
        QString name = parseToQString(child, "name", "", false); // mandatory
        bool pole = false;

        QStringList nameParts = name.split("_"); // return only name in type.typeSubclass-subtype_name_p

        if (nameParts.size() > 1)
        {
            name = nameParts.at(1);
            /*		if (name == nameParts.at(0))		// only to remove the bugs
			{
				name = "";
			}
*/

            if (name == "p")
            {
                name = "";
                pole = true;
            }
            else if (nameParts.size() > 2)
            {
                pole = true;
            }
        }
        else
        {
            name = "";
        }

        double s = parseToDouble(child, "s", 0.0, false); // mandatory
        double t = parseToDouble(child, "t", 0.0, false); // mandatory

        QString dynamicString = parseToQString(child, "dynamic", "no", false); // mandatory

        bool dynamic = false;
        if (dynamicString == "yes")
        {
            dynamic = true;
        }
        else if (dynamicString == "no")
        {
            dynamic = false;
        }
        QString orientationString = parseToQString(child, "orientation", "+", false); // mandatory

        double zOffset = parseToDouble(child, "zOffset", 0.0, false); // mandatory
        QString country = parseToQString(child, "country", "Germany", false); // mandatory
        QString type = parseToQString(child, "type", "-1", false); // mandatory
        
		QString subtype = parseToQString(child, "subtype", "-1", false); // optional
        double value = parseToDouble(child, "value", 0.0, false); // optional

		QString unit = parseToQString(child, "unit", "km/h", true); //optional
        

		double pitch, width, height, roll, hOffset;
		QString text;
		if (opendriveVersion_ > 1.3)
		{
			pitch = parseToDouble(child, "pitch", 0.0, false); // mandatory
			text = parseToQString(child, "text", "", false);//mandatory
			width = parseToDouble(child, "width", 0.0, false);//mandatory
			height = parseToDouble(child, "height", 0.0, false);//mandatory
			roll = parseToDouble(child, "roll", 0.0, false); // mandatory
			hOffset = parseToDouble(child, "hOffset", 0.0, false); // mandatory
		}
		else
		{
			hOffset = parseToDouble(child, "hOffset", 0.0, true); // optional
			pitch = parseToDouble(child, "pitch", 0.0, true); // optional
			text = parseToQString(child, "text", "", true);//optional
			width = parseToDouble(child, "width", 0.0, true);//optional
			height = parseToDouble(child, "height", 0.0, true);//optional
			roll = parseToDouble(child, "roll", 0.0, true); // optional
		}

        // Get validity record

        int fromLane = 0;
        int toLane = 0;
        QDomElement objectChild = child.firstChildElement("validity");
        if (!objectChild.isNull())
        {
            fromLane = parseToInt(objectChild, "fromLane", 0, false); // mandatory
            toLane = parseToInt(objectChild, "toLane", 0, false); // mandatory
        }
        else
        {
            toLane = road->getValidLane(s, t);
        }

        Signal *signal = NULL;

        int size = parseToInt(child, "size", 2, true); // mandatory
        QString typeSubclass = parseToQString(child, "subclass", "", true);
        // conversion: we now need a String
        if (typeSubclass == "-1")
        {
            typeSubclass = "";
        }

        QDomElement ancillary = child.firstChildElement("userData");
        double crossProb = 0.0;
        double resetTime = 0.0;
        while (!ancillary.isNull())
        {
            QString code = parseToQString(ancillary, "code", "", true);
            QString value = parseToQString(ancillary, "value", "", true);

            // Pedestrian Crossing has additional ancillary data
            //
            if (code == "crossProb")
            {
                crossProb = value.toDouble();
            }
            else if (code == "resetTime")
            {
                resetTime = value.toDouble();
            }
            else if (code == "subclass")
            {
                typeSubclass = value;
            }
            else if (code == "size")
            {
                size = value.toInt();
            }

            ancillary = ancillary.nextSiblingElement("userData");
        }

		odrID ID;
		StringToID(id, ID, odrID::ID_Object,tileID);
        if ((type == "625") && (subtype == "10") && (typeSubclass == "20"))
        {
            hOffset = name.toDouble();

            // Construct signal object
            signal = new Signal(ID, "", s, t, dynamic, Signal::parseOrientationType(orientationString), zOffset, country, type, typeSubclass, subtype, value, hOffset, pitch  * 180.0 / (M_PI), roll  * 180.0 / (M_PI), unit, text, width, height, pole, size, fromLane, toLane, crossProb, resetTime);
        }
        else
        {
            // Construct signal object
            signal = new Signal(ID, name, s, t, dynamic, Signal::parseOrientationType(orientationString), zOffset, country, type, typeSubclass, subtype, value, hOffset * 180.0 / (M_PI), pitch  * 180.0 / (M_PI), roll  * 180.0 / (M_PI), unit, text, width, height, pole, size, fromLane, toLane, crossProb, resetTime);
        }


		tileSystem_->addTileIfNecessary(ID);

        // Add to road
        road->addSignal(signal);

        // Attempt to locate another signal
        child = child.nextSiblingElement("signal");
    } // Find all signals (unlimited)

	child = element.firstChildElement("signalReference");

	while (!child.isNull())
	{
		int tileID = parseTileID(child);
		// Get mandatory attributes
		QString id = parseToQString(child, "id", "", false); // mandatory
		double s = parseToDouble(child, "s", 0.0, false);
		double t = parseToDouble(child, "t", 0.0, false);
		QString orientation = parseToQString(child, "orientation", "both", false);

		// Get validity record
		QList<Signal::Validity> validities;
		QDomElement objectChild = child.firstChildElement("validity");
		while (!objectChild.isNull())
		{
			int fromLane = parseToInt(objectChild, "fromLane", 0, false); // mandatory
			int toLane = parseToInt(objectChild, "toLane", 0, false); // mandatory

			validities.append(Signal::Validity{ fromLane, toLane });

			objectChild = objectChild.nextSiblingElement("validity");
		}


		odrID refID;
		StringToID(id, refID, odrID::ID_Signal,tileID);
		odrID ID;

		SignalReference *signalReference = new SignalReference(ID, NULL, refID, s, t, Signal::parseOrientationType(orientation), validities);
		road->addSignalReference(signalReference);

		// Attempt to locate another signal
		child = child.nextSiblingElement("signalReference");
	}

    // Return successfully
    return true;
}
/*! \brief Parses a road:objects element.
*
*/
bool
DomParser::parseSensorsElement(QDomElement &element, RSystemElementRoad *road)
{
    QDomElement child = element.firstChildElement("sensor");
    while (!child.isNull())
    {
        // Get mandatory attributes
        QString id = parseToQString(child, "id", "", false); // mandatory
        double s = parseToDouble(child, "s", 0.0, false); // mandatory

        // Construct sensor object
        Sensor *sensor = new Sensor(id, s);

        // Add to road
        road->addSensor(sensor);

        // Attempt to locate another sensor
        child = child.nextSiblingElement("sensor");
    } // Find all sensors (unlimited)

    // Return successfully
    return true;
}

/*! \brief Parses a road:elevationProfile:elevation element.
*
*/
bool
DomParser::parseElevationElement(QDomElement &element, RSystemElementRoad *road)
{
    double s = parseToDouble(element, "s", 0.0, false); // mandatory
    double a = parseToDouble(element, "a", 0.0, false); // mandatory
    double b = parseToDouble(element, "b", 0.0, false); // mandatory
    double c = parseToDouble(element, "c", 0.0, false); // mandatory
    double d = parseToDouble(element, "d", 0.0, false); // mandatory

    ElevationSection *section = new ElevationSection(s, a, b, c, d);
    road->addElevationSection(section);

    return true;
}

/*! \briefParses a road:lateralProfile:superelevation element.
*
*/
bool
DomParser::parseSuperelevationElement(QDomElement &element, RSystemElementRoad *road)
{
    double s = parseToDouble(element, "s", 0.0, false); // mandatory
    double a = parseToDouble(element, "a", 0.0, false) * 180.0 / (M_PI); // mandatory
    double b = parseToDouble(element, "b", 0.0, false) * 180.0 / (M_PI); // mandatory
    double c = parseToDouble(element, "c", 0.0, false) * 180.0 / (M_PI); // mandatory
    double d = parseToDouble(element, "d", 0.0, false) * 180.0 / (M_PI); // mandatory

    SuperelevationSection *section = new SuperelevationSection(s, a, b, c, d);
    road->addSuperelevationSection(section);

    return true;
}

/*! \brief Parses a road:lateralProfile:crossfall element.
*
*/
bool
DomParser::parseCrossfallElement(QDomElement &element, RSystemElementRoad *road)
{
    QString side = parseToQString(element, "side", "both", true); // optional
    double s = parseToDouble(element, "s", 0.0, false); // mandatory
    double a = parseToDouble(element, "a", 0.0, false) * 180.0 / (M_PI); // mandatory
    double b = parseToDouble(element, "b", 0.0, false) * 180.0 / (M_PI); // mandatory
    double c = parseToDouble(element, "c", 0.0, false) * 180.0 / (M_PI); // mandatory
    double d = parseToDouble(element, "d", 0.0, false) * 180.0 / (M_PI); // mandatory

    CrossfallSection *section = new CrossfallSection(CrossfallSection::parseCrossfallSide(side), s, a, b, c, d);
    road->addCrossfallSection(section);

    return true;
}

bool 
DomParser::parseShapeElement(QDomElement &element, RSystemElementRoad *road)
{

	double s = parseToDouble(element, "s", 0.0, false); // mandatory
	double t = parseToDouble(element, "t", 0.0, false); // mandatory
	double a = parseToDouble(element, "a", 0.0, false); // mandatory
	double b = parseToDouble(element, "b", 0.0, false); // mandatory
	double c = parseToDouble(element, "c", 0.0, false); // mandatory
	double d = parseToDouble(element, "d", 0.0, false); // mandatory

	PolynomialLateralSection *polynomialLateralSection = new PolynomialLateralSection(t, a, b, c, d);
	ShapeSection *section = road->getShapeSection(s);
	if (!section || (abs(section->getSStart() - s) > NUMERICAL_ZERO3))
	{
		section = new ShapeSection(s, t, polynomialLateralSection);
		road->addShapeSection(section);
	}
	else
	{
		section->addShape(t, polynomialLateralSection);
	}
	return true;
}

/*! \brief Parses a (road) <surface> element.
*
*/
bool
DomParser::parseSurfaceElement(QDomElement &surface, RSystemElementRoad *road)
{
    QDomElement child = surface.firstChildElement("CRG");

    if (!child.isNull())
    {
        SurfaceSection *section = new SurfaceSection();

		while (!child.isNull())
		{
			// <CRG> (optional, max count: unlimited) //
			//
			QString file = parseToQString(child, "file", "", true);			// TODO: most parameters are not optional
			QString sStart = parseToQString(child, "sStart", "", true);
			QString sEnd = parseToQString(child, "sEnd", "", true);
			QString orientation = parseToQString(child, "orientation", "", true);
			QString mode = parseToQString(child, "mode", "", true);
			QString purpose = "";
			if (opendriveVersion_ >= 1.4)
			{
				purpose = parseToQString(child, "purpose", "", false);
			}
            QString sOffset = parseToQString(child, "sOffset", "", true);
            QString tOffset = parseToQString(child, "tOffset", "", true);
            QString zOffset = parseToQString(child, "zOffset", "", true);
            QString zScale = parseToQString(child, "zScale", "", true);
            QString hOffset = parseToQString(child, "hOffset", "", true);

            section->addCRG(file,
                            sStart,
                            sEnd,
                            orientation,
                            mode,
						    purpose,
                            sOffset,
                            tOffset,
                            zOffset,
                            zScale,
                            hOffset);

            child = child.nextSiblingElement("CRG");
        }

        road->addSurfaceSection(section);
    }

    return true;
}

/*! \brief Parses a (road) <planView><geometry> element.
*
*/
bool
DomParser::parseGeometryElement(QDomElement &geometry, RSystemElementRoad *road)
{
    // <geometry> (mandatory, max count: unlimited) //
    //
    double s = parseToDouble(geometry, "s", 0.0, false); // mandatory
    double x = parseToDouble(geometry, "x", 0.0, false); // mandatory
    double y = parseToDouble(geometry, "y", 0.0, false); // mandatory
    double hdg = parseToDouble(geometry, "hdg", 0.0, false); // mandatory
    double length = parseToDouble(geometry, "length", 0.0, false); // mandatory

    // <geometry> child nodes //
    //
    QDomElement child = geometry.firstChildElement();
    QString name = child.nodeName();
    if (name == "line")
    {
        // <line> //
        TrackElementLine *line = new TrackElementLine(x, y, hdg / ( M_PI) * 180.0, s, length);
        road->addTrackComponent(line);
    }
    else if (name == "spiral")
    {
        // <spiral> //
        double curvStart = parseToDouble(child, "curvStart", 0.0, false); // mandatory
        double curvEnd = parseToDouble(child, "curvEnd", 0.0, false); // mandatory
        TrackElementSpiral *spiral = new TrackElementSpiral(x, y, hdg / ( M_PI) * 180.0, s, length, curvStart, curvEnd);
        road->addTrackComponent(spiral);
    }
    else if (name == "arc")
    {
        // <arc> //
        double curvature = parseToDouble(child, "curvature", 0.0, false); // mandatory
        if (curvature == 0.0)
        {
            // TODO //
            qDebug("FEHLER BEIM PARSEN VON KREIS");
        }
        TrackElementArc *arc = new TrackElementArc(x, y, hdg / ( M_PI) * 180.0, s, length, curvature);
        road->addTrackComponent(arc);
    }
    else if (name == "poly3")
    {
        // <poly3> //
        double a = parseToDouble(child, "a", 0.0, false); // mandatory
        double b = parseToDouble(child, "b", 0.0, false); // mandatory
        double c = parseToDouble(child, "c", 0.0, false); // mandatory
        double d = parseToDouble(child, "d", 0.0, false); // mandatory
        TrackElementPoly3 *poly = new TrackElementPoly3(x, y, hdg / ( M_PI) * 180.0, s, length, a, b, c, d);
        road->addTrackComponent(poly);
    }
	else if (name == "paramPoly3")
	{
		// <poly3> //
		double aU = parseToDouble(child, "aU", 0.0, false); // mandatory
		double bU = parseToDouble(child, "bU", 0.0, false); // mandatory
		double cU = parseToDouble(child, "cU", 0.0, false); // mandatory
		double dU = parseToDouble(child, "dU", 0.0, false); // mandatory
		double aV = parseToDouble(child, "aV", 0.0, false); // mandatory
		double bV = parseToDouble(child, "bV", 0.0, false); // mandatory
		double cV = parseToDouble(child, "cV", 0.0, false); // mandatory
		double dV = parseToDouble(child, "dV", 0.0, false); // mandatory
		QString pRange = parseToQString(child, "pRange", "normalized", false); // mandatory

		Polynomial *polyU = new Polynomial(aU, bU, cU, dU);
		Polynomial *polyV = new Polynomial(aV, bV, cV, dV);
		TrackElementCubicCurve *cubicCurve = new TrackElementCubicCurve(x, y, hdg / (M_PI) * 180.0, s, length, polyU, polyV, pRange);
		road->addTrackComponent(cubicCurve);

		qDebug() << "paramPoly3: " << cubicCurve->getPolynomialU()->getA() << "+" << cubicCurve->getPolynomialU()->getB() << "t+" << cubicCurve->getPolynomialU()->getC() << "t2+" << cubicCurve->getPolynomialU()->getD() << "t3";
		qDebug() << "paramPoly3: " << cubicCurve->getPolynomialV()->getA() << "+" << cubicCurve->getPolynomialV()->getB() << "t+" << cubicCurve->getPolynomialV()->getC() << "t2+" << cubicCurve->getPolynomialV()->getD() << "t3";

		QPointF p = cubicCurve->getPoint(272,0);
		qDebug() << "paramPoly3: " << "Point for s=272: " << p.x() << "," << p.y() << " Heading: " << cubicCurve->getHeading(272);
		p = cubicCurve->getPoint(0, 0);
		qDebug() << "paramPoly3: " << "Point for s=0: " << p.x() << "," << p.y() << " Heading: " << cubicCurve->getHeading(0);
		p = cubicCurve->getPoint(36, 0);
		double curv = cubicCurve->getCurvature(36.0);
		qDebug() << "paramPoly3: " << "Point for s=36: " << p.x() << "," << p.y() << " Curvature: " << curv << " Heading: " << cubicCurve->getHeading(36);

/*		cubicCurve->setLocalStartPoint(QPointF(147, 131));
		cubicCurve->setLocalEndPoint(QPointF(216, 149));
		cubicCurve->setLocalStartHeading(30.0);  */
	}
    else
    {
        // error! unknown //
        // TODO //
    }
    return true;
}

/** Parses a (road) <lanes><laneSection> element.
*
*/
bool
DomParser::parseLaneSectionElement(QDomElement &laneSectionElement, RSystemElementRoad *road)
{
	// <laneSection> (mandatory, max count: unlimited) //
	//
	double s = parseToDouble(laneSectionElement, "s", 0.0, false); // mandatory

	QString side = "false";
	if (opendriveVersion_ > 1.3)
	{
		side = parseToQString(laneSectionElement, "singleSide", "false", false);
	}
	LaneSection *laneSection = new LaneSection(s, false);

	if (side == "true")
	{
		laneSection->setSide(true);
	}

	// Add LaneSection to Road //
	//
	road->addLaneSection(laneSection);

    // <laneSection><left/center/right> //
    //
    QDomElement child = laneSectionElement.firstChildElement();

    while (!child.isNull())
    {
        QDomElement lane = child.firstChildElement("lane");
        if (lane.isNull())
        {

            // TODO: NOT OPTIONAL
            qDebug() << "NOT OPTIONAL: <laneSection><left/center/right><lane> at road: " << road->getID().speakingName() << " " << road->getName();
        }
        while (!lane.isNull())
        {
            parseLaneElement(lane, laneSection);
            lane = lane.nextSiblingElement("lane");
        }

        child = child.nextSiblingElement();
    }

    laneSection->checkAndFixLanes();

    return true;
}

/** Parses a (road) <lanes><laneSection><left/center/right><lane> element.
*
*/
bool
DomParser::parseLaneElement(QDomElement &laneElement, LaneSection *laneSection)
{
    // <lane> (mandatory, max count: unlimited) //
    //
    int id = parseToInt(laneElement, "id", 0, false); // mandatory
    QString type = parseToQString(laneElement, "type", "driving", false); // mandatory
    QString level = parseToQString(laneElement, "level", "false", true); // optional

    // <lane><link> //
    //
    int predecessorId = Lane::NOLANE;
    int successorId = Lane::NOLANE;

    QDomElement child = laneElement.firstChildElement("link"); // optional
    if (!child.isNull())
    {
        // <lane><link><predecessor> //
        QDomElement linkChild = child.firstChildElement("predecessor"); // optional
        if (!linkChild.isNull())
        {
            predecessorId = parseToInt(linkChild, "id", 0, false); // mandatory
        }

        // <lane><link><successor> //
        linkChild = child.firstChildElement("successor"); // optional
        if (!linkChild.isNull())
        {
            successorId = parseToInt(linkChild, "id", 0, false); // mandatory
        }
    }

    Lane *lane = new Lane(id, Lane::parseLaneType(type), (level == "true"), predecessorId, successorId);

    // <lane><width> //
    //
    child = laneElement.firstChildElement("width"); // optional for id=0 (center)
    while (!child.isNull())
    {
        double sOffset = parseToDouble(child, "sOffset", 0.0, false); // mandatory
        double a = parseToDouble(child, "a", 0.0, false); // mandatory
        double b = parseToDouble(child, "b", 0.0, false); // mandatory
        double c = parseToDouble(child, "c", 0.0, false); // mandatory
        double d = parseToDouble(child, "d", 0.0, false); // mandatory

        LaneWidth *widthEntry = new LaneWidth(sOffset, a, b, c, d);
        lane->addWidthEntry(widthEntry);

        child = child.nextSiblingElement("width");
    }

	// <lane><border> //
	//
	child = laneElement.firstChildElement("border"); // optional 
	QMap<double, LaneWidth *> widthEntries = lane->getWidthEntries();
	bool isWidthEntry = !widthEntries.isEmpty();
	while (!child.isNull())
	{
		double sOffset = parseToDouble(child, "sOffset", 0.0, false); // mandatory
		LaneBorder *border = NULL;
		bool addEntry = false;

		if (isWidthEntry)
		{
			QMap<double, LaneWidth *>::const_iterator it = widthEntries.upperBound(sOffset);
			if (it != widthEntries.constBegin())
			{
				it--;
			}

			if (sOffset > it.key())
			{
				border = dynamic_cast<LaneBorder *>(it.value());
			}
			else if (it.key() - sOffset > NUMERICAL_ZERO3)
			{
				addEntry = true;
			}
		}

		if (!isWidthEntry || border || addEntry)
		{

			double a = parseToDouble(child, "a", 0.0, false); // mandatory
			double b = parseToDouble(child, "b", 0.0, false); // mandatory
			double c = parseToDouble(child, "c", 0.0, false); // mandatory
			double d = parseToDouble(child, "d", 0.0, false); // mandatory

			LaneBorder *borderEntry = new LaneBorder(sOffset, a, b, c, d);
			lane->addWidthEntry(borderEntry);
		}

		child = child.nextSiblingElement("border");
	}

	if (lane->getWidthEntries().isEmpty())
	{
		if (id != 0)
		{
			// TODO: NOT OPTIONAL
			warning( tr("ODD: XML Parser Error"),
				tr("NOT OPTIONAL: <width> or <border> of <lane>  %1 in laneSection %2 of road %3")
					.arg(id)
					.arg(laneSection->getSStart())
					.arg(laneSection->getParentRoad()->getID().speakingName()));
		}
	}

    // <lane><roadMark> //
    //
    child = laneElement.firstChildElement("roadMark");
    if (child.isNull())
    {
        //default
        //TODO, OPTIONAL
        // it is optional	qDebug() << "NOT OPTIONAL: <roadMark>" << laneElement.lineNumber();
    }
    while (!child.isNull())
    {
        double sOffset = parseToDouble(child, "sOffset", 0.0, false); // mandatory
        QString type = parseToQString(child, "type", "none", false); // mandatory
        QString weight = parseToQString(child, "weight", "standard", true); // optional
        QString color = parseToQString(child, "color", "standard", true); // optional
        double width = parseToDouble(child, "width", -1.0, true); // optional
        QString laneChange = parseToQString(child, "laneChange", "both", true); // optional

		QString material;
		double height;
		if (opendriveVersion_ < 1.4)
		{
			material = parseToQString(child, "material", "none", true);
			height = parseToDouble(child, "height", 0.0, true);
		}
		else
		{
			material = parseToQString(child, "material", "none", false);
			height = parseToDouble(child, "height", 0.0, false);
		}

        LaneRoadMark *roadMarkEntry = new LaneRoadMark(
            sOffset,
            LaneRoadMark::parseRoadMarkType(type),
            LaneRoadMark::parseRoadMarkWeight(weight),
            LaneRoadMark::parseRoadMarkColor(color),
            width,
            LaneRoadMark::parseRoadMarkLaneChange(laneChange),
			material, height);
        lane->addRoadMarkEntry(roadMarkEntry);

		QDomElement typeChild = child.firstChildElement("type");
		if (!typeChild.isNull())
		{
			QString name = parseToQString(typeChild, "name", "", false);
			double width = parseToDouble(typeChild, "width", 0.0, false);

			QDomElement lineChild = typeChild.firstChildElement("line");
			if (lineChild.isNull())
			{
				//default
				//TODO, OPTIONAL
				qDebug() << "NOT OPTIONAL: <line>" << lineChild.lineNumber();
			}
			else
			{
				LaneRoadMarkType *roadMarkType = new LaneRoadMarkType(name, width);
				roadMarkEntry->setUserType(roadMarkType);

				while (!lineChild.isNull())
				{
					double length = parseToDouble(lineChild, "length", 0.0, false);
					double space = parseToDouble(lineChild, "space", 0.0, false);
					double tOffset = parseToDouble(lineChild, "tOffset", 0.0, false);
					double sOffset = parseToDouble(lineChild, "sOffset", 0.0, false);
					QString rule = parseToQString(lineChild, "rule", "none", false);
					double width = parseToDouble(lineChild, "width", 0.0, false);

					roadMarkType->addRoadMarkTypeLine(roadMarkEntry, length, space, tOffset, sOffset, rule, width);

					lineChild = lineChild.nextSiblingElement("line");
				}
			}


		}

        child = child.nextSiblingElement("roadMark");
    }

    // <lane><height> //
    // Optional, unlimited, not allowed for center lane (id=0)
    //
    child = laneElement.firstChildElement("height");
    while (!child.isNull() && id != 0)
    {
        double sOffset = parseToDouble(child, "sOffset", 0.0, false); // mandatory
        double inner = parseToDouble(child, "inner", 0.0, false); // mandatory
        double outer = parseToDouble(child, "outer", 0.0, false); // mandatory

        LaneHeight *heightEntry = new LaneHeight(sOffset, inner, outer);
        lane->addHeightEntry(heightEntry);

        child = child.nextSiblingElement("height");
    }

	// <lane><speed> //
	// Optional, unlimited, not allowed for center lane (id=0)
	//
	child = laneElement.firstChildElement("speed");
	while (!child.isNull() && id != 0)
	{
		double sOffset = parseToDouble(child, "sOffset", 0.0, false); // mandatory
		double max = parseToDouble(child, "max", 0.0, false); // mandatory
		QString unit = parseToQString(child, "unit", "m/s", true); // optional

		LaneSpeed *speedEntry = new LaneSpeed(sOffset, max, unit);
		lane->addSpeedEntry(speedEntry);

		qDebug() << "lane speed:" << max << "Unit:" << unit;

		child = child.nextSiblingElement("speed");
	}

	// <lane><rule> //
	//
	child = laneElement.firstChildElement("rule");
	if (child.isNull())
	{
		//default
		//TODO, OPTIONAL
		// it is optional	qDebug() << "NOT OPTIONAL: <rule>" << laneElement.lineNumber();
	}
	while (!child.isNull())
	{
		double sOffset = parseToDouble(child, "sOffset", 0.0, false); // mandatory
		QString value = parseToQString(child, "value", LaneRule::KNOWNVALUES.at(0), false); // mandatory

		LaneRule *ruleEntry = new LaneRule(sOffset, value);

		lane->addLaneRuleEntry(ruleEntry);

		child = child.nextSiblingElement("rule");
	}

	// <lane><access> //
	//
	child = laneElement.firstChildElement("access");
	if (child.isNull())
	{
		//default
		//TODO, OPTIONAL
		// it is optional	qDebug() << "NOT OPTIONAL: <access>" << laneElement.lineNumber();
	}
	while (!child.isNull())
	{
		double sOffset = parseToDouble(child, "sOffset", 0.0, false); // mandatory
		QString restriction = parseToQString(child, "restriction", "none", false); // mandatory

		LaneAccess *accessEntry = new LaneAccess(sOffset, LaneAccess::parseLaneRestriction(restriction));

		lane->addLaneAccessEntry(accessEntry);

		child = child.nextSiblingElement("access");
	}

    // Add Lane To LaneSection //
    //
    laneSection->addLane(lane);

    return true;
}

//################//
// CONTROLLER     //
//################//

/** Parses a <controller> element.
*
*/
bool
DomParser::parseControllerElement(QDomElement &controllerElement)
{
	int tileID = parseTileID(controllerElement);
    // Get mandatory attributes
    QString name = parseToQString(controllerElement, "name", "", false); // mandatory
    QString id = parseToQString(controllerElement, "id", "", false); // mandatory
    int sequence = parseToInt(controllerElement, "sequence", 0, true);

    QString script = "";
    double cycleTime = 0;
    QDomElement ancillary = controllerElement.firstChildElement("userData");
    if (ancillary.isNull())
    {
        script = parseToQString(controllerElement, "script", "", false);        // for old files
        cycleTime = parseToDouble(controllerElement, "cycleTime", 0.0, false);
    }
    while (!ancillary.isNull())
    {
        QString code = parseToQString(ancillary, "code", "", true);

        if (code == "cycleTime")
        {
            cycleTime = parseToDouble(ancillary, "value", 0.0, true);
        }
        else
        {
            script = parseToQString(ancillary, "value", "", true);
        }

        ancillary = ancillary.nextSiblingElement("userData");
    }
    
    // Corners of the outline of the object //
    //
    QList<ControlEntry *> controlEntries;
    QDomElement control = controllerElement.firstChildElement("control");
    while (!control.isNull())
    {
        QString signalId = parseToQString(control, "signalId", "", false);
        QString type = parseToQString(control, "type", "", false);

		odrID signalID;
		StringToID(signalId, signalID, odrID::ID_Signal,tileID);
        ControlEntry *entry = new ControlEntry(signalID, type);
        controlEntries.append(entry);

        control = control.nextSiblingElement("control");
    }

    // Construct Controller
    //

	odrID ID;
	StringToID(id, ID, odrID::ID_Controller,tileID);
    RSystemElementController *controller = new RSystemElementController(name, ID, sequence, script, cycleTime, controlEntries);

	tileSystem_->addTileIfNecessary(ID);
    roadSystem_->addController(controller);

    return true;
}

//################//
// JUNCTION       //
//################//

/** Parses a <junction> element.
*
*/
bool
DomParser::parseJunctionElement(QDomElement &element)
{

	int tileID = parseTileID(element);
    // 1.) Parse Attributes //
    //
    QString name = parseToQString(element, "name", "Untitled", true); // optional
    QString id = parseToQString(element, "id", "", false); // mandatory

	odrID ID;
	StringToID(id, ID, odrID::ID_Junction,tileID);
    RSystemElementJunction *junction = new RSystemElementJunction(name, ID);

    // 2.) Parse children //
    //
    QDomElement child;

    // <link> <predecessor> (optional, max count: 1) //
    child = element.firstChildElement("connection");
    while (!child.isNull())
    {
        QString childId = parseToQString(child, "id", "", false); // mandatory
        QString incomingRoad = parseToQString(child, "incomingRoad", "", false); // mandatory
        QString connectingRoad = parseToQString(child, "connectingRoad", "", false); // mandatory
        QString contactPoint = parseToQString(child, "contactPoint", "", false); // mandatory
        double numerator = parseToDouble(child, "numerator", 1.0, true); // optional

        QDomElement ancillary = child.firstChildElement("userData");
        while (!ancillary.isNull())
        {
            numerator = parseToDouble(ancillary, "value", 1.0, true);
            ancillary = ancillary.nextSiblingElement("userData");
        }


		odrID incommingID;
		odrID connectingID;
		StringToID(incomingRoad, incommingID, odrID::ID_Road,tileID);
		StringToID(connectingRoad, connectingID, odrID::ID_Road,tileID);
        // <laneLink> //
        JunctionConnection *connection = new JunctionConnection(childId, incommingID, connectingID, JunctionConnection::parseContactPoint(contactPoint), numerator);

        QDomElement link;
        link = child.firstChildElement("laneLink");
        while (!link.isNull())
        {
            int from = parseToInt(link, "from", 0, false);
            int to = parseToInt(link, "to", 0, false);
            connection->addLaneLink(from, to);
            link = link.nextSiblingElement("laneLink");
        }

        junction->addConnection(connection);

        child = child.nextSiblingElement("connection");
    }

	if(tileSystem_!=NULL)
	{
		tileSystem_->addTileIfNecessary(ID);
	}
    roadSystem_->addJunction(junction);


    return true;
}

//################//
// JUNCTIONGROUP      //
//################//

/** Parses a <junction> element.
*
*/
bool
DomParser::parseJunctionGroupElement(QDomElement &element)
{

	int tileID = parseTileID(element);
	// 1.) Parse Attributes //
	//
	QString name = parseToQString(element, "name", "Untitled", true); // optional
	QString id = parseToQString(element, "id", "", false); // mandatory
	QString type = parseToQString(element, "type", "unknown", false); // mandatory

	odrID ID;
	StringToID(id, ID, odrID::ID_Junction,tileID);
	RSystemElementJunctionGroup *junctionGroup = new RSystemElementJunctionGroup(name, ID, type);

	// 2.) Parse children //
	//
	QDomElement child;

	// junction reference, min count: 1) //
	child = element.firstChildElement("junctionReference");
	while (!child.isNull())
	{
		QString junctionId = parseToQString(child, "junction", "", false); // mandatory
		junctionGroup->addJunction(junctionId);

		child = child.nextSiblingElement("junctionReference");
	}

	tileSystem_->addTileIfNecessary(ID);
	roadSystem_->addJunctionGroup(junctionGroup);


	return true;
}

//################//
// FIDDLEYARD     //
//################//

/** Parses a <fiddleyard> element.
*
*/
bool
DomParser::parseFiddleyardElement(QDomElement &element)
{

	int tileID = parseTileID(element);
    // 1.) Parse Attributes //
    //
    QString name = parseToQString(element, "name", "Untitled", true); // optional
    QString id = parseToQString(element, "id", "", false); // mandatory

    // 2.) Parse children //
    //
    QDomElement child;

    // <link>(mandatory, max count: 1) //
    child = element.firstChildElement("link");
    QString elementType;
    QString elementId;
    QString contactPoint;
    if (!child.isNull())
    {
        elementType = parseToQString(child, "elementType", "road", false); // mandatory
        elementId = parseToQString(child, "elementId", "", false); // mandatory
        contactPoint = parseToQString(child, "contactPoint", "start", false); // mandatory
    }
    else
    {
        // TODO: mandatory
        qDebug("NOT OPTIONAL: <fiddleyard><link>");
    }

	odrID ID;
	StringToID(id, ID, odrID::ID_Fiddleyard,tileID);
	odrID elementID;
	StringToID(elementId, elementID, odrID::ID_Road,tileID);
    RSystemElementFiddleyard *fiddleyard = new RSystemElementFiddleyard(name, ID, elementType, elementID, contactPoint);

    // <source>(optional, max count: unlimited) //
    child = element.firstChildElement("source");
    while (!child.isNull())
    {
        QString id = parseToQString(child, "id", "", false); // mandatory
        int lane = parseToInt(child, "lane", 0, false); // mandatory
        double startTime = parseToDouble(child, "startTime", 0.0, false); // mandatory
        double repeatTime = parseToDouble(child, "repeatTime", 10.0, false); // mandatory
        double velocity = parseToDouble(child, "velocity", 20.0, false); // mandatory
        double velocityDeviance = parseToDouble(child, "velocityDeviance", 0.0, false); // mandatory

		odrID ID;
		StringToID(id, ID, odrID::ID_Fiddleyard,tileID);
        FiddleyardSource *source = new FiddleyardSource(ID, lane, startTime, repeatTime, velocity, velocityDeviance);

        QDomElement subchild = child.firstChildElement("vehicle");
        while (!subchild.isNull())
        {
            QString id = parseToQString(subchild, "id", "", false); // mandatory
            double numerator = parseToDouble(subchild, "numerator", 1.0, true); // optional

			odrID ID;
			StringToID(id, ID, odrID::ID_Vehicle,tileID);
            source->addVehicle(ID, numerator);
            subchild = subchild.nextSiblingElement("vehicle");
        }

        fiddleyard->addSource(source);

        child = child.nextSiblingElement("source");
    }

    // <sink>(optional, max count: unlimited) //
    child = element.firstChildElement("sink");
    while (!child.isNull())
    {
        QString id = parseToQString(child, "id", "", false); // mandatory
        int lane = parseToInt(child, "lane", 0, false); // mandatory

		odrID ID;
		StringToID(id, ID, odrID::ID_Fiddleyard,tileID);
        fiddleyard->addSink(new FiddleyardSink(ID, lane));

        child = child.nextSiblingElement("sink");
    }

	tileSystem_->addTileIfNecessary(ID);
    roadSystem_->addFiddleyard(fiddleyard);


    return true;
}

//################//
// PEDFIDDLEYARD  //
//################//

/** Parses a <pedFiddleyard> element.
*
*/
bool
DomParser::parsePedFiddleyardElement(QDomElement &element)
{

	int tileID = parseTileID(element);
    // 1.) Parse Attributes //
    //
    QString id = parseToQString(element, "id", "", false); // mandatory
    QString name = parseToQString(element, "name", "", false); // mandatory
    QString roadId = parseToQString(element, "roadId", "", false); // mandatory

	odrID ID;
	StringToID(id, ID, odrID::ID_PedFiddleyard,tileID);
	odrID roadID;
	StringToID(roadId, roadID, odrID::ID_Road,tileID);
    RSystemElementPedFiddleyard *fiddleyard = new RSystemElementPedFiddleyard(ID, name, roadID);

    // 2.) Parse children //
    //
    QDomElement child;

    // <source>(optional, max count: unlimited) //
    child = element.firstChildElement("source");
    while (!child.isNull())
    {
        // Parse mandatory attributes
        QString id = parseToQString(child, "id", "", false); // mandatory
        int lane = parseToInt(child, "lane", 0, false); // mandatory
        double velocity = parseToDouble(child, "velocity", 0.0, false); // mandatory

		odrID ID;
		StringToID(id, ID, odrID::ID_PedFiddleyard,tileID);
        // Create source
        PedFiddleyardSource *source = new PedFiddleyardSource(ID,
                                                              lane,
                                                              velocity);

        // Add optional attributes
        if (parseToQString(child, "startTime", "", true).length() > 0)
            source->setStartTime(parseToDouble(child, "startTime", 0.0, true));
        if (parseToQString(child, "repeatTime", "", true).length() > 0)
            source->setRepeatTime(parseToDouble(child, "repeatTime", 60.0, true));
        if (parseToQString(child, "timeDeviance", "", true).length() > 0)
            source->setTimeDeviance(parseToDouble(child, "timeDeviance", 0.0, true));
        if (parseToQString(child, "direction", "", true).length() > 0)
            source->setDirection(parseToInt(child, "direction", 0, true));
        if (parseToQString(child, "sOffset", "", true).length() > 0)
            source->setSOffset(parseToDouble(child, "sOffset", 0.0, true));
        if (parseToQString(child, "vOffset", "", true).length() > 0)
            source->setVOffset(parseToDouble(child, "vOffset", 0.0, true));
        if (parseToQString(child, "velocityDeviance", "", true).length() > 0)
            source->setVelocityDeviance(parseToDouble(child, "velocityDeviance", 0.0, true));
        if (parseToQString(child, "acceleration", "", true).length() > 0)
            source->setAcceleration(parseToDouble(child, "acceleration", velocity / 2, true));
        if (parseToQString(child, "accelerationDeviance", "", true).length() > 0)
            source->setAccelerationDeviance(parseToDouble(child, "accelerationDeviance", 0.0, true));

        // Parse children
        QDomElement subchild = child.firstChildElement("ped");
        while (!subchild.isNull())
        {
            QString id = parseToQString(subchild, "templateId", "", true); // optional
            double numerator = parseToDouble(subchild, "numerator", 1.0, true); // optional
            source->addPedestrian(id, numerator);
            subchild = subchild.nextSiblingElement("ped");
        }

        // Add to pedFiddleyard
        fiddleyard->addSource(source);

        // Get next source
        child = child.nextSiblingElement("source");
    }

    // <sink>(optional, max count: unlimited) //
    child = element.firstChildElement("sink");
    while (!child.isNull())
    {
        // Parse mandatory attributes
        QString id = parseToQString(child, "id", "", false); // mandatory
        int lane = parseToInt(child, "lane", 0, false); // mandatory

		odrID ID;
		StringToID(id, ID, odrID::ID_PedFiddleyard,tileID);
        // Create sink
        PedFiddleyardSink *sink = new PedFiddleyardSink(ID, lane);

        // Add optional attributes
        if (parseToQString(child, "sinkProb", "", true).length() > 0)
            sink->setSinkProb(parseToDouble(child, "sinkProb", 1.0, true));
        if (parseToQString(child, "direction", "", true).length() > 0)
            sink->setDirection(parseToInt(child, "direction", 0, true));
        if (parseToQString(child, "sOffset", "", true).length() > 0)
            sink->setSOffset(parseToDouble(child, "sOffset", 0.0, true));
        if (parseToQString(child, "vOffset", "", true).length() > 0)
            sink->setVOffset(parseToDouble(child, "vOffset", 0.0, true));

        // Add to pedFiddleyard
        fiddleyard->addSink(sink);

        // Get next sink
        child = child.nextSiblingElement("sink");
    }

	tileSystem_->addTileIfNecessary(ID);
    roadSystem_->addPedFiddleyard(fiddleyard);


    return true;
}

//################//
// CARPOOL       //
//################//

/*! \brief Parses a <carpool> element.
*
*/
bool
DomParser::parseCarPool(const QDomElement &element)
{
    // carpool //
    //
    CarPool *carPool = new CarPool();
    vehicleSystem_->setCarPool(carPool);

    // carpool:pool //
    //
    QDomElement child = element.firstChildElement("pool");
    while (!child.isNull())
    {
        carPool->addPool(parsePool(child));

        child = child.nextSiblingElement("pool");
    }

    return true;
}

/*! \brief Parses a <pool> element.
*
*/
Pool *
DomParser::parsePool(const QDomElement &element)
{

	int tileID = parseTileID(element);
    // pool //
    //
    QString id = parseToQString(element, "id", "", false); // mandatory
    QString elementName = parseToQString(element, "name", "", true); // optional
    double velocity = parseToDouble(element, "velocity", 33.0, true); // optional
    double velocityDeviance = parseToDouble(element, "velocityDeviance", 5.0, true); // optional
    double numerator = parseToDouble(element, "numerator", 20, true); // optional

	odrID ID;
	StringToID(id, ID, odrID::ID_PedFiddleyard,tileID);
    Pool *pool = new Pool(elementName, ID, velocity, velocityDeviance, numerator);

    // carpool:pool::vehicle //
    //
    QDomElement child = element.firstChildElement("vehicle");
    while (!child.isNull())
    {
        pool->addVehicle(parsePoolVehicle(child));

        child = child.nextSiblingElement("vehicle");
    }

    return pool;
}
/*! \brief Parses a <pool/vehicle> element.
*
*/
PoolVehicle *
DomParser::parsePoolVehicle(const QDomElement &element)
{
    // carpool //
    //
    QString elementId = parseToQString(element, "id", "", false); // mandatory
    double numerator = parseToDouble(element, "numerator", 20, true); // optional
    PoolVehicle *vehicle = new PoolVehicle(elementId, numerator);

    return vehicle;
}
//################//
// VEHICLES       //
//################//

/*! \brief Parses the VehicleSystem.
*
*/
bool
DomParser::parseVehicleSystem(QDomElement &root)
{
    QDomElement child = root.firstChildElement("vehicles");
    while (!child.isNull())
    {
        parseVehiclesElement(child);
        child = child.nextSiblingElement("vehicles");
    }

    return true;
}

/*! \brief Parses a <vehicles> element.
*
*/
bool
DomParser::parseVehiclesElement(const QDomElement &element)
{
    // vehicles //
    //
    double rangeLOD = parseToDouble(element, "rangeLOD", VehicleGroup::defaultRangeLOD, true); // optional
    VehicleGroup *vehicleGroup = new VehicleGroup(rangeLOD);
    vehicleSystem_->addVehicleGroup(vehicleGroup);

    // Get passThreshold (if set)
    if (parseToQString(element, "passThreshold", "", true).length() > 0)
        vehicleGroup->setPassThreshold(parseToDouble(element, "passThreshold", 0.5, true));

    // vehicles:roadVehicle //
    //
    QDomElement child = element.firstChildElement("roadVehicle");
    while (!child.isNull())
    {
        vehicleGroup->addRoadVehicle(parseRoadVehicleElement(child));

        child = child.nextSiblingElement("roadVehicle");
    }

    return true;
}

/*! \brief Parses a <vehicles><roadVehicle> element.
*
*/
RoadVehicle *
DomParser::parseRoadVehicleElement(const QDomElement &element)
{
    // <roadVehicle> //
    //
    QString elementId = parseToQString(element, "id", "", false); // mandatory
    QString elementName = parseToQString(element, "name", "", true); // optional
    RoadVehicle::RoadVehicleType elementType = RoadVehicle::parseRoadVehicleType(parseToQString(element, "type", "agent", false)); // mandatory

    QDomElement child;

    RoadVehicle::RoadVehicleIntelligenceType intelligenceType = RoadVehicle::DRVI_AGENT;
    QString modelFile = "";

    double maxAcceleration = RoadVehicle::defaultMaxAcceleration;
    double indicatoryVelocity = RoadVehicle::defaultIndicatoryVelocity;
    double maxCrossAcceleration = RoadVehicle::defaultMaxCrossAcceleration;

    double minimumGap = RoadVehicle::defaultMinimumGap;
    double pursueTime = RoadVehicle::defaultPursueTime;
    double comfortableDecelaration = RoadVehicle::defaultComfortableDecelaration;
    double saveDeceleration = RoadVehicle::defaultSaveDeceleration;
    double approachFactor = RoadVehicle::defaultApproachFactor;
    double laneChangeThreshold = RoadVehicle::defaultLaneChangeThreshold;
    double politenessFactor = RoadVehicle::defaultPolitenessFactor;

    // <roadVehicle><intelligence> //
    //
    child = element.firstChildElement("intelligence");
    if (!child.isNull())
    {
        intelligenceType = RoadVehicle::parseRoadVehicleIntelligenceType(parseToQString(child, "type", "agent", false)); // mandatory
    }

    // <roadVehicle><geometry> //
    //
    child = element.firstChildElement("geometry");
    if (!child.isNull())
    {
        modelFile = parseToQString(child, "modelFile", "", false); // mandatory
    }

    // <roadVehicle><initialState> //
    //
    child = element.firstChildElement("initialState");
    if (!child.isNull())
    {
        qDebug("<roadVehicle><initialState> NOT IMPLEMENTED YET!");
    }

    // <roadVehicle><route> //
    //
    child = element.firstChildElement("route");
    if (!child.isNull())
    {
        qDebug("<roadVehicle><route> NOT IMPLEMENTED YET!");
    }

    // <roadVehicle><dynamics> //
    //
    child = element.firstChildElement("dynamics");
    if (!child.isNull())
    {
        QDomElement subChild;

        subChild = child.firstChildElement("maximumAcceleration");
        if (!subChild.isNull())
        {
            maxAcceleration = parseToDouble(subChild, "value", RoadVehicle::defaultMaxAcceleration, true); // optional
        }

        subChild = child.firstChildElement("indicatoryVelocity");
        if (!subChild.isNull())
        {
            indicatoryVelocity = parseToDouble(subChild, "value", RoadVehicle::defaultIndicatoryVelocity, true); // optional
        }

        subChild = child.firstChildElement("maximumCrossAcceleration");
        if (!subChild.isNull())
        {
            maxCrossAcceleration = parseToDouble(subChild, "value", RoadVehicle::defaultMaxCrossAcceleration, true); // optional
        }
    }

    // <roadVehicle><behaviour> //
    //
    child = element.firstChildElement("behaviour");
    if (!child.isNull())
    {
        QDomElement subChild;

        subChild = child.firstChildElement("minimumGap");
        if (!subChild.isNull())
        {
            minimumGap = parseToDouble(subChild, "value", RoadVehicle::defaultMinimumGap, true); // optional
        }

        subChild = child.firstChildElement("pursueTime");
        if (!subChild.isNull())
        {
            pursueTime = parseToDouble(subChild, "value", RoadVehicle::defaultPursueTime, true); // optional
        }

        subChild = child.firstChildElement("comfortableDecelaration");
        if (!subChild.isNull())
        {
            comfortableDecelaration = parseToDouble(subChild, "value", RoadVehicle::defaultComfortableDecelaration, true); // optional
        }

        subChild = child.firstChildElement("saveDeceleration");
        if (!subChild.isNull())
        {
            saveDeceleration = parseToDouble(subChild, "value", RoadVehicle::defaultSaveDeceleration, true); // optional
        }

        subChild = child.firstChildElement("approachFactor");
        if (!subChild.isNull())
        {
            approachFactor = parseToDouble(subChild, "value", RoadVehicle::defaultApproachFactor, true); // optional
        }

        subChild = child.firstChildElement("laneChangeTreshold"); // NOTE: typo in specification
        if (!subChild.isNull())
        {
            laneChangeThreshold = parseToDouble(subChild, "value", RoadVehicle::defaultLaneChangeThreshold, true); // optional
        }

        subChild = child.firstChildElement("politenessFactor");
        if (!subChild.isNull())
        {
            politenessFactor = parseToDouble(subChild, "value", RoadVehicle::defaultPolitenessFactor, true); // optional
        }
    }

    RoadVehicle *vehicle = new RoadVehicle(
        elementName,
        elementId,
        elementType,
        intelligenceType,
        modelFile,
        maxAcceleration,
        indicatoryVelocity,
        maxCrossAcceleration,
        minimumGap,
        pursueTime,
        comfortableDecelaration,
        saveDeceleration,
        approachFactor,
        laneChangeThreshold,
        politenessFactor);

    return vehicle;
}

//################//
// PEDESTRIANS    //
//################//

/*! \brief Parses the PedestrianSystem.
*
*/
bool
DomParser::parsePedestrianSystem(QDomElement &root)
{
    QDomElement child = root.firstChildElement("pedestrians");
    while (!child.isNull())
    {
        parsePedestriansElement(child);
        child = child.nextSiblingElement("pedestrians");
    }

    return true;
}

/*! \brief Parses a <pedestrians> element.
*
*/
bool
DomParser::parsePedestriansElement(const QDomElement &element)
{
    // pedestrians //
    //
    PedestrianGroup *pedestrianGroup = new PedestrianGroup();
    pedestrianSystem_->addPedestrianGroup(pedestrianGroup);

    // Get spawnRange (if set)
    if (parseToQString(element, "spawnRange", "", true).length() > 0)
        pedestrianGroup->setSpawnRange(parseToDouble(element, "spawnRange", -1.0, true));

    // Get maximum (if set)
    if (parseToQString(element, "maxPeds", "", true).length() > 0)
        pedestrianGroup->setMaxPeds(parseToInt(element, "maxPeds", -1, true));

    // Get report interval (if set)
    if (parseToQString(element, "reportInterval", "", true).length() > 0)
        pedestrianGroup->setReportInterval(parseToDouble(element, "reportInterval", -1.0, true));

    // Get avoidCount (if set)
    if (parseToQString(element, "avoidCount", "", true).length() > 0)
        pedestrianGroup->setAvoidCount(parseToInt(element, "avoidCount", 3, true));

    // Get avoidTime (if set)
    if (parseToQString(element, "avoidTime", "", true).length() > 0)
        pedestrianGroup->setAvoidTime(parseToDouble(element, "avoidTime", 3.0, true));

    // Get autoFiddle (if set)
    if (parseToQString(element, "autoFiddle", "", true).length() > 0)
        pedestrianGroup->setAutoFiddle(parseToQString(element, "autoFiddle", "", true).compare("true") == 0);

    // Get movingFiddle (if set)
    if (parseToQString(element, "movingFiddle", "", true).length() > 0)
        pedestrianGroup->setMovingFiddle(parseToQString(element, "movingFiddle", "", true).compare("true") == 0);

    QDomElement child;

    // pedestrians:default //
    //
    child = element.firstChildElement("default");
    while (!child.isNull())
    {
        pedestrianGroup->addPedestrian(parsePedElement(child, true, false));
        child = child.nextSiblingElement("default");
    }

    // pedestrians:template //
    //
    child = element.firstChildElement("template");
    while (!child.isNull())
    {
        pedestrianGroup->addPedestrian(parsePedElement(child, false, true));
        child = child.nextSiblingElement("template");
    }

    // pedestrians:ped //
    //
    child = element.firstChildElement("ped");
    while (!child.isNull())
    {
        pedestrianGroup->addPedestrian(parsePedElement(child, false, false));
        child = child.nextSiblingElement("ped");
    }

    return true;
}

/*! \brief Parses a <pedestrians> <default>/<template>/<ped> element.
*
*/
Pedestrian *
DomParser::parsePedElement(const QDomElement &element, bool defaultPed, bool templatePed)
{
    QString id = "";
    QString name = "";
    QString templateId = "";
    QString rangeLOD = "";
    QString debugLvl = "";
    QString modelFile = "";
    QString scale = "";
    QString heading = "";
    QString startRoadId = "";
    QString startLane = "";
    QString startSOffset = "";
    QString startVOffset = "";
    QString startDir = "";
    QString startVel = "";
    QString startAcc = "";
    QString idleIdx = "";
    QString idleVel = "";
    QString slowIdx = "";
    QString slowVel = "";
    QString walkIdx = "";
    QString walkVel = "";
    QString jogIdx = "";
    QString jogVel = "";
    QString lookIdx = "";
    QString waveIdx = "";

    if (!defaultPed)
    {
        // <template/ped> //
        //
        id = parseToQString(element, "id", "", false); // mandatory

        if (!templatePed)
        {
            // <ped> //
            //
            name = parseToQString(element, "name", "", false); // mandatory
            templateId = parseToQString(element, "templateId", "", true); // optional
        }
    }
    // <default/template/ped> //
    //
    rangeLOD = parseToQString(element, "rangeLOD", "", true);
    debugLvl = parseToQString(element, "debugLvl", "", true);

    QDomElement child;

    // <default/template/ped><geometry> //
    //
    child = element.firstChildElement("geometry");
    if (!child.isNull())
    {
        modelFile = parseToQString(child, "modelFile", "", true); // optional
        scale = parseToQString(child, "scale", "", true); // optional
        heading = parseToQString(child, "heading", "", true); // optional
    }

    // <default/template/ped><start> //
    //
    child = element.firstChildElement("start");
    if (!child.isNull())
    {
        startRoadId = parseToQString(child, "roadId", "", true); // optional
        startLane = parseToQString(child, "lane", "", true); // optional
        startDir = parseToQString(child, "direction", "", true); // optional
        startSOffset = parseToQString(child, "sOffset", "", true); // optional
        startVOffset = parseToQString(child, "vOffset", "", true); // optional
        startVel = parseToQString(child, "velocity", "", true); // optional
        startAcc = parseToQString(child, "acceleration", "", true); // optional
    }

    // <default/template/ped><animations> //
    //
    child = element.firstChildElement("animations");
    if (!child.isNull())
    {
        QDomElement anim;

        anim = child.firstChildElement("idle");
        if (!anim.isNull())
        {
            idleIdx = parseToQString(anim, "index", "", true); // optional
            idleVel = parseToQString(anim, "velocity", "", true); // optional
        }

        anim = child.firstChildElement("slow");
        if (!anim.isNull())
        {
            slowIdx = parseToQString(anim, "index", "", true); // optional
            slowVel = parseToQString(anim, "velocity", "", true); // optional
        }

        anim = child.firstChildElement("walk");
        if (!anim.isNull())
        {
            walkIdx = parseToQString(anim, "index", "", true); // optional
            walkVel = parseToQString(anim, "velocity", "", true); // optional
        }

        anim = child.firstChildElement("jog");
        if (!anim.isNull())
        {
            jogIdx = parseToQString(anim, "index", "", true); // optional
            jogVel = parseToQString(anim, "velocity", "", true); // optional
        }

        anim = child.firstChildElement("look");
        if (!anim.isNull())
        {
            lookIdx = parseToQString(anim, "index", "", true); // optional
        }

        anim = child.firstChildElement("wave");
        if (!anim.isNull())
        {
            waveIdx = parseToQString(anim, "index", "", true); // optional
        }
    }

    Pedestrian *ped = new Pedestrian(
        defaultPed,
        templatePed,
        id,
        name,
        templateId,
        rangeLOD,
        debugLvl,
        modelFile,
        scale,
        heading,
        startRoadId,
        startLane,
        startDir,
        startSOffset,
        startVOffset,
        startVel,
        startAcc,
        idleIdx,
        idleVel,
        slowIdx,
        slowVel,
        walkIdx,
        walkVel,
        jogIdx,
        jogVel,
        lookIdx,
        waveIdx);

    return ped;
}

//################//
// SCENERY        //
//################//

/*! \brief Parses a <scenery> element.
*
*/
bool
DomParser::parseScenerySystem(QDomElement &root)
{
    QDomElement child = root.firstChildElement("scenery");
    while (!child.isNull())
    {
        parseSceneryElement(child);
        child = child.nextSiblingElement("scenery");
    }

    child = root.firstChildElement("environment");
    while (!child.isNull())
    {
        parseEnvironmentElement(child);
        child = child.nextSiblingElement("environment");
    }

    return true;
}

/*! \brief Parses a <scenery> element.
*
*/
bool
DomParser::parseSceneryElement(QDomElement &element)
{

    // 3D File //
    //
    QString sceneryFilename = parseToQString(element, "file", "", true); // optional
    if (!sceneryFilename.isEmpty())
    {
        scenerySystem_->addSceneryFile(sceneryFilename);
    }
    else
    {
        // Image File //
        //
        QDomElement child = element.firstChildElement("map");
        while (!child.isNull())
        {
            QString filename = parseToQString(child, "filename", "", false); // mandatory
            QString id = parseToQString(child, "id", "", false); // mandatory
            double width = parseToDouble(child, "width", 1.0, false); // mandatory
            double height = parseToDouble(child, "height", 1.0, false); // mandatory
            SceneryMap *map = new SceneryMap(id, filename, width, height, SceneryMap::DMT_Aerial);

            double x = parseToDouble(child, "x", 0.0, true); // optional
            map->setX(x);

            double y = parseToDouble(child, "y", 0.0, true); // optional
            map->setY(y);

            double opacity = parseToDouble(child, "opacity", 1.0, true); // optional
            map->setOpacity(opacity);

            scenerySystem_->addSceneryMap(map);

            child = child.nextSiblingElement("map");
        }

        // Heightmap File //
        //
        child = element.firstChildElement("heightmap");
        while (!child.isNull())
        {
            QString filename = parseToQString(child, "filename", "", false); // mandatory
            QString dataFilename = parseToQString(child, "data", "", false); // mandatory
            QString id = parseToQString(child, "id", "", false); // mandatory
            double width = parseToDouble(child, "width", 1.0, false); // mandatory
            double height = parseToDouble(child, "height", 1.0, false); // mandatory
            Heightmap *map = new Heightmap(id, filename, width, height, SceneryMap::DMT_Heightmap, dataFilename);

            double x = parseToDouble(child, "x", 0.0, true); // optional
            map->setX(x);

            double y = parseToDouble(child, "y", 0.0, true); // optional
            map->setY(y);

            double opacity = parseToDouble(child, "opacity", 1.0, true); // optional
            map->setOpacity(opacity);

            scenerySystem_->addHeightmap(map);

            child = child.nextSiblingElement("heightmap");
        }
    }

    return true;
}

//################//
// ENVIRONMENT    //
//################//

/*! \brief Parses a <environment> element.
*
*/
bool
DomParser::parseEnvironmentElement(QDomElement &environment)
{
    SceneryTesselation *tesselationSettings = new SceneryTesselation();
    QString name = parseToQString(environment, "tessellateRoads", "true", true); // optional
    if (name == "true" || name == "1")
    {
        tesselationSettings->setTesselateRoads(true);
    }
    else if (name == "false" || name == "0")
    {
        tesselationSettings->setTesselateRoads(false);
    }
    else
    {
        qDebug("ERROR 1011181023! tessellateRoads must be true or false");
    }
    name = parseToQString(environment, "tessellatePaths", "true", true); // optional
    if (name == "true" || name == "1")
    {
        tesselationSettings->setTesselatePaths(true);
    }
    else if (name == "false" || name == "0")
    {
        tesselationSettings->setTesselatePaths(false);
    }
    else
    {
        qDebug("ERROR 1011181024! tessellatePaths must be true or false");
    }
    scenerySystem_->setSceneryTesselation(tesselationSettings);
    return true;
}

//################//
// SIGNALS    //
//################//

/*! \brief Opens a signal file/device and parses the signals.
*
*/
bool
DomParser::parseSignals(QIODevice *source)
{
    // Mode //
    //
    mode_ = DomParser::MODE_NONE;

    // Open file and parse tree //
    //
    QString errorStr = "";
    int errorLine = 0;
    int errorColumn = 0;

    if (!doc_->setContent(source, true, &errorStr, &errorLine, &errorColumn))
    {
        warning( tr("ODD: XML Parser Error"),
                             tr("Parse error at line %1, column %2:\n%3")
                                 .arg(errorLine)
                                 .arg(errorColumn)
                                 .arg(errorStr));
        return false;
    }

#ifdef WIN32
    char *pValue;
    size_t len;
    errno_t err = _dupenv_s(&pValue, &len, "ODDLOTDIR");
    if (err || pValue==NULL || strlen(pValue)==0)
        err = _dupenv_s(&pValue, &len, "COVISEDIR");
    if (err)
        return false;
    QString covisedir = pValue;
#else
    QString covisedir = getenv("ODDLOTDIR");
    if (covisedir == "")
        covisedir = getenv("COVISEDIR");
#endif

    // Root Element //
    //
    QDomElement root = doc_->documentElement();
    if (root.tagName() != "ODDLot")
    {
        warning( tr("ODD: XML Parser Error"),
                             tr("Root element is not <ODDLot>!"));
        return false;
    }

    // Signals //
    //
    QDomElement signalsRoot;
    signalsRoot = root.firstChildElement("signalsObjects");
    if (signalsRoot.isNull())
    {
        warning( tr("ODD: XML Parser Error"),
                             tr("Missing <signals> element!"));
        return false;
    }

    QDomElement country = signalsRoot.firstChildElement("country");
    QString countryName = "OpenDRIVE";

    while (!country.isNull())
    {

        countryName = parseToQString(country, "name", "noname", false);
		ODD::mainWindow()->getSignalManager()->addCountry(countryName);

        QDomElement element = country.firstChildElement("signals");
        if (!element.isNull())
        {
			QDomElement category = element.firstChildElement("category");
			while (!category.isNull())
			{		
				QString categoryName = parseToQString(category, "name", "noname", false);
				parseSignalPrototypes(category, categoryName, countryName);
				category = category.nextSiblingElement("category");
			}
        }

        element = country.firstChildElement("objects");
        if (!element.isNull())
        {
			QDomElement category = element.firstChildElement("category");
			while (!category.isNull())
			{		
				QString categoryName = parseToQString(category, "name", "noname", false);
				parseObjectPrototypes(category, categoryName, countryName);
				category = category.nextSiblingElement("category");
			}
        }

        country = country.nextSiblingElement("country");
    }

    // TODO Objects //
    //

    // Postprocessing //
    //
    //	postprocess();

    return true;
}

bool
DomParser::parseSignalPrototypes(const QDomElement &element, const QString &categoryName, const QString &countryName)
{
    QDomElement sign = element.firstChildElement("sign");
    while (!sign.isNull())
    {
        // Name and Icon //
        //
        QString name = parseToQString(sign, "name", "", false); // mandatory
        QString icon = ":/signalIcons/" + parseToQString(sign, "icon", "", true); // optional
        QString type = parseToQString(sign, "type", "-1", false);
        QString typeSubclass = parseToQString(sign, "subclass", "", true);
        QString subType = parseToQString(sign, "subtype", "-1", true);
        double value = parseToDouble(sign, "value", 0.0, true);
        double distance = parseToDouble(sign, "distance", 0.0, true);
		double heightOffset = parseToDouble(sign, "heightOffset", 0.0, true);
		QString unit = parseToQString(sign, "unit", "km/h", true);
		QString text = parseToQString(sign, "text", "", true);
		double width = parseToDouble(sign, "width", 0.0, true);
		double height = parseToDouble(sign, "height", 0.0, true);

		SignalManager *signalManager = ODD::mainWindow()->getSignalManager();
		signalManager->addSignal(countryName, name, QIcon(icon), categoryName, type, typeSubclass, subType, value, distance, heightOffset, unit, text, width, height);
		signalManager->addCategory(categoryName);

        sign = sign.nextSiblingElement("sign");
    }

    // Return successfully
    return true;
}

bool
DomParser::parseObjectPrototypes(const QDomElement &element, const QString &categoryName, const QString &countryName)
{
    QDomElement object = element.firstChildElement("object");
    while (!object.isNull())
    {
        // Name and Icon //
        //
        QString name = parseToQString(object, "name", "", false); // mandatory
        QString file = parseToQString(object, "file", "", true); // optional 
        QString icon = ":/signalIcons/" + parseToQString(object, "icon", "", true); // optional
        QString type = parseToQString(object, "type", "", true);
        double length = parseToDouble(object, "length", 0.0, true);
        double width = parseToDouble(object, "width", 0.0, true);
        double height = parseToDouble(object, "height", 0.0, true);
        double radius = parseToDouble(object, "radius", 0.0, true);
        double distance = parseToDouble(object, "distance", 0.0, true);
        double heading = parseToDouble(object, "heading", 0.0, true);
        double repeatDistance = parseToDouble(object, "repeatDistance", 0.0, true);

        // Corners of the outline of the object //
        //
        QList<ObjectCorner *> corners;
        QDomElement corner = object.firstChildElement("corner");
        while (!corner.isNull())
        {
            double u = parseToDouble(corner, "u", 0.0, true);
            double v = parseToDouble(corner, "v", 0.0, true);
            double z = parseToDouble(corner, "z", 0.0, true);
            double height = parseToDouble(corner, "height", 0.0, true);

            ObjectCorner *objectCorner = new ObjectCorner(u, v, z, height, true);
            corners.append(objectCorner);

            corner = corner.nextSiblingElement("corner");
        }

		SignalManager *manager = ODD::mainWindow()->getSignalManager();
        manager->addObject(countryName, name, file, QIcon(icon), categoryName, type, length, width, radius, height, distance, heading, repeatDistance, corners);
		manager->addCategory(categoryName);

        object = object.nextSiblingElement("object");
    }

    // Return successfully
    return true;
}
