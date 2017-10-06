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

#include "src/data/roadsystem/sections/objectobject.hpp"
#include "src/data/roadsystem/sections/crosswalkobject.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"
#include "src/data/roadsystem/sections/sensorobject.hpp"
#include "src/data/roadsystem/sections/surfaceobject.hpp"
#include "src/data/roadsystem/sections/bridgeobject.hpp"

#include "src/data/roadsystem/roadlink.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"

#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementspiral.hpp"
#include "src/data/roadsystem/track/trackelementpoly3.hpp"

#include "src/data/roadsystem/sections/typesection.hpp"
#include "src/data/roadsystem/sections/surfacesection.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"
#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"
#include "src/data/roadsystem/sections/laneroadmark.hpp"
#include "src/data/roadsystem/sections/lanespeed.hpp"
#include "src/data/roadsystem/sections/laneheight.hpp"

#include "src/data/roadsystem/sections/elevationsection.hpp"
#include "src/data/roadsystem/sections/superelevationsection.hpp"
#include "src/data/roadsystem/sections/crossfallsection.hpp"

#include "src/data/roadsystem/sections/objectobject.hpp"

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
{
    doc_ = new QDomDocument();
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
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
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
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
                             tr("Root element is not <OpenDRIVE>!"));
        return false;
    }

    // <OpenDRIVE><header> //
    //
    QDomElement child;
    child = root.firstChildElement("header");
    if (child.isNull())
    {
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
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

    if (!elementIDs_.empty())
    {
        roadSystem_->checkIDs(elementIDs_);
        elementIDs_.clear();
    }
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

    // Open file and parse tree //
    //
    QString errorStr = "";
    int errorLine = 0;
    int errorColumn = 0;

    if (!doc_->setContent(source, true, &errorStr, &errorLine, &errorColumn))
    {
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
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
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
                             tr("Root element is not <ODDLot>!"));
        return false;
    }

    // Prototypes //
    //
    QDomElement prototypesRoot;
    prototypesRoot = root.firstChildElement("prototypes");
    if (prototypesRoot.isNull())
    {
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
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
            QString p = "p";
            RSystemElementRoad *road = parseRoadElement(child, p);
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
            QString p = "p";
            RSystemElementRoad *road = parseRoadElement(child, p);
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
            QString p = "p";
            RSystemElementRoad *road = parseRoadElement(child, p);
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
            QString p = "p";
            RSystemElementRoad *road = parseRoadElement(child, p);
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
            QString p = "p";
            RSystemElementRoad *road = parseRoadElement(child, p);
            road->accept(spArcSMergeVisitor);

            ODD::mainWindow()->getPrototypeManager()->addRoadPrototype(name, QIcon(icon), road, PrototypeManager::PTP_CrossfallPrototype,system,type,lanes);

            prototype = prototype.nextSiblingElement("crossfallPrototype");
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
            QString p = "p";
            RSystemElementRoad *road = parseRoadElement(child, p);
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
                    QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
                                         tr("Prototype has a road without Tracks (PlanView)!"));
                    return false;
                }

                if (road->getLaneSections().isEmpty())
                {
                    QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
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
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
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

/* New Element ID [Tilenumber]_[Elementnumber]_[UserDefined Name]
*/

void
DomParser::setTile(const QString &id, QString &oldTileId)
{

    if (id.count("_") >= 2)
    {
        bool number = false;
        QStringList parts = id.split("_");

        QString tileId = parts.at(0);
        if (oldTileId.isEmpty())
        {
            oldTileId = tileId;
        }

        int tileNumber = tileId.toInt(&number);
        if (number)
        {
            parts.at(1).toInt(&number);

            if (number) // IDs are formatted
            {
                if (tileId != oldTileId) //New Tile
                {
                    if (!tileCounts_.contains(tileNumber))
                    {
                        oldTileId = tileId;
                        tileCount_++;
                        QString tileName = QString("Tile%1").arg(tileCount_);
                        tileCounts_.insert(tileNumber, tileCount_);
                        Tile *tile = new Tile(tileName, QString("%1").arg(tileCount_));

                        tileSystem_->addTile(tile);
                    }
                    else if (!tileSystem_->getTile(tileId))
                    {
                        tileCount_++;
                        QString tileName = QString("Tile%1").arg(tileCount_);
                        Tile *tile = new Tile(tileName, QString("%1").arg(tileCount_));

                        tileSystem_->addTile(tile);
                    }

                    tileSystem_->setCurrentTile(tileSystem_->getTile(tileId));
                    oldTileId = tileId;
                }
                else if (!tileCounts_.contains(tileNumber))
                {
                    tileCounts_.insert(tileNumber, tileCount_);
                }
            }
        }
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
    float version = parseToFloat(element, "version", 1.0f, true); // optional
	
	if ((version > ODD::getVersion()) || (revMajor > ODD::getRevMajor()) || (revMinor > ODD::getRevMinor()))
	{
		qDebug() << "Oddlot only supports OpenDrive versions up to 1.3";
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
    projectData_->setVersion(version);
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

    /*    QDomElement child = root.firstChildElement("tile");
    if(child.isNull())
    {*/
    // Create a tile
    //
    QString id = QString("%1").arg(tileCount_);
    QString tileName = QString("Tile%1").arg(id);
    Tile *tile = new Tile(tileName, id);

    tileSystem_->addTile(tile);

    /*   }
    else
    {
        while(!child.isNull())
        {*/
    //           QString name = parseToQString(child,"name","Untitled", true);
    //           QString id = parseToQString(child,"id","",true);
    /*			QString id = QString("%1").arg(projectData_->getTileSystem()->getTileCount());
           if (id.isEmpty())
           {
               id = parseToQString(child,"ID","",false);
           }
		   QString tileName = QString("Tile%1").arg(id);
           Tile * tile = new Tile(tileName, id);*/
    // TODO: Tile getUniqueID

    /*          QDomElement child2 = child.firstChildElement("road");
           while(!child2.isNull()){
               RSystemElementRoad * road = parseRoadElement(child2);
               QString id2 = road->getID();
               QString idNew = QString("%1_%2_%3").arg(id).arg(id2).arg(road->getName());
               road->setID(idNew);
               roadSystem_->addRoad(road);
//               tile->getRoadSystem()->addRoad(road);
//               fprintf(stderr,"%s != %s\n",idNew.toUtf8().constData(),road->getID().toUtf8().constData());
               if(idNew != road->getID())
               {
				   idsChanged = 1;
				   qDebug() << "Domparser: Warning! Road ID assigned twice: %1. This won't go so smoothly with road linkage, so you'd better check your file..." << id;
               }
               child2 = child2.nextSiblingElement("road");
           }*/

    /*           child = child.nextSiblingElement("tile");
        }

    }*/
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
    QString oldTileId = QString();

    // <OpenDRIVE><road> //
    //

    QDomElement child = root.firstChildElement("road");
    if (child.isNull())
    {
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
                             tr("Missing <road> element!"));
        return false;
    }
    else
    {
        while (!child.isNull())
        {
            RSystemElementRoad *road = parseRoadElement(child, oldTileId);

            child = child.nextSiblingElement("road");
        }
    }

    // Optional Elements //
    //
    child = root.firstChildElement("controller");
    while (!child.isNull())
    {
        parseControllerElement(child, oldTileId);
        child = child.nextSiblingElement("controller");
    }

    child = root.firstChildElement("junction");
    while (!child.isNull())
    {
        parseJunctionElement(child, oldTileId);
        child = child.nextSiblingElement("junction");
    }

    child = root.firstChildElement("fiddleyard");
    while (!child.isNull())
    {
        parseFiddleyardElement(child, oldTileId);
        child = child.nextSiblingElement("fiddleyard");
    }

    child = root.firstChildElement("pedFiddleyard");
    while (!child.isNull())
    {
        parsePedFiddleyardElement(child, oldTileId);
        child = child.nextSiblingElement("pedFiddleyard");
    }

    return true;
}

/** Parses a <road> element.
*
*/
RSystemElementRoad *
DomParser::parseRoadElement(QDomElement &element, QString &oldTileId)
{
    // 1.) Parse Attributes //
    //
    QString name = parseToQString(element, "name", "Untitled", true); // optional
    //	double length = parseToDouble(element, "length", 0.0, true);			// optional
    QString id = parseToQString(element, "id", "", true); // "id" is optional...
    if (id.isEmpty())
    {
        id = parseToQString(element, "ID", "", false); // ...but at least "ID" should be there
    }
    QString junction = parseToQString(element, "junction", "-1", true); // optional

    // For Testing //
    //
    //	if(id != "17")
    //	{
    //		return NULL;
    //	}

    RSystemElementRoad *road = new RSystemElementRoad(name, id, junction);

    if (projectData_) // Change ids not for Prototypes
    {
        // Check if the ids have the format [Tilenumber]_[Elementnumber]_[Name]
        //
        QString id = road->getID();

        setTile(id, oldTileId);

        roadSystem_->addRoad(road); // This may change the ID!
        if (id != road->getID())
        {
			RoadSystem::IdType el = {road->getID(), "road"};
            elementIDs_.insert(id, el);
        }
    }
    else if (roadSystem_)
    {
        road->setID(QString("p_%1_%2").arg(prototypeElementCount_++).arg(road->getName()));
        roadSystem_->addRoad(road); // This may change the ID!
    }

    // 2.) Parse children //
    //
    QDomElement child;

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
            RoadLink *roadLink = new RoadLink(elementType, elementId, contactPoint);
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
            RoadLink *roadLink = new RoadLink(elementType, elementId, contactPoint);
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

    if (!foundLateral && (mode_ != DomParser::MODE_PROTOTYPES))
    {
        SuperelevationSection *sSection = new SuperelevationSection(0.0, 0.0, 0.0, 0.0, 0.0);
        road->addSuperelevationSection(sSection);

        CrossfallSection *cSection = new CrossfallSection(CrossfallSection::DCF_SIDE_BOTH, 0.0, 0.0, 0.0, 0.0, 0.0);
        road->addCrossfallSection(cSection);
    }

    // <lanes> //
    child = element.firstChildElement("lanes");
    if (!child.isNull())
    {
        child = child.firstChildElement("laneSection");
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
    }
    else
    {
        // TODO: NOT OPTIONAL
        //		qDebug("NOT OPTIONAL: <lanes>");
    }

    // <objects>                //
    // (optional, max count: 1) //
    child = element.firstChildElement("objects");
    if (!child.isNull())
    {
        parseObjectsElement(child, road, oldTileId);

        // Check that the maximum (1) is not exceeded
        child = child.nextSiblingElement("objects");
        if (!child.isNull())
            qDebug("WARNING: maximum of one <objects> element, ignoring any subsequent ones");
    }

    // <signals> //
    child = element.firstChildElement("signals");
    if (!child.isNull())
    {
        parseSignalsElement(child, road, oldTileId);

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

    SpeedRecord *sr = new SpeedRecord(max, unit);
    type->setSpeedRecord(sr);

    return true;
}

/*! \brief Parses a road:objects element.
*
*/
bool
DomParser::parseObjectsElement(QDomElement &element, RSystemElementRoad *road, QString &oldTileId)
{
    // Find all objects (unlimited)
    QDomElement child = element.firstChildElement("object");
    while (!child.isNull())
    {
        // Get mandatory attributes
        QString type = parseToQString(child, "type", "", false); // mandatory

        // Don't create objects for the simple poles of signs and the traffic lights

        if (type != "simplePole")
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
            double t = parseToDouble(child, "t", 0.0, false); // mandatory
            double zOffset = parseToDouble(child, "zOffset", 0.0, true); // optional
            double validLength = parseToDouble(child, "validLength", 0.0, true); // optional
            QString orientationString = parseToQString(child, "orientation", "+", true); // optional
            Object::ObjectOrientation orientation;
            if (orientationString == "+")
            {
                orientation = Object::POSITIVE_TRACK_DIRECTION;
            }
            else if (orientationString == "-")
            {
                orientation = Object::NEGATIVE_TRACK_DIRECTION;
            }
            else
            {
                orientation = Object::BOTH_DIRECTIONS;
            }
            double length = parseToDouble(child, "length", 0.0, true); // optional
            double width = parseToDouble(child, "width", 0.0, true); // optional
            double radius = parseToDouble(child, "radius", 0.0, true); // optional
            double height = parseToDouble(child, "height", 0.0, true); // optional
            double hdg = parseToDouble(child, "hdg", 0.0, true); // optional
            double pitch = parseToDouble(child, "pitch", 0.0, true); // optional
            double roll = parseToDouble(child, "roll", 0.0, true); // optional

            double repeatS = -1.0;
            double repeatLength = -1.0;
            double repeatDistance = -1.0;

            // Get <repeat> record
            QDomElement objectChild = child.firstChildElement("repeat");
            if (!objectChild.isNull())
            {
                repeatS = parseToDouble(objectChild, "s", 0.0, false); // mandatory
                repeatLength = parseToDouble(objectChild, "length", 0.0, false); // mandatory
                repeatDistance = parseToDouble(objectChild, "distance", 0.0, false); // mandatory
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

            // Construct object object
            if (type != "")
            {
                Object *object = new Object(id, name, type, s, t, zOffset, validLength, orientation,
                    length, width, radius, height, hdg * 180.0 / (M_PI), pitch  * 180.0 / (M_PI), roll  * 180.0 / (M_PI), pole, repeatS, repeatLength, repeatDistance, textureFile);

                setTile(id, oldTileId);

                // Add to road
                road->addObject(object);

                if (id != object->getId())
                {
					RoadSystem::IdType el = {object->getId(), "object"};
                    elementIDs_.insert(id, el);
                }
            }
       }

        // Attempt to locate another object
        child = child.nextSiblingElement("object");
    }

    // Find all bridges (unlimited)
    child = element.firstChildElement("bridge");
    while (!child.isNull())
    {
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

        int typenr = 0; //"concrete"
        if (type == "steel")
        {
            typenr = 1;
        }
        else if (type == "brick")
        {
            typenr = 2;
        }
        else if (type == "wood")
        {
            typenr = 3;
        }

        // Construct bridge object
        Bridge *bridge = new Bridge(id, modelFile, name, typenr, s, length);

        setTile(id, oldTileId);

        // Add to road
        road->addBridge(bridge);

        if (id != bridge->getId())
        {
			RoadSystem::IdType el = {bridge->getId(), "bridge"};
            elementIDs_.insert(id, el);
        }

        // Attempt to locate another object
        child = child.nextSiblingElement("bridge");
    }

    // Find all tunnels (unlimited)
    child = element.firstChildElement("tunnel");
    while (!child.isNull())
    {
        // Get mandatory attributes
        QString type = parseToQString(child, "type", "", false); // mandatory
        QString name = parseToQString(child, "name", "", false); // mandatory
        QString modelFile = parseToQString(child, "modelFile", name, true); // optional

        QString id = parseToQString(child, "id", "", false); // mandatory
        double s = parseToDouble(child, "s", 0.0, false); // mandatory
        double length = parseToDouble(child, "length", 0.0, false); // mandatory

        // Construct tunnel object
        Bridge *bridge = new Bridge(id, modelFile, name, 0, s, length);

        setTile(id, oldTileId);

        // Add to road
        road->addBridge(bridge);

        if (id != bridge->getId())
        {
			RoadSystem::IdType el = {bridge->getId(), "bridge"};
            elementIDs_.insert(id, el);
        }

        // Attempt to locate another object
        child = child.nextSiblingElement("tunnel");
    }

    // Find all crosswalks (unlimited) and create a signal. Conversion for old files.
    //
    child = element.firstChildElement("crosswalk");
    while (!child.isNull())
    {
        // Get mandatory attributes
        QString id = parseToQString(child, "id", "", false); // mandatory
        QString name = parseToQString(child, "name", "", false); // mandatory
        double s = parseToDouble(child, "s", 0.0, false); // mandatory
        double length = parseToDouble(child, "length", 0.0, false); // mandatory

        // Construct crosswalk object
        Crosswalk *crosswalk = new Crosswalk(id, name, s, length);

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

        // Construct signal object
        Signal *signal = new Signal(id, name, s, 0.0, "no", Signal::POSITIVE_TRACK_DIRECTION, 0.0, "Germany", 293, "", -1, length, 0.0, 0.0, 0.0, "km/h", "", 0.0, 0.0, false, 2, crosswalk->getFromLane(), crosswalk->getToLane(), crosswalk->getCrossProb(), crosswalk->getResetTime());
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
DomParser::parseSignalsElement(QDomElement &element, RSystemElementRoad *road, QString &oldTileId)
{
    QDomElement child = element.firstChildElement("signal");
    while (!child.isNull())
    {
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
        Signal::OrientationType orientation = Signal::BOTH_DIRECTIONS;
        if (orientationString == "+")
        {
            orientation = Signal::POSITIVE_TRACK_DIRECTION;
        }
        else if (orientationString == "-")
        {
            orientation = Signal::NEGATIVE_TRACK_DIRECTION;
        }
        double zOffset = parseToDouble(child, "zOffset", 0.0, false); // mandatory
        QString country = parseToQString(child, "country", "Germany", false); // mandatory
        int type = parseToInt(child, "type", 0, false); // mandatory
        
        int subtype = parseToInt(child, "subtype", -1, true); // optional
        double value = parseToDouble(child, "value", 0.0, true); // optional
        double hOffset = parseToDouble(child, "hOffset", 0.0, true); // optional
        double pitch = parseToDouble(child, "pitch", 0.0, true); // optional
		QString unit = parseToQString(child, "unit", "km/h", true); //optional
		QString text = parseToQString(child, "text", "", true);//optional
		double width = parseToDouble(child, "width", 0.0, true);//optional
		double height = parseToDouble(child, "height", 0.0, true);//optional
        double roll = parseToDouble(child, "roll", 0.0, true); // optional

        // Get validity record

        int fromLane = 0;
        int toLane = 0;
        QDomElement objectChild = child.firstChildElement("validity");
        if (!objectChild.isNull())
        {
            fromLane = parseToInt(objectChild, "fromLane", 0, false); // mandatory
            toLane = parseToInt(objectChild, "toLane", 0.0, false); // mandatory
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

        if ((type == 625) && (subtype == 10) && (typeSubclass == "20"))
        {
            hOffset = name.toDouble();

            // Construct signal object
            signal = new Signal(id, "", s, t, dynamic, orientation, zOffset, country, type, typeSubclass, subtype, value, hOffset, pitch  * 180.0 / (M_PI), roll  * 180.0 / (M_PI), unit, text, width, height, pole, size, fromLane, toLane, crossProb, resetTime);
        }
        else
        {
            // Construct signal object
            signal = new Signal(id, name, s, t, dynamic, orientation, zOffset, country, type, typeSubclass, subtype, value, hOffset * 180.0 / (M_PI), pitch  * 180.0 / (M_PI), roll  * 180.0 / (M_PI), unit, text, width, height, pole, size, fromLane, toLane, crossProb, resetTime);
        }


        setTile(id, oldTileId);

        // Add to road
        road->addSignal(signal);
        if (id != signal->getId())
        {
			RoadSystem::IdType el = {signal->getId(), "signal"};
            elementIDs_.insert(id, el);
        }

        // Attempt to locate another signal
        child = child.nextSiblingElement("signal");
    } // Find all signals (unlimited)

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
            QString file = parseToQString(child, "file", "", true);
            QString sStart = parseToQString(child, "sStart", "", true);
            QString sEnd = parseToQString(child, "sEnd", "", true);
            QString orientation = parseToQString(child, "orientation", "", true);
            QString mode = parseToQString(child, "mode", "", true);
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

    LaneSection *laneSection = new LaneSection(s);

    // <laneSection><left/center/right> //
    //
    QDomElement child = laneSectionElement.firstChildElement();

    while (!child.isNull())
    {
        QDomElement lane = child.firstChildElement("lane");
        if (lane.isNull())
        {

            // TODO: NOT OPTIONAL
            qDebug() << "NOT OPTIONAL: <laneSection><left/center/right><lane> at road: " << road->getID() << " " << road->getName();
        }
        while (!lane.isNull())
        {
            parseLaneElement(lane, laneSection);
            lane = lane.nextSiblingElement("lane");
        }

        child = child.nextSiblingElement();
    }

    // Add LaneSection to Road //
    //
    road->addLaneSection(laneSection);

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
    if (child.isNull() && id != 0)
    {

        // NOT OPTIONAL
    }
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

        LaneRoadMark *roadMarkEntry = new LaneRoadMark(
            sOffset,
            LaneRoadMark::parseRoadMarkType(type),
            LaneRoadMark::parseRoadMarkWeight(weight),
            LaneRoadMark::parseRoadMarkColor(color),
            width,
            LaneRoadMark::parseRoadMarkLaneChange(laneChange));
        lane->addRoadMarkEntry(roadMarkEntry);

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
DomParser::parseControllerElement(QDomElement &controllerElement, QString &oldTileId)
{
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

        ControlEntry *entry = new ControlEntry(signalId, type);
        controlEntries.append(entry);

        control = control.nextSiblingElement("control");
    }

    // Construct Controller
    //
    RSystemElementController *controller = new RSystemElementController(name, id, sequence, script, cycleTime, controlEntries);

    setTile(id, oldTileId);
    roadSystem_->addController(controller);


    if (id != controller->getID())
    {
		RoadSystem::IdType el = {controller->getID(), "controller"};
        elementIDs_.insert(id, el);
    }

    return true;
}

//################//
// JUNCTION       //
//################//

/** Parses a <junction> element.
*
*/
bool
DomParser::parseJunctionElement(QDomElement &element, QString &oldTileId)
{

    // 1.) Parse Attributes //
    //
    QString name = parseToQString(element, "name", "Untitled", true); // optional
    QString id = parseToQString(element, "id", "", false); // mandatory

    RSystemElementJunction *junction = new RSystemElementJunction(name, id);

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

        // <laneLink> //
        JunctionConnection *connection = new JunctionConnection(childId, incomingRoad, connectingRoad, contactPoint, numerator);

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

    setTile(id, oldTileId);
    roadSystem_->addJunction(junction);

    if (id != junction->getID())
    {
		RoadSystem::IdType el = {junction->getID(), "junction"};
        elementIDs_.insert(id, el);
    }

    return true;
}

//################//
// FIDDLEYARD     //
//################//

/** Parses a <fiddleyard> element.
*
*/
bool
DomParser::parseFiddleyardElement(QDomElement &element, QString &oldTileId)
{

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

    RSystemElementFiddleyard *fiddleyard = new RSystemElementFiddleyard(name, id, elementType, elementId, contactPoint);

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

        FiddleyardSource *source = new FiddleyardSource(id, lane, startTime, repeatTime, velocity, velocityDeviance);

        QDomElement subchild = child.firstChildElement("vehicle");
        while (!subchild.isNull())
        {
            QString id = parseToQString(subchild, "id", "", false); // mandatory
            double numerator = parseToDouble(subchild, "numerator", 1.0, true); // optional
            source->addVehicle(id, numerator);
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

        fiddleyard->addSink(new FiddleyardSink(id, lane));

        child = child.nextSiblingElement("sink");
    }

    setTile(id, oldTileId);
    roadSystem_->addFiddleyard(fiddleyard);

    if (id != fiddleyard->getID())
    {
		RoadSystem::IdType el = {fiddleyard->getID(), "fiddleyard"};
        elementIDs_.insert(id, el);
    }

    return true;
}

//################//
// PEDFIDDLEYARD  //
//################//

/** Parses a <pedFiddleyard> element.
*
*/
bool
DomParser::parsePedFiddleyardElement(QDomElement &element, QString &oldTileId)
{

    // 1.) Parse Attributes //
    //
    QString id = parseToQString(element, "id", "", false); // mandatory
    QString name = parseToQString(element, "name", "", false); // mandatory
    QString roadId = parseToQString(element, "roadId", "", false); // mandatory

    RSystemElementPedFiddleyard *fiddleyard = new RSystemElementPedFiddleyard(id, name, roadId);

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

        // Create source
        PedFiddleyardSource *source = new PedFiddleyardSource(id,
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

        // Create sink
        PedFiddleyardSink *sink = new PedFiddleyardSink(id, lane);

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

    setTile(id, oldTileId);
    roadSystem_->addPedFiddleyard(fiddleyard);

    if (id != fiddleyard->getID())
    {
		RoadSystem::IdType el = {fiddleyard->getID(), "pedFiddleyard"};
        elementIDs_.insert(id, el);
    }

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

    // pool //
    //
    QString elementId = parseToQString(element, "id", "", false); // mandatory
    QString elementName = parseToQString(element, "name", "", true); // optional
    double velocity = parseToDouble(element, "velocity", 33.0, true); // optional
    double velocityDeviance = parseToDouble(element, "velocityDeviance", 5.0, true); // optional
    double numerator = parseToDouble(element, "numerator", 20, true); // optional

    Pool *pool = new Pool(elementId, elementName, velocity, velocityDeviance, numerator);

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
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
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
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
                             tr("Root element is not <ODDLot>!"));
        return false;
    }

    // Signals //
    //
    QDomElement signalsRoot;
    signalsRoot = root.firstChildElement("signalsObjects");
    if (signalsRoot.isNull())
    {
        QMessageBox::warning(NULL, tr("ODD: XML Parser Error"),
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
        int type = parseToInt(sign, "type", 0, false);
        QString typeSubclass = parseToQString(sign, "subclass", "", true);
        int subType = parseToInt(sign, "subtype", -1, true);
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

            ObjectCorner *objectCorner = new ObjectCorner(u, v, z, height);
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
