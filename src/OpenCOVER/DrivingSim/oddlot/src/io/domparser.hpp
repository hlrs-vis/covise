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

#ifndef DOMPARSER_HPP
#define DOMPARSER_HPP

#include <QObject>
// because of tr()
#include <QMap>
#include <QStringList>

#include "../data/roadsystem/roadsystem.hpp"

class QIODevice;
class QDomDocument;
class QDomElement;

//class QMap;
//class QStringList;

class ProjectData;

class Tile;

class RSystemElementRoad;
class LaneSection;

class VehicleSystem;
class VehicleGroup;
class RoadVehicle;

class PedestrianSystem;
class PedestrianGroup;
class Pedestrian;

class CarPool;
class Pool;
class PoolVehicle;

class ScenerySystem;

class TileSystem;

class TypeSection;

class DomParser : public QObject
{
    Q_OBJECT

    //################//
    // STATIC         //
    //################//

    enum Mode
    {
        MODE_NONE,
        MODE_XODR,
        MODE_PROTOTYPES
    };


    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit DomParser(ProjectData *, QObject *parent = NULL);
    ~DomParser();

    // XODR //
    //
    bool parseXODR(QIODevice *device);

    // Prototypes //
    //
    bool parsePrototypes(QIODevice *device);

    // Signals  and Objects prototypes //
    //
    bool parseSignals(QIODevice *device);
    bool parseSignalPrototypes(const QDomElement &element, const QString &categoryName, const QString &countryName);
    bool parseObjectPrototypes(const QDomElement &element, const QString &categoryName, const QString &countryName);

    // Postprocessing //
    //
    void postprocess();

    // Support Functions //
    //
    int parseToInt(const QDomElement &element, const QString &attributeName, int defaultValue, bool optional = false);
    float parseToFloat(const QDomElement &element, const QString &attributeName, float defaultValue, bool optional = false);
    double parseToDouble(const QDomElement &element, const QString &attributeName, double defaultValue, bool optional = false);
    QString parseToQString(const QDomElement &element, const QString &attributeName, const QString &defaultValue, bool optional = false);

    // Parse Elements //
    //
    bool parseHeaderElement(QDomElement &child);

    bool parseTile(QDomElement &element); // returns -1 (no Tile), 0 (ids not changed), 1 (ids changed)
    bool parseRoadSystem(QDomElement &element);

    bool parseVehicleSystem(QDomElement &element);
    bool parsePedestrianSystem(QDomElement &element);
    bool parseScenerySystem(QDomElement &element);

    bool parseCarPool(const QDomElement &element);
    PoolVehicle *parsePoolVehicle(const QDomElement &element);
    Pool *parsePool(const QDomElement &element);

    RSystemElementRoad *parseRoadElement(QDomElement &child, QString &oldTileId);
    bool parseTypeElement(QDomElement &element, RSystemElementRoad *road);
    bool parseSpeedElement(QDomElement &element, TypeSection *type);
    bool parseSurfaceElement(QDomElement &element, RSystemElementRoad *road);
    bool parseObjectsElement(QDomElement &element, RSystemElementRoad *road, QString &oldTileId);
    bool parseSignalsElement(QDomElement &element, RSystemElementRoad *road, QString &oldTileId);
    bool parseSensorsElement(QDomElement &element, RSystemElementRoad *road);
    bool parseGeometryElement(QDomElement &element, RSystemElementRoad *road);
    bool parseElevationElement(QDomElement &element, RSystemElementRoad *road);
    bool parseSuperelevationElement(QDomElement &element, RSystemElementRoad *road);
    bool parseCrossfallElement(QDomElement &element, RSystemElementRoad *road);
	bool parseShapeElement(QDomElement &element, RSystemElementRoad *road);

    bool parseLaneSectionElement(QDomElement &element, RSystemElementRoad *road);
    bool parseLaneElement(QDomElement &element, LaneSection *laneSection);

    bool parseControllerElement(QDomElement &child, QString &oldTileId);
    bool parseJunctionElement(QDomElement &child, QString &oldTileId);
	bool parseJunctionGroupElement(QDomElement &child, QString &oldTileId);
    bool parseFiddleyardElement(QDomElement &child, QString &oldTileId);
    bool parsePedFiddleyardElement(QDomElement &child, QString &oldTileId);
    bool parseSceneryElement(QDomElement &child);
    bool parseEnvironmentElement(QDomElement &child);

    bool parseVehiclesElement(const QDomElement &element);
    RoadVehicle *parseRoadVehicleElement(const QDomElement &element);
    bool parsePedestriansElement(const QDomElement &element);
    Pedestrian *parsePedElement(const QDomElement &element, bool defaultPed, bool templatePed);

protected:
    //	DomParser(){ /* not allowed */ };

    //################//
    // PROPERTIES     //
    //################//

private:
    QDomDocument *doc_;

    bool check(bool success, const QDomElement &element, const QString &attributeName, const QString &type);
    void setTile(const QString &id, QString &oldId);

    ProjectData *projectData_;

    //    QMap<QString, Tile *>   tiles_;
    int prototypeElementCount_;
    int elementCount_;
    int tileCount_;

    RoadSystem *roadSystem_;
    TileSystem *tileSystem_;
    VehicleSystem *vehicleSystem_;
    PedestrianSystem *pedestrianSystem_;
    ScenerySystem *scenerySystem_;

    DomParser::Mode mode_;

    QMultiMap<QString, RoadSystem::IdType> elementIDs_;
    QMap<int, int> tileCounts_;
};

#endif // DOMPARSER_HPP
