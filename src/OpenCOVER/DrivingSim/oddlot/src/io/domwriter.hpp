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

#ifndef DOMWRITER_HPP
#define DOMWRITER_HPP

#include "../data/acceptor.hpp"
#include "../data/roadsystem/odrID.hpp"
#include "src/gui/exportsettings.hpp"

#include <QDomElement>
#include <QMap>
#include <QSet>

class QDomDocument;
class ProjectData;
class SignalManager;

class SurfaceSection;
class CarPool;
class Pool;

class GeoReference;
class TileSystem;


class DomWriter : public Visitor
{
public:
    DomWriter(ProjectData *projectData);
    virtual ~DomWriter()
    {
    }

    void runToTheHills();
    QDomDocument *getDomDocument()
    {
        return doc_;
    }

    virtual void visit(Acceptor * /*acceptor*/)
    { /* does nothing by default */
    }

	void addTileInfo(QDomElement element, uint32_t tileID);

    virtual void visit(RoadSystem *);
    virtual void visit(RSystemElementRoad *);
    virtual void visit(RSystemElementController *);
    virtual void visit(RSystemElementFiddleyard *);
    virtual void visit(RSystemElementPedFiddleyard *);

    virtual void visit(TypeSection *);

    virtual void visit(TrackSpiralArcSpiral *);
    virtual void visit(TrackElement *);
    virtual void visit(TrackElementLine *);
    virtual void visit(TrackElementArc *);
    virtual void visit(TrackElementSpiral *);
    virtual void visit(TrackElementPoly3 *);
	virtual void visit(TrackElementCubicCurve *);

    virtual void visit(SurfaceSection *);

    virtual void visit(ElevationSection *);
    virtual void visit(SuperelevationSection *);
    virtual void visit(CrossfallSection *);
    virtual void visit(ShapeSection *);

    virtual void visit(LaneSection *);
    virtual void visit(Lane *);
    virtual void visit(LaneWidth *);
	virtual void visit(LaneBorder *);
    virtual void visit(LaneRoadMark *);
    virtual void visit(LaneSpeed *);
    virtual void visit(LaneHeight *);
	virtual void visit(LaneRule *);
	virtual void visit(LaneAccess *);

    virtual void visit(Object *);
	virtual void visit(ObjectReference *);
    virtual void visit(Bridge *);
    virtual void visit(Tunnel *);
    virtual void visit(Crosswalk *);
    virtual void visit(Signal *);
	virtual void visit (SignalReference *);
    virtual void visit(Sensor *);

    virtual void visit(FiddleyardSink *);
    virtual void visit(FiddleyardSource *);

    virtual void visit(PedFiddleyardSink *);
    virtual void visit(PedFiddleyardSource *);

    virtual void visit(RSystemElementJunction *);
    virtual void visit(JunctionConnection *);

	virtual void visit(RSystemElementJunctionGroup *);

    // VehicleSystem //
    //
    virtual void visit(VehicleSystem *);
    virtual void visit(VehicleGroup *);
    virtual void visit(RoadVehicle *);
    virtual void visit(CarPool *);
    virtual void visit(Pool *);

    // PedestrianSystem //
    //
    virtual void visit(PedestrianSystem *);
    virtual void visit(PedestrianGroup *);
    virtual void visit(Pedestrian *);

    // ScenerySystem //
    //
    virtual void visit(ScenerySystem *);
    virtual void visit(SceneryMap *);
    virtual void visit(Heightmap *);
    virtual void visit(SceneryTesselation *);

	// Georeference //
	//
	virtual void visit(GeoReference *);

private:
    DomWriter()
        : Visitor()
    {
    }
	ExportSettings::ExportIDVariants exportIDvar;
	///write original ID if possible, otherwise create a unique ID based on the original one
	QString getIDString(const odrID &ID, const QString &name);

	QMap<odrID, QString> writtenIDs[odrID::NUM_IDs];
	QSet<QString> writtenIDStrings[odrID::NUM_IDs];

    QDomDocument *doc_;
    QDomElement root_;

	QDomElement header_;

    QDomElement currentRoad_;
    QDomElement currentPVElement_;

    QDomElement currentElevationProfileElement_;
    QDomElement currentLateralProfileElement_;

    QDomElement currentLanesElement_;
    QDomElement currentLaneSectionElement_;
    QDomElement currentLeftLaneElement_;
    QDomElement currentCenterLaneElement_;
    QDomElement currentRightLaneElement_;
    QDomElement currentLaneElement_;

    QDomElement currentTrackElement_;
    QDomElement currentFiddleyardElement_;
    QDomElement currentPedFiddleyardElement_;

    QDomElement currentJunctionElement_;

    QDomElement currentControllerElement_;

    QDomElement currentObjectsElement_;
    QDomElement currentSignalsElement_;
    QDomElement currentSensorsElement_;

    QDomElement currentVehicleGroupElement_;
    QDomElement currentCarPoolElement_;

    QDomElement currentPedestrianGroupElement_;

    ProjectData *projectData_;
	TileSystem *tileSystem_;

    SignalManager *signalManager_;
};

#endif // DOMWRITER_HPP
