/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   04.02.2010
**
**************************************************************************/

#ifndef VISITOR_HPP
#define VISITOR_HPP

// Forward Declarations //
//
// NOTE: Do not forget to link
class Acceptor;

class ProjectData;

// RoadSystem //
//
class RoadSystem;

class RSystemElementController;
class RSystemElementFiddleyard;
class RSystemElementPedFiddleyard;

class RSystemElementRoad;

class TypeSection;

class SurfaceSection;

class TrackComponent;
class TrackComposite;
class TrackElement;
class TrackSpiralArcSpiral;
class TrackElementLine;
class TrackElementArc;
class TrackElementSpiral;
class TrackElementPoly3;

class ElevationSection;
class SuperelevationSection;
class CrossfallSection;

class Object;
class Bridge;
class Tunnel;
class Crosswalk;
class Signal;
class Sensor;
class Surface;

class LaneSection;
class Lane;
class LaneWidth;
class LaneRoadMark;
class LaneSpeed;
class LaneHeight;

class FiddleyardSource;
class FiddleyardSink;

class PedFiddleyardSource;
class PedFiddleyardSink;

class JunctionConnection;
class RSystemElementJunction;

// TileSystem //
//
class TileSystem;
class Tile;

// VehicleSystem //
//
class VehicleSystem;
class VehicleGroup;
class RoadVehicle;

// PedestrianSystem //
//
class PedestrianSystem;
class PedestrianGroup;
class Pedestrian;

// ScenerySystem //
//
class ScenerySystem;
class SceneryMap;
class Heightmap;
class SceneryTesselation;

/** Visitor base class.
*
*/
class Visitor
{
public:
    explicit Visitor();
    virtual ~Visitor()
    {
    }

    virtual void visit(Acceptor *)
    {
    } /* not allowed: implement for baseclass! */

    virtual void visit(ProjectData *)
    {
    }

    // RoadSystem //
    //
    virtual void visit(RoadSystem *)
    {
    }

    virtual void visit(RSystemElementController *)
    {
    }
    virtual void visit(RSystemElementFiddleyard *)
    {
    }
    virtual void visit(RSystemElementPedFiddleyard *)
    {
    }

    // Roads //
    //
    virtual void visit(RSystemElementRoad *)
    {
    }

    virtual void visit(TypeSection *)
    {
    }

    virtual void visit(SurfaceSection *)
    {
    }

    virtual void visit(TrackComponent *);
    virtual void visit(TrackComposite *);
    virtual void visit(TrackElement *);
    virtual void visit(TrackElementLine *);
    virtual void visit(TrackElementArc *);
    virtual void visit(TrackElementSpiral *);
    virtual void visit(TrackElementPoly3 *);
    virtual void visit(TrackSpiralArcSpiral *);

    virtual void visit(ElevationSection *)
    {
    }
    virtual void visit(SuperelevationSection *)
    {
    }
    virtual void visit(CrossfallSection *)
    {
    }

    virtual void visit(Object *)
    {
    }
    virtual void visit(Crosswalk *)
    {
    }
    virtual void visit(Signal *)
    {
    }
    virtual void visit(Sensor *)
    {
    }
    virtual void visit(Surface *)
    {
    }
    virtual void visit(Bridge *)
    {
    }
	virtual void visit(Tunnel *)
    {
    }

    virtual void visit(LaneSection *)
    {
    }
    virtual void visit(Lane *)
    {
    }
    virtual void visit(LaneWidth *)
    {
    }
    virtual void visit(LaneRoadMark *)
    {
    }
    virtual void visit(LaneSpeed *)
    {
    }
    virtual void visit(LaneHeight *)
    {
    }

    virtual void visit(FiddleyardSink *)
    {
    }
    virtual void visit(FiddleyardSource *)
    {
    }

    virtual void visit(PedFiddleyardSink *)
    {
    }
    virtual void visit(PedFiddleyardSource *)
    {
    }

    // Junctions //
    //
    virtual void visit(RSystemElementJunction *)
    {
    }
    virtual void visit(JunctionConnection *)
    {
    }

    // VehicleSystem //
    //
    virtual void visit(VehicleSystem *)
    {
    }
    virtual void visit(VehicleGroup *)
    {
    }
    virtual void visit(RoadVehicle *)
    {
    }

    // PedestrianSystem //
    //
    virtual void visit(PedestrianSystem *)
    {
    }
    virtual void visit(PedestrianGroup *)
    {
    }
    virtual void visit(Pedestrian *)
    {
    }

    // ScenerySystem //
    //
    virtual void visit(ScenerySystem *)
    {
    }
    virtual void visit(SceneryMap *);
    virtual void visit(Heightmap *);
    virtual void visit(SceneryTesselation *)
    {
    }

private:
    Visitor(const Visitor &); /* not allowed */
};

#endif // VISITOR_HPP
