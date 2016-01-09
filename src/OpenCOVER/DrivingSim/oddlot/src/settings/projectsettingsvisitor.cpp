/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/13/2010
**
**************************************************************************/

#include "projectsettingsvisitor.hpp"

// Data //
//
#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementspiral.hpp"
#include "src/data/roadsystem/track/trackelementpoly3.hpp"
#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"
#include "src/data/roadsystem/sections/signalobject.hpp"

// OpenScenario //
//
#include "src/data/oscsystem/oscelement.hpp"

#include "src/data/scenerysystem/heightmap.hpp"

// Settings //
//
#include "widgets/projectdatasettings.hpp"

#include "widgets/roadsettings.hpp"

#include "widgets/scenerymapsettings.hpp"
#include "widgets/trackcomponentsettings.hpp"

#include "widgets/lanesettings.hpp"
#include "widgets/laneroadmarksettings.hpp"

#include "widgets/signalsettings.hpp"

#include "widgets/objectsettings.hpp"

#include "widgets/bridgesettings.hpp"
#include "widgets/tunnelsettings.hpp"

#include "widgets/junctionsettings.hpp"

#include "widgets/typesectionsettings.hpp"

#include "widgets/elevationsettings.hpp"

#include "widgets/controllersettings.hpp"

#include "widgets/oscobjectsettingsstack.hpp"

ProjectSettingsVisitor::ProjectSettingsVisitor(ProjectSettings *projectSettings)
    : Visitor()
    , projectSettings_(projectSettings)
    , settingsElement_(NULL)
{
}

ProjectSettingsVisitor::~ProjectSettingsVisitor()
{
}

// RoadSystem //
//
void
ProjectSettingsVisitor::visit(ProjectData *element)
{
    settingsElement_ = new ProjectDataSettings(projectSettings_, NULL, element);
}

// RoadSystem //
//
void
ProjectSettingsVisitor::visit(RoadSystem *)
{
}

void
ProjectSettingsVisitor::visit(RSystemElementController *acceptor)
{
    settingsElement_ = new ControllerSettings(projectSettings_, NULL, acceptor);
}

void
ProjectSettingsVisitor::visit(RSystemElementFiddleyard *)
{
}

// Roads //
//
void
ProjectSettingsVisitor::visit(RSystemElementRoad *acceptor)
{
    settingsElement_ = new RoadSettings(projectSettings_, NULL, acceptor);
}

void
ProjectSettingsVisitor::visit(TypeSection *acceptor)
{
    settingsElement_ = new TypeSectionSettings(projectSettings_, NULL, acceptor);
}

void
ProjectSettingsVisitor::visit(TrackSpiralArcSpiral *acceptor)
{
    settingsElement_ = new TrackComponentSettings(projectSettings_, NULL, acceptor);
}

void
ProjectSettingsVisitor::visit(TrackElement *)
{
}

void
ProjectSettingsVisitor::visit(TrackElementLine *acceptor)
{
    settingsElement_ = new TrackComponentSettings(projectSettings_, NULL, acceptor);
}

void
ProjectSettingsVisitor::visit(TrackElementArc *acceptor)
{
    settingsElement_ = new TrackComponentSettings(projectSettings_, NULL, acceptor);
}

void
ProjectSettingsVisitor::visit(TrackElementSpiral *acceptor)
{
    settingsElement_ = new TrackComponentSettings(projectSettings_, NULL, acceptor);
}

void
ProjectSettingsVisitor::visit(TrackElementPoly3 *acceptor)
{
    settingsElement_ = new TrackComponentSettings(projectSettings_, NULL, acceptor);
}

void
ProjectSettingsVisitor::visit(ElevationSection *section)
{
    settingsElement_ = new ElevationSettings(projectSettings_, NULL, section);
}

void
ProjectSettingsVisitor::visit(SuperelevationSection *)
{
}

void
ProjectSettingsVisitor::visit(CrossfallSection *)
{
}

void
ProjectSettingsVisitor::visit(LaneSection *)
{
}

void
ProjectSettingsVisitor::visit(Lane *acceptor)
{
    settingsElement_ = new LaneSettings(projectSettings_, NULL, acceptor);
}

void
ProjectSettingsVisitor::visit(LaneWidth *)
{
}

void
ProjectSettingsVisitor::visit(LaneRoadMark *acceptor)
{
    settingsElement_ = new LaneRoadMarkSettings(projectSettings_, NULL, acceptor);
}

void
ProjectSettingsVisitor::visit(LaneSpeed *)
{
}

void
ProjectSettingsVisitor::visit(FiddleyardSink *)
{
}

void
ProjectSettingsVisitor::visit(FiddleyardSource *)
{
}

// Signals //
//
void
ProjectSettingsVisitor::visit(Signal *acceptor)
{
    settingsElement_ = new SignalSettings(projectSettings_, NULL, acceptor);
}

// Objects //
//
void
ProjectSettingsVisitor::visit(Object *acceptor)
{
    settingsElement_ = new ObjectSettings(projectSettings_, NULL, acceptor);
}

// Bridges //
//
void
ProjectSettingsVisitor::visit(Bridge *acceptor)
{
    settingsElement_ = new BridgeSettings(projectSettings_, NULL, acceptor);
}

// Tunnels //
//
void
ProjectSettingsVisitor::visit(Tunnel *acceptor)
{
    settingsElement_ = new TunnelSettings(projectSettings_, NULL, acceptor);
}

// Junctions //
//
void
ProjectSettingsVisitor::visit(RSystemElementJunction *acceptor)
{
    settingsElement_ = new JunctionSettings(projectSettings_, NULL, acceptor);
}

void
ProjectSettingsVisitor::visit(JunctionConnection *)
{
}

void
ProjectSettingsVisitor::visit(OSCElement *acceptor)
{
    settingsElement_ = new OSCObjectSettingsStack(projectSettings_, NULL, acceptor);
}

// VehicleSystem //
//
void
ProjectSettingsVisitor::visit(VehicleSystem *)
{
}

void
ProjectSettingsVisitor::visit(VehicleGroup *)
{
}

void
ProjectSettingsVisitor::visit(RoadVehicle *)
{
}

// ScenerySystem //
//
void
ProjectSettingsVisitor::visit(ScenerySystem *)
{
}

void
ProjectSettingsVisitor::visit(SceneryMap *map)
{
    settingsElement_ = new SceneryMapSettings(projectSettings_, NULL, map);
}

void
ProjectSettingsVisitor::visit(Heightmap *map)
{
    settingsElement_ = new SceneryMapSettings(projectSettings_, NULL, map);
}
