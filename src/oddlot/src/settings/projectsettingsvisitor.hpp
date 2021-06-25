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

#ifndef PROJECTSETTINGSVISITOR_HPP
#define PROJECTSETTINGSVISITOR_HPP

#include "src/data/visitor.hpp"

class ProjectSettings;
class SettingsElement;

// Qt //
//
class QWidget;

class ProjectSettingsVisitor : public Visitor
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProjectSettingsVisitor(ProjectSettings *projectSettings);
    virtual ~ProjectSettingsVisitor();

    SettingsElement *getSettingsElement() const
    {
        return settingsElement_;
    }

    virtual void visit(ProjectData *);

    // RoadSystem //
    //
    virtual void visit(RoadSystem *);

    virtual void visit(RSystemElementController *);
    virtual void visit(RSystemElementFiddleyard *);

    // Roads //
    //
    virtual void visit(RSystemElementRoad *);

    virtual void visit(TypeSection *);

    virtual void visit(TrackSpiralArcSpiral *);
    virtual void visit(TrackElement *);
    virtual void visit(TrackElementLine *);
    virtual void visit(TrackElementArc *);
    virtual void visit(TrackElementSpiral *);
    virtual void visit(TrackElementPoly3 *);

    virtual void visit(ElevationSection *);
    virtual void visit(SuperelevationSection *);
    virtual void visit(CrossfallSection *);

    virtual void visit(LaneSection *);
    virtual void visit(Lane *);
    virtual void visit(LaneWidth *);
    virtual void visit(LaneRoadMark *);
    virtual void visit(LaneSpeed *);

    virtual void visit(FiddleyardSink *);
    virtual void visit(FiddleyardSource *);

    // Signals and Objects //
    //
    virtual void visit(Signal *);
    virtual void visit(Object *);
    virtual void visit(Bridge *);
	virtual void visit(Tunnel *);

    // Junctions //
    //
    virtual void visit(RSystemElementJunction *);
    virtual void visit(JunctionConnection *);

	// OpenScenario Elements //
	//
	virtual void visit(OSCElement *);

    // VehicleSystem //
    //
    virtual void visit(VehicleSystem *);
    virtual void visit(VehicleGroup *);
    virtual void visit(RoadVehicle *);

    // ScenerySystem //
    //
    virtual void visit(ScenerySystem *);
    virtual void visit(SceneryMap *);
    virtual void visit(Heightmap *);

private:
    ProjectSettingsVisitor(const ProjectSettingsVisitor &); /* not allowed */
    ProjectSettingsVisitor &operator=(const ProjectSettingsVisitor &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    ProjectSettings *projectSettings_;

    SettingsElement *settingsElement_;
};

#endif // PROJECTSETTINGSVISITOR_HPP
