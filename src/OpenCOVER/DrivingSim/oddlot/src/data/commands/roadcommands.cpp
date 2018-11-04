/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   21.05.2010
**
**************************************************************************/

#include "roadcommands.hpp"
#include "trackcommands.hpp"
#include <QMessageBox>

#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/rsystemelementfiddleyard.hpp"
#include "src/data/roadsystem/junctionconnection.hpp"
#include "src/data/roadsystem/roadlink.hpp"

#include "src/data/roadsystem/sections/typesection.hpp"
#include "src/data/roadsystem/track/trackcomponent.hpp"
#include "src/data/roadsystem/track/trackcomposite.hpp"
#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"
#include "src/data/roadsystem/sections/elevationsection.hpp"
#include "src/data/roadsystem/sections/superelevationsection.hpp"
#include "src/data/roadsystem/sections/crossfallsection.hpp"
#include "src/data/roadsystem//sections/shapesection.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"

#include "src/data/roadsystem/sections/lane.hpp"
#include "src/data/roadsystem/sections/laneroadmark.hpp"
#include "src/data/roadsystem/sections/lanespeed.hpp"
#include "src/data/roadsystem/sections/lanewidth.hpp"

#include "math.h"

//#########################//
// AppendRoadPrototypeCommand //
//#########################//

AppendRoadPrototypeCommand::AppendRoadPrototypeCommand(RSystemElementRoad *road, RSystemElementRoad *prototype, bool atStart, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , prototype_(prototype)
    , atStart_(atStart)
{
    // Check for validity //
    //
    if (!road_ || !prototype_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Append (invalid!)"));
        return;
    }

    prototypeLength_ = prototype_->getLength();

    oldTypeSections_ = road->getTypeSections();
    oldElevationSections_ = road->getElevationSections();
    oldSuperelevationSections_ = road->getSuperelevationSections();
    oldCrossfallSections_ = road->getCrossfallSections();
    oldLaneSections_ = road->getLaneSections();
	oldShapeSections_ = road->getShapeSections();

    if (atStart_)
    {
        // Append at start //
        //
        //
        prototypeLength_ = prototype_->getLength();

        // Track Components //
        //
        QTransform roadTrafo = road_->getGlobalTransform(0.0);
        QTransform protoTrafo = prototype_->getGlobalTransform(prototypeLength_);
        double roadHeading = road_->getGlobalHeading(0.0);
        double protoHeading = prototype_->getGlobalHeading(prototypeLength_);

        foreach (TrackComponent *track, prototype_->getTrackSections())
        {
            // Clone //
            //
            TrackComponent *clone = track->getClone();

            // Set Parameters //
            //
            //clone->setSStart(clone->getSStart() - prototypeLength); // note: s will be negative until rebuildTrackComponentList()
            clone->setGlobalTranslation(roadTrafo.map(protoTrafo.inverted().map(clone->getGlobalPoint(clone->getSStart()))));
            clone->setGlobalRotation(clone->getGlobalHeading(clone->getSStart()) - protoHeading + roadHeading);

            // Save in list //
            //
            trackSections_.insert(clone->getSStart(), clone);
        }
    }
    else
    {
        // Append at back //
        //

        // Delta s //
        //
        double oldLength = road_->getLength();

        // Track Components //
        //
        QTransform roadTrafo = road_->getGlobalTransform(oldLength);
        QTransform protoTrafo = prototype_->getGlobalTransform(0.0);
        double roadHeading = road_->getGlobalHeading(oldLength);
        double protoHeading = prototype_->getGlobalHeading(0.0);

        foreach (TrackComponent *track, prototype_->getTrackSections())
        {
            // Clone //
            //
            TrackComponent *clone = track->getClone();

            // Set Parameters //
            //
            // Transform start points back to inertial system and then to the new one
            clone->setSStart(clone->getSStart() + oldLength);
            clone->setGlobalTranslation(roadTrafo.map(protoTrafo.inverted().map(clone->getGlobalPoint(clone->getSStart()))));
            clone->setGlobalRotation(clone->getGlobalHeading(clone->getSStart()) - protoHeading + roadHeading);

            // Save in list //
            //
            trackSections_.insert(clone->getSStart(), clone);
        }
    }

    double deltaS = road_->getLength();

    // TypeSections //
    //
    foreach (TypeSection *section, prototype_->getTypeSections())
    {
        TypeSection *clone = section->getClone();
        if (!atStart_)
        {
            clone->setSStart(clone->getSStart() + deltaS);
        }
        newTypeSections_.insert(clone->getSStart(), clone);
    }

    // ElevationSections //
    //
    foreach (ElevationSection *section, prototype_->getElevationSections())
    {
        ElevationSection *clone = section->getClone();
        if (!atStart_)
        {
            clone->setSStart(clone->getSStart() + deltaS);
        }
        newElevationSections_.insert(clone->getSStart(), clone);
    }

    // SuperelevationSections //
    //
    foreach (SuperelevationSection *section, prototype_->getSuperelevationSections())
    {
        SuperelevationSection *clone = section->getClone();
        if (!atStart_)
        {
            clone->setSStart(clone->getSStart() + deltaS);
        }
        newSuperelevationSections_.insert(clone->getSStart(), clone);
    }

    // CrossfallSections //
    //
    foreach (CrossfallSection *section, prototype_->getCrossfallSections())
    {
        CrossfallSection *clone = section->getClone();
        if (!atStart_)
        {
            clone->setSStart(clone->getSStart() + deltaS);
        }
        newCrossfallSections_.insert(clone->getSStart(), clone);
    }

    // LaneSections //
    //
    foreach (LaneSection *section, prototype_->getLaneSections())
    {
        LaneSection *clone = section->getClone();
        if (!atStart_)
        {
            clone->setSStart(clone->getSStart() + deltaS);
        }
        newLaneSections_.insert(clone->getSStart(), clone);
    }

	// ShapeSections //
	//
	foreach(ShapeSection *section, prototype_->getShapeSections())
	{
		ShapeSection *clone = section->getClone();
		if (!atStart_)
		{
			clone->setSStart(clone->getSStart() + deltaS);
		}
		newShapeSections_.insert(clone->getSStart(), clone);
	}

    // Done //
    //
    setValid();
    setText(QObject::tr("Append"));
}

/*! \brief .
*
*/
AppendRoadPrototypeCommand::~AppendRoadPrototypeCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        foreach (TrackComponent *section, trackSections_)
        {
            delete section; // delete sections
        }
        foreach (TypeSection *section, newTypeSections_)
        {
            delete section; // delete sections
        }
        foreach (ElevationSection *section, newElevationSections_)
        {
            delete section; // delete sections
        }
        foreach (SuperelevationSection *section, newSuperelevationSections_)
        {
            delete section; // delete sections
        }
        foreach (CrossfallSection *section, newCrossfallSections_)
        {
            delete section; // delete sections
        }
        foreach (LaneSection *section, newLaneSections_)
        {
            delete section; // delete sections
        }
		foreach(ShapeSection *section, newShapeSections_)
		{
			delete section; // delete sections
		}
    }
    else
    {
        // nothing to be done (tracks are now owned by the road)
    }
}

/*! \brief .
*
*/
void
AppendRoadPrototypeCommand::redo()
{
    if (atStart_)
    {
        foreach (TrackComponent *track, trackSections_)
        {
            track->setSStart(track->getSStart() - prototypeLength_); // note: s will be negative until rebuildTrackComponentList()
            road_->addTrackComponent(track);
        }
        road_->rebuildTrackComponentList();

        // Move/Add RoadSections //
        //
        if (!newTypeSections_.isEmpty())
        {
            // Move old sections //
            //
            QMap<double, TypeSection *> newSections = newTypeSections_;
            foreach (TypeSection *section, oldTypeSections_)
            {
                section->setSStart(section->getSStart() + prototypeLength_);
                newSections.insert(section->getSStart(), section);
            }
            road_->setTypeSections(newSections);
        }
        if (!newElevationSections_.isEmpty())
        {
            // Move old sections //
            //
            QMap<double, ElevationSection *> newSections = newElevationSections_;
            foreach (ElevationSection *section, oldElevationSections_)
            {
                section->setSStart(section->getSStart() + prototypeLength_);
                newSections.insert(section->getSStart(), section);
            }
            road_->setElevationSections(newSections);
        }
        if (!newSuperelevationSections_.isEmpty())
        {
            // Move old sections //
            //
            QMap<double, SuperelevationSection *> newSections = newSuperelevationSections_;
            foreach (SuperelevationSection *section, oldSuperelevationSections_)
            {
                section->setSStart(section->getSStart() + prototypeLength_);
                newSections.insert(section->getSStart(), section);
            }
            road_->setSuperelevationSections(newSections);
        }
        if (!newCrossfallSections_.isEmpty())
        {
            // Move old sections //
            //
            QMap<double, CrossfallSection *> newSections = newCrossfallSections_;
            foreach (CrossfallSection *section, oldCrossfallSections_)
            {
                section->setSStart(section->getSStart() + prototypeLength_);
                newSections.insert(section->getSStart(), section);
            }
            road_->setCrossfallSections(newSections);
        }
        if (!newLaneSections_.isEmpty())
        {
            // Move old sections //
            //
            QMap<double, LaneSection *> newSections = newLaneSections_;
            foreach (LaneSection *section, oldLaneSections_)
            {
                section->setSStart(section->getSStart() + prototypeLength_);
                newSections.insert(section->getSStart(), section);
            }
            road_->setLaneSections(newSections);
        }
		if (!newShapeSections_.isEmpty())
		{
			// Move old sections //
			//
			QMap<double, ShapeSection *> newSections = newShapeSections_;
			foreach(ShapeSection *section, oldShapeSections_)
			{
				section->setSStart(section->getSStart() + prototypeLength_);
				newSections.insert(section->getSStart(), section);
			}
			road_->setShapeSections(newSections);
		}
    }
    else
    {
        foreach (TrackComponent *track, trackSections_)
        {
            road_->addTrackComponent(track);
        }
        foreach (TypeSection *section, newTypeSections_)
        {
            road_->addTypeSection(section);
        }
        foreach (ElevationSection *section, newElevationSections_)
        {
            road_->addElevationSection(section);
        }
        foreach (SuperelevationSection *section, newSuperelevationSections_)
        {
            road_->addSuperelevationSection(section);
        }
        foreach (CrossfallSection *section, newCrossfallSections_)
        {
            road_->addCrossfallSection(section);
        }
        foreach (LaneSection *section, newLaneSections_)
        {
            road_->addLaneSection(section);
        }
		foreach(ShapeSection *section, newShapeSections_)
		{
			road_->addShapeSection(section);
		}
    }

    setRedone();
}

/*! \brief
*
*/
void
AppendRoadPrototypeCommand::undo()
{
    if (atStart_)
    {
        // TrackComponents //
        //
        foreach (TrackComponent *track, trackSections_)
        {
            road_->delTrackComponent(track);
        }
        road_->rebuildTrackComponentList();

        // TypeSections //
        //
        QMap<double, TypeSection *> newTypeSections;
        foreach (TypeSection *section, oldTypeSections_)
        {
            section->setSStart(section->getSStart() - prototypeLength_);
            newTypeSections.insert(section->getSStart(), section);
        }
        road_->setTypeSections(newTypeSections);

        // ElevationSections //
        //
        QMap<double, ElevationSection *> newElevationSections;
        foreach (ElevationSection *section, oldElevationSections_)
        {
            section->setSStart(section->getSStart() - prototypeLength_);
            newElevationSections.insert(section->getSStart(), section);
        }
        road_->setElevationSections(newElevationSections);

        // SuperelevationSections //
        //
        QMap<double, SuperelevationSection *> newSuperelevationSections;
        foreach (SuperelevationSection *section, oldSuperelevationSections_)
        {
            section->setSStart(section->getSStart() - prototypeLength_);
            newSuperelevationSections.insert(section->getSStart(), section);
        }
        road_->setSuperelevationSections(newSuperelevationSections);

        // CrossfallSections //
        //
        QMap<double, CrossfallSection *> newCrossfallSections;
        foreach (CrossfallSection *section, oldCrossfallSections_)
        {
            section->setSStart(section->getSStart() - prototypeLength_);
            newCrossfallSections.insert(section->getSStart(), section);
        }
        road_->setCrossfallSections(newCrossfallSections);

        // LaneSections //
        //
        QMap<double, LaneSection *> newLaneSections;
        foreach (LaneSection *section, oldLaneSections_)
        {
            section->setSStart(section->getSStart() - prototypeLength_);
            newLaneSections.insert(section->getSStart(), section);
        }
        road_->setLaneSections(newLaneSections);

		// ShapeSections //
		//
		QMap<double, ShapeSection *> newShapeSections;
		foreach(ShapeSection *section, oldShapeSections_)
		{
			section->setSStart(section->getSStart() - prototypeLength_);
			newShapeSections.insert(section->getSStart(), section);
		}
		road_->setShapeSections(newShapeSections);
    }
    else
    {
        foreach (TrackComponent *track, trackSections_)
        {
            road_->delTrackComponent(track);
        }
        foreach (TypeSection *section, newTypeSections_)
        {
            road_->delTypeSection(section);
        }
        foreach (ElevationSection *section, newElevationSections_)
        {
            road_->delElevationSection(section);
        }
        foreach (SuperelevationSection *section, newSuperelevationSections_)
        {
            road_->delSuperelevationSection(section);
        }
        foreach (CrossfallSection *section, newCrossfallSections_)
        {
            road_->delCrossfallSection(section);
        }
        foreach (LaneSection *section, newLaneSections_)
        {
            road_->delLaneSection(section);
        }
		foreach(ShapeSection *section, newShapeSections_)
		{
			road_->delShapeSection(section);
		}
    }

    setUndone();
}

//#########################//
// MergeRoadsCommand //
//#########################//

MergeRoadsCommand::MergeRoadsCommand(RSystemElementRoad *road1, RSystemElementRoad *road2, bool firstSt, bool secondSt, DataCommand *parent)
    : DataCommand(parent)
    , road1_(road1)
    , road2_(road2)
    , firstStart(firstSt)
    , secondStart(secondSt)
{
    // Check for validity //
    //
    if (!road1_ || !road2_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Append (invalid!)"));
        return;
    }
    
    if (firstStart==true && secondStart == false)
    {
        firstStart=false;
        secondStart=true;
        RSystemElementRoad *tmpRoad = road1_;
        road1_=road2_;
        road2_=tmpRoad;
    }
    roadSystem_ = road2_->getRoadSystem();

    secondRoadLength_ = road2->getLength();

    oldTypeSections_ = road1->getTypeSections();
    oldElevationSections_ = road1->getElevationSections();
    oldSuperelevationSections_ = road1->getSuperelevationSections();
    oldCrossfallSections_ = road1->getCrossfallSections();
    oldLaneSections_ = road1->getLaneSections();
	oldShapeSections_ = road1->getShapeSections();
    

    QTransform road1Trafo, road2Trafo;

    double deltaS = road1_->getLength();
    QPointF dPos;
    if (firstStart==true && secondStart == true)
    {
        //TODO: reverse first road
        setInvalid(); // Invalid
        setText(QObject::tr("Merge invalid! Can only connect start to end"));
        return;
    }
    if (firstStart==false && secondStart == false)
    {
        //TODO: reverse second road
        setInvalid(); // Invalid
        setText(QObject::tr("Merge invalid! Can only connect start to end"));
        return;
    }
    if (firstStart==false && secondStart == true)
    {
        dPos = road1_->getGlobalPoint(road1_->getLength()) - road2_->getGlobalPoint(0.0);
    }
    else // this should not happen
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Append (invalid!)"));
        return;
    }

    // TrackSections //
    //
    foreach (TrackComponent *track, road2_->getTrackSections())
    {
        TrackComponent *clone = track->getClone();
        clone->setLocalTranslation(clone->getGlobalPoint(clone->getSStart()) + dPos);
        clone->setSStart(clone->getSStart() + deltaS);
        trackSections_.insert(clone->getSStart(), clone);
    }

    // TypeSections //
    //
    foreach (TypeSection *section, road2_->getTypeSections())
    {
        TypeSection *clone = section->getClone();
        clone->setSStart(clone->getSStart() + deltaS);
        newTypeSections_.insert(clone->getSStart(), clone);
    }

    // ElevationSections //
    //
    foreach (ElevationSection *section, road2_->getElevationSections())
    {
        ElevationSection *clone = section->getClone();
        clone->setSStart(clone->getSStart() + deltaS);
        newElevationSections_.insert(clone->getSStart(), clone);
    }

    // SuperelevationSections //
    //
    foreach (SuperelevationSection *section, road2_->getSuperelevationSections())
    {
        SuperelevationSection *clone = section->getClone();
        clone->setSStart(clone->getSStart() + deltaS);
        newSuperelevationSections_.insert(clone->getSStart(), clone);
    }

    // CrossfallSections //
    //
    foreach (CrossfallSection *section, road2_->getCrossfallSections())
    {
        CrossfallSection *clone = section->getClone();
        clone->setSStart(clone->getSStart() + deltaS);
        newCrossfallSections_.insert(clone->getSStart(), clone);
    }

    // LaneSections //
    //
    foreach (LaneSection *section, road2_->getLaneSections())
    {
        LaneSection *clone = section->getClone();
        clone->setSStart(clone->getSStart() + deltaS);
        newLaneSections_.insert(clone->getSStart(), clone);
    }

	// ShapeSections //
	//
	foreach(ShapeSection *section, road2_->getShapeSections())
	{
		ShapeSection *clone = section->getClone();
		clone->setSStart(clone->getSStart() + deltaS);
		newShapeSections_.insert(clone->getSStart(), clone);
	}

    // Done //
    //
    setValid();
    setText(QObject::tr("Merge"));
}

/*! \brief .
*
*/
MergeRoadsCommand::~MergeRoadsCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        foreach (TrackComponent *section, trackSections_)
        {
            delete section; // delete sections
        }
        foreach (TypeSection *section, newTypeSections_)
        {
            delete section; // delete sections
        }
        foreach (ElevationSection *section, newElevationSections_)
        {
            delete section; // delete sections
        }
        foreach (SuperelevationSection *section, newSuperelevationSections_)
        {
            delete section; // delete sections
        }
        foreach (CrossfallSection *section, newCrossfallSections_)
        {
            delete section; // delete sections
        }
        foreach (LaneSection *section, newLaneSections_)
        {
            delete section; // delete sections
        }
		foreach(ShapeSection *section, newShapeSections_)
		{
			delete section; // delete sections
		}
    }
    else
    {
        // nothing to be done (tracks are now owned by the road)
    }
}

/*! \brief .
*
*/
void
MergeRoadsCommand::redo()
{
    roadSystem_->delRoad(road2_);

    foreach (TrackComponent *track, trackSections_)
    {
        road1_->addTrackComponent(track);
    }
    foreach (TypeSection *section, newTypeSections_)
    {
        road1_->addTypeSection(section);
    }
    foreach (ElevationSection *section, newElevationSections_)
    {
        road1_->addElevationSection(section);
    }
    foreach (SuperelevationSection *section, newSuperelevationSections_)
    {
        road1_->addSuperelevationSection(section);
    }
    foreach (CrossfallSection *section, newCrossfallSections_)
    {
        road1_->addCrossfallSection(section);
    }
    foreach (LaneSection *section, newLaneSections_)
    {
        road1_->addLaneSection(section);
    }
	foreach(ShapeSection *section, newShapeSections_)
	{
		road1_->addShapeSection(section);
	}

    setRedone();
}

/*! \brief
*
*/
void
MergeRoadsCommand::undo()
{
    
    foreach (TrackComponent *track, trackSections_)
    {
        road1_->delTrackComponent(track);
    }
    foreach (TypeSection *section, newTypeSections_)
    {
        road1_->delTypeSection(section);
    }
    foreach (ElevationSection *section, newElevationSections_)
    {
        road1_->delElevationSection(section);
    }
    foreach (SuperelevationSection *section, newSuperelevationSections_)
    {
        road1_->delSuperelevationSection(section);
    }
    foreach (CrossfallSection *section, newCrossfallSections_)
    {
        road1_->delCrossfallSection(section);
    }
    foreach (LaneSection *section, newLaneSections_)
    {
        road1_->delLaneSection(section);
    }
	foreach(ShapeSection *section, newShapeSections_)
	{
		road1_->delShapeSection(section);
	}
    roadSystem_->addRoad(road2_);

    setUndone();
}

//#########################//
// SnapRoadsCommand //
//#########################//

SnapRoadsCommand::SnapRoadsCommand(RSystemElementRoad *road1, RSystemElementRoad *road2, short int pos, DataCommand *parent)
    : DataCommand(parent)
    , road1_(road1)
    , road2_(road2)
    , pos_(pos)
{
    // Check for validity //
    //
    if (!road1_ || !road2_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Snap (invalid!)"));
        return;
    }

    bool mergeRoads = false;
    switch (pos) // always change track of road2 and append road2 to road1
    {
    case SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadStart:
    {
        track_ = road2_->getTrackComponent(0.0); // roads are not merged
        newPoint_ = road1_->getGlobalPoint(0.0);
        newHeading_ = road1_->getGlobalHeading(0.0) + 180;
        oldPoint_ = road2_->getGlobalPoint(0.0);
        oldHeading_ = road2->getGlobalHeading(0.0);
    }
    break;
    case SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart:
    {
        track_ = road2_->getTrackComponent(0.0);
        newPoint_ = road1_->getGlobalPoint(road1_->getLength());
        newHeading_ = road1_->getGlobalHeading(road1_->getLength());
        mergeRoads = true;
    }
    break;
    case SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadEnd:
    {
        track_ = road2_->getTrackComponent(road2_->getLength()); // roads are not merged
        newPoint_ = road1_->getGlobalPoint(road1_->getLength());
        newHeading_ = road1_->getGlobalHeading(road1_->getLength()) + 180;
        oldPoint_ = road2_->getGlobalPoint(road2_->getLength());
        oldHeading_ = road2->getGlobalHeading(road2_->getLength());
    }
    break;
    case SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadEnd:
    {
        track_ = road2_->getTrackComponent(road2_->getLength());
        newPoint_ = road1_->getGlobalPoint(0.0);
        newHeading_ = road1_->getGlobalHeading(0.0);
        mergeRoads = true;
    }
    }

    if (mergeRoads)
    {
        roadSystem_ = road2_->getRoadSystem();

        oldTypeSections_ = road1->getTypeSections();
        oldElevationSections_ = road1->getElevationSections();
        oldSuperelevationSections_ = road1->getSuperelevationSections();
        oldCrossfallSections_ = road1->getCrossfallSections();
        oldLaneSections_ = road1->getLaneSections();
		oldShapeSections_ = road1->getShapeSections();

        // TrackSections //
        //
        foreach (TrackComponent *track, road2_->getTrackSections())
        {
            TrackComponent *clone = track->getClone();
            if (track == track_)
            {
                track_ = clone;
            }
            if (pos & (SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart))
            {
                clone->setSStart(road1->getLength() + clone->getSStart());
            }

            trackSections_.insert(clone->getSStart(), clone);
        }

        // TypeSections //
        //
        foreach (TypeSection *section, road2_->getTypeSections())
        {
            TypeSection *clone = section->getClone();
            if (pos & (SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart))
            {
                clone->setSStart(road1->getLength() + clone->getSStart());
            }
            newTypeSections_.insert(clone->getSStart(), clone);
        }

        // ElevationSections //
        //
        foreach (ElevationSection *section, road2_->getElevationSections())
        {
            ElevationSection *clone = section->getClone();
            if (pos & (SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart))
            {
                clone->setSStart(road1->getLength() + clone->getSStart());
            }
            newElevationSections_.insert(clone->getSStart(), clone);
        }

        // SuperelevationSections //
        //
        foreach (SuperelevationSection *section, road2_->getSuperelevationSections())
        {
            SuperelevationSection *clone = section->getClone();
            if (pos & (SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart))
            {
                clone->setSStart(road1->getLength() + clone->getSStart());
            }
            newSuperelevationSections_.insert(clone->getSStart(), clone);
        }

        // CrossfallSections //
        //
        foreach (CrossfallSection *section, road2_->getCrossfallSections())
        {
            CrossfallSection *clone = section->getClone();
            if (pos & (SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart))
            {
                clone->setSStart(road1->getLength() + clone->getSStart());
            }
            newCrossfallSections_.insert(clone->getSStart(), clone);
        }

        // LaneSections //
        //
        foreach (LaneSection *section, road2_->getLaneSections())
        {
            LaneSection *clone = section->getClone();
            if (pos & (SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart))
            {
                clone->setSStart(road1->getLength() + clone->getSStart());
            }
            newLaneSections_.insert(clone->getSStart(), clone);
        }

		// ShapeSections //
		//
		foreach(ShapeSection *section, road2_->getShapeSections())
		{
			ShapeSection *clone = section->getClone();
			if (pos & (SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart))
			{
				clone->setSStart(road1->getLength() + clone->getSStart());
			}
			newShapeSections_.insert(clone->getSStart(), clone);
		}
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Snap"));
}

/*! \brief .
*
*/
SnapRoadsCommand::~SnapRoadsCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        if ((pos_ == (SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadEnd)) || (pos_ == (SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart)))
        {
            foreach (TrackComponent *section, trackSections_)
            {
                delete section; // delete sections
            }
            foreach (TypeSection *section, newTypeSections_)
            {
                delete section; // delete sections
            }
            foreach (ElevationSection *section, newElevationSections_)
            {
                delete section; // delete sections
            }
            foreach (SuperelevationSection *section, newSuperelevationSections_)
            {
                delete section; // delete sections
            }
            foreach (CrossfallSection *section, newCrossfallSections_)
            {
                delete section; // delete sections
            }
            foreach (LaneSection *section, newLaneSections_)
            {
                delete section; // delete sections
            }
			foreach(ShapeSection *section, newShapeSections_)
			{
				delete section; // delete sections
			}
        }
    }
    else
    {
        // nothing to be done (tracks are now owned by the road)
    }
}

/*! \brief .
*
*/
void
SnapRoadsCommand::redo()
{
    switch (pos_)
    {
    case SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadStart:
    {
        track_->setGlobalStartPoint(newPoint_);
        track_->setGlobalStartHeading(newHeading_);
    }
    break;
    case SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart:
    {
        track_->setGlobalStartPoint(newPoint_);
        track_->setGlobalStartHeading(newHeading_);
    }
    break;
    case SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadEnd:
    {
        track_->setGlobalEndPoint(newPoint_);
        track_->setGlobalEndHeading(newHeading_);
    }
    break;
    case SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadEnd:
    {
        track_->setGlobalEndPoint(newPoint_);
        track_->setGlobalEndHeading(newHeading_);
    }
    break;
    }

    if (pos_ == (SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadEnd))
    {
        QMap<double, TrackComponent *> newTracks = trackSections_;
        foreach (TrackComponent *track, road1_->getTrackSections())
        {
            track->setSStart(track->getSStart() + road2_->getLength());
            newTracks.insert(track->getSStart(), track);
        }

        road1_->setTrackComponents(newTracks);

        // Move/Add RoadSections //
        //
        if (!newTypeSections_.isEmpty())
        {
            // Move old sections //
            //
            QMap<double, TypeSection *> newSections = newTypeSections_;
            foreach (TypeSection *section, oldTypeSections_)
            {
                section->setSStart(section->getSStart() + road2_->getLength());
                newSections.insert(section->getSStart(), section);
            }
            road1_->setTypeSections(newSections);
        }
        if (!newElevationSections_.isEmpty())
        {
            // Move old sections //
            //
            QMap<double, ElevationSection *> newSections = newElevationSections_;
            foreach (ElevationSection *section, oldElevationSections_)
            {
                section->setSStart(section->getSStart() + road2_->getLength());
                newSections.insert(section->getSStart(), section);
            }
            road1_->setElevationSections(newSections);
        }
        if (!newSuperelevationSections_.isEmpty())
        {
            // Move old sections //
            //
            QMap<double, SuperelevationSection *> newSections = newSuperelevationSections_;
            foreach (SuperelevationSection *section, oldSuperelevationSections_)
            {
                section->setSStart(section->getSStart() + road2_->getLength());
                newSections.insert(section->getSStart(), section);
            }
            road1_->setSuperelevationSections(newSections);
        }
        if (!newCrossfallSections_.isEmpty())
        {
            // Move old sections //
            //
            QMap<double, CrossfallSection *> newSections = newCrossfallSections_;
            foreach (CrossfallSection *section, oldCrossfallSections_)
            {
                section->setSStart(section->getSStart() + road2_->getLength());
                newSections.insert(section->getSStart(), section);
            }
            road1_->setCrossfallSections(newSections);
        }
        if (!newLaneSections_.isEmpty())
        {
            // Move old sections //
            //
            QMap<double, LaneSection *> newSections = newLaneSections_;
            foreach (LaneSection *section, oldLaneSections_)
            {
                section->setSStart(section->getSStart() + road2_->getLength());
                newSections.insert(section->getSStart(), section);
            }
            road1_->setLaneSections(newSections);
        }
		if (!newShapeSections_.isEmpty())
		{
			// Move old sections //
			//
			QMap<double, ShapeSection *> newSections = newShapeSections_;
			foreach(ShapeSection *section, oldShapeSections_)
			{
				section->setSStart(section->getSStart() + road2_->getLength());
				newSections.insert(section->getSStart(), section);
			}
			road1_->setShapeSections(newSections);
		}

        roadSystem_->delRoad(road2_);
    }
    else if (pos_ == (SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart))
    {
        foreach (TrackComponent *track, trackSections_)
        {
            road1_->addTrackComponent(track);
        }
        foreach (TypeSection *section, newTypeSections_)
        {
            road1_->addTypeSection(section);
        }
        foreach (ElevationSection *section, newElevationSections_)
        {
            road1_->addElevationSection(section);
        }
        foreach (SuperelevationSection *section, newSuperelevationSections_)
        {
            road1_->addSuperelevationSection(section);
        }
        foreach (CrossfallSection *section, newCrossfallSections_)
        {
            road1_->addCrossfallSection(section);
        }
        foreach (LaneSection *section, newLaneSections_)
        {
            road1_->addLaneSection(section);
        }
		foreach(ShapeSection *section, newShapeSections_)
		{
			road1_->addShapeSection(section);
		}

        roadSystem_->delRoad(road2_);
    }

    setRedone();
}

/*! \brief
*
*/
void
SnapRoadsCommand::undo()
{
    switch (pos_)
    {
    case SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadEnd:
    {

        // TrackComponents //
        //
        foreach (TrackComponent *track, trackSections_)
        {
            road1_->delTrackComponent(track);
        }
        road1_->rebuildTrackComponentList();

        // TypeSections //
        //
        QMap<double, TypeSection *> newTypeSections;
        foreach (TypeSection *section, oldTypeSections_)
        {
            section->setSStart(section->getSStart() - road2_->getLength());
            newTypeSections.insert(section->getSStart(), section);
        }
        road1_->setTypeSections(newTypeSections);

        // ElevationSections //
        //
        QMap<double, ElevationSection *> newElevationSections;
        foreach (ElevationSection *section, oldElevationSections_)
        {
            section->setSStart(section->getSStart() - road2_->getLength());
            newElevationSections.insert(section->getSStart(), section);
        }
        road1_->setElevationSections(newElevationSections);

        // SuperelevationSections //
        //
        QMap<double, SuperelevationSection *> newSuperelevationSections;
        foreach (SuperelevationSection *section, oldSuperelevationSections_)
        {
            section->setSStart(section->getSStart() - road2_->getLength());
            newSuperelevationSections.insert(section->getSStart(), section);
        }
        road1_->setSuperelevationSections(newSuperelevationSections);

        // CrossfallSections //
        //
        QMap<double, CrossfallSection *> newCrossfallSections;
        foreach (CrossfallSection *section, oldCrossfallSections_)
        {
            section->setSStart(section->getSStart() - road2_->getLength());
            newCrossfallSections.insert(section->getSStart(), section);
        }
        road1_->setCrossfallSections(newCrossfallSections);

        // LaneSections //
        //
        QMap<double, LaneSection *> newLaneSections;
        foreach (LaneSection *section, oldLaneSections_)
        {
            section->setSStart(section->getSStart() - road2_->getLength());
            newLaneSections.insert(section->getSStart(), section);
        }
        road1_->setLaneSections(newLaneSections);

        roadSystem_->addRoad(road2_);

		// ShapeSections //
		//
		QMap<double, ShapeSection *> newShapeSections;
		foreach(ShapeSection *section, oldShapeSections_)
		{
			section->setSStart(section->getSStart() - road2_->getLength());
			newShapeSections.insert(section->getSStart(), section);
		}
		road1_->setShapeSections(newShapeSections);
    }
    break;
    case SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadStart:
    {
        foreach (TrackComponent *track, trackSections_)
        {
            road1_->delTrackComponent(track);
        }
        road1_->rebuildTrackComponentList();

        foreach (TypeSection *section, newTypeSections_)
        {
            road1_->delTypeSection(section);
        }
        foreach (ElevationSection *section, newElevationSections_)
        {
            road1_->delElevationSection(section);
        }
        foreach (SuperelevationSection *section, newSuperelevationSections_)
        {
            road1_->delSuperelevationSection(section);
        }
        foreach (CrossfallSection *section, newCrossfallSections_)
        {
            road1_->delCrossfallSection(section);
        }
        foreach (LaneSection *section, newLaneSections_)
        {
            road1_->delLaneSection(section);
        }
		foreach(ShapeSection *section, newShapeSections_)
		{
			road1_->delShapeSection(section);
		}

        roadSystem_->addRoad(road2_);
    }
    break;
    case SetRoadLinkRoadsCommand::FirstRoadStart | SetRoadLinkRoadsCommand::SecondRoadStart:
    {
        track_->setGlobalStartPoint(oldPoint_);
        track_->setGlobalStartHeading(oldHeading_);
    }
    break;
    case SetRoadLinkRoadsCommand::FirstRoadEnd | SetRoadLinkRoadsCommand::SecondRoadEnd:
    {
        track_->setGlobalEndPoint(oldPoint_);
        track_->setGlobalEndHeading(oldHeading_);
    }
    break;
    }

    setUndone();
}

//#########################//
// ChangeLanePrototypeCommand //
//#########################//

ChangeLanePrototypeCommand::ChangeLanePrototypeCommand(RSystemElementRoad *road, RSystemElementRoad *prototype, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , prototype_(prototype)
{

    // Check for validity //
    //
    if (!road_ || !prototype_ || prototype_->getLaneSections().empty())
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Change (invalid!)"));
        return;
    }

    oldLaneSections_ = road_->getLaneSections();

    // Delete old LaneSections //
    //
    foreach (LaneSection *laneSection, oldLaneSections_)
    {
        road_->delLaneSection(laneSection);
    }

    // New LaneSections //
    //
    if (road_->getLaneSections().empty())
    {

        foreach (LaneSection *section, prototype_->getLaneSections())
        {
            road_->addLaneSection(section->getClone());
        }
    }
    newLaneSections_ = road_->getLaneSections();

    // Done //
    //
    setValid();
    setText(QObject::tr("Change"));
}

/*! \brief .
*
*/
ChangeLanePrototypeCommand::~ChangeLanePrototypeCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        foreach (LaneSection *section, newLaneSections_)
        {
            delete section; // delete sections
        }
    }
    else
    {
        // nothing to be done (tracks are now owned by the road)
    }
}

/*! \brief .
*
*/
void
ChangeLanePrototypeCommand::redo()
{

    if (!newLaneSections_.isEmpty())
    {
        // Delete old LaneSections //
        //
        foreach (LaneSection *laneSection, oldLaneSections_)
        {
            road_->delLaneSection(laneSection);
        }

        // New LaneSections //
        //

        foreach (LaneSection *section, newLaneSections_)
        {
            road_->addLaneSection(section);
        }
    }

    setRedone();
}

/*! \brief .
*
*/
void
ChangeLanePrototypeCommand::undo()
{
    QMap<double, LaneSection *> newLaneSections;

    // Delete old LaneSections //
    //
    foreach (LaneSection *laneSection, newLaneSections_)
    {
        road_->delLaneSection(laneSection);
    }

    // New LaneSections //
    //

    foreach (LaneSection *section, oldLaneSections_)
    {
        road_->addLaneSection(section);
    }

    setUndone();
}

//#########################//
// ChangeRoadTypePrototypeCommand //
//#########################//

ChangeRoadTypePrototypeCommand::ChangeRoadTypePrototypeCommand(RSystemElementRoad *road, RSystemElementRoad *prototype, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , prototype_(prototype)
{

    // Check for validity //
    //
    if (!road_ || !prototype_ || prototype_->getTypeSections().empty())
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Change (invalid!)"));
        return;
    }

    oldRoadTypeSections_ = road_->getTypeSections();

    // Delete old RoadTypeSections //
    //
    foreach (TypeSection *RoadTypeSection, oldRoadTypeSections_)
    {
        road_->delTypeSection(RoadTypeSection);
    }

    // New RoadTypeSections //
    //
    if (road_->getTypeSections().empty())
    {
        foreach (TypeSection *section, prototype_->getTypeSections())
        {
            road_->addTypeSection(section->getClone());
        }
    }
    newRoadTypeSections_ = road_->getTypeSections();

    // Done //
    //
    setValid();
    setText(QObject::tr("Change"));
}

/*! \brief .
*
*/
ChangeRoadTypePrototypeCommand::~ChangeRoadTypePrototypeCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        foreach (TypeSection *section, newRoadTypeSections_)
        {
            delete section; // delete sections
        }
    }
    else
    {
        // nothing to be done (tracks are now owned by the road)
    }
}

/*! \brief .
*
*/
void
ChangeRoadTypePrototypeCommand::redo()
{

    if (!newRoadTypeSections_.isEmpty())
    {
        // Delete old RoadTypeSections //
        //
        foreach (TypeSection *RoadTypeSection, oldRoadTypeSections_)
        {
            road_->delTypeSection(RoadTypeSection);
        }

        // New RoadTypeSections //
        //

        foreach (TypeSection *section, newRoadTypeSections_)
        {
            road_->addTypeSection(section);
        }
    }

    setRedone();
}

/*! \brief .
*
*/
void
ChangeRoadTypePrototypeCommand::undo()
{
    QMap<double, TypeSection *> newRoadTypeSections;

    // Delete old RoadTypeSections //
    //
    foreach (TypeSection *RoadTypeSection, newRoadTypeSections_)
    {
        road_->delTypeSection(RoadTypeSection);
    }

    // New RoadTypeSections //
    //

    foreach (TypeSection *section, oldRoadTypeSections_)
    {
        road_->addTypeSection(section);
    }

    setUndone();
}

//#########################//
// ChangeElevationPrototypeCommand //
//#########################//

ChangeElevationPrototypeCommand::ChangeElevationPrototypeCommand(RSystemElementRoad *road, RSystemElementRoad *prototype, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , prototype_(prototype)
{

    // Check for validity //
    //
    if (!road_ || !prototype_ || prototype_->getElevationSections().empty())
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Change (invalid!)"));
        return;
    }

    oldElevationSections_ = road_->getElevationSections();

    // Delete old ElevationSections //
    //
    foreach (ElevationSection *ElevationSection, oldElevationSections_)
    {
        road_->delElevationSection(ElevationSection);
    }

    // New ElevationSections //
    //
    if (road_->getElevationSections().empty())
    {

        foreach (ElevationSection *section, prototype_->getElevationSections())
        {
            road_->addElevationSection(section->getClone());
        }
    }
    newElevationSections_ = road_->getElevationSections();

    // Done //
    //
    setValid();
    setText(QObject::tr("Change"));
}

/*! \brief .
*
*/
ChangeElevationPrototypeCommand::~ChangeElevationPrototypeCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        foreach (ElevationSection *section, newElevationSections_)
        {
            delete section; // delete sections
        }
    }
    else
    {
        // nothing to be done (tracks are now owned by the road)
    }
}

/*! \brief .
*
*/
void
ChangeElevationPrototypeCommand::redo()
{

    if (!newElevationSections_.isEmpty())
    {
        // Delete old ElevationSections //
        //
        foreach (ElevationSection *ElevationSection, oldElevationSections_)
        {
            road_->delElevationSection(ElevationSection);
        }

        // New ElevationSections //
        //

        foreach (ElevationSection *section, newElevationSections_)
        {
            road_->addElevationSection(section);
        }
    }

    setRedone();
}

/*! \brief .
*
*/
void
ChangeElevationPrototypeCommand::undo()
{
    QMap<double, ElevationSection *> newElevationSections;

    // Delete old ElevationSections //
    //
    foreach (ElevationSection *ElevationSection, newElevationSections_)
    {
        road_->delElevationSection(ElevationSection);
    }

    // New ElevationSections //
    //

    foreach (ElevationSection *section, oldElevationSections_)
    {
        road_->addElevationSection(section);
    }

    setUndone();
}

//#########################//
// ChangeSuperelevationPrototypeCommand //
//#########################//

ChangeSuperelevationPrototypeCommand::ChangeSuperelevationPrototypeCommand(RSystemElementRoad *road, RSystemElementRoad *prototype, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , prototype_(prototype)
{

    // Check for validity //
    //
    if (!road_ || !prototype_ || prototype_->getSuperelevationSections().empty())
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Change (invalid!)"));
        return;
    }

    oldSuperelevationSections_ = road_->getSuperelevationSections();

    // Delete old SuperelevationSections //
    //
    foreach (SuperelevationSection *SuperelevationSection, oldSuperelevationSections_)
    {
        road_->delSuperelevationSection(SuperelevationSection);
    }

    // New SuperelevationSections //
    //
    if (road_->getSuperelevationSections().empty())
    {

        foreach (SuperelevationSection *section, prototype_->getSuperelevationSections())
        {
            road_->addSuperelevationSection(section->getClone());
        }
    }
    newSuperelevationSections_ = road_->getSuperelevationSections();

    // Done //
    //
    setValid();
    setText(QObject::tr("Change"));
}

/*! \brief .
*
*/
ChangeSuperelevationPrototypeCommand::~ChangeSuperelevationPrototypeCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        foreach (SuperelevationSection *section, newSuperelevationSections_)
        {
            delete section; // delete sections
        }
    }
    else
    {
        // nothing to be done (tracks are now owned by the road)
    }
}

/*! \brief .
*
*/
void
ChangeSuperelevationPrototypeCommand::redo()
{

    if (!newSuperelevationSections_.isEmpty())
    {
        // Delete old SuperelevationSections //
        //
        foreach (SuperelevationSection *SuperelevationSection, oldSuperelevationSections_)
        {
            road_->delSuperelevationSection(SuperelevationSection);
        }

        // New SuperelevationSections //
        //

        foreach (SuperelevationSection *section, newSuperelevationSections_)
        {
            road_->addSuperelevationSection(section);
        }
    }

    setRedone();
}

/*! \brief .
*
*/
void
ChangeSuperelevationPrototypeCommand::undo()
{
    QMap<double, SuperelevationSection *> newSuperelevationSections;

    // Delete old SuperelevationSections //
    //
    foreach (SuperelevationSection *SuperelevationSection, newSuperelevationSections_)
    {
        road_->delSuperelevationSection(SuperelevationSection);
    }

    // New SuperelevationSections //
    //

    foreach (SuperelevationSection *section, oldSuperelevationSections_)
    {
        road_->addSuperelevationSection(section);
    }

    setUndone();
}

//#########################//
// ChangeCrossfallPrototypeCommand //
//#########################//

ChangeCrossfallPrototypeCommand::ChangeCrossfallPrototypeCommand(RSystemElementRoad *road, RSystemElementRoad *prototype, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , prototype_(prototype)
{

    // Check for validity //
    //
    if (!road_ || !prototype_ || prototype_->getCrossfallSections().empty())
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Change (invalid!)"));
        return;
    }

    oldCrossfallSections_ = road_->getCrossfallSections();

    // Delete old CrossfallSections //
    //
    foreach (CrossfallSection *CrossfallSection, oldCrossfallSections_)
    {
        road_->delCrossfallSection(CrossfallSection);
    }

    // New CrossfallSections //
    //
    if (road_->getCrossfallSections().empty())
    {

        foreach (CrossfallSection *section, prototype_->getCrossfallSections())
        {
            road_->addCrossfallSection(section->getClone());
        }
    }
    newCrossfallSections_ = road_->getCrossfallSections();

    // Done //
    //
    setValid();
    setText(QObject::tr("Change"));
}

/*! \brief .
*
*/
ChangeCrossfallPrototypeCommand::~ChangeCrossfallPrototypeCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        foreach (CrossfallSection *section, newCrossfallSections_)
        {
            delete section; // delete sections
        }
    }
    else
    {
        // nothing to be done (tracks are now owned by the road)
    }
}

/*! \brief .
*
*/
void
ChangeCrossfallPrototypeCommand::redo()
{

    if (!newCrossfallSections_.isEmpty())
    {
        // Delete old CrossfallSections //
        //
        foreach (CrossfallSection *CrossfallSection, oldCrossfallSections_)
        {
            road_->delCrossfallSection(CrossfallSection);
        }

        // New CrossfallSections //
        //

        foreach (CrossfallSection *section, newCrossfallSections_)
        {
            road_->addCrossfallSection(section);
        }
    }

    setRedone();
}

/*! \brief .
*
*/
void
ChangeCrossfallPrototypeCommand::undo()
{
    QMap<double, CrossfallSection *> newCrossfallSections;

    // Delete old CrossfallSections //
    //
    foreach (CrossfallSection *CrossfallSection, newCrossfallSections_)
    {
        road_->delCrossfallSection(CrossfallSection);
    }

    // New CrossfallSections //
    //

    foreach (CrossfallSection *section, oldCrossfallSections_)
    {
        road_->addCrossfallSection(section);
    }

    setUndone();
}

//#########################//
// ChangeShapePrototypeCommand //
//#########################//

ChangeShapePrototypeCommand::ChangeShapePrototypeCommand(RSystemElementRoad *road, RSystemElementRoad *prototype, DataCommand *parent)
	: DataCommand(parent)
	, road_(road)
	, prototype_(prototype)
{

	// Check for validity //
	//
	if (!road_ || !prototype_ || prototype_->getShapeSections().empty())
	{
		setInvalid(); // Invalid
		setText(QObject::tr("Change (invalid!)"));
		return;
	}

	oldShapeSections_ = road_->getShapeSections();

	// Delete old ShapeSections //
	//
	foreach(ShapeSection *ShapeSection, oldShapeSections_)
	{
		road_->delShapeSection(ShapeSection);
	}

	// New ShapeSections //
	//
	if (road_->getShapeSections().empty())
	{

		foreach(ShapeSection *section, prototype_->getShapeSections())
		{
			road_->addShapeSection(section->getClone());
		}
	}
	newShapeSections_ = road_->getShapeSections();

	// Done //
	//
	setValid();
	setText(QObject::tr("Change"));
}

/*! \brief .
*
*/
ChangeShapePrototypeCommand::~ChangeShapePrototypeCommand()
{
	// Clean up //
	//
	if (isUndone())
	{
		foreach(ShapeSection *section, newShapeSections_)
		{
			delete section; // delete sections
		}
	}
	else
	{
		// nothing to be done (tracks are now owned by the road)
	}
}

/*! \brief .
*
*/
void
ChangeShapePrototypeCommand::redo()
{

	if (!newShapeSections_.isEmpty())
	{
		// Delete old ShapeSections //
		//
		foreach(ShapeSection *ShapeSection, oldShapeSections_)
		{
			road_->delShapeSection(ShapeSection);
		}

		// New ShapeSections //
		//

		foreach(ShapeSection *section, newShapeSections_)
		{
			road_->addShapeSection(section);
		}
	}

	setRedone();
}

/*! \brief .
*
*/
void
ChangeShapePrototypeCommand::undo()
{
	QMap<double, ShapeSection *> newShapeSections;

	// Delete old ShapeSections //
	//
	foreach(ShapeSection *ShapeSection, newShapeSections_)
	{
		road_->delShapeSection(ShapeSection);
	}

	// New ShapeSections //
	//

	foreach(ShapeSection *section, oldShapeSections_)
	{
		road_->addShapeSection(section);
	}

	setUndone();
}

//#########################//
// RemoveTrackCommand //
//#########################//

RemoveTrackCommand::RemoveTrackCommand(RSystemElementRoad *road, TrackComponent *trackComponent, bool atStart, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , atStart_(atStart)

{
    // Check for validity //
    //
    if (!road_ || !trackComponent)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Remove Tracks (invalid!)"));
        return;
    }

    if (atStart_)
    {
        // TODO: implement atStart!
        //
        // Problem: The RoadSection containing sCut_ must be moved (and moved back at undo)
        setInvalid();
        setText(QObject::tr("Remove Tracks not possible at start."));
        return;
    }

    cutLength_ = trackComponent->getSEnd();

    if (atStart_)
    {
        //		// Coordinate of the cut //
        //		//
        //		sCut_ = trackComponent->getSEnd();
        //
        //		// TrackSections //
        //		//
        //		foreach(TrackComponent * track, road_->getTrackSections())
        //		{
        //			if(track->getSStart() < sCut_)
        //			{
        //				trackSections_.insert(track->getSStart(), track);
        //			}
        //		}
        //
        //		// TypeSections //
        //		//
        //		foreach(TypeSection * section, road_->getTypeSections())
        //		{
        //			if(section->getSStart() < sCut_)
        //			{
        //				typeSections_.insert(section->getSStart(), section);
        //			}
        //		}
        //
        //		// ElevationSections //
        //		//
        //		foreach(ElevationSection * section, road_->getElevationSections())
        //		{
        //			if(section->getSStart() < sCut_)
        //			{
        //				elevationSections_.insert(section->getSStart(), section);
        //			}
        //		}
        //
        //		// SuperelevationSections //
        //		//
        //		foreach(SuperelevationSection * section, road_->getSuperelevationSections())
        //		{
        //			if(section->getSStart() < sCut_)
        //			{
        //				superelevationSections_.insert(section->getSStart(), section);
        //			}
        //		}
        //
        //		// CrossfallSections //
        //		//
        //		foreach(CrossfallSection * section, road_->getCrossfallSections())
        //		{
        //			if(section->getSStart() < sCut_)
        //			{
        //				crossfallSections_.insert(section->getSStart(), section);
        //			}
        //		}
        //
        //		// LaneSections //
        //		//
        //		foreach(LaneSection * section, road_->getLaneSections())
        //		{
        //			if(section->getSStart() < sCut_)
        //			{
        //				laneSections_.insert(section->getSStart(), section);
        //			}
        //		}
    }
    else
    {
        // Coordinate of the cut //
        //
        sCut_ = trackComponent->getSStart();

        // TrackSections //
        //
        foreach (TrackComponent *track, road_->getTrackSections())
        {
            if (track->getSStart() >= sCut_)
            {
                trackSections_.insert(track->getSStart(), track);
            }
        }

        // TypeSections //
        //
        foreach (TypeSection *section, road_->getTypeSections())
        {
            if (section->getSStart() >= sCut_)
            {
                typeSections_.insert(section->getSStart(), section);
            }
        }

        // ElevationSections //
        //
        foreach (ElevationSection *section, road_->getElevationSections())
        {
            if (section->getSStart() >= sCut_)
            {
                elevationSections_.insert(section->getSStart(), section);
            }
        }

        // SuperelevationSections //
        //
        foreach (SuperelevationSection *section, road_->getSuperelevationSections())
        {
            if (section->getSStart() >= sCut_)
            {
                superelevationSections_.insert(section->getSStart(), section);
            }
        }

        // CrossfallSections //
        //
        foreach (CrossfallSection *section, road_->getCrossfallSections())
        {
            if (section->getSStart() >= sCut_)
            {
                crossfallSections_.insert(section->getSStart(), section);
            }
        }

        // LaneSections //
        //
        foreach (LaneSection *section, road_->getLaneSections())
        {
            if (section->getSStart() >= sCut_)
            {
                laneSections_.insert(section->getSStart(), section);
            }
        }

		// ShapeSections //
		//
		foreach(ShapeSection *section, road_->getShapeSections())
		{
			if (section->getSStart() >= sCut_)
			{
				shapeSections_.insert(section->getSStart(), section);
			}
		}
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Remove Tracks"));
}

/*! \brief .
*
*/
RemoveTrackCommand::~RemoveTrackCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (tracks/sections are still owned by the road)
    }
    else
    {
        foreach (TrackComponent *section, trackSections_)
        {
            delete section; // delete sections
        }
        foreach (TypeSection *section, typeSections_)
        {
            delete section; // delete sections
        }
        foreach (ElevationSection *section, elevationSections_)
        {
            delete section; // delete sections
        }
        foreach (SuperelevationSection *section, superelevationSections_)
        {
            delete section; // delete sections
        }
        foreach (CrossfallSection *section, crossfallSections_)
        {
            delete section; // delete sections
        }
        foreach (LaneSection *section, laneSections_)
        {
            delete section; // delete sections
        }
		foreach(ShapeSection *section, shapeSections_)
		{
			delete section; // delete sections
		}
    }
}

/*! \brief .
*
*/
void
RemoveTrackCommand::redo()
{
    if (atStart_)
    {
        //		// TrackComponents //
        //		//
        //		foreach(TrackComponent * track, trackSections_)
        //		{
        //			road_->delTrackComponent(track);
        //		}
        //		road_->rebuildTrackComponentList();
        //
        //
        //		// TypeSections //
        //		//
        //		QMap<double, TypeSection *> newTypeSections;
        //		foreach(TypeSection * section, road_->getTypeSections())
        //		{
        //			if(section->getSStart() < sCut_)
        //			{
        //				section->addDataElementChanges(DataElement::CDE_DataElementRemoved);
        //			}
        //			else
        //			{
        //				section->setSStart(section->getSStart() - cutLength_);
        //				newTypeSections.insert(section->getSStart(), section);
        //			}
        //		}
        //		road_->setTypeSections(newTypeSections);
        //
        //		// ElevationSections //
        //		//
        //		QMap<double, ElevationSection *> newElevationSections;
        //		foreach(ElevationSection * section, road_->getElevationSections())
        //		{
        //			if(section->getSStart() < sCut_)
        //			{
        //				section->addDataElementChanges(DataElement::CDE_DataElementRemoved);
        //			}
        //			else
        //			{
        //				section->setSStart(section->getSStart() - cutLength_);
        //				newElevationSections.insert(section->getSStart(), section);
        //			}
        //		}
        //		road_->setElevationSections(newElevationSections);
        //
        //		// SuperelevationSections //
        //		//
        //		QMap<double, SuperelevationSection *> newSuperelevationSections;
        //		foreach(SuperelevationSection * section, road_->getSuperelevationSections())
        //		{
        //			if(section->getSStart() < sCut_)
        //			{
        //				section->addDataElementChanges(DataElement::CDE_DataElementRemoved);
        //			}
        //			else
        //			{
        //				section->setSStart(section->getSStart() - cutLength_);
        //				newSuperelevationSections.insert(section->getSStart(), section);
        //			}
        //		}
        //		road_->setSuperelevationSections(newSuperelevationSections);
        //
        //		// CrossfallSections //
        //		//
        //		QMap<double, CrossfallSection *> newCrossfallSections;
        //		foreach(CrossfallSection * section, road_->getCrossfallSections())
        //		{
        //			if(section->getSStart() < sCut_)
        //			{
        //				section->addDataElementChanges(DataElement::CDE_DataElementRemoved);
        //			}
        //			else
        //			{
        //				section->setSStart(section->getSStart() - cutLength_);
        //				newCrossfallSections.insert(section->getSStart(), section);
        //			}
        //		}
        //		road_->setCrossfallSections(newCrossfallSections);
        //
        //		// LaneSections //
        //		//
        //		QMap<double, LaneSection *> newLaneSections;
        //		foreach(LaneSection * section, road_->getLaneSections())
        //		{
        //			if(section->getSStart() < sCut_)
        //			{
        //				section->addDataElementChanges(DataElement::CDE_DataElementRemoved);
        //			}
        //			else
        //			{
        //				section->setSStart(section->getSStart() - cutLength_);
        //				newLaneSections.insert(section->getSStart(), section);
        //			}
        //		}
        //		road_->setLaneSections(newLaneSections);
    }
    else
    {
        foreach (TypeSection *section, typeSections_)
        {
            road_->delTypeSection(section);
        }
        foreach (ElevationSection *section, elevationSections_)
        {
            road_->delElevationSection(section);
        }
        foreach (SuperelevationSection *section, superelevationSections_)
        {
            road_->delSuperelevationSection(section);
        }
        foreach (CrossfallSection *section, crossfallSections_)
        {
            road_->delCrossfallSection(section);
        }
        foreach (LaneSection *section, laneSections_)
        {
            road_->delLaneSection(section);
        }
		foreach(ShapeSection *section, shapeSections_)
		{
			road_->delShapeSection(section);
		}

        QMap<double, TrackComponent *>::const_iterator i = trackSections_.constEnd();
        while (i != trackSections_.constBegin())
        {
            --i;
            road_->delTrackComponent(i.value());
        }
        //foreach(TrackComponent * track, trackSections_)
        //{
        //	road_->delTrackComponent(track);
        //}
    }

    setRedone();
}

/*! \brief
*
*/
void
RemoveTrackCommand::undo()
{
    if (atStart_)
    {

        //		foreach(TrackComponent * track, trackSections_)
        //		{
        //			track->setSStart(track->getSStart() - prototypeLength_); // note: s will be negative until rebuildTrackComponentList()
        //			road_->addTrackComponent(track);
        //		}
        //		road_->rebuildTrackComponentList();
        //
        //		// Move/Add RoadSections //
        //		//
        //		if(!newTypeSections_.isEmpty())
        //		{
        //			// Move old sections //
        //			//
        //			QMap<double, TypeSection *> newSections = newTypeSections_;
        //			foreach(TypeSection * section, newSections)
        //			{
        //				section->addDataElementChanges(DataElement::CDE_DataElementAdded);
        //			}
        //			foreach(TypeSection * section, oldTypeSections_)
        //			{
        //				section->setSStart(section->getSStart() + prototypeLength_);
        //				newSections.insert(section->getSStart(), section);
        //			}
        //			road_->setTypeSections(newSections);
        //		}
        //		if(!newElevationSections_.isEmpty())
        //		{
        //			// Move old sections //
        //			//
        //			QMap<double, ElevationSection *> newSections = newElevationSections_;
        //			foreach(ElevationSection * section, newSections)
        //			{
        //				section->addDataElementChanges(DataElement::CDE_DataElementAdded);
        //			}
        //			foreach(ElevationSection * section, oldElevationSections_)
        //			{
        //				section->setSStart(section->getSStart() + prototypeLength_);
        //				newSections.insert(section->getSStart(), section);
        //			}
        //			road_->setElevationSections(newSections);
        //		}
        //		if(!newSuperelevationSections_.isEmpty())
        //		{
        //			// Move old sections //
        //			//
        //			QMap<double, SuperelevationSection *> newSections = newSuperelevationSections_;
        //			foreach(SuperelevationSection * section, newSections)
        //			{
        //				section->addDataElementChanges(DataElement::CDE_DataElementAdded);
        //			}
        //			foreach(SuperelevationSection * section, oldSuperelevationSections_)
        //			{
        //				section->setSStart(section->getSStart() + prototypeLength_);
        //				newSections.insert(section->getSStart(), section);
        //			}
        //			road_->setSuperelevationSections(newSections);
        //		}
        //		if(!newCrossfallSections_.isEmpty())
        //		{
        //			// Move old sections //
        //			//
        //			QMap<double, CrossfallSection *> newSections = newCrossfallSections_;
        //			foreach(CrossfallSection * section, newSections)
        //			{
        //				section->addDataElementChanges(DataElement::CDE_DataElementAdded);
        //			}
        //			foreach(CrossfallSection * section, oldCrossfallSections_)
        //			{
        //				section->setSStart(section->getSStart() + prototypeLength_);
        //				newSections.insert(section->getSStart(), section);
        //			}
        //			road_->setCrossfallSections(newSections);
        //		}
        //		if(!newLaneSections_.isEmpty())
        //		{
        //			// Move old sections //
        //			//
        //			QMap<double, LaneSection *> newSections = newLaneSections_;
        //			foreach(LaneSection * section, newSections)
        //			{
        //				section->addDataElementChanges(DataElement::CDE_DataElementAdded);
        //			}
        //			foreach(LaneSection * section, oldLaneSections_)
        //			{
        //				section->setSStart(section->getSStart() + prototypeLength_);
        //				newSections.insert(section->getSStart(), section);
        //			}
        //			road_->setLaneSections(newSections);
        //		}
    }
    else
    {
        foreach (TrackComponent *track, trackSections_)
        {
            road_->addTrackComponent(track);
        }
        foreach (TypeSection *section, typeSections_)
        {
            road_->addTypeSection(section);
        }
        foreach (ElevationSection *section, elevationSections_)
        {
            road_->addElevationSection(section);
        }
        foreach (SuperelevationSection *section, superelevationSections_)
        {
            road_->addSuperelevationSection(section);
        }
        foreach (CrossfallSection *section, crossfallSections_)
        {
            road_->addCrossfallSection(section);
        }
        foreach (LaneSection *section, laneSections_)
        {
            road_->addLaneSection(section);
        }
		foreach(ShapeSection *section, shapeSections_)
		{
			road_->addShapeSection(section);
		}
    }

    setUndone();
}

//#########################//
// NewRoadCommand //
//#########################//

NewRoadCommand::NewRoadCommand(RSystemElementRoad *newRoad, RoadSystem *roadSystem, DataCommand *parent)
    : DataCommand(parent)
    , newRoad_(newRoad)
    , roadSystem_(roadSystem)
{
    // Check for validity //
    //
    if (!newRoad || !roadSystem_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("NewRoadCommand: Internal error! No new road specified."));
        return;
    }
    else
    {
        setValid();
        setText(QObject::tr("New Road"));
    }
}

/*! \brief .
*
*/
NewRoadCommand::~NewRoadCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        delete newRoad_;
    }
    else
    {
        // nothing to be done (road is now owned by the roadsystem)
    }
}

/*! \brief .
*
*/
void
NewRoadCommand::redo()
{
    roadSystem_->addRoad(newRoad_);

    setRedone();
}

/*! \brief
*
*/
void
NewRoadCommand::undo()
{
    roadSystem_->delRoad(newRoad_);

    setUndone();
}

//#########################//
// RemoveRoadCommand //
//#########################//

RemoveRoadCommand::RemoveRoadCommand(RSystemElementRoad *road, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
{
    // Check for validity //
    //
    if (!road_ || !road_->getRoadSystem())
    {
        setInvalid(); // Invalid
        setText(QObject::tr("RemoveRoadCommand: Internal error! No road specified."));
        return;
    }
    else
    {
        roadSystem_ = road->getRoadSystem();

        setValid();
        setText(QObject::tr("Remove Road"));
    }
}

/*! \brief .
*
*/
RemoveRoadCommand::~RemoveRoadCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (road is still owned by the roadsystem)
    }
    else
    {
        delete road_;
    }
}

/*! \brief .
*
*/
void
RemoveRoadCommand::redo()
{
    roadSystem_->delRoad(road_);

    setRedone();
}

/*! \brief
*
*/
void
RemoveRoadCommand::undo()
{
    roadSystem_->addRoad(road_);

    setUndone();
}

//#########################//
// MoveRoadCommand //
//#########################//

MoveRoadCommand::MoveRoadCommand(const QList<RSystemElementRoad *> &roads, const QPointF &dPos, DataCommand *parent)
    : DataCommand(parent)
    , roads_(roads)
    , dPos_(dPos)
{
    // Check for validity //
    //
    if (dPos_.isNull())
    {
        setInvalid();
        setText(QObject::tr("MoveRoadCommand: Nothing to be done."));
        return;
    }

    if (roads_.isEmpty())
    {
        setInvalid();
        setText(QObject::tr("MoveRoadCommand: No road selected."));
        return;
    }

    // Old Points //
    //
    //	foreach(RSystemElementRoad * road, roads_)
    //	{
    //		oldStartPoints_.append(road->getGlobalPoint(0.0));
    //	}

    // Done //
    //
    setValid();
    setText(QObject::tr("Move Road"));
}

/*! \brief .
*
*/
MoveRoadCommand::~MoveRoadCommand()
{
}

/*! \brief .
*
*/
void
MoveRoadCommand::redo()
{
    foreach (RSystemElementRoad *road, roads_)
    {
        foreach (TrackComponent *track, road->getTrackSections())
        {
            track->setLocalTranslation(track->getGlobalPoint(track->getSStart()) + dPos_);
        }
    }

    setRedone();
}

/*! \brief
*
*/
void
MoveRoadCommand::undo()
{
    int i = 0;
    foreach (RSystemElementRoad *road, roads_)
    {
        foreach (TrackComponent *track, road->getTrackSections())
        {
            track->setLocalTranslation(track->getGlobalPoint(track->getSStart()) - dPos_);
        }

        ++i;
    }

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
MoveRoadCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const MoveRoadCommand *command = static_cast<const MoveRoadCommand *>(other);

    // Check tracks //
    //
    if (roads_.size() != command->roads_.size())
    {
        return false;
    }
    for (int i = 0; i < roads_.size(); ++i)
    {
        if (roads_[i] != command->roads_[i])
        {
            return false;
        }
    }

    // Success //
    //
    dPos_ += command->dPos_; // adjust to new point, then let the undostack kill the new command

    return true;
}

//#########################//
// RotateRoadAroundPointCommand //
//#########################//

RotateRoadAroundPointCommand::RotateRoadAroundPointCommand(const QList<RSystemElementRoad *> &roads, const QPointF &pivotPoint, double angleDegrees, DataCommand *parent)
    : DataCommand(parent)
    , roads_(roads)
    , pivotPoint_(pivotPoint)
    , angleDegrees_(angleDegrees)
{
    // Check for validity //
    //
    if (angleDegrees_ < NUMERICAL_ZERO8)
    {
        setInvalid();
        setText(QObject::tr("RotateRoadAroundPointCommand: Nothing to be done."));
        return;
    }

    if (roads_.isEmpty())
    {
        setInvalid();
        setText(QObject::tr("RotateRoadAroundPointCommand: No road selected."));
        return;
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Rotate Roads"));
}

RotateRoadAroundPointCommand::RotateRoadAroundPointCommand(RSystemElementRoad *road, const QPointF &pivotPoint, double angleDegrees, DataCommand *parent)
    : DataCommand(parent)
    , pivotPoint_(pivotPoint)
    , angleDegrees_(angleDegrees)
{
    // Check for validity //
    //
    if (angleDegrees_ < NUMERICAL_ZERO8)
    {
        setInvalid();
        setText(QObject::tr("RotateRoadAroundPointCommand: Nothing to be done."));
        return;
    }

    if (!road)
    {
        setInvalid();
        setText(QObject::tr("RotateRoadAroundPointCommand: No road selected."));
        return;
    }

    roads_.append(road);

    // Done //
    //
    setValid();
    setText(QObject::tr("Rotate Roads"));
}

/*! \brief .
*
*/
RotateRoadAroundPointCommand::~RotateRoadAroundPointCommand()
{
}

/*! \brief .
*
*/
void
RotateRoadAroundPointCommand::redo()
{
    QTransform redoTrafo;
    redoTrafo.rotate(angleDegrees_);

    foreach (RSystemElementRoad *road, roads_)
    {
        foreach (TrackComponent *track, road->getTrackSections())
        {
            QPointF q = track->getGlobalPoint(track->getSStart()) - pivotPoint_;
            track->setLocalTranslation(pivotPoint_ + redoTrafo.map(q));
            track->setLocalRotation(track->getGlobalHeading(track->getSStart()) + angleDegrees_);
        }
    }

    setRedone();
}

/*! \brief
*
*/
void
RotateRoadAroundPointCommand::undo()
{
    QTransform undoTrafo;
    undoTrafo.rotate(-angleDegrees_);

    foreach (RSystemElementRoad *road, roads_)
    {
        foreach (TrackComponent *track, road->getTrackSections())
        {
            QPointF q = track->getGlobalPoint(track->getSStart()) - pivotPoint_;
            track->setLocalTranslation(pivotPoint_ + undoTrafo.map(q));
            track->setLocalRotation(track->getGlobalHeading(track->getSStart()) - angleDegrees_);
        }
    }

    setUndone();
}

/*! \brief Attempts to merge this command with other. Returns true on success; otherwise returns false.
*
*/
bool
RotateRoadAroundPointCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const RotateRoadAroundPointCommand *command = static_cast<const RotateRoadAroundPointCommand *>(other);

    // Check tracks //
    //
    if (roads_.size() != command->roads_.size())
    {
        return false;
    }
    for (int i = 0; i < roads_.size(); ++i)
    {
        if (roads_[i] != command->roads_[i])
        {
            return false;
        }
    }
    if (pivotPoint_ != command->pivotPoint_)
    {
        return false;
    }

    // Success //
    //
    angleDegrees_ += command->angleDegrees_; // adjust to new angle, then let the undostack kill the new command

    return true;
}

//#########################//
// SplitRoadCommand //
//#########################//

SplitRoadCommand::SplitRoadCommand(RSystemElementRoad *road, double s, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , newRoadA_(NULL)
    , newRoadB_(NULL)
    , splitS_(s)
{
    // Check for validity //
    //
    if (!road)
    {
        setInvalid();
        setText(QObject::tr("SplitRoadCommand: Nothing to be done."));
        return;
    }

    if (s < 0.5)
    {
        setInvalid();
        setText(QObject::tr("SplitRoadCommand: s < 0.5."));
        return;
    }

    if (s > road_->getLength() - 0.5)
    {
        setInvalid();
        setText(QObject::tr("SplitRoadCommand: s > length - 0.5."));
        return;
    }

    roadSystem_ = road_->getRoadSystem();

    TrackComponent *splitTrack = road_->getTrackComponent(splitS_);
    //	if(splitTrack->getTrackType() == )
    //qDebug(QString("splitS: %1").arg(splitS_).toUtf8());
    //qDebug(QString("splitTrack->getSStart(): %1").arg(splitTrack->getSStart()).toUtf8());
    //qDebug(QString("splitTrack->getSEnd(): %1").arg(splitTrack->getSEnd()).toUtf8());
    if (fabs(splitS_ - splitTrack->getSStart()) < NUMERICAL_ZERO6)
    {
        // Exact match, split here //
        //
        splitBefore(splitTrack->getSStart());
    }
    else
    {
        // Distance to start/end of section //
        //
        double sToStart = splitS_ - splitTrack->getSStart();
        double sToEnd = splitTrack->getSEnd() - splitS_;
        if (sToStart < sToEnd)
        {
            if (splitTrack->getSStart() < NUMERICAL_ZERO6)
            {
                setInvalid();
                setText(QObject::tr("SplitRoadCommand: too close to the start of the road."));
                return;
            }
            splitBefore(splitTrack->getSStart());
        }
        else
        {
            if (fabs(splitTrack->getSEnd() - road_->getLength()) < NUMERICAL_ZERO6)
            {
                setInvalid();
                setText(QObject::tr("SplitRoadCommand: too close to the end of the road."));
                return;
            }
            else
            {
                splitBefore(splitTrack->getSEnd());
            }
        }
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Split Road"));
}

/*! \brief .
*
*/
SplitRoadCommand::~SplitRoadCommand()
{
    if (isUndone())
    {
        delete newRoadA_;
        delete newRoadB_;
    }
    else
    {
        delete road_;
    }
}

/*! \brief .
*
*/
void
SplitRoadCommand::splitBefore(double s)
{
    //	qDebug(QString("split at %1").arg(s).toUtf8());

    newRoadA_ = new RSystemElementRoad(road_->getName(), odrID::invalidID(), road_->getJunction());
    newRoadB_ = new RSystemElementRoad(road_->getName(), odrID::invalidID(), road_->getJunction());

    // planView //
    QMap<double, TrackComponent *> trackSectionsA;
    QMap<double, TrackComponent *> trackSectionsB;
    foreach (TrackComponent *section, road_->getTrackSections())
    {
        if (section->getSStart() < s)
        {
            section = section->getClone();
            trackSectionsA.insert(section->getSStart(), section);
        }
        else
        {
            section = section->getClone();
            section->setSStart(section->getSStart() - s);
            trackSectionsB.insert(section->getSStart(), section);
        }
    }
    newRoadA_->setTrackComponents(trackSectionsA);
    newRoadB_->setTrackComponents(trackSectionsB);

    // typesections //
    QMap<double, TypeSection *> typeSectionsA;
    QMap<double, TypeSection *> typeSectionsB;
    if (!road_->getTypeSections().isEmpty())
    {
        foreach (TypeSection *section, road_->getTypeSections())
        {
            bool exactMatch = false;

            // First road part //
            //
            if (section->getSStart() < s)
            {
                section = section->getClone(); // clone the section
                typeSectionsA.insert(section->getSStart(), section);
            }

            // Second road part //
            //
            else if (fabs(section->getSStart() - s) < NUMERICAL_ZERO6)
            {
                exactMatch = true;
                section = section->getClone();
                section->setSStart(0.0); // set s coordinate
                typeSectionsB.insert(0.0, section);
            }
            else
            {
                if (!exactMatch)
                {
                    // Split the section containing the split coordinate
                    TypeSection *splittedSection = road_->getTypeSection(s);
                    splittedSection = splittedSection->getClone();
                    splittedSection->setSStart(0.0); // set s coordinate
                    typeSectionsB.insert(0.0, splittedSection);
                    exactMatch = true;
                }
                section = section->getClone();
                section->setSStart(section->getSStart() - s); // set s coordinate
                typeSectionsB.insert(section->getSStart(), section);
            }
        }
        if (typeSectionsB.isEmpty())
        {
            TypeSection *splittedSection = road_->getTypeSection(s);
            splittedSection = splittedSection->getClone();
            splittedSection->setSStart(0.0);
            typeSectionsB.insert(0.0, splittedSection);
        }
        newRoadA_->setTypeSections(typeSectionsA);
        newRoadB_->setTypeSections(typeSectionsB);
    }

    // elevation //
    QMap<double, ElevationSection *> elevationSectionsA;
    QMap<double, ElevationSection *> elevationSectionsB;
    if (!road_->getElevationSections().isEmpty())
    {
        foreach (ElevationSection *section, road_->getElevationSections())
        {
            bool exactMatch = false;
            if (section->getSStart() < s)
            {
                section = section->getClone();
                elevationSectionsA.insert(section->getSStart(), section);
            }
            else if (fabs(section->getSStart() - s) < NUMERICAL_ZERO6)
            {
                exactMatch = true;
                section = section->getClone();
                section->setSStart(0.0);
                elevationSectionsB.insert(0.0, section);
            }
            else
            {
                if (!exactMatch)
                {
                    ElevationSection *originalSection = road_->getElevationSection(s);
                    ElevationSection *splittedSection = new ElevationSection(0.0, originalSection->getElevation(s), originalSection->getSlope(s), originalSection->getCurvature(s) / 2.0, originalSection->getD()); // fancy calculation, don't try this at home
                    elevationSectionsB.insert(0.0, splittedSection);
                    exactMatch = true;
                }
                section = section->getClone();
                section->setSStart(section->getSStart() - s);
                //				qDebug(QString("sstart: %1").arg(section->getSStart()).toUtf8());
                elevationSectionsB.insert(section->getSStart(), section);
            }
        }
        if (elevationSectionsB.isEmpty())
        {
            ElevationSection *splittedSection = road_->getElevationSection(s);
            splittedSection = splittedSection->getClone();
            splittedSection->setSStart(0.0);
            elevationSectionsB.insert(0.0, splittedSection);
        }
        newRoadA_->setElevationSections(elevationSectionsA);
        newRoadB_->setElevationSections(elevationSectionsB);
    }

    // superelevation //
    QMap<double, SuperelevationSection *> superelevationSectionsA;
    QMap<double, SuperelevationSection *> superelevationSectionsB;
    if (!road_->getSuperelevationSections().isEmpty())
    {
        foreach (SuperelevationSection *section, road_->getSuperelevationSections())
        {
            bool exactMatch = false;
            if (section->getSStart() < s)
            {
                section = section->getClone();
                superelevationSectionsA.insert(section->getSStart(), section);
            }
            else if (fabs(section->getSStart() - s) < NUMERICAL_ZERO6)
            {
                exactMatch = true;
                section = section->getClone();
                section->setSStart(0.0);
                superelevationSectionsB.insert(0.0, section);
            }
            else
            {
                if (!exactMatch)
                {
                    SuperelevationSection *originalSection = road_->getSuperelevationSection(s);
                    SuperelevationSection *splittedSection = new SuperelevationSection(0.0, originalSection->getSuperelevationDegrees(s), originalSection->getSuperelevationSlopeDegrees(s), originalSection->getSuperelevationCurvatureDegrees(s) / 2.0, originalSection->getD()); // fancy calculation, don't try this at home
                    superelevationSectionsB.insert(0.0, splittedSection);
                    exactMatch = true;
                }
                section = section->getClone();
                section->setSStart(section->getSStart() - s);
                //				qDebug(QString("sstart: %1").arg(section->getSStart()).toUtf8());
                superelevationSectionsB.insert(section->getSStart(), section);
            }
        }
        if (superelevationSectionsB.isEmpty())
        {
            SuperelevationSection *splittedSection = road_->getSuperelevationSection(s);
            splittedSection = splittedSection->getClone();
            splittedSection->setSStart(0.0);
            superelevationSectionsB.insert(0.0, splittedSection);
        }
        newRoadA_->setSuperelevationSections(superelevationSectionsA);
        newRoadB_->setSuperelevationSections(superelevationSectionsB);
    }

    // crossfall //
    QMap<double, CrossfallSection *> crossfallSectionsA;
    QMap<double, CrossfallSection *> crossfallSectionsB;
    if (!road_->getCrossfallSections().isEmpty())
    {
        foreach (CrossfallSection *section, road_->getCrossfallSections())
        {
            bool exactMatch = false;
            if (section->getSStart() < s)
            {
                section = section->getClone();
                crossfallSectionsA.insert(section->getSStart(), section);
            }
            else if (fabs(section->getSStart() - s) < NUMERICAL_ZERO6)
            {
                exactMatch = true;
                section = section->getClone();
                section->setSStart(0.0);
                crossfallSectionsB.insert(0.0, section);
            }
            else
            {
                if (!exactMatch)
                {
                    CrossfallSection *originalSection = road_->getCrossfallSection(s);
                    CrossfallSection *splittedSection = new CrossfallSection(originalSection->getSide(), 0.0, originalSection->getCrossfallDegrees(s), originalSection->getCrossfallSlopeDegrees(s), originalSection->getCrossfallCurvatureDegrees(s) / 2.0, originalSection->getD()); // fancy calculation, don't try this at home
                    crossfallSectionsB.insert(0.0, splittedSection);
                    exactMatch = true;
                }
                section = section->getClone();
                section->setSStart(section->getSStart() - s);
                //				qDebug(QString("sstart: %1").arg(section->getSStart()).toUtf8());
                crossfallSectionsB.insert(section->getSStart(), section);
            }
        }
        if (crossfallSectionsB.isEmpty())
        {
            CrossfallSection *splittedSection = road_->getCrossfallSection(s);
            splittedSection = splittedSection->getClone();
            // TODO, to do what?
            splittedSection->setSStart(0.0);
            crossfallSectionsB.insert(0.0, splittedSection);
        }
        newRoadA_->setCrossfallSections(crossfallSectionsA);
        newRoadB_->setCrossfallSections(crossfallSectionsB);
    }

    // lanes //
    QMap<double, LaneSection *> laneSectionsA;
    QMap<double, LaneSection *> laneSectionsB;
    if (!road_->getLaneSections().isEmpty())
    {
        foreach (LaneSection *section, road_->getLaneSections())
        {
            if (section->getSStart() < s) // starts before split
            {
                if (section->getSEnd() < s) // ends before split
                {
                    // Add completely to A //
                    //
                    section = section->getClone();
                    laneSectionsA.insert(section->getSStart(), section);
                }
                else // starts before split but ends afterwards
                {
                    LaneSection *newSection;

                    // Add first part to A //
                    //
                    newSection = section->getClone(section->getSStart(), s); // add until split coordinate
                    laneSectionsA.insert(newSection->getSStart(), newSection);

                    // Add second part to B //
                    //
                    newSection = section->getClone(s, section->getSEnd());
                    newSection->setSStart(0.0);
                    laneSectionsB.insert(0.0, newSection);
                }
            }
            else // starts (and ends) before or at split coordinate
            {
                // Add completely to B //
                //
                double newSStart = section->getSStart() - s;
                if (fabs(newSStart) < NUMERICAL_ZERO6)
                {
                    newSStart = 0.0; // round to zero, so it isn't negative (e.g. -1e-7)
                }
                section = section->getClone();
                section->setSStart(newSStart);
                laneSectionsB.insert(newSStart, section);
            }
        }
    }
    newRoadA_->setLaneSections(laneSectionsA);
    newRoadB_->setLaneSections(laneSectionsB);

	// shape //
	QMap<double, ShapeSection *> shapeSectionsA;
	QMap<double, ShapeSection *> shapeSectionsB;
	if (!road_->getShapeSections().isEmpty())
	{
		foreach(ShapeSection *section, road_->getShapeSections())
		{
			bool exactMatch = false;
			if (section->getSStart() < s)
			{
				section = section->getClone();
				shapeSectionsA.insert(section->getSStart(), section);
			}
			else if (fabs(section->getSStart() - s) < NUMERICAL_ZERO6)
			{
				exactMatch = true;
				section = section->getClone();
				section->setSStart(0.0);
				shapeSectionsB.insert(0.0, section);
			}

		}
		if (shapeSectionsB.isEmpty())
		{
			ShapeSection *splittedSection = road_->getShapeSection(s);
			splittedSection = splittedSection->getClone();
			splittedSection->setSStart(0.0);
			shapeSectionsB.insert(0.0, splittedSection);
		}
		newRoadA_->setShapeSections(shapeSectionsA);
		newRoadB_->setShapeSections(shapeSectionsB);
	}
}

/*! \brief .
*
*/
void
SplitRoadCommand::redo()
{
    roadSystem_->delRoad(road_);

    roadSystem_->addRoad(newRoadA_);
    roadSystem_->addRoad(newRoadB_);

    // predecessor/successor //
    //
    if (road_->getPredecessor())
    {
        if (road_->getPredecessor()->getElementId() == road_->getID())
        {
            newRoadA_->setPredecessor(new RoadLink("road", newRoadB_->getID(), JunctionConnection::JCP_END));
        }
        else
        {
            newRoadA_->setPredecessor(road_->getPredecessor()->getClone());
        }
    }
    newRoadA_->setSuccessor(new RoadLink("road", newRoadB_->getID(), JunctionConnection::JCP_START));
    if (road_->getSuccessor())
    {
        if (road_->getSuccessor()->getElementId() == road_->getID())
        {
            newRoadB_->setSuccessor(new RoadLink("road", newRoadA_->getID(), JunctionConnection::JCP_START));
        }
        else
        {
            newRoadB_->setSuccessor(road_->getSuccessor()->getClone());
        }
    }
    newRoadB_->setPredecessor(new RoadLink("road", newRoadA_->getID(), JunctionConnection::JCP_END));
    updateLinks(road_, road_, newRoadA_, newRoadB_);

    setRedone();
}

/*! \brief
*
*/
void
SplitRoadCommand::undo()
{
    roadSystem_->delRoad(newRoadA_);
    roadSystem_->delRoad(newRoadB_);

    roadSystem_->addRoad(road_);
    updateLinks(newRoadA_, newRoadB_, road_, road_);

    setUndone();
}

void SplitRoadCommand::updateLinks(RSystemElementRoad *currentA, RSystemElementRoad *currentB, RSystemElementRoad *newA, RSystemElementRoad *newB)
{

    RoadLink *rl = currentA->getPredecessor();
    if (rl)
    {
        if (rl->getElementType() == "road")
        {
            RSystemElementRoad *r = roadSystem_->getRoad(rl->getElementId());
            if (r)
            {
                RoadLink *rl2;
                if (rl->getContactPoint() == JunctionConnection::JCP_END)
                    rl2 = r->getSuccessor();
                else
                    rl2 = r->getPredecessor();
                if (rl2)
                {
                    rl2->setElementId(newA->getID());
                }
            }
        }
        else if (rl->getElementType() == "junction")
        {
            RSystemElementJunction *j = roadSystem_->getJunction(rl->getElementId());
            if (j)
            {
                auto connections = j->getConnections();
                foreach (JunctionConnection *connection, connections)
                {
                    if (connection->getIncomingRoad() == currentA->getID())
                    {
                        connection->setIncomingRoad(newA->getID());
                        RSystemElementRoad *connectingRoad = roadSystem_->getRoad(connection->getConnectingRoad());
                        RoadLink *rl2;
                        if (connection->getContactPoint() == JunctionConnection::JCP_START)
                        {
                            rl2 = connectingRoad->getPredecessor();
                        }
                        else
                        {
                            rl2 = connectingRoad->getSuccessor();
                        }
                        rl2->setElementId(newA->getID());
                    }
                }
            }
        }
        else if (rl->getElementType() == "fiddleyard")
        {
            RSystemElementFiddleyard *f = roadSystem_->getFiddleyard(rl->getElementId());
            if (f)
            {
                f->setElementId(newA->getID());
            }
        }
    }
    rl = currentB->getSuccessor();
    if (rl)
    {
        if (rl->getElementType() == "road")
        {
            RSystemElementRoad *r = roadSystem_->getRoad(rl->getElementId());
            if (r)
            {
                RoadLink *rl2;
                if (rl->getContactPoint() == JunctionConnection::JCP_END)
                    rl2 = r->getSuccessor();
                else
                    rl2 = r->getPredecessor();
                if (rl2)
                {
                    rl2->setElementId(newB->getID());
                }
            }
        }
        else if (rl->getElementType() == "junction")
        {
            RSystemElementJunction *j = roadSystem_->getJunction(rl->getElementId());
            if (j)
            {
                auto connections = j->getConnections();
                foreach (JunctionConnection *connection, connections)
                {
                    if (connection->getIncomingRoad() == currentB->getID())
                    {
                        connection->setIncomingRoad(newB->getID());
                        RSystemElementRoad *connectingRoad = roadSystem_->getRoad(connection->getConnectingRoad());
                        RoadLink *rl2;
                        if (connection->getContactPoint() == JunctionConnection::JCP_START)
                        {
                            rl2 = connectingRoad->getPredecessor();
                        }
                        else
                        {
                            rl2 = connectingRoad->getSuccessor();
                        }
                        rl2->setElementId(newB->getID());
                    }
                }
            }
        }
        else if (rl->getElementType() == "fiddleyard")
        {
            RSystemElementFiddleyard *f = roadSystem_->getFiddleyard(rl->getElementId());
            if (f)
            {
                f->setElementId(newB->getID());
            }
        }
    }
}

//#########################//
// SplitTrackRoadCommand //
//#########################//

SplitTrackRoadCommand::SplitTrackRoadCommand(RSystemElementRoad *road, double s, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , splitTrackComponentCommand_(NULL)
    , splitRoadCommand_(NULL)
    , splitS_(s)
{

    // Check for validity //
    //
    if (!road)
    {
        setInvalid();
        setText(QObject::tr("SplitTrackRoadCommand: Nothing to be done."));
        return;
    }

    TrackComponent *splitTrack = road_->getTrackComponent(splitS_);
    if (!splitTrack)
    {
        setInvalid();
        setText(QObject::tr("SplitTrackRoadCommand: Can not find track."));
        return;
    }

    TrackComposite *splitComposite = dynamic_cast<TrackComposite *>(splitTrack);
    if (splitComposite)
    {
        TrackComponent *childTrack = splitComposite->getChild(splitS_);
        if (childTrack)
        {
            splitTrack = childTrack;
        }
    }

    splitTrackComponentCommand_ = new SplitTrackComponentCommand(splitTrack, splitS_, this);
    if (splitTrackComponentCommand_->isValid())
    {
        splitTrackComponentCommand_->redo(); // have to call redo, otherwise the road does not contain the new tracks
    }

    splitRoadCommand_ = new SplitRoadCommand(road_, splitS_, this);
    if (!splitRoadCommand_->isValid())
    {
        setInvalid();
        setText(QObject::tr("SplitTrackRoadCommand: Can not split road."));
        return;
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Split Track and Road"));
}

/*! \brief .
*
*/
SplitTrackRoadCommand::~SplitTrackRoadCommand()
{
}

//#########################//
// SetRoadLinkCommand //
//#########################//

SetRoadLinkCommand::SetRoadLinkCommand(RSystemElementRoad *road, RoadLink::RoadLinkType roadLinkType, RoadLink *roadLink, JunctionConnection *newConnection, RSystemElementJunction *junction, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , roadLinkType_(roadLinkType)
    , oldRoadLink_(NULL)
    , newRoadLink_(roadLink)
    , newConnection_(newConnection)
    , junction_(junction)
    , roadSystem_(NULL)
{
    // Check for validity //
    //
    if (!road || !roadLink || roadLinkType_ == RoadLink::DRL_UNKNOWN)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Set Road Link: invalid parameters!"));
        return;
    }

    if (roadLinkType_ == RoadLink::DRL_PREDECESSOR)
    {
        oldRoadLink_ = road->getPredecessor();
    }
    else
    {
        oldRoadLink_ = road->getSuccessor();
    }

    roadSystem_ = road->getRoadSystem();

    // Done //
    //
    setValid();
    setText(QObject::tr("Set Road Link"));
}

SetRoadLinkCommand::~SetRoadLinkCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        delete newRoadLink_;
    }
    else
    {
        delete oldRoadLink_;
    }
}

void
SetRoadLinkCommand::redo()
{

    if (roadLinkType_ == RoadLink::DRL_PREDECESSOR)
    {
        road_->setPredecessor(newRoadLink_);
    }
    else
    {
        road_->setSuccessor(newRoadLink_);
    }
    if (junction_)
    {
        junction_->addConnection(newConnection_);
        junction_->notifyObservers();
    }
    else
    {
        if (roadLinkType_ == RoadLink::DRL_PREDECESSOR)
        {
            LaneSection *laneSection = road_->getLaneSection(0.0);
            RSystemElementRoad *linkRoad = roadSystem_->getRoad(newRoadLink_->getElementId());
			if (linkRoad == NULL)
			{
				QMessageBox msgBox;
				msgBox.setIcon(QMessageBox::Critical);
				msgBox.setWindowTitle("link error");
				msgBox.setText("wrong road link");
				msgBox.exec();
				return;
			}
            if (newRoadLink_->getContactPoint() == JunctionConnection::JCP_START)
            {
                LaneSection *linkSection = linkRoad->getLaneSection(0.0);
                QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                foreach (Lane *lane, laneSection->getLanes())
                {
                    if (lane->getId() > 0)
                    {
                        if (linkSectionLanes.contains(-lane->getId()))
                        {
                            lane->setPredecessor(-lane->getId());
                            linkSectionLanes.value(-lane->getId())->setPredecessor(lane->getId());
                        }
                    }
                }
            }
            else
            {
                LaneSection *linkSection = linkRoad->getLaneSection(linkRoad->getLength());
                QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                foreach (Lane *lane, laneSection->getLanes())
                {
                    if (lane->getId() > 0)
                    {
                        if (linkSectionLanes.contains(lane->getId()))
                        {
                            lane->setPredecessor(lane->getId());
                            linkSectionLanes.value(lane->getId())->setSuccessor(lane->getId());
                        }
                    }
                }
            }
        }
        else
        {
            LaneSection *laneSection = road_->getLaneSection(road_->getLength());
            RSystemElementRoad *linkRoad = roadSystem_->getRoad(newRoadLink_->getElementId());
            if (newRoadLink_->getContactPoint() == JunctionConnection::JCP_START)
            {
                LaneSection *linkSection = linkRoad->getLaneSection(0.0);
                QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                foreach (Lane *lane, laneSection->getLanes())
                {
                    if (lane->getId() < 0)
                    {
                        if (linkSectionLanes.contains(lane->getId()))
                        {
                            lane->setSuccessor(lane->getId());
                            linkSectionLanes.value(lane->getId())->setPredecessor(lane->getId());
                        }
                    }
                }
            }
            else
            {
                LaneSection *linkSection = linkRoad->getLaneSection(linkRoad->getLength());
                QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                foreach (Lane *lane, laneSection->getLanes())
                {
                    if (lane->getId() < 0)
                    {
                        if (linkSectionLanes.contains(-lane->getId()))
                        {
                            lane->setSuccessor(-lane->getId());
                            linkSectionLanes.value(-lane->getId())->setSuccessor(lane->getId());
                        }
                    }
                }
            }
        }
    }

    setRedone();
}

void
SetRoadLinkCommand::undo()
{
    if (roadLinkType_ == RoadLink::DRL_PREDECESSOR)
    {
        road_->setPredecessor(oldRoadLink_);
    }
    else
    {
        road_->setSuccessor(oldRoadLink_);
    }

    if (!junction_)
    {
        if (roadLinkType_ == RoadLink::DRL_PREDECESSOR)
        {
            if (oldRoadLink_)
            {
                LaneSection *laneSection = road_->getLaneSection(0.0);
                RSystemElementRoad *linkRoad = roadSystem_->getRoad(oldRoadLink_->getElementId());
                if (oldRoadLink_->getContactPoint() == JunctionConnection::JCP_START)
                {
                    LaneSection *linkSection = linkRoad->getLaneSection(0.0);
                    QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                    foreach (Lane *lane, laneSection->getLanes())
                    {
                        if (lane->getId() > 0)
                        {
                            if (linkSectionLanes.contains(-lane->getId()))
                            {
                                lane->setPredecessor(-lane->getId());
                                linkSectionLanes.value(-lane->getId())->setPredecessor(lane->getId());
                            }
                        }
                    }
                }
                else
                {
                    LaneSection *linkSection = linkRoad->getLaneSection(linkRoad->getLength());
                    QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                    foreach (Lane *lane, laneSection->getLanes())
                    {
                        if (lane->getId() > 0)
                        {
                            if (linkSectionLanes.contains(lane->getId()))
                            {
                                lane->setPredecessor(lane->getId());
                                linkSectionLanes.value(lane->getId())->setSuccessor(lane->getId());
                            }
                        }
                    }
                }
            }
            else
            {
                LaneSection *laneSection = road_->getLaneSection(0.0);
                RSystemElementRoad *linkRoad = roadSystem_->getRoad(newRoadLink_->getElementId());
                if (newRoadLink_->getContactPoint() == JunctionConnection::JCP_START)
                {
                    LaneSection *linkSection = linkRoad->getLaneSection(0.0);
                    QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                    foreach (Lane *lane, laneSection->getLanes())
                    {
                        if (lane->getId() > 0)
                        {
                            if (linkSectionLanes.contains(-lane->getId()))
                            {
                                lane->setPredecessor(lane->getId());
                                linkSectionLanes.value(-lane->getId())->setPredecessor(-lane->getId());
                            }
                        }
                    }
                }
                else
                {
                    // nothing to change, lanes are linked to lanes with their own id
                }
            }
        }
        else
        {
            if (oldRoadLink_)
            {
                LaneSection *laneSection = road_->getLaneSection(road_->getLength());
                RSystemElementRoad *linkRoad = roadSystem_->getRoad(oldRoadLink_->getElementId());
                if (oldRoadLink_->getContactPoint() == JunctionConnection::JCP_START)
                {
                    LaneSection *linkSection = linkRoad->getLaneSection(0.0);
                    QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                    foreach (Lane *lane, laneSection->getLanes())
                    {
                        if (lane->getId() < 0)
                        {
                            if (linkSectionLanes.contains(lane->getId()))
                            {
                                lane->setSuccessor(lane->getId());
                                linkSectionLanes.value(lane->getId())->setPredecessor(lane->getId());
                            }
                        }
                    }
                }
                else
                {
                    LaneSection *linkSection = linkRoad->getLaneSection(linkRoad->getLength());
                    QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                    foreach (Lane *lane, laneSection->getLanes())
                    {
                        if (lane->getId() < 0)
                        {
                            if (linkSectionLanes.contains(-lane->getId()))
                            {
                                lane->setSuccessor(-lane->getId());
                                linkSectionLanes.value(-lane->getId())->setSuccessor(lane->getId());
                            }
                        }
                    }
                }
            }
            else
            {
                LaneSection *laneSection = road_->getLaneSection(road_->getLength());
                RSystemElementRoad *linkRoad = roadSystem_->getRoad(newRoadLink_->getElementId());
                if (newRoadLink_->getContactPoint() == JunctionConnection::JCP_START)
                {
                    // nothing to change, lanes are linked to lanes with their own id
                }
                else
                {
                    LaneSection *linkSection = linkRoad->getLaneSection(linkRoad->getLength());
                    QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                    foreach (Lane *lane, laneSection->getLanes())
                    {
                        if (lane->getId() < 0)
                        {
                            if (linkSectionLanes.contains(-lane->getId()))
                            {
                                lane->setSuccessor(lane->getId());
                                linkSectionLanes.value(-lane->getId())->setSuccessor(-lane->getId());
                            }
                        }
                    }
                }
            }
        }
    }

    setUndone();
}

//#########################//
// SetRoadLinkRoadsCommand //
//#########################//

SetRoadLinkRoadsCommand::SetRoadLinkRoadsCommand(const QList<RSystemElementRoad *> &roads, double threshold, DataCommand *parent)
    : DataCommand(parent)
    , roads_(roads)
{

    // Check for validity //
    //
    if (roads_.size() < 1)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Set Road Link Roads: invalid parameters!"));
        return;
    }

    // Road Pairs are stored in distRoadPairs, sorted by distance
    QMultiMap<double, RoadPair> distRoadPairs;

    for (int i = 0; i < roads_.size(); i++)
    {
        if (roads_.at(i)->getJunction().getID() != -1)
        {

            for (int k = i + 1; k < roads_.size(); k++)
            {
                if (roads_.at(k)->getJunction().getID() == -1) // Otherwise we have two junctions
                {
                    distanceRoads(roads_.at(i), roads_.at(k), threshold, &distRoadPairs);
                }
            }
        }
        else
        {
            distanceRoad(roads_.at(i), threshold, &distRoadPairs);
            for (int k = i + 1; k < roads_.size(); k++)
            {
                if (roads_.at(k)->getJunction().getID() != -1)
                {
                    distanceRoads(roads_.at(k), roads_.at(i), threshold, &distRoadPairs);
                }
                else
                {
                    distanceRoads(roads_.at(i), roads_.at(k), threshold, &distRoadPairs);
                }
            }
        }
    }

    //connect all, except if both roads are inside a junction
    QMultiMap<double, RoadPair>::iterator it = distRoadPairs.begin();
    while (it != distRoadPairs.end())
    {
        RoadPair roadPair = it.value(); // make Connections
        if (roadPair.road1->getJunction().getID() == -1|| roadPair.road2->getJunction().getID() == -1) // at least one of the roads has to be outside
        {
            findConnections(roadPair.road1, roadPair.road2, roadPair.positionIndex);
        }
        it++;
    }
    /*
		// Keep only the pairs with shortest distances, one pair for the start of the road and one for the end
		QMultiMap<double, RoadPair>::iterator it = distRoadPairs.begin();
		while (it != distRoadPairs.end())
		{
			RoadPair roadPairMin = it.value(); // make Connections
			findConnections(roadPairMin.road1, roadPairMin.road2, roadPairMin.positionIndex);
			it = distRoadPairs.erase(it);

			while (it != distRoadPairs.end()) 
			{
				RoadPair roadPair = it.value();
				if (roadPair.road1->getJunction().isValid())
				{
					if ((roadPair.road1 == roadPairMin.road1) && 
						(((roadPair.positionIndex & FirstRoadStart) == (roadPairMin.positionIndex & FirstRoadStart)) || 
						((roadPair.positionIndex & FirstRoadEnd) == (roadPairMin.positionIndex & FirstRoadEnd))))
					{
						it = distRoadPairs.erase(it);
					}
					else
					{
						it++;
					}
				}
				else
				{

					if ((roadPair.road1 == roadPairMin.road1) && 
						(((roadPair.positionIndex & FirstRoadStart) == (roadPairMin.positionIndex & FirstRoadStart)) || 
						((roadPair.positionIndex & FirstRoadEnd) == (roadPairMin.positionIndex & FirstRoadEnd))))
					{
						it = distRoadPairs.erase(it);
					}
					else if ((roadPair.road1 == roadPairMin.road2) && 
						(((roadPair.positionIndex & FirstRoadStart) == ((roadPairMin.positionIndex & SecondRoadStart) >> 2)) || 
						((roadPair.positionIndex & FirstRoadEnd) == ((roadPairMin.positionIndex & SecondRoadEnd) >> 2))))
					{
						it = distRoadPairs.erase(it);
					}
					else if ((roadPair.road2 == roadPairMin.road1) && 
						((((roadPair.positionIndex & SecondRoadStart) >> 2) == (roadPairMin.positionIndex & FirstRoadStart)) || 
						(((roadPair.positionIndex & SecondRoadEnd) >> 2) == (roadPairMin.positionIndex & FirstRoadEnd))))
					{
						it = distRoadPairs.erase(it);
					}
					else if ((roadPair.road2 == roadPairMin.road2) && 
						(((roadPair.positionIndex >> 2) & (SecondRoadStart >> 2)) == ((roadPairMin.positionIndex >> 2) & (SecondRoadStart >> 2))))
					{

						it = distRoadPairs.erase(it);

					}
					else 
					{
						it++;
					}
				}
			}

			it = distRoadPairs.begin();
		};
		*/

    // Done //
    //
    setValid();
    setText(QObject::tr("Set Road Link"));
}

SetRoadLinkRoadsCommand::SetRoadLinkRoadsCommand(const QList<RoadPair *> &roadPairs, DataCommand *parent)
    : DataCommand(parent)
{

    // Check for validity //
    //
    if (roadPairs.size() < 1)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("Set Road Link Roads: invalid parameters!"));
        return;
    }

    //connect all, except if both roads are inside a junction

    for (int i = 0; i < roadPairs.size(); i++)
    {
        RoadPair *roadPair = roadPairs.at(i); // make Connections

        if (roadPair->road1->getJunction().getID() == -1|| roadPair->road2->getJunction().getID() == -1) // at least one of the roads has to be outside
        {
            findConnections(roadPair->road1, roadPair->road2, roadPair->positionIndex);
        }

        if (!roads_.contains(roadPair->road1))
        {
            roads_.append(roadPair->road1);
        }
        if (!roads_.contains(roadPair->road2))
        {
            roads_.append(roadPair->road2);
        }
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Set Road Link"));
}

SetRoadLinkRoadsCommand::~SetRoadLinkRoadsCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        newRoadLinks_.clear();
    }
    else
    {
        oldRoadLinks_.clear();
    }
}

void
SetRoadLinkRoadsCommand::redo()
{
    if (isUndone())
    {
        foreach (RSystemElementRoad *road, roads_)
        {
            QList<RoadLink *> newRoadLinks = newRoadLinks_.values(road);
            QList<RoadLink::RoadLinkType> roadLinkTypes = roadLinkTypes_.values(road);

            for (int i = 0; i < roadLinkTypes.size(); i++)
            {
                if (roadLinkTypes.at(i) == RoadLink::DRL_PREDECESSOR)
                {
                    road->setPredecessor(newRoadLinks.at(i));
                    /*
                    // Lane Predecessors
                    LaneSection *laneSection = road->getLaneSection(0.0);
                    RSystemElementRoad *linkRoad = road->getRoadSystem()->getRoad(newRoadLinks.at(i)->getElementId());
                    if (linkRoad != NULL) // else road ends at a junction
                    {
                    if (newRoadLinks.at(i)->getContactPoint() == "start")
                    {
                    LaneSection *linkSection = linkRoad->getLaneSection(0.0);
                    QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                    foreach (Lane *lane, laneSection->getLanes())
                    {
                    if (linkSectionLanes.contains(-lane->getId()))
                    {
                    lane->setPredecessor(-lane->getId());
                    linkSectionLanes.value(-lane->getId())->setPredecessor(lane->getId());
                    }
                    }
                    }
                    else
                    {
                    LaneSection *linkSection = linkRoad->getLaneSection(linkRoad->getLength());
                    QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                    foreach (Lane *lane, laneSection->getLanes())
                    {
                    if (linkSectionLanes.contains(lane->getId()))
                    {
                    lane->setPredecessor(lane->getId());
                    linkSectionLanes.value(lane->getId())->setSuccessor(lane->getId());
                    }
                    }
                    }
                    } */
                }

                else
                {
                    road->setSuccessor(newRoadLinks.at(i));
                    // Lane Successors

                    /*          LaneSection *laneSection = road->getLaneSection(road->getLength());
                    RSystemElementRoad *linkRoad = road->getRoadSystem()->getRoad(newRoadLinks.at(i)->getElementId());
                    if (linkRoad != NULL) // else road ends at a junction
                    {
                    if (newRoadLinks.at(i)->getContactPoint() == "start")
                    {
                    LaneSection *linkSection = linkRoad->getLaneSection(0.0);
                    QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                    foreach (Lane *lane, laneSection->getLanes())
                    {
                    if (linkSectionLanes.contains(lane->getId()))
                    {
                    lane->setSuccessor(lane->getId());
                    linkSectionLanes.value(lane->getId())->setPredecessor(lane->getId());
                    }
                    }
                    }
                    else
                    {
                    LaneSection *linkSection = linkRoad->getLaneSection(linkRoad->getLength());
                    QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                    foreach (Lane *lane, laneSection->getLanes())
                    {
                    if (linkSectionLanes.contains(-lane->getId()))
                    {
                    lane->setSuccessor(-lane->getId());
                    linkSectionLanes.value(-lane->getId())->setSuccessor(lane->getId());
                    }
                    }
                    }
                    } */
                }
            }
        }

        foreach (RSystemElementJunction *junction, junctions_)
        {
            QList<JunctionConnection *> junctionConnections = newConnections_.values(junction);
            for (int i = 0; i < junctionConnections.size(); ++i)
            {
                junction->addConnection(junctionConnections.at(i));
            }
        }

        setRedone();
    }
}

void
SetRoadLinkRoadsCommand::undo()
{
    foreach (RSystemElementRoad *road, roads_)
    {
        QList<RoadLink *> oldRoadLinks = oldRoadLinks_.values(road);
        QList<RoadLink::RoadLinkType> roadLinkTypes = roadLinkTypes_.values(road);

        for (int i = 0; i < roadLinkTypes.size(); i++)
        {
            if (roadLinkTypes.at(i) == RoadLink::DRL_PREDECESSOR)
            {
                road->setPredecessor(oldRoadLinks.at(i));
                // Lane Predecessors

      /*          if (oldRoadLinks.at(i))
                {
                    LaneSection *laneSection = road->getLaneSection(0.0);
                    RSystemElementRoad *linkRoad = road->getRoadSystem()->getRoad(oldRoadLinks.at(i)->getElementId());
                    if (linkRoad != NULL) // else road ends at a junction
                    {
                        if (oldRoadLinks.at(i)->getContactPoint() == "start")
                        {
                            LaneSection *linkSection = linkRoad->getLaneSection(0.0);
                            QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                            foreach (Lane *lane, laneSection->getLanes())
                            {
                                if (lane->getId() > 0)
                                {
                                    if (linkSectionLanes.contains(-lane->getId()))
                                    {
                                        lane->setPredecessor(-lane->getId());
                                        linkSectionLanes.value(-lane->getId())->setPredecessor(lane->getId());
                                    }
                                }
                            }
                        }
                        else
                        {
                            LaneSection *linkSection = linkRoad->getLaneSection(linkRoad->getLength());
                            QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                            foreach (Lane *lane, laneSection->getLanes())
                            {
                                if (lane->getId() > 0)
                                {
                                    if (linkSectionLanes.contains(lane->getId()))
                                    {
                                        lane->setPredecessor(lane->getId());
                                        linkSectionLanes.value(lane->getId())->setSuccessor(lane->getId());
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    LaneSection *laneSection = road->getLaneSection(0.0);
                    foreach (Lane *lane, laneSection->getLanes())
                    {
                        lane->setPredecessor(lane->getId());
                    }
                }*/
            }
            else
            {
                road->setSuccessor(oldRoadLinks.at(i));
                // Lane Successors

         /*      if (oldRoadLinks.at(i))
                {
                    LaneSection *laneSection = road->getLaneSection(road->getLength());
                    RSystemElementRoad *linkRoad = road->getRoadSystem()->getRoad(oldRoadLinks.at(i)->getElementId());
                    if (linkRoad != NULL) // else road ends at a junction
                    {
                        if (oldRoadLinks.at(i)->getContactPoint() == "start")
                        {
                            LaneSection *linkSection = linkRoad->getLaneSection(0.0);
                            QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                            foreach (Lane *lane, laneSection->getLanes())
                            {
                                if (lane->getId() < 0)
                                {
                                    if (linkSectionLanes.contains(lane->getId()))
                                    {
                                        lane->setSuccessor(lane->getId());
                                        linkSectionLanes.value(lane->getId())->setPredecessor(lane->getId());
                                    }
                                }
                            }
                        }
                        else
                        {
                            LaneSection *linkSection = linkRoad->getLaneSection(linkRoad->getLength());
                            QMap<int, Lane *> linkSectionLanes = linkSection->getLanes();
                            foreach (Lane *lane, laneSection->getLanes())
                            {
                                if (lane->getId() < 0)
                                {
                                    if (linkSectionLanes.contains(-lane->getId()))
                                    {
                                        lane->setSuccessor(-lane->getId());
                                        linkSectionLanes.value(-lane->getId())->setSuccessor(lane->getId());
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    LaneSection *laneSection = road->getLaneSection(road->getLength());
                    foreach (Lane *lane, laneSection->getLanes())
                    {
                        lane->setSuccessor(lane->getId());
                    }
                } */
            }
        }
    }

    foreach (RSystemElementJunction *junction, junctions_)
    {
        QList<JunctionConnection *> junctionConnections = newConnections_.values(junction);
        for (int i = 0; i < junctionConnections.size(); ++i)
        {
            junction->delConnection(junctionConnections.at(i));
        }
    }

    setUndone();
}

void
SetRoadLinkRoadsCommand::findConnections(RSystemElementRoad *road1, RSystemElementRoad *road2, short int index)
{

    if (road1->getJunction().getID() != -1)
    {
        RSystemElementJunction *junction = road1->getRoadSystem()->getJunction(road1->getJunction());
        findPathContactPoint(road1, road2, index, junction);
    }
    else if (road2->getJunction().getID() != -1)
    {
        RSystemElementJunction *junction = road2->getRoadSystem()->getJunction(road2->getJunction());
        findPathContactPoint(road1, road2, index, junction);
    }
    else
    {
        findRoadContactPoints(road1, road2, index);
    }
}

void
SetRoadLinkRoadsCommand::findRoadContactPoints(RSystemElementRoad *road1, RSystemElementRoad *road2, unsigned int posIndex)
{
    // Generate up to four RoadLinks

    switch (posIndex)
    {
    case FirstRoadStart | SecondRoadStart:
    {
        newRoadLinks_.insert(road1, new RoadLink("road", road2->getID(), JunctionConnection::JCP_START));
        oldRoadLinks_.insert(road1, road1->getPredecessor());
        roadLinkTypes_.insert(road1, RoadLink::DRL_PREDECESSOR);
        newRoadLinks_.insert(road2, new RoadLink("road", road1->getID(), JunctionConnection::JCP_START));
        oldRoadLinks_.insert(road2, road2->getPredecessor());
        roadLinkTypes_.insert(road2, RoadLink::DRL_PREDECESSOR);
    }
    break;
    case FirstRoadEnd | SecondRoadStart:
    {
        newRoadLinks_.insert(road1, new RoadLink("road", road2->getID(), JunctionConnection::JCP_START));
        oldRoadLinks_.insert(road1, road1->getSuccessor());
        roadLinkTypes_.insert(road1, RoadLink::DRL_SUCCESSOR);
        newRoadLinks_.insert(road2, new RoadLink("road", road1->getID(), JunctionConnection::JCP_END));
        oldRoadLinks_.insert(road2, road2->getPredecessor());
        roadLinkTypes_.insert(road2, RoadLink::DRL_PREDECESSOR);
    }
    break;
    case FirstRoadEnd | SecondRoadEnd:
    {
        newRoadLinks_.insert(road1, new RoadLink("road", road2->getID(), JunctionConnection::JCP_END));
        oldRoadLinks_.insert(road1, road1->getSuccessor());
        roadLinkTypes_.insert(road1, RoadLink::DRL_SUCCESSOR);
        newRoadLinks_.insert(road2, new RoadLink("road", road1->getID(), JunctionConnection::JCP_END));
        oldRoadLinks_.insert(road2, road2->getSuccessor());
        roadLinkTypes_.insert(road2, RoadLink::DRL_SUCCESSOR);
    }
    break;
    case FirstRoadStart | SecondRoadEnd:
    {
        newRoadLinks_.insert(road1, new RoadLink("road", road2->getID(), JunctionConnection::JCP_END));
        oldRoadLinks_.insert(road1, road1->getPredecessor());
        roadLinkTypes_.insert(road1, RoadLink::DRL_PREDECESSOR);
        newRoadLinks_.insert(road2, new RoadLink("road", road1->getID(), JunctionConnection::JCP_START));
        oldRoadLinks_.insert(road2, road2->getSuccessor());
        roadLinkTypes_.insert(road2, RoadLink::DRL_SUCCESSOR);
    }
    break;
    }
}

void
SetRoadLinkRoadsCommand::findPathContactPoint(RSystemElementRoad *road1, RSystemElementRoad *road2, unsigned int posIndex, RSystemElementJunction *junction)
{
    // Generate two RoadLinks
    // road1 is part of a junction

    JunctionConnection::ContactPointValue contactPoint;
    RoadLink::RoadLinkType roadLinkType;

    switch (posIndex)
    {
    case FirstRoadStart | SecondRoadStart:
    {
        newRoadLinks_.insert(road1, new RoadLink("road", road2->getID(), JunctionConnection::JCP_START));
        roadLinkTypes_.insert(road1, RoadLink::DRL_PREDECESSOR);
        oldRoadLinks_.insert(road1, road1->getPredecessor());
        roadLinkType = RoadLink::DRL_PREDECESSOR;
        oldRoadLinks_.insert(road2, road2->getPredecessor());
        contactPoint = JunctionConnection::JCP_START;
    }
    break;
    case FirstRoadEnd | SecondRoadStart:
    {
        newRoadLinks_.insert(road1, new RoadLink("road", road2->getID(), JunctionConnection::JCP_START));
        oldRoadLinks_.insert(road1, road1->getSuccessor());
        roadLinkTypes_.insert(road1, RoadLink::DRL_SUCCESSOR);
        oldRoadLinks_.insert(road2, road2->getPredecessor());
        roadLinkType = RoadLink::DRL_PREDECESSOR;
        contactPoint = JunctionConnection::JCP_END;
    }
    break;
    case FirstRoadEnd | SecondRoadEnd:
    {
        newRoadLinks_.insert(road1, new RoadLink("road", road2->getID(), JunctionConnection::JCP_END));
        oldRoadLinks_.insert(road1, road1->getSuccessor());
        roadLinkTypes_.insert(road1, RoadLink::DRL_SUCCESSOR);
        oldRoadLinks_.insert(road2, road2->getSuccessor());
        roadLinkType = RoadLink::DRL_SUCCESSOR;
        contactPoint = JunctionConnection::JCP_END;
    }
    break;
    case FirstRoadStart | SecondRoadEnd:
    {
        newRoadLinks_.insert(road1, new RoadLink("road", road2->getID(), JunctionConnection::JCP_END));
        roadLinkTypes_.insert(road1, RoadLink::DRL_PREDECESSOR);
        oldRoadLinks_.insert(road1, road1->getPredecessor());
        roadLinkType = RoadLink::DRL_SUCCESSOR;
        oldRoadLinks_.insert(road2, road2->getSuccessor());
        contactPoint = JunctionConnection::JCP_START;
    }
    break;
    }

    JunctionConnection *newConnection = new JunctionConnection(QString("0"), road2->getID(), road1->getID(), contactPoint, 1);

/*    QMap<double, LaneSection *> lanes = road1->getLaneSections();
    LaneSection *lsC = *lanes.begin();
    bool even = false;
    if (contactPoint == "start")
    {
        even = true;
        lsC = *lanes.begin();
    }
    else
    {
        even = false;
        QMap<double, LaneSection *>::iterator it = lanes.end();
        it--;
        lsC = *(it);
    }
    if (roadLinkType == RoadLink::DRL_PREDECESSOR)
    {
        if (even)
        {
            foreach (Lane *lane, lsC->getLanes())
            {
                newConnection->addLaneLink(lane->getId(), -lane->getId());
            }
        }
        else
        {
            foreach (Lane *lane, lsC->getLanes())
            {
                newConnection->addLaneLink(lane->getId(), lane->getId());
            }
        }
    }
    else
    {
        if (even)
        {
            foreach (Lane *lane, lsC->getLanes())
            {
                newConnection->addLaneLink(lane->getId(), lane->getId());
            }
        }
        else
        {
            foreach (Lane *lane, lsC->getLanes())
            {
                newConnection->addLaneLink(lane->getId(), -lane->getId());
            }
        }
    }*/

    newRoadLinks_.insert(road2, new RoadLink("junction", road1->getJunction(), contactPoint));
    roadLinkTypes_.insert(road2, roadLinkType);

 //   junction->addConnection(newConnection);
    newConnections_.insert(junction, newConnection);

    if (!junctions_.contains(junction))
    {
        junctions_.append(junction);
    }
}

void
SetRoadLinkRoadsCommand::distanceRoad(RSystemElementRoad *road, double threshold, QMultiMap<double, RoadPair> *distRoadPairs)
{
    // Check if the road's start and end are closer than the threshold
    double lineLength;

    lineLength = QVector2D(road->getGlobalPoint(0.0) - road->getGlobalPoint(road->getLength())).length();

    if (lineLength <= threshold)
    {
        RoadPair roadPair = { road, road, FirstRoadStart | SecondRoadEnd };
        distRoadPairs->insert(lineLength, roadPair);
    }
}

void
SetRoadLinkRoadsCommand::distanceRoads(RSystemElementRoad *road1, RSystemElementRoad *road2, double threshold, QMultiMap<double, RoadPair> *distRoadPairs)
{
    // Find closest positions of the two roads
    double lineLength[4];

    lineLength[0] = QVector2D(road1->getGlobalPoint(0.0) - road2->getGlobalPoint(0.0)).length();
    if (lineLength[0] - road1->getLength() - road2->getLength() > threshold) // If roads are further apart than their added length
    { // plus the threshold they are out of focus
        return;
    }

    lineLength[1] = QVector2D(road1->getGlobalPoint(road1->getLength()) - road2->getGlobalPoint(0.0)).length();
    lineLength[2] = QVector2D(road1->getGlobalPoint(road1->getLength()) - road2->getGlobalPoint(road2->getLength())).length();
    lineLength[3] = QVector2D(road1->getGlobalPoint(0.0) - road2->getGlobalPoint(road2->getLength())).length();

    // two distances less the threshold
    short int k = 0;
    for (; k < 4; k++)
    {
        if (lineLength[k] <= threshold)
        {
            break;
        }
    }

    if (k >= 4)
    {
        return;
    }

    short int min[2] = { k, k };

    for (++k; k < 4; k++)
    {
        if (lineLength[k] <= threshold)
        {
            if (lineLength[k] < lineLength[min[0]])
            {
                min[1] = min[0];
                min[0] = k;
            }
            else if ((min[0] != min[1]) && (lineLength[k] < lineLength[min[1]]))
            {
                min[1] = k;
            }
            else
            {
                min[1] = k;
            }
        }
    }

    short int roadPosition[4] = { FirstRoadStart | SecondRoadStart, FirstRoadEnd | SecondRoadStart, FirstRoadEnd | SecondRoadEnd, FirstRoadStart | SecondRoadEnd };

    for (int k = 0; k < 2; k++)
    {
        RoadPair roadPair = { road1, road2, roadPosition[min[k]] };
        distRoadPairs->insert(lineLength[min[k]], roadPair);

        if (min[k + 1] == min[k])
        {
            break;
        }
    }
}

//#########################//
// RemoveRoadLinkCommand //
//#########################//

RemoveRoadLinkCommand::RemoveRoadLinkCommand(RSystemElementRoad *road, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , predecessor_(NULL)
    , successor_(NULL)
    , predecessorLink_(NULL)
    , successorLink_(NULL)
    , junction_(NULL)
{
    // Check for validity //
    //
     if (!road_ || !road_->getRoadSystem())
    {
        setInvalid(); // Invalid
        setText(QObject::tr("RemoveRoadLinkCommand: Internal error! No road specified."));
        return;
    }
    else
    {
        predecessor_ = road_->getPredecessor();
        successor_ = road_->getSuccessor();
        roadSystem_ = road_->getRoadSystem();

        if (road_->getJunction().getID() != -1)
        {
            junction_ = roadSystem_->getJunction(road_->getJunction());
            if (junction_)
            {
                junctionConnections_ = junction_->getConnectingRoadConnections(road_->getID());
            }
        }

        LaneSection *laneSection = road_->getLaneSection(0.0);
        QMap<int, int> laneMapPredecessor;
        foreach (Lane *lane, laneSection->getLanes())
        {
            const int laneId = lane->getId();
            const int predecessorId = lane->getPredecessor();
            if (predecessorId != Lane::NOLANE)
            {
                laneMapPredecessor.insert(laneId, predecessorId);
            }
        }
        if (laneMapPredecessor.count() > 0)
        {
            laneLinksRoadStart_.insert(road_->getID(), laneMapPredecessor);
        }

        if (predecessor_)
        {
            if (predecessor_->getElementType() == "junction")
            {
                RSystemElementJunction *junction = roadSystem_->getJunction(predecessor_->getElementId());
                if (junction)
                {
                    saveConnectingRoadLanes(junction);
                }
            }
            else
            {
                RSystemElementRoad *linkRoad = roadSystem_->getRoad(predecessor_->getElementId());

                if (linkRoad)
                {
                    if (predecessor_->getContactPoint() == JunctionConnection::JCP_START)
                    {
                        predecessorLink_ = linkRoad->getPredecessor();
                    }
                    else
                    {
                        predecessorLink_ = linkRoad->getSuccessor();
                    }
                }
            }
        }

        laneSection = road_->getLaneSection(road_->getLength());
        QMap<int, int> laneMapSuccessor;
        foreach (Lane *lane, laneSection->getLanes())
        {
            const int laneId = lane->getId();
            const int successorId = lane->getSuccessor();
            if (successorId != Lane::NOLANE)
            {
                laneMapSuccessor.insert(laneId, successorId);
            }
        }
        if (laneMapSuccessor.count() > 0)
        {
            laneLinksRoadEnd_.insert(road_->getID(), laneMapSuccessor);
        }

        if (successor_)
        {
            if (successor_->getElementType() == "junction")
            {
                RSystemElementJunction *junction = roadSystem_->getJunction(successor_->getElementId());
                if (junction)
                {
                    saveConnectingRoadLanes(junction);
                }
            }
            else
            {
                RSystemElementRoad *linkRoad = roadSystem_->getRoad(successor_->getElementId());
                if (linkRoad)
                {
                    if (successor_->getContactPoint() == JunctionConnection::JCP_START)
                    {
                        successorLink_ = linkRoad->getPredecessor();
                    }
                    else
                    {
                        successorLink_ = linkRoad->getSuccessor();
                    }
                }
            }
        }

        setValid();
        setText(QObject::tr("Remove Links"));
    }
}

/*! \brief .
*
*/
RemoveRoadLinkCommand::~RemoveRoadLinkCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (road is still owned by the roadsystem)
    }
    else
    {
        junctionConnections_.clear();
        laneLinksRoadStart_.clear();
        laneLinksRoadEnd_.clear();
    }
}

/*! \brief .
*
*/
void
RemoveRoadLinkCommand::redo()
{
    foreach (JunctionConnection *connection, junctionConnections_)
    {
        RSystemElementJunction * junction = connection->getParentJunction();
        if (junction != junction_)
        {
            RSystemElementRoad * connectingRoad = roadSystem_->getRoad(connection->getConnectingRoad());
            if (connectingRoad)
            {
                removeLinkRoadLink(connectingRoad, connection->getContactPoint());
            }
        }
        junction->delConnection(connection);
    }

    if (predecessor_)
    {
        road_->delPredecessor();
    }

    if (successor_)
    {
        road_->delSuccessor();
    }


    if (predecessor_)
    {
        RSystemElementRoad *linkRoad = roadSystem_->getRoad(predecessor_->getElementId());
        if (linkRoad)
        {
            removeLinkRoadLink(linkRoad, predecessor_->getContactPoint());
        }
    }

    LaneSection *laneSection = road_->getLaneSection(0.0);
    foreach (Lane *lane, laneSection->getLanes())
    {
        lane->setPredecessor(Lane::NOLANE);
    }


    if (successor_)
    {
        RSystemElementRoad *linkRoad = roadSystem_->getRoad(successor_->getElementId());

        if (linkRoad)
        {
            removeLinkRoadLink(linkRoad, successor_->getContactPoint());
        }
    }

    laneSection = road_->getLaneSection(road_->getLength());
    foreach (Lane *lane, laneSection->getLanes())
    {
        lane->setSuccessor(Lane::NOLANE);
    }


    setRedone();
}

/*! \brief
*
*/
void
RemoveRoadLinkCommand::undo()
{
    road_->setPredecessor(predecessor_);
    road_->setSuccessor(successor_);

    if (junction_)
    {
        foreach (JunctionConnection *connection, junctionConnections_)
        {
            junction_->addConnection(connection);
        }
    }
    else
    {

        foreach (JunctionConnection *connection, junctionConnections_)
        {
            odrID junctionID = roadSystem_->getRoad(connection->getConnectingRoad())->getJunction();

            RSystemElementJunction * junction = roadSystem_->getJunction(junctionID);
            if (junction)
            {
                junction->addConnection(connection);
                odrID connectingRoadID = connection->getConnectingRoad();
                RSystemElementRoad * connectingRoad = roadSystem_->getRoad(connectingRoadID);
                if (connectingRoad)
                {
                    RoadLink * roadLink;
                    if (predecessor_ && (predecessor_->getElementId() == junctionID))
                    {
                        roadLink = new RoadLink("road", road_->getID(), JunctionConnection::JCP_START);
                        if (connection->getContactPoint() == JunctionConnection::JCP_START)
                        {
                            connectingRoad->setPredecessor(roadLink);
                        }
                        else
                        {
                            connectingRoad->setSuccessor(roadLink);
                        }


                        LaneSection *linkSection = connectingRoad->getLaneSection(0.0);

                        QMap<int,int> connectingLanes = laneLinksRoadStart_.value(connectingRoadID);
                        QMap<int,int>::ConstIterator it = connectingLanes.constBegin();
                        while (it != connectingLanes.constEnd())
                        {
                            linkSection->getLane(it.value())->setPredecessor(it.key());
                            it++;
                        }
                    }
                    else
                    {
                        roadLink = new RoadLink("road", road_->getID(), JunctionConnection::JCP_END);
                        if (connection->getContactPoint() == JunctionConnection::JCP_START)
                        {
                            connectingRoad->setPredecessor(roadLink);
                        }
                        else
                        {
                            connectingRoad->setSuccessor(roadLink);
                        }

                        LaneSection *linkSection = connectingRoad->getLaneSection(connectingRoad->getLength());

                        QMap<int,int> connectingLanes = laneLinksRoadStart_.value(connectingRoadID);
                        QMap<int,int>::ConstIterator it = connectingLanes.constBegin();
                        while (it != connectingLanes.constEnd())
                        {
                            linkSection->getLane(it.value())->setSuccessor(it.key());
                            it++;
                        }
                    }
                }
            }
        }
    }

    if (predecessor_)
    {
        LaneSection *laneSection = road_->getLaneSection(0.0);       
        RSystemElementRoad *linkRoad = roadSystem_->getRoad(predecessor_->getElementId());

        if (linkRoad)
        {
            if (predecessor_->getContactPoint() == JunctionConnection::JCP_START)
            {
                linkRoad->setPredecessor(predecessorLink_);
                LaneSection *linkSection = linkRoad->getLaneSection(0.0);
                
                QMap<int,int>::ConstIterator it = laneLinksRoadStart_.value(road_->getID()).constBegin();
                while (it != laneLinksRoadStart_.value(road_->getID()).constEnd())
                {
                    laneSection->getLane(it.key())->setPredecessor(it.value());
                    linkSection->getLane(it.value())->setPredecessor(it.key());
                    it++;
                }
            }
            else
            {
                linkRoad->setSuccessor(predecessorLink_);
                LaneSection *linkSection = linkRoad->getLaneSection(linkRoad->getLength());
                QMap<int,int>::ConstIterator it = laneLinksRoadStart_.value(road_->getID()).constBegin();
                while (it != laneLinksRoadStart_.value(road_->getID()).constEnd())
                {
                    laneSection->getLane(it.key())->setPredecessor(it.value());
                    linkSection->getLane(it.value())->setSuccessor(it.key());
                    it++;
                }
            }
        }
    }

    if (successor_)
    {
        LaneSection *laneSection = road_->getLaneSection(road_->getLength());
        RSystemElementRoad *linkRoad = roadSystem_->getRoad(successor_->getElementId());

        if (linkRoad)
        {
            if (successor_->getContactPoint() == JunctionConnection::JCP_START)
            {
                linkRoad->setPredecessor(successorLink_);
                LaneSection *linkSection = linkRoad->getLaneSection(0.0);

                QMap<int,int>::ConstIterator it = laneLinksRoadEnd_.value(road_->getID()).constBegin();
                while (it != laneLinksRoadEnd_.value(road_->getID()).constEnd())
                {
                    laneSection->getLane(it.key())->setSuccessor(it.value());
                    linkSection->getLane(it.value())->setPredecessor(it.key());
                    it++;
                }
            }
            else
            {
                linkRoad->setSuccessor(successorLink_);
                LaneSection *linkSection = linkRoad->getLaneSection(linkRoad->getLength());
                QMap<int,int>::ConstIterator it = laneLinksRoadEnd_.value(road_->getID()).constBegin();
                while (it != laneLinksRoadEnd_.value(road_->getID()).constEnd())
                {
                    laneSection->getLane(it.key())->setSuccessor(it.value());
                    linkSection->getLane(it.value())->setSuccessor(it.key());
                    it++;
                }
            }
        }
    }

    setUndone();
}

void
    RemoveRoadLinkCommand::saveConnectingRoadLanes(RSystemElementJunction * junction)
{
    QList<JunctionConnection *> connections = junction->getConnections(road_->getID());
    foreach (JunctionConnection * connection, connections)
    {

        RSystemElementRoad * connectingRoad = roadSystem_->getRoad(connection->getConnectingRoad());
        QMap<int, int> connectingRoadLaneMap;
        if (connection->getContactPoint() == JunctionConnection::JCP_START)
        {
            connectingRoad->delPredecessor();

            LaneSection *laneSection = connectingRoad->getLaneSection(0.0);
            foreach (Lane *lane, laneSection->getLanes())
            {
                const int laneId = lane->getId();
                const int predecessorId = lane->getPredecessor();
                if (predecessorId != Lane::NOLANE)
                {
                    connectingRoadLaneMap.insert(laneId, predecessorId);
                }
            }

            if (connectingRoadLaneMap.count() > 0)
            {
                laneLinksRoadStart_.insert(connection->getConnectingRoad(), connectingRoadLaneMap);
            }
        }
        else
        {
            connectingRoad->delSuccessor();

            LaneSection *laneSection = connectingRoad->getLaneSection(connectingRoad->getLength());
            foreach (Lane *lane, laneSection->getLanes())
            {
                const int laneId = lane->getId();
                const int successorId = lane->getSuccessor();
                if (successorId != Lane::NOLANE)
                {
                    connectingRoadLaneMap.insert(laneId, successorId);
                }
            }
            if (connectingRoadLaneMap.count() > 0)
            {
                laneLinksRoadEnd_.insert(connection->getConnectingRoad(), connectingRoadLaneMap);
            }
        }
    }
    junctionConnections_.append(connections);
}

void 
RemoveRoadLinkCommand::removeLinkRoadLink(RSystemElementRoad * linkRoad, JunctionConnection::ContactPointValue contactPoint)
{
    LaneSection *linkSection;
    int connectionCount = 0;
    if (junction_)
    {
        connectionCount = junction_->getConnections(linkRoad->getID()).count();
    }

    if ((contactPoint == JunctionConnection::JCP_START) && linkRoad->getPredecessor() && ((linkRoad->getPredecessor()->getElementType() != "junction") || (connectionCount < 1)))
    {
        linkRoad->delPredecessor();
        linkSection = linkRoad->getLaneSection(0.0);
        foreach (Lane * lane, linkSection->getLanes())
        {
            lane->setPredecessor(Lane::NOLANE);
        }
    }
    else if ((contactPoint == JunctionConnection::JCP_END) && linkRoad->getSuccessor() && ((linkRoad->getSuccessor()->getElementType() != "junction") || (connectionCount < 1)))
    {
        linkRoad->delSuccessor();
        linkSection = linkRoad->getLaneSection(linkRoad->getLength());
        foreach (Lane * lane, linkSection->getLanes())
        {
            lane->setSuccessor(Lane::NOLANE);
        }
    }
}

//#########################//
// CreateInnerLaneLinksCommand //
//#########################//

CreateInnerLaneLinksCommand::CreateInnerLaneLinksCommand(RSystemElementRoad *road, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
{
    // Check for validity //
    //
    if (!road_ || !road_->getRoadSystem())
    {
        setInvalid(); // Invalid
        setText(QObject::tr("CreateInnerLaneLinksCommand: Internal error! No road specified."));
        return;
    }

    QMap<double, LaneSection *>::const_iterator it = road_->getLaneSections().constBegin();
    while (it != road_->getLaneSections().constEnd() - 1)
    {
        LaneSection * laneSection = it.value();
        double s = laneSection->getSStart();
        LaneSection * nextLaneSection = road_->getLaneSectionNext(s);
        LaneSection * lastLaneSection = road_->getLaneSectionBefore(s);

        double nextS = nextLaneSection->getSStart();

        foreach (Lane * lane, laneSection->getLanes())
        {
            int id = lane->getId();
            int nextLaneId = lane->getPredecessor();
            double laneWidth = laneSection->getLaneWidth(id, 0.0);
            if (((laneWidth < NUMERICAL_ZERO3) || !lastLaneSection) && (nextLaneId != Lane::NOLANE))
            {
                LaneLinkPair oldLaneLinkPair = {id, nextLaneId};
                oldPredecessorLaneLinks_.insert(s, oldLaneLinkPair);
                LaneLinkPair newLaneLinkPair = {id, Lane::NOLANE};
                newPredecessorLaneLinks_.insert(s, newLaneLinkPair); 
            }

            nextLaneId = lane->getSuccessor();
            laneWidth = laneSection->getLaneWidth(id, laneSection->getSEnd());
            if ((laneWidth < NUMERICAL_ZERO3) || !nextLaneSection)
            {
                if (nextLaneId != Lane::NOLANE)
                {
                    LaneLinkPair oldLaneLinkPair = {id, nextLaneId};
                    oldSuccessorLaneLinks_.insert(s, oldLaneLinkPair);
                    LaneLinkPair newLaneLinkPair = {id, Lane::NOLANE};
                    newSuccessorLaneLinks_.insert(s, newLaneLinkPair); 
                }
                continue;
            }
            double t = laneSection->getTValue(lane, laneSection->getSEnd(), laneWidth);

            int laneId = nextLaneSection->getLaneId(0, t);
            if (laneId != nextLaneId)
            {
                LaneLinkPair oldLaneLinkPair = {id, nextLaneId};
                oldSuccessorLaneLinks_.insert(s, oldLaneLinkPair);
                LaneLinkPair newLaneLinkPair = {id, laneId};
                newSuccessorLaneLinks_.insert(s, newLaneLinkPair);
            }

            int nextLanePredecessor = nextLaneSection->getLane(laneId)->getPredecessor();
            if (nextLanePredecessor != id)
            {
                LaneLinkPair oldNextLaneLinkPair = {laneId, nextLanePredecessor};
                oldPredecessorLaneLinks_.insert(nextS, oldNextLaneLinkPair);
                LaneLinkPair newNextLaneLinkPair = {laneId, id};
                newPredecessorLaneLinks_.insert(nextS, newNextLaneLinkPair);
            }

        }
        it++;
    }

    LaneSection * laneSection = road_->getLaneSection(road_->getLength());
    double s = laneSection->getSStart();
    LaneSection * lastLaneSection = road_->getLaneSectionBefore(s);

    foreach (Lane * lane, laneSection->getLanes())
    {
        int id = lane->getId();
        int nextLaneId = lane->getPredecessor();
        double laneWidth = laneSection->getLaneWidth(id, 0.0);
        if (((laneWidth < NUMERICAL_ZERO3) || !lastLaneSection) && (nextLaneId != Lane::NOLANE))
        {
            LaneLinkPair oldLaneLinkPair = {id, nextLaneId};
            oldPredecessorLaneLinks_.insert(s, oldLaneLinkPair);
            LaneLinkPair newLaneLinkPair = {id, Lane::NOLANE};
            newPredecessorLaneLinks_.insert(s, newLaneLinkPair); 
        }

        nextLaneId = lane->getSuccessor();
        if ((nextLaneId != Lane::NOLANE) && !road_->getSuccessor())
        {
            LaneLinkPair oldLaneLinkPair = {id, nextLaneId};
            oldSuccessorLaneLinks_.insert(s, oldLaneLinkPair);
            LaneLinkPair newLaneLinkPair = {id, Lane::NOLANE};
            newSuccessorLaneLinks_.insert(s, newLaneLinkPair); 
        }
    }


    setValid();
    setText(QObject::tr("Create Links between lane sections"));
}

/*! \brief .
*
*/
CreateInnerLaneLinksCommand::~CreateInnerLaneLinksCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (road is still owned by the roadsystem)
    }
    else
    {
        oldSuccessorLaneLinks_.clear();
        newSuccessorLaneLinks_.clear();
        oldPredecessorLaneLinks_.clear();
        newPredecessorLaneLinks_.clear();
    }
}

/*! \brief .
*
*/
void
CreateInnerLaneLinksCommand::redo()
{
    QMultiMap<double, LaneLinkPair>::const_iterator it = newSuccessorLaneLinks_.constBegin();
    while (it != newSuccessorLaneLinks_.constEnd())
    {
        double s = it.key();
        LaneLinkPair laneLinkPair = it.value();
        LaneSection * laneSection = road_->getLaneSection(s);
        laneSection->getLane(laneLinkPair.laneId)->setSuccessor(laneLinkPair.linkId);

        it++;
    }

    it = newPredecessorLaneLinks_.constBegin();
    while (it != newPredecessorLaneLinks_.constEnd())
    {
        double s = it.key();
        LaneLinkPair laneLinkPair = it.value();
        LaneSection * laneSection = road_->getLaneSection(s);
        laneSection->getLane(laneLinkPair.laneId)->setPredecessor(laneLinkPair.linkId);

        it++;
    }

    setRedone();
}

/*! \brief
*
*/
void
CreateInnerLaneLinksCommand::undo()
{
    QMultiMap<double, LaneLinkPair>::ConstIterator it = oldSuccessorLaneLinks_.constBegin();
    while (it != oldSuccessorLaneLinks_.constEnd())
    {
        double s = it.key();
        LaneLinkPair laneLinkPair = it.value();
        LaneSection * laneSection = road_->getLaneSection(s);
        laneSection->getLane(laneLinkPair.laneId)->setSuccessor(laneLinkPair.linkId);

        it++;
    }
    it = oldPredecessorLaneLinks_.constBegin();
    while (it != oldPredecessorLaneLinks_.constEnd())
    {
        double s = it.key();
        LaneLinkPair laneLinkPair = it.value();
        LaneSection * laneSection = road_->getLaneSection(s);
        laneSection->getLane(laneLinkPair.laneId)->setPredecessor(laneLinkPair.linkId);

        it++;
    }

    setUndone();
}

//#########################//
// CreateNextRoadLaneLinksCommand //
//#########################//

CreateNextRoadLaneLinksCommand::CreateNextRoadLaneLinksCommand(RoadSystem *roadSystem, RSystemElementRoad *road, DataCommand *parent)
    : DataCommand(parent)
    , roadSystem_(roadSystem)
    , road_(road)
    , junction_(NULL)
    , junctionPredecessorConnection_(NULL)
    , junctionSuccessorConnection_(NULL)
{
    // Check for validity //
    //
    if (!road_ || !roadSystem_)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("CreateNextRoadLaneLinksCommand: Internal error! No road specified."));
        return;
    }

    RoadLink * roadPredecessor = road_->getPredecessor();
    RoadLink * roadSuccessor = road_->getSuccessor();        // Successor

    if (road_->getJunction().isValid())
    {
        junction_ = roadSystem_->getJunction(road->getJunction());

        if (roadPredecessor)
        {
            junctionPredecessorConnection_ = junction_->getConnection(roadPredecessor->getElementId(), road->getID());
        }
        if (roadSuccessor)
        {
            junctionSuccessorConnection_ = junction_->getConnection(roadSuccessor->getElementId(), road->getID());
        }

    }

    // Predecessor
    LaneSection * laneSection = road_->getLaneSection(0.0);
    if (!roadPredecessor || (roadPredecessor->getElementType() == "junction"))
    {
        foreach (Lane * lane, laneSection->getLanes())
        {
            int id = lane->getId();
            double laneWidth = laneSection->getLaneWidth(id, 0.0);
            if ((lane->getPredecessor() != Lane::NOLANE) || (laneWidth < NUMERICAL_ZERO3))
            {
                LaneLinkPair laneLinkPairOld = { id, lane->getPredecessor()};
                oldPredecessorLaneLinks_.insert(road->getID(), laneLinkPairOld);
                LaneLinkPair laneLinkPairNew = { id, Lane::NOLANE};
                newPredecessorLaneLinks_.insert(road->getID(), laneLinkPairNew);
            }
        }
    }
    else
    {
        RSystemElementRoad * predRoad = roadSystem_->getRoad(roadPredecessor->getElementId());
        if (predRoad)
        {
            QPointF laneMid = road_->getGlobalPoint(0.0, 0.0);
            QVector2D vecStart = QVector2D(laneMid - predRoad->getGlobalPoint(0.0));
            QVector2D vecEnd = QVector2D(laneMid - predRoad->getGlobalPoint(predRoad->getLength()));

            double nextStart;
            bool start = true;
            QMultiMap<odrID, LaneLinkPair> oldLaneLinks;
            QMultiMap<odrID, LaneLinkPair> newLaneLinks;
            if (vecStart.length() > vecEnd.length())
            {
                start = false;
                nextStart = predRoad->getLength();
                oldLaneLinks = oldSuccessorLaneLinks_;
                newLaneLinks = newSuccessorLaneLinks_;
            }
            else
            {
                nextStart = 0.0;
                oldLaneLinks = oldPredecessorLaneLinks_;
                newLaneLinks = newPredecessorLaneLinks_;
            }
            LaneSection * nextLaneSection = predRoad->getLaneSection(nextStart);

            foreach (Lane * lane, laneSection->getLanes())
            {
                int id = lane->getId();
                int nextLaneId = lane->getPredecessor();
                double laneWidth = laneSection->getLaneWidth(id, 0.0);
                if (laneWidth < NUMERICAL_ZERO3) 
                {
                    if (nextLaneId != Lane::NOLANE)
                    {
                        LaneLinkPair laneLinkPairOld = { id, nextLaneId};
                        oldPredecessorLaneLinks_.insert(road->getID(), laneLinkPairOld);
                        LaneLinkPair laneLinkPairNew = {nextLaneId, Lane::NOLANE};
                        newPredecessorLaneLinks_.insert(road->getID(), laneLinkPairNew);
                    }
                }
                else // find the connecting lane 
                {
                    double t = laneSection->getTValue(lane, 0.0, laneWidth);
                    QVector2D vecT = QVector2D(road_->getGlobalPoint(0.0, t) - predRoad->getGlobalPoint(nextStart));
                    float dist = vecT.length();
                    float dotVec = QVector2D::dotProduct(predRoad->getGlobalNormal(nextStart), vecT.normalized());

                    if (dist < NUMERICAL_ZERO3)
                    {
                        if (id > 0)
                        {
                            dist = -dist;
                        }
                    }
                    else if (dotVec > 0)
                    {
                        dist = -dist;
                    }

                    int laneId = nextLaneSection->getLaneId(nextStart, dist);

                    if (laneId != nextLaneId)
                    {
                        if (laneId == 0)
                        {
                            LaneLinkPair laneLinkPairOld = { id, nextLaneId};
                            oldPredecessorLaneLinks_.insert(road->getID(), laneLinkPairOld);
                            LaneLinkPair laneLinkPairNew = {id, Lane::NOLANE};
                            newPredecessorLaneLinks_.insert(road->getID(), laneLinkPairNew);
                            continue;
                        }

                        LaneLinkPair laneLinkPairOld = { id, nextLaneId};
                        oldPredecessorLaneLinks_.insert(road->getID(), laneLinkPairOld);
                        LaneLinkPair laneLinkPairNew = {id, laneId};
                        newPredecessorLaneLinks_.insert(road->getID(), laneLinkPairNew);

                        odrID predRoadId = predRoad->getID();
                        LaneLinkPair laneLinkNextOld = { laneId, nextLaneSection->getLane(laneId)->getPredecessor()};
                        oldLaneLinks.insert(predRoadId, laneLinkNextOld);
                        if (!junctionPredecessorConnection_)
                        {
                            LaneLinkPair laneLinkPairNext = {laneId, id};
                            newLaneLinks.insert(predRoadId, laneLinkPairNext);

                        }
                        else
                        {
                            LaneLinkPair laneLinkPairNext = {laneId, Lane::NOLANE};
                            newLaneLinks.insert(predRoadId, laneLinkPairNext);
                        }
                    }
                }
            }
        }
    }

    // Successor
    double roadLength = road_->getLength();
    laneSection = road_->getLaneSection(roadLength);
    if (!roadSuccessor || (roadSuccessor->getElementType() == "junction"))
    {
        foreach (Lane * lane, laneSection->getLanes())
        {
            int id = lane->getId();
            double laneWidth = laneSection->getLaneWidth(id, roadLength);
            if ((lane->getSuccessor() != Lane::NOLANE) || (laneWidth < NUMERICAL_ZERO3))
            {
                LaneLinkPair laneLinkPairOld = { id, lane->getSuccessor()};
                oldSuccessorLaneLinks_.insert(road->getID(), laneLinkPairOld);
                LaneLinkPair laneLinkPairNew = { id, Lane::NOLANE};
                newSuccessorLaneLinks_.insert(road->getID(), laneLinkPairNew);
            }
        }
    }
    else
    {
        RSystemElementRoad * succRoad = roadSystem_->getRoad(roadSuccessor->getElementId());
        if (succRoad)
        {
            QPointF laneMid = road->getGlobalPoint(roadLength, 0.0);
            QVector2D vecStart = QVector2D(laneMid - succRoad->getGlobalPoint(0.0));
            QVector2D vecEnd = QVector2D(laneMid - succRoad->getGlobalPoint(succRoad->getLength()));

            bool start = true;
            double nextStart;
            QMultiMap<odrID, LaneLinkPair> oldLaneLinks;
            QMultiMap<odrID, LaneLinkPair> newLaneLinks;
            if (vecStart.length() > vecEnd.length())
            {
                start = false;
                nextStart = succRoad->getLength();
                oldLaneLinks = oldSuccessorLaneLinks_;
                newLaneLinks = newSuccessorLaneLinks_;
            }
            else
            {
                nextStart = 0.0;
                oldLaneLinks = oldPredecessorLaneLinks_;
                newLaneLinks = newPredecessorLaneLinks_;
            }
            LaneSection * nextLaneSection = succRoad->getLaneSection(nextStart);

            foreach (Lane * lane, laneSection->getLanes())
            {
                int id = lane->getId();
                int nextLaneId = lane->getSuccessor();
                double laneWidth = laneSection->getLaneWidth(id, roadLength);
                if (laneWidth < NUMERICAL_ZERO3) 
                {
                    if (nextLaneId != Lane::NOLANE)
                    {
                        LaneLinkPair laneLinkPairOld = { id, nextLaneId};
                        oldLaneLinks.insert(road->getID(), laneLinkPairOld);
                        LaneLinkPair laneLinkPairNew = {nextLaneId, Lane::NOLANE};
                        newLaneLinks.insert(road->getID(), laneLinkPairNew);
                    }
                }
                else // find the connecting lane 
                {
                    double t = laneSection->getTValue(lane, roadLength, laneWidth);
                    QVector2D vecT = QVector2D(road_->getGlobalPoint(roadLength, t) - succRoad->getGlobalPoint(nextStart));
                    float dist = vecT.length();
                    float dotVec = QVector2D::dotProduct(succRoad->getGlobalNormal(nextStart), vecT.normalized());

                    if (dist < NUMERICAL_ZERO3)
                    {
                        if (id > 0)
                        {
                            dist = -dist;
                        }
                    }
                    else if (dotVec > 0)
                    {
                        dist = -dist;
                    }

                    int laneId = nextLaneSection->getLaneId(nextStart, dist);

                    if (laneId != nextLaneId)
                    {
                        if (laneId == 0)
                        {
                            LaneLinkPair laneLinkPairOld = { id, nextLaneId};
                            oldSuccessorLaneLinks_.insert(road->getID(), laneLinkPairOld);
                            LaneLinkPair laneLinkPairNew = {id, Lane::NOLANE};
                            newSuccessorLaneLinks_.insert(road->getID(), laneLinkPairNew);
                            continue;
                        }

                        LaneLinkPair laneLinkPairOld = { id, nextLaneId};
                        oldSuccessorLaneLinks_.insert(road->getID(), laneLinkPairOld);
                        LaneLinkPair laneLinkPairNew = {id, laneId};
                        newSuccessorLaneLinks_.insert(road->getID(), laneLinkPairNew);

                        odrID succRoadId = succRoad->getID();

                        LaneLinkPair laneLinkNextOld = { laneId, nextLaneSection->getLane(laneId)->getPredecessor()};
                        oldLaneLinks.insert(succRoadId, laneLinkNextOld);

                        if (!junctionSuccessorConnection_)
                        {
                            LaneLinkPair laneLinkPairNext = {laneId, id};
                            newLaneLinks.insert(succRoadId, laneLinkPairNext);
                        }
                        else
                        {
                            LaneLinkPair laneLinkPairNext = {laneId, Lane::NOLANE};
                            newLaneLinks.insert(succRoadId, laneLinkPairNext);
                        }
                    }
                }
            }
        }
    }

/*    qDebug() << road_->getID();
    QMultiMap<QString, LaneLinkPair>::const_iterator it = newPredecessorLaneLinks_.constBegin();
    while (it != newPredecessorLaneLinks_.constEnd())
    {
        LaneLinkPair laneLinkPair = it.value();
        qDebug() << it.key() << " Predecessor: " << laneLinkPair.laneId << " " << laneLinkPair.linkId;
        it++;
    }
    it = newSuccessorLaneLinks_.constBegin();
    while (it != newSuccessorLaneLinks_.constEnd())
    {
        LaneLinkPair laneLinkPair = it.value();
        qDebug() << it.key() << "Successor: " << laneLinkPair.laneId << " " << laneLinkPair.linkId;
        it++;
    }*/

    setValid();
    setText(QObject::tr("Create Lane Links"));
}

/*! \brief .
*
*/
CreateNextRoadLaneLinksCommand::~CreateNextRoadLaneLinksCommand()
{
    // Clean up //
    //
    if (isUndone())
    {
        // nothing to be done (road is still owned by the roadsystem)
    }
    else
    {
        oldSuccessorLaneLinks_.clear();
        newSuccessorLaneLinks_.clear();
        oldPredecessorLaneLinks_.clear();
        newPredecessorLaneLinks_.clear();
    }
}

/*! \brief .
*
*/
void
CreateNextRoadLaneLinksCommand::redo()
{
    auto it = newSuccessorLaneLinks_.constBegin();
    while (it != newSuccessorLaneLinks_.constEnd())
    {
        RSystemElementRoad * road = roadSystem_->getRoad(it.key());
        LaneSection * laneSection = road->getLaneSection(road->getLength());
        LaneLinkPair laneLinkPair = it.value();
        int oldSuccessor = laneSection->getLane(laneLinkPair.laneId)->getSuccessor();
        laneSection->getLane(laneLinkPair.laneId)->setSuccessor(laneLinkPair.linkId);
        if (junctionSuccessorConnection_)
        {
            if ((laneLinkPair.laneId != 0) && (laneLinkPair.linkId != Lane::NOLANE))
            {
                junctionSuccessorConnection_->addLaneLink(laneLinkPair.linkId, laneLinkPair.laneId);
            }
            else
            {
                junctionSuccessorConnection_->removeLaneLink(oldSuccessor);
            }
        }

        it++;
    }

    if (junctionSuccessorConnection_)
    {
       /* if (junctionSuccessorConnection_->getLaneLinks().count() == 0)
        {
            junction_->delConnection(junctionSuccessorConnection_);
        }
        else */ if (!junction_->getConnectingRoadConnections(road_->getID()).contains(junctionSuccessorConnection_))
        {
            junction_->addConnection(junctionSuccessorConnection_);
        }
    }


    it = newPredecessorLaneLinks_.constBegin();
    while (it != newPredecessorLaneLinks_.constEnd())
    {
        RSystemElementRoad * road = roadSystem_->getRoad(it.key());
        LaneSection * laneSection = road->getLaneSection(0.0);
        LaneLinkPair laneLinkPair = it.value();
        int oldPredecessor = laneSection->getLane(laneLinkPair.laneId)->getPredecessor();
        laneSection->getLane(laneLinkPair.laneId)->setPredecessor(laneLinkPair.linkId);
        if (junctionPredecessorConnection_) 
        {
            if ((laneLinkPair.laneId != 0) && (laneLinkPair.linkId != Lane::NOLANE))
            {
                junctionPredecessorConnection_->addLaneLink(laneLinkPair.linkId, laneLinkPair.laneId);
            }
            else
            {
                junctionPredecessorConnection_->removeLaneLink(oldPredecessor);
            }
        }

        it++;
    }

    if (junctionPredecessorConnection_)
    {
   /*     if (junctionPredecessorConnection_->getLaneLinks().count() == 0)
        {
            junction_->delConnection(junctionPredecessorConnection_);
        }
        else*/ if (!junction_->getConnectingRoadConnections(road_->getID()).contains(junctionPredecessorConnection_))
        {
            junction_->addConnection(junctionPredecessorConnection_);
        }
    }

    setRedone();
}

/*! \brief
*
*/
void
CreateNextRoadLaneLinksCommand::undo()
{
    auto it = oldSuccessorLaneLinks_.constBegin();
    while (it != oldSuccessorLaneLinks_.constEnd())
    {
        RSystemElementRoad * road = roadSystem_->getRoad(it.key());
        LaneSection * laneSection = road->getLaneSection(road->getLength());
        LaneLinkPair laneLinkPair = it.value();
        int newSuccessor = laneSection->getLane(laneLinkPair.laneId)->getSuccessor();
        laneSection->getLane(laneLinkPair.laneId)->setSuccessor(laneLinkPair.linkId);
        if (junctionSuccessorConnection_)
        {
            if ((laneLinkPair.laneId != 0) && (laneLinkPair.linkId != Lane::NOLANE))
            {
                junctionSuccessorConnection_->addLaneLink(laneLinkPair.linkId, laneLinkPair.laneId);
            }
            else
            {
                junctionSuccessorConnection_->removeLaneLink(newSuccessor);
            }
        }

        it++;
    }

    if (junctionSuccessorConnection_)
    {
        if (junctionSuccessorConnection_->getLaneLinks().count() == 0)
        {
            junction_->delConnection(junctionSuccessorConnection_);
        }
        else if (!junction_->getConnectingRoadConnections(road_->getID()).contains(junctionSuccessorConnection_))
        {
            junction_->addConnection(junctionSuccessorConnection_);
        }
    }

    it = oldPredecessorLaneLinks_.constBegin();
    while (it != oldPredecessorLaneLinks_.constEnd())
    {
        odrID predRoadId = it.key();
        RSystemElementRoad * road = roadSystem_->getRoad(it.key());
        LaneSection * laneSection = road->getLaneSection(0.0);
        LaneLinkPair laneLinkPair = it.value();
        int newPredecessor = laneSection->getLane(laneLinkPair.laneId)->getPredecessor();
        laneSection->getLane(laneLinkPair.laneId)->setPredecessor(laneLinkPair.linkId);
        if (junctionPredecessorConnection_) 
        {
            if ((laneLinkPair.laneId != 0) && (laneLinkPair.linkId != Lane::NOLANE))
            {
                junctionPredecessorConnection_->addLaneLink(laneLinkPair.linkId, laneLinkPair.laneId);
            }
            else
            {
                junctionPredecessorConnection_->removeLaneLink(newPredecessor);
            }
        }

        it++;
    }

    if (junctionPredecessorConnection_)
    {
        if (junctionPredecessorConnection_->getLaneLinks().count() == 0)
        {
            junction_->delConnection(junctionPredecessorConnection_);
        }
        else if (!junction_->getConnectingRoadConnections(road_->getID()).contains(junctionPredecessorConnection_))
        {
            junction_->addConnection(junctionPredecessorConnection_);
        }
    }

    setUndone();
}

//#########################//
// LinkRoadsAndLanesCommand //
//#########################//

LinkRoadsAndLanesCommand::LinkRoadsAndLanesCommand(const QList<RSystemElementRoad *> &roads, double threshold, DataCommand *parent)
    : DataCommand(parent)
    , roads_(roads)
    , threshold_(threshold)
    , setRoadLinkRoadsCommand_(NULL)
    , createInnerLaneLinksCommand_(NULL)
    , createNextRoadLaneLinksCommand_(NULL)
{
    setRoadLinkRoadsCommand_ = new SetRoadLinkRoadsCommand(roads_, threshold_, this);
    if (setRoadLinkRoadsCommand_->isValid())
    {
        setRoadLinkRoadsCommand_->redo(); // have to call redo, otherwise the roads are not linked
    }
    else
    {
        return;
    }

    // Check for validity //
    //
    RoadSystem * roadSystem = roads_.first()->getRoadSystem();
    if (!roadSystem)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("LinkRoadsAndLanesCommand: Internal error! No valid roadsystem."));
        return;
    }

    foreach (RSystemElementRoad * road, roads_)
    {
        createInnerLaneLinksCommand_ = new CreateInnerLaneLinksCommand(road, this);
        if (!createInnerLaneLinksCommand_->isValid())
        {
            return;
        }

        createNextRoadLaneLinksCommand_ = new CreateNextRoadLaneLinksCommand(roadSystem, road, this);
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Link roads and lanes"));
}

LinkRoadsAndLanesCommand::LinkRoadsAndLanesCommand(const QList<SetRoadLinkRoadsCommand::RoadPair *> &roadPairs, DataCommand *parent)
    : DataCommand(parent)
    , roadPairs_(roadPairs)
    , setRoadLinkRoadsCommand_(NULL)
    , createInnerLaneLinksCommand_(NULL)
    , createNextRoadLaneLinksCommand_(NULL)
{

    // Check for validity //
    //
    RoadSystem * roadSystem = roadPairs.first()->road1->getRoadSystem();
    if (!roadSystem)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("LinkRoadsAndLanesCommand: Internal error! No valid roadsystem."));
        return;
    }

    setRoadLinkRoadsCommand_ = new SetRoadLinkRoadsCommand(roadPairs_, this);
    if (setRoadLinkRoadsCommand_->isValid())
    {
        setRoadLinkRoadsCommand_->redo(); // have to call redo, otherwise the roads are not linked
    }
    else
    {
        return;
    }

    foreach (RSystemElementRoad * road, roads_)
    {
        createInnerLaneLinksCommand_ = new CreateInnerLaneLinksCommand(road, this);
        if (!createInnerLaneLinksCommand_->isValid())
        {
            return;
        }

        createNextRoadLaneLinksCommand_ = new CreateNextRoadLaneLinksCommand(roadSystem, road, this);
    }

    // Done //
    //
    setValid();
    setText(QObject::tr("Link roads and lanes"));
}

/*! \brief .
*
*/
LinkRoadsAndLanesCommand::~LinkRoadsAndLanesCommand()
{
}

//#########################//
// LinkLanesCommand //
//#########################//

LinkLanesCommand::LinkLanesCommand(RSystemElementRoad *road, DataCommand *parent)
    : DataCommand(parent)
    , road_(road)
    , createInnerLaneLinksCommand_(NULL)
    , createNextRoadLaneLinksCommand_(NULL)
{

    // Check for validity //
    //
    RoadSystem * roadSystem = road_->getRoadSystem();
    if (!roadSystem)
    {
        setInvalid(); // Invalid
        setText(QObject::tr("LinkLanesCommand: Internal error! No valid roadsystem."));
        return;
    }

    createInnerLaneLinksCommand_ = new CreateInnerLaneLinksCommand(road_, this);
    if (!createInnerLaneLinksCommand_->isValid())
    {
        return;
    }

    createNextRoadLaneLinksCommand_ = new CreateNextRoadLaneLinksCommand(roadSystem, road, this);

    // Done //
    //
    setValid();
    setText(QObject::tr("Link lanes"));
}

/*! \brief .
*
*/
LinkLanesCommand::~LinkLanesCommand()
{
}

