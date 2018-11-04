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

#ifndef ROADCOMMANDS_HPP
#define ROADCOMMANDS_HPP

// 1000

#include "datacommand.hpp"

#include <QMap>
#include <QPointF>

class RoadSystem;
class RSystemElementRoad;

#include "src/data/roadsystem/roadlink.hpp"

class TrackComponent;
class TypeSection;
class ElevationSection;
class SuperelevationSection;
class CrossfallSection;
class ShapeSection;
class LaneSection;
class SplitTrackComponentCommand;

//#########################//
// AppendRoadPrototypeCommand //
//#########################//

class AppendRoadPrototypeCommand : public DataCommand
{
public:
    explicit AppendRoadPrototypeCommand(RSystemElementRoad *road, RSystemElementRoad *prototype, bool atStart, DataCommand *parent = NULL);
    virtual ~AppendRoadPrototypeCommand();

    virtual int id() const
    {
        return 0x1001;
    }

    virtual void undo();
    virtual void redo();

private:
    AppendRoadPrototypeCommand(); /* not allowed */
    AppendRoadPrototypeCommand(const AppendRoadPrototypeCommand &); /* not allowed */
    AppendRoadPrototypeCommand &operator=(const AppendRoadPrototypeCommand &); /* not allowed */

private:
    RSystemElementRoad *road_; // linked
    RSystemElementRoad *prototype_; // now owned

    bool atStart_;

    double prototypeLength_;

    QMap<double, TypeSection *> newTypeSections_;
    QMap<double, TypeSection *> oldTypeSections_;

    QMap<double, TrackComponent *> trackSections_;

    QMap<double, ElevationSection *> newElevationSections_;
    QMap<double, ElevationSection *> oldElevationSections_;

    QMap<double, SuperelevationSection *> newSuperelevationSections_;
    QMap<double, SuperelevationSection *> oldSuperelevationSections_;

    QMap<double, CrossfallSection *> newCrossfallSections_;
    QMap<double, CrossfallSection *> oldCrossfallSections_;

    QMap<double, LaneSection *> newLaneSections_;
    QMap<double, LaneSection *> oldLaneSections_;

	QMap<double, ShapeSection *> newShapeSections_;
	QMap<double, ShapeSection *> oldShapeSections_;
};

//#########################//
// MergeRoadsCommand //
//#########################//

class MergeRoadsCommand : public DataCommand
{
public:
    explicit MergeRoadsCommand(RSystemElementRoad *road1, RSystemElementRoad *road2, bool firstStart, bool secondStart, DataCommand *parent = NULL);
    virtual ~MergeRoadsCommand();

    virtual int id() const
    {
        return 0x1012;
    }

    virtual void undo();
    virtual void redo();

private:
    MergeRoadsCommand(); /* not allowed */
    MergeRoadsCommand(const MergeRoadsCommand &); /* not allowed */
    MergeRoadsCommand &operator=(const MergeRoadsCommand &); /* not allowed */

private:
    RSystemElementRoad *road1_; // linked
    RSystemElementRoad *road2_; // now owned
    RoadSystem *roadSystem_;
    
    bool firstStart;//merge start of first road ( end of first road if false)
    bool secondStart;//to start of second road ( end of second road if false)

    double secondRoadLength_;
    QPointF secondEnd_;

    QMap<double, TypeSection *> newTypeSections_;
    QMap<double, TypeSection *> oldTypeSections_;

    QMap<double, TrackComponent *> trackSections_;

    QMap<double, ElevationSection *> newElevationSections_;
    QMap<double, ElevationSection *> oldElevationSections_;

    QMap<double, SuperelevationSection *> newSuperelevationSections_;
    QMap<double, SuperelevationSection *> oldSuperelevationSections_;

    QMap<double, CrossfallSection *> newCrossfallSections_;
    QMap<double, CrossfallSection *> oldCrossfallSections_;

    QMap<double, LaneSection *> newLaneSections_;
    QMap<double, LaneSection *> oldLaneSections_;

	QMap<double, ShapeSection *> newShapeSections_;
	QMap<double, ShapeSection *> oldShapeSections_;
};

//#########################//
// SnapRoadsCommand //
//#########################//

class SnapRoadsCommand : public DataCommand
{
public:
    explicit SnapRoadsCommand(RSystemElementRoad *road1, RSystemElementRoad *road2, short int pos, DataCommand *parent = NULL);
    virtual ~SnapRoadsCommand();

    virtual int id() const
    {
        return 0x1012;
    }

    virtual void undo();
    virtual void redo();

private:
    SnapRoadsCommand(); /* not allowed */
    SnapRoadsCommand(const SnapRoadsCommand &); /* not allowed */
    SnapRoadsCommand &operator=(const SnapRoadsCommand &); /* not allowed */

private:
    RSystemElementRoad *road1_; // linked
    RSystemElementRoad *road2_; // now owned
    RoadSystem *roadSystem_;
    TrackComponent *track_;
    short int pos_;

    double oldHeading_;
    double newHeading_;
    QPointF oldPoint_;
    QPointF newPoint_;

    QMap<double, TypeSection *> newTypeSections_;
    QMap<double, TypeSection *> oldTypeSections_;

    QMap<double, TrackComponent *> trackSections_;

    QMap<double, ElevationSection *> newElevationSections_;
    QMap<double, ElevationSection *> oldElevationSections_;

    QMap<double, SuperelevationSection *> newSuperelevationSections_;
    QMap<double, SuperelevationSection *> oldSuperelevationSections_;

    QMap<double, CrossfallSection *> newCrossfallSections_;
    QMap<double, CrossfallSection *> oldCrossfallSections_;

    QMap<double, LaneSection *> newLaneSections_;
    QMap<double, LaneSection *> oldLaneSections_;

	QMap<double, ShapeSection *> newShapeSections_;
	QMap<double, ShapeSection *> oldShapeSections_;
};

//#########################//
// ChangeLanePrototypeCommand //
//#########################//

class ChangeLanePrototypeCommand : public DataCommand
{
public:
    explicit ChangeLanePrototypeCommand(RSystemElementRoad *road, RSystemElementRoad *prototype, DataCommand *parent = NULL);
    virtual ~ChangeLanePrototypeCommand();

    virtual int id() const
    {
        return 0x1001;
    }

    virtual void undo();
    virtual void redo();

private:
    ChangeLanePrototypeCommand(); /* not allowed */
    ChangeLanePrototypeCommand(const ChangeLanePrototypeCommand &); /* not allowed */
    ChangeLanePrototypeCommand &operator=(const ChangeLanePrototypeCommand &); /* not allowed */

private:
    RSystemElementRoad *road_; // linked
    RSystemElementRoad *prototype_; // now owned

    QMap<double, LaneSection *> newLaneSections_;
    QMap<double, LaneSection *> oldLaneSections_;
};

//#########################//
// ChangeRoadTypePrototypeCommand //
//#########################//

class ChangeRoadTypePrototypeCommand : public DataCommand
{
public:
    explicit ChangeRoadTypePrototypeCommand(RSystemElementRoad *road, RSystemElementRoad *prototype, DataCommand *parent = NULL);
    virtual ~ChangeRoadTypePrototypeCommand();

    virtual int id() const
    {
        return 0x1001;
    }

    virtual void undo();
    virtual void redo();

private:
    ChangeRoadTypePrototypeCommand(); /* not allowed */
    ChangeRoadTypePrototypeCommand(const ChangeRoadTypePrototypeCommand &); /* not allowed */
    ChangeRoadTypePrototypeCommand &operator=(const ChangeRoadTypePrototypeCommand &); /* not allowed */

private:
    RSystemElementRoad *road_; // linked
    RSystemElementRoad *prototype_; // now owned

    QMap<double, TypeSection *> newRoadTypeSections_;
    QMap<double, TypeSection *> oldRoadTypeSections_;
};

//#########################//
// ChangeElevationPrototypeCommand //
//#########################//

class ChangeElevationPrototypeCommand : public DataCommand
{
public:
    explicit ChangeElevationPrototypeCommand(RSystemElementRoad *road, RSystemElementRoad *prototype, DataCommand *parent = NULL);
    virtual ~ChangeElevationPrototypeCommand();

    virtual int id() const
    {
        return 0x1001;
    }

    virtual void undo();
    virtual void redo();

private:
    ChangeElevationPrototypeCommand(); /* not allowed */
    ChangeElevationPrototypeCommand(const ChangeElevationPrototypeCommand &); /* not allowed */
    ChangeElevationPrototypeCommand &operator=(const ChangeElevationPrototypeCommand &); /* not allowed */

private:
    RSystemElementRoad *road_; // linked
    RSystemElementRoad *prototype_; // now owned

    QMap<double, ElevationSection *> newElevationSections_;
    QMap<double, ElevationSection *> oldElevationSections_;
};

//#########################//
// ChangeSuperelevationPrototypeCommand //
//#########################//

class ChangeSuperelevationPrototypeCommand : public DataCommand
{
public:
    explicit ChangeSuperelevationPrototypeCommand(RSystemElementRoad *road, RSystemElementRoad *prototype, DataCommand *parent = NULL);
    virtual ~ChangeSuperelevationPrototypeCommand();

    virtual int id() const
    {
        return 0x1001;
    }

    virtual void undo();
    virtual void redo();

private:
    ChangeSuperelevationPrototypeCommand(); /* not allowed */
    ChangeSuperelevationPrototypeCommand(const ChangeSuperelevationPrototypeCommand &); /* not allowed */
    ChangeSuperelevationPrototypeCommand &operator=(const ChangeSuperelevationPrototypeCommand &); /* not allowed */

private:
    RSystemElementRoad *road_; // linked
    RSystemElementRoad *prototype_; // now owned

    QMap<double, SuperelevationSection *> newSuperelevationSections_;
    QMap<double, SuperelevationSection *> oldSuperelevationSections_;
};

//#########################//
// ChangeCrossfallPrototypeCommand //
//#########################//

class ChangeCrossfallPrototypeCommand : public DataCommand
{
public:
    explicit ChangeCrossfallPrototypeCommand(RSystemElementRoad *road, RSystemElementRoad *prototype, DataCommand *parent = NULL);
    virtual ~ChangeCrossfallPrototypeCommand();

    virtual int id() const
    {
        return 0x1001;
    }

    virtual void undo();
    virtual void redo();

private:
    ChangeCrossfallPrototypeCommand(); /* not allowed */
    ChangeCrossfallPrototypeCommand(const ChangeCrossfallPrototypeCommand &); /* not allowed */
    ChangeCrossfallPrototypeCommand &operator=(const ChangeCrossfallPrototypeCommand &); /* not allowed */

private:
    RSystemElementRoad *road_; // linked
    RSystemElementRoad *prototype_; // now owned

    QMap<double, CrossfallSection *> newCrossfallSections_;
    QMap<double, CrossfallSection *> oldCrossfallSections_;
};

//#########################//
// ChangeShapePrototypeCommand //
//#########################//

class ChangeShapePrototypeCommand : public DataCommand
{
public:
	explicit ChangeShapePrototypeCommand(RSystemElementRoad *road, RSystemElementRoad *prototype, DataCommand *parent = NULL);
	virtual ~ChangeShapePrototypeCommand();

	virtual int id() const
	{
		return 0x1001;
	}

	virtual void undo();
	virtual void redo();

private:
	ChangeShapePrototypeCommand(); /* not allowed */
	ChangeShapePrototypeCommand(const ChangeShapePrototypeCommand &); /* not allowed */
	ChangeShapePrototypeCommand &operator=(const ChangeShapePrototypeCommand &); /* not allowed */

private:
	RSystemElementRoad *road_; // linked
	RSystemElementRoad *prototype_; // now owned

	QMap<double, ShapeSection *> newShapeSections_;
	QMap<double, ShapeSection *> oldShapeSections_;
};

//#########################//
// RemoveTrackCommand //
//#########################//

class RemoveTrackCommand : public DataCommand
{
public:
    explicit RemoveTrackCommand(RSystemElementRoad *road, TrackComponent *trackComponent, bool atStart, DataCommand *parent = NULL);
    virtual ~RemoveTrackCommand();

    virtual int id() const
    {
        return 0x1002;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveTrackCommand(); /* not allowed */
    RemoveTrackCommand(const RemoveTrackCommand &); /* not allowed */
    RemoveTrackCommand &operator=(const RemoveTrackCommand &); /* not allowed */

private:
    RSystemElementRoad *road_; // linked

    bool atStart_;

    double cutLength_;
    double sCut_;

    QMap<double, TypeSection *> typeSections_;
    QMap<double, TrackComponent *> trackSections_;
    QMap<double, ElevationSection *> elevationSections_;
    QMap<double, SuperelevationSection *> superelevationSections_;
    QMap<double, CrossfallSection *> crossfallSections_;
    QMap<double, LaneSection *> laneSections_;
	QMap<double, ShapeSection *> shapeSections_;
};

//#########################//
// NewRoadCommand //
//#########################//

class NewRoadCommand : public DataCommand
{
public:
    explicit NewRoadCommand(RSystemElementRoad *newRoad, RoadSystem *roadSystem, DataCommand *parent = NULL);
    virtual ~NewRoadCommand();

    virtual int id() const
    {
        return 0x1004;
    }

    virtual void undo();
    virtual void redo();

private:
    NewRoadCommand(); /* not allowed */
    NewRoadCommand(const NewRoadCommand &); /* not allowed */
    NewRoadCommand &operator=(const NewRoadCommand &); /* not allowed */

private:
    RSystemElementRoad *newRoad_;
    RoadSystem *roadSystem_;
};

//#########################//
// RemoveRoadCommand //
//#########################//

class RemoveRoadCommand : public DataCommand
{
public:
    explicit RemoveRoadCommand(RSystemElementRoad *road, DataCommand *parent = NULL);
    virtual ~RemoveRoadCommand();

    virtual int id() const
    {
        return 0x1008;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveRoadCommand(); /* not allowed */
    RemoveRoadCommand(const RemoveRoadCommand &); /* not allowed */
    RemoveRoadCommand &operator=(const RemoveRoadCommand &); /* not allowed */

private:
    RSystemElementRoad *road_;
    RoadSystem *roadSystem_;
};

//#########################//
// MoveRoadCommand //
//#########################//

class MoveRoadCommand : public DataCommand
{
public:
    explicit MoveRoadCommand(const QList<RSystemElementRoad *> &roads, const QPointF &dPos, DataCommand *parent = NULL);
    virtual ~MoveRoadCommand();

    virtual int id() const
    {
        return 0x1010;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    MoveRoadCommand(); /* not allowed */
    MoveRoadCommand(const MoveRoadCommand &); /* not allowed */
    MoveRoadCommand &operator=(const MoveRoadCommand &); /* not allowed */

private:
    QList<RSystemElementRoad *> roads_;
    QPointF dPos_;

    //	QList<QPointF>			oldStartPoints_;
};

//#########################//
// RotateRoadAroundPointCommand //
//#########################//

class RotateRoadAroundPointCommand : public DataCommand
{
public:
    explicit RotateRoadAroundPointCommand(const QList<RSystemElementRoad *> &roads, const QPointF &pivotPoint, double angleDegrees, DataCommand *parent = NULL);
    explicit RotateRoadAroundPointCommand(RSystemElementRoad *road, const QPointF &pivotPoint, double angleDegrees, DataCommand *parent = NULL);
    virtual ~RotateRoadAroundPointCommand();

    virtual int id() const
    {
        return 0x1020;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    RotateRoadAroundPointCommand(); /* not allowed */
    RotateRoadAroundPointCommand(const RotateRoadAroundPointCommand &); /* not allowed */
    RotateRoadAroundPointCommand &operator=(const RotateRoadAroundPointCommand &); /* not allowed */

private:
    QList<RSystemElementRoad *> roads_;
    QPointF pivotPoint_;

    double angleDegrees_;
};

//#########################//
// SplitRoadCommand //
//#########################//

class SplitRoadCommand : public DataCommand
{
public:
    explicit SplitRoadCommand(RSystemElementRoad *road, double s, DataCommand *parent = NULL);
    virtual ~SplitRoadCommand();

    virtual int id() const
    {
        return 0x1040;
    }

    RSystemElementRoad *getFirstNewRoad()
    {
        return newRoadA_;
    }
    RSystemElementRoad *getSecondNewRoad()
    {
        return newRoadB_;
    }

    virtual void undo();
    virtual void redo();
    void updateLinks(RSystemElementRoad *currentA, RSystemElementRoad *currentB, RSystemElementRoad *newA, RSystemElementRoad *newB);

    //	virtual bool			mergeWith(const QUndoCommand * other);

private:
    SplitRoadCommand(); /* not allowed */
    SplitRoadCommand(const SplitRoadCommand &); /* not allowed */
    SplitRoadCommand &operator=(const SplitRoadCommand &); /* not allowed */

    void splitBefore(double s);

private:
    RSystemElementRoad *road_;

    RSystemElementRoad *newRoadA_;
    RSystemElementRoad *newRoadB_;

    double splitS_;

    RoadSystem *roadSystem_;
};

//#########################//
// SplitTrackRoadCommand //
//#########################//

class SplitTrackRoadCommand : public DataCommand
{
public:
    explicit SplitTrackRoadCommand(RSystemElementRoad *road, double s, DataCommand *parent = NULL);
    virtual ~SplitTrackRoadCommand();

    virtual int id() const
    {
        return 0x1042;
    }

    SplitRoadCommand *getSplitRoadCommand()
    {
        return splitRoadCommand_;
    };

    virtual void undo()
    {
        QUndoCommand::undo();
    };
    virtual void redo()
    {
        QUndoCommand::redo();
    };

private:
    SplitTrackRoadCommand(); /* not allowed */
    SplitTrackRoadCommand(const SplitTrackRoadCommand &); /* not allowed */
    SplitTrackRoadCommand &operator=(const SplitTrackRoadCommand &); /* not allowed */

private:
    SplitTrackComponentCommand *splitTrackComponentCommand_;
    SplitRoadCommand *splitRoadCommand_;

    RSystemElementRoad *road_;
    double splitS_;
};

//#########################//
// SetRoadLinkCommand //
//#########################//

class SetRoadLinkCommand : public DataCommand
{
public:
    explicit SetRoadLinkCommand(RSystemElementRoad *road, RoadLink::RoadLinkType roadLinkType, RoadLink *roadLink, JunctionConnection *newConnection, RSystemElementJunction *junction, DataCommand *parent = NULL);
    virtual ~SetRoadLinkCommand();

    virtual int id() const
    {
        return 0x1050;
    }

    virtual void undo();
    virtual void redo();

private:
    SetRoadLinkCommand(); /* not allowed */
    SetRoadLinkCommand(const SetRoadLinkCommand &); /* not allowed */
    SetRoadLinkCommand &operator=(const SetRoadLinkCommand &); /* not allowed */

private:
    RSystemElementRoad *road_;
    RoadLink::RoadLinkType roadLinkType_;
    RSystemElementJunction *junction_;
    RoadSystem *roadSystem_;

    RoadLink *oldRoadLink_;
    RoadLink *newRoadLink_;
    JunctionConnection *oldConnection_;
    JunctionConnection *newConnection_;
};

//#########################//
// SetRoadLinkRoadsCommand //
//#########################//

class SetRoadLinkRoadsCommand : public DataCommand
{

public:
    enum RoadPosition
    {
        FirstRoadStart = 1,
        FirstRoadEnd = 2,
        SecondRoadStart = 4,
        SecondRoadEnd = 8
    };

    struct RoadPair
    {
        RSystemElementRoad *road1;
        RSystemElementRoad *road2;
        short int positionIndex; // 0 (start,start), 1 (end,start), 2 (end,end), 3 (start,end)
    };

    explicit SetRoadLinkRoadsCommand(const QList<RSystemElementRoad *> &roads, double threshold, DataCommand *parent = NULL);
    explicit SetRoadLinkRoadsCommand(const QList<RoadPair *> &roadPairs, DataCommand *parent = NULL); // Road pairs that have to be linked are already selected
    virtual ~SetRoadLinkRoadsCommand();

    virtual int id() const
    {
        return 0x1052;
    }

    virtual void undo();
    virtual void redo();

private:
    SetRoadLinkRoadsCommand(); /* not allowed */
    SetRoadLinkRoadsCommand(const SetRoadLinkCommand &); /* not allowed */
    SetRoadLinkRoadsCommand &operator=(const SetRoadLinkCommand &); /* not allowed */

    void distanceRoad(RSystemElementRoad *road, double threshold, QMultiMap<double, RoadPair> *distRoadPairs);
    void distanceRoads(RSystemElementRoad *road1, RSystemElementRoad *road2, double threshold, QMultiMap<double, RoadPair> *distRoadPairs);
    void findRoadContactPoints(RSystemElementRoad *road1, RSystemElementRoad *road2, unsigned int posIndex);
    void findPathContactPoint(RSystemElementRoad *road1, RSystemElementRoad *road2, unsigned int posIndex, RSystemElementJunction *junction);
    void findConnections(RSystemElementRoad *road1, RSystemElementRoad *road2, short int index);

private:
    QList<RSystemElementRoad *> roads_;

    QMultiMap<RSystemElementRoad *, RoadLink *> oldRoadLinks_;
    QMultiMap<RSystemElementRoad *, RoadLink *> newRoadLinks_;
    QMultiMap<RSystemElementRoad *, RoadLink::RoadLinkType> roadLinkTypes_;

    QList<RSystemElementJunction *> junctions_;
    QMultiMap<RSystemElementJunction *, JunctionConnection *> oldConnections_;
    QMultiMap<RSystemElementJunction *, JunctionConnection *> newConnections_;
};

//#########################//
// RemoveRoadLinkCommand //
//#########################//

class RemoveRoadLinkCommand : public DataCommand
{
public:
    explicit RemoveRoadLinkCommand(RSystemElementRoad *road, DataCommand *parent = NULL);
    virtual ~RemoveRoadLinkCommand();

    virtual int id() const
    {
        return 0x1054;
    }

    virtual void undo();
    virtual void redo();

    void removeLinkRoadLink(RSystemElementRoad * linkRoad, JunctionConnection::ContactPointValue contactPoint);
    void saveConnectingRoadLanes(RSystemElementJunction * junction);

private:
    RemoveRoadLinkCommand(); /* not allowed */
    RemoveRoadLinkCommand(const RemoveRoadLinkCommand &); /* not allowed */
    RemoveRoadLinkCommand &operator=(const RemoveRoadLinkCommand &); /* not allowed */

private:
    RoadSystem * roadSystem_;
    RSystemElementRoad *road_;
    RoadLink *predecessor_;
    RoadLink *successor_;
    RoadLink *predecessorLink_;
    RoadLink *successorLink_;
    RSystemElementJunction *junction_;
    QList<JunctionConnection *> junctionConnections_;
    QMap<odrID, QMap<int,int>> laneLinksRoadStart_;
    QMap<odrID, QMap<int,int>> laneLinksRoadEnd_;
};


//#########################//
// CreateInnerLaneLinksCommand //
//#########################//

class CreateInnerLaneLinksCommand : public DataCommand
{
    struct LaneLinkPair
    {
        int laneId;
        int linkId;
    };

public:
    explicit CreateInnerLaneLinksCommand(RSystemElementRoad *road, DataCommand *parent = NULL);
    virtual ~CreateInnerLaneLinksCommand();

    virtual int id() const
    {
        return 0x1056;
    }

    virtual void undo();
    virtual void redo();

private:
    CreateInnerLaneLinksCommand(); /* not allowed */
    CreateInnerLaneLinksCommand(const CreateInnerLaneLinksCommand &); /* not allowed */
    CreateInnerLaneLinksCommand &operator=(const CreateInnerLaneLinksCommand &); /* not allowed */

private:
    RSystemElementRoad *road_;
    QMultiMap<double,LaneLinkPair> newSuccessorLaneLinks_;
    QMultiMap<double,LaneLinkPair> oldSuccessorLaneLinks_;
    QMultiMap<double,LaneLinkPair> newPredecessorLaneLinks_;
    QMultiMap<double,LaneLinkPair> oldPredecessorLaneLinks_;
};

//#########################//
// CreateNextRoadLaneLinksCommand //
//#########################//

class CreateNextRoadLaneLinksCommand : public DataCommand
{
    struct LaneLinkPair
    {
        int laneId;
        int linkId;
    };

public:
    explicit CreateNextRoadLaneLinksCommand(RoadSystem *roadSystem, RSystemElementRoad *road, DataCommand *parent = NULL);
    virtual ~CreateNextRoadLaneLinksCommand();

    virtual int id() const
    {
        return 0x1058;
    }

    virtual void undo();
    virtual void redo();

private:
    CreateNextRoadLaneLinksCommand(); /* not allowed */
    CreateNextRoadLaneLinksCommand(const CreateNextRoadLaneLinksCommand &); /* not allowed */
    CreateNextRoadLaneLinksCommand &operator=(const CreateNextRoadLaneLinksCommand &); /* not allowed */

private:
    RoadSystem *roadSystem_;
    RSystemElementRoad *road_;
    RSystemElementJunction *junction_;
    JunctionConnection *junctionPredecessorConnection_;
    JunctionConnection *junctionSuccessorConnection_;
    QMultiMap<odrID, LaneLinkPair> newSuccessorLaneLinks_;
    QMultiMap<odrID, LaneLinkPair> oldSuccessorLaneLinks_;
    QMultiMap<odrID, LaneLinkPair> newPredecessorLaneLinks_;
    QMultiMap<odrID, LaneLinkPair> oldPredecessorLaneLinks_;
};

//#########################//
// LinkRoadsAndLanesCommand //
//#########################//

class LinkRoadsAndLanesCommand : public DataCommand
{
public:
    explicit LinkRoadsAndLanesCommand(const QList<RSystemElementRoad *> &roads, double threshold, DataCommand *parent = NULL);
    explicit LinkRoadsAndLanesCommand(const QList<SetRoadLinkRoadsCommand::RoadPair *> &roadPairs, DataCommand *parent = NULL);
    virtual ~LinkRoadsAndLanesCommand();

    virtual int id() const
    {
        return 0x1060;
    }

    virtual void undo()
    {
        QUndoCommand::undo();
    };
    virtual void redo()
    {
        QUndoCommand::redo();
    };

private:
    LinkRoadsAndLanesCommand(); /* not allowed */
    LinkRoadsAndLanesCommand(const LinkRoadsAndLanesCommand &); /* not allowed */
    LinkRoadsAndLanesCommand &operator=(const LinkRoadsAndLanesCommand &); /* not allowed */

private:
    QList<RSystemElementRoad *> roads_;
    QList<SetRoadLinkRoadsCommand::RoadPair *> roadPairs_;
    double threshold_;

    SetRoadLinkRoadsCommand *setRoadLinkRoadsCommand_;
    CreateInnerLaneLinksCommand *createInnerLaneLinksCommand_;
    CreateNextRoadLaneLinksCommand *createNextRoadLaneLinksCommand_;
};

//#########################//
// LinkLanesCommand //
//#########################//

class LinkLanesCommand : public DataCommand
{
public:
    explicit LinkLanesCommand(RSystemElementRoad *road, DataCommand *parent = NULL);
    virtual ~LinkLanesCommand();

    virtual int id() const
    {
        return 0x1062;
    }

    virtual void undo()
    {
        QUndoCommand::undo();
    };
    virtual void redo()
    {
        QUndoCommand::redo();
    };

private:
    LinkLanesCommand(); /* not allowed */
    LinkLanesCommand(const LinkLanesCommand &); /* not allowed */
    LinkLanesCommand &operator=(const LinkLanesCommand &); /* not allowed */

private:
    RSystemElementRoad * road_;

    CreateInnerLaneLinksCommand *createInnerLaneLinksCommand_;
    CreateNextRoadLaneLinksCommand *createNextRoadLaneLinksCommand_;
};




#endif // ROADCOMMANDS_HPP
