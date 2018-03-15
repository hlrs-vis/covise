/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   09.04.2010
**
**************************************************************************/

#ifndef TRACKCOMMANDS_HPP
#define TRACKCOMMANDS_HPP

// 100

#include "datacommand.hpp"

#include <QPointF>
#include <QMultiMap>

class RoadSystem;
class RSystemElementRoad;
class TrackComponent;
class TrackComposite;
class TrackElementLine;
class TrackElementArc;
class TrackElementSpiral;
class TrackElementPoly3;
class TrackSpiralArcSpiral;
class UngroupTrackCompositeCommand;
class TypeSection;
class ElevationSection;
class SuperelevationSection;
class CrossfallSection;
class LaneSection;
class ShapeSection;

//##########################//
// TrackComponentGlobalPointsCommand //
//##########################//

class TrackComponentGlobalPointsCommand : public DataCommand
{
public:
    explicit TrackComponentGlobalPointsCommand(const QList<TrackComponent *> &endPointTracks, const QList<TrackComponent *> &startPointTracks, const QPointF &dPos, DataCommand *parent = NULL);
    virtual ~TrackComponentGlobalPointsCommand();

    virtual int id() const
    {
        return 0x101;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

    QPointF getDPos() const
    {
        return dPos_;
    }

private:
    TrackComponentGlobalPointsCommand(); /* not allowed */
    TrackComponentGlobalPointsCommand(const TrackComponentGlobalPointsCommand &); /* not allowed */

private:
    QList<TrackComponent *> endPointTracks_;
    QList<TrackComponent *> startPointTracks_;

    QList<RSystemElementRoad *> roads_;

    QList<QPointF> oldEndPoints_;
    QList<QPointF> oldStartPoints_;

    QPointF dPos_;
};

//##########################//
// TrackComponentPointAndHeadingCommand //
//##########################//

class TrackComponentPointAndHeadingCommand : public DataCommand
{
public:
    explicit TrackComponentPointAndHeadingCommand(TrackComponent *track, const QPointF &globalPoint, double globalHeadingDeg, bool isStart, DataCommand *parent = NULL);
    virtual ~TrackComponentPointAndHeadingCommand();

    virtual int id() const
    {
        return 0x102;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    TrackComponentPointAndHeadingCommand(); /* not allowed */
    TrackComponentPointAndHeadingCommand(const TrackComponentPointAndHeadingCommand &); /* not allowed */
    TrackComponentPointAndHeadingCommand &operator=(const TrackComponentPointAndHeadingCommand &); /* not allowed */

private:
    TrackComponent *track_;

    QPointF oldPoint_;
    QPointF newPoint_;

    double oldHeading_;
    double newHeading_;

    bool isStart_;
};

//##########################//
// TrackComponentSingleHeadingCommand //
//##########################//

class TrackComponentSingleHeadingCommand : public DataCommand
{
public:
    explicit TrackComponentSingleHeadingCommand(TrackComponent *lowSlotTrack, TrackComponent *highSlotTrack, double newHeading, DataCommand *parent = NULL);
    virtual ~TrackComponentSingleHeadingCommand();

    virtual int id() const
    {
        return 0x104;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    TrackComponentSingleHeadingCommand(); /* not allowed */
    TrackComponentSingleHeadingCommand(const TrackComponentSingleHeadingCommand &); /* not allowed */

private:
    TrackComponent *lowSlotTrack_;
    TrackComponent *highSlotTrack_;

    double oldHeading_;
    double newHeading_;
};

//##################//
// SplitTrackComponentCommand //
//##################//

class SplitTrackComponentCommand : public DataCommand
{
public:
    explicit SplitTrackComponentCommand(TrackComponent *component, double sSplit, DataCommand *parent = NULL);
    virtual ~SplitTrackComponentCommand();

    virtual int id() const
    {
        return 0x108;
    }

    virtual void undo();
    virtual void redo();

private:
    SplitTrackComponentCommand(); /* not allowed */
    SplitTrackComponentCommand(const SplitTrackComponentCommand &); /* not allowed */
    SplitTrackComponentCommand &operator=(const SplitTrackComponentCommand &); /* not allowed */

private:
    RSystemElementRoad *road_;

    TrackComponent *oldComponent_;
    TrackComposite *oldComposite_;

    TrackComponent *newComponentA_;
    TrackComponent *newComponentB_;
    QList<TrackComponent *> newComponents_;
};

//##########################//
// SetSpArcSFactorCommand //
//##########################//

class SetSpArcSFactorCommand : public DataCommand
{
public:
    explicit SetSpArcSFactorCommand(TrackSpiralArcSpiral *sparcs, double newFactor, DataCommand *parent = NULL);
    virtual ~SetSpArcSFactorCommand();

    virtual int id() const
    {
        return 0x110;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    SetSpArcSFactorCommand(); /* not allowed */
    SetSpArcSFactorCommand(const SetSpArcSFactorCommand &); /* not allowed */

private:
    TrackSpiralArcSpiral *sparcs_;

    double oldFactor_;
    double newFactor_;
};

//##########################//
// TrackComponentSinglePointCommand //
//##########################//
#if 0
class TrackComponentSinglePointCommand : public DataCommand
{
public:
	explicit TrackComponentSinglePointCommand(TrackComponent * lowSlotTrack, TrackComponent * highSlotTrack, const QPointF & dPos, DataCommand * parent = NULL);
	virtual ~TrackComponentSinglePointCommand();

	virtual int				id() const { return 0x111; }

	virtual void			undo();
	virtual void			redo();

	virtual bool			mergeWith(const QUndoCommand * other);


private:
	TrackComponentSinglePointCommand():DataCommand(){ /* not allowed */ }
	TrackComponentSinglePointCommand(const TrackComponentSinglePointCommand &):DataCommand(){ /* not allowed */ }


private:
	TrackComponent *		lowSlotTrack_;
	TrackComponent *		highSlotTrack_;

	QPointF					oldPoint_;
	QPointF					dPos_;

};
#endif

//##########################//
// SetGlobalTrackPosCommand //
//##########################//

class SetGlobalTrackPosCommand : public DataCommand
{
public:
    explicit SetGlobalTrackPosCommand(TrackComponent *track, const QPointF &newPos, DataCommand *parent = NULL);
    virtual ~SetGlobalTrackPosCommand();

    virtual int id() const
    {
        return 0x112;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    SetGlobalTrackPosCommand()
        : DataCommand()
    { /* not allowed */
    }
    SetGlobalTrackPosCommand(const SetGlobalTrackPosCommand &)
        : DataCommand()
    { /* not allowed */
    }
    SetGlobalTrackPosCommand &operator=(const SetGlobalTrackPosCommand &); /* not allowed */

private:
    TrackComponent *track_;

    QPointF oldPos_;
    QPointF newPos_;
};


//##########################//
// SetGlobalTrackPosCommand //
//##########################//

class SetTrackLengthCommand : public DataCommand
{
public:
    explicit SetTrackLengthCommand(TrackComponent *track, double newLength, DataCommand *parent = NULL);
    virtual ~SetTrackLengthCommand();

    virtual int id() const
    {
        return 0x113;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    SetTrackLengthCommand()
        : DataCommand()
    { /* not allowed */
    }
    SetTrackLengthCommand(const SetTrackLengthCommand &)
        : DataCommand()
    { /* not allowed */
    }
    SetTrackLengthCommand &operator=(const SetTrackLengthCommand &); /* not allowed */

private:
    TrackComponent *track_;

    double oldLength_;
    double newLength_;
};
//##########################//
// TrackComponentHeadingCommand //
//##########################//
#if 0
class TrackComponentHeadingCommand : public DataCommand
{
public:
	explicit TrackComponentHeadingCommand(TrackComponent * track, double newHeading, DataCommand * parent = NULL);
	virtual ~TrackComponentHeadingCommand();

	virtual int				id() const { return 0x114; }

	virtual void			undo();
	virtual void			redo();


private:
	TrackComponentHeadingCommand():DataCommand(){ /* not allowed */ }
	TrackComponentHeadingCommand(const TrackComponentHeadingCommand &):DataCommand(){ /* not allowed */ }


private:
	TrackComponent *		track_;

	double					oldHeading_;
	double					newHeading_;

};
#endif

//#########################//
// ArcCurvatureCommand        //
//#########################//
#if 0
class ArcCurvatureCommand : public DataCommand
{
public:
	explicit ArcCurvatureCommand(TrackElementArc * arc, double newCurvature, DataCommand * parent = NULL);
	virtual ~ArcCurvatureCommand();

	virtual int				id() const { return 0x118; }

	virtual void			undo();
	virtual void			redo();


private:
	ArcCurvatureCommand():DataCommand(){ /* not allowed */ }
	ArcCurvatureCommand(const ArcCurvatureCommand &):DataCommand(){ /* not allowed */ }


private:
	TrackElementArc *		arc_;

	double					oldCurvature_;
	double					newCurvature_;

};
#endif

//##########################//
// MoveTrackCommand //
//##########################//

class MoveTrackCommand : public DataCommand
{
public:
    explicit MoveTrackCommand(TrackComponent *track, const QPointF &newPos, DataCommand *parent = NULL);
    virtual ~MoveTrackCommand();

    virtual int id() const
    {
        return 0x120;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    MoveTrackCommand()
        : DataCommand()
    { /* not allowed */
    }
    MoveTrackCommand(const MoveTrackCommand &)
        : DataCommand()
    { /* not allowed */
    }
    MoveTrackCommand &operator=(const MoveTrackCommand &); /* not allowed */

    bool buildSubCommand();

private:
    TrackComponent *track_;

    TrackComponent *preTrack_;
    TrackComponent *postTrack_;

    QPointF oldPos_;
    QPointF newPos_;

    TrackComponentGlobalPointsCommand *subCommand_;
};

//##########################//
// MorphIntoPoly3Command //
//##########################//

class MorphIntoPoly3Command : public DataCommand
{
public:
    explicit MorphIntoPoly3Command(TrackComponent *track, DataCommand *parent = NULL);
    virtual ~MorphIntoPoly3Command();

    virtual int id() const
    {
        return 0x150;
    }

    virtual void undo();
    virtual void redo();

private:
    MorphIntoPoly3Command()
        : DataCommand()
    { /* not allowed */
    }
    MorphIntoPoly3Command(const MorphIntoPoly3Command &)
        : DataCommand()
    { /* not allowed */
    }
    MorphIntoPoly3Command &operator=(const MorphIntoPoly3Command &); /* not allowed */

private:
    TrackComponent *oldTrack_;
    TrackElementPoly3 *newTrack_;

    RSystemElementRoad *parentRoad_;
};

//##################//
// UngroupTrackCompositeCommand //
//##################//

class UngroupTrackCompositeCommand : public DataCommand
{
public:
    explicit UngroupTrackCompositeCommand(TrackComposite *composite, DataCommand *parent = NULL);
    virtual ~UngroupTrackCompositeCommand();

    virtual int id() const
    {
        return 0x108;
    }

    virtual void undo();
    virtual void redo();

private:
    UngroupTrackCompositeCommand(); /* not allowed */
    UngroupTrackCompositeCommand(const UngroupTrackCompositeCommand &); /* not allowed */
    UngroupTrackCompositeCommand &operator=(const UngroupTrackCompositeCommand &); /* not allowed */

private:
    RSystemElementRoad *parentRoad_;

    TrackComponent *oldComposite_;
    QList<TrackComponent *> newComponents_;
};

//##########################//
// TranslateTrackComponentsCommand //
//##########################//

class TranslateTrackComponentsCommand : public DataCommand
{
private:
    enum TransformType
    {
        TT_MOVE = 1,
        TT_ROTATE = 2
    };

    struct TrackMoveProperties
    {
        TrackComponent *highSlot;
        TrackComponent *lowSlot;
        QPointF dPos;
        double heading;
        double oldHeading;
        short int transform;
    };

public:
    explicit TranslateTrackComponentsCommand(const QMultiMap<TrackComponent *, bool> &selectedTrackComponents, const QPointF &mousePos, const QPointF &pressPos, DataCommand *parent = NULL);
    virtual ~TranslateTrackComponentsCommand();

    virtual int id() const
    {
        return 0x109;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    TranslateTrackComponentsCommand(); /* not allowed */
    TranslateTrackComponentsCommand(const TranslateTrackComponentsCommand &); /* not allowed */

    bool validate(TrackMoveProperties *props);
    void translate(TrackMoveProperties *props);
    void undoTranslate(TrackMoveProperties *props);

private:
    QList<TrackMoveProperties *> translateTracks_;
    QMultiMap<TrackComponent *, bool> selectedTrackComponents_;

    QList<RSystemElementRoad *> roads_;
    // Sections to remove //
    //
    QMap<int, TypeSection *> typeSections_;
    QMap<int, ElevationSection *> elevationSections_;
    QMap<int, SuperelevationSection *> superelevationSections_;
    QMap<int, CrossfallSection *> crossfallSections_;
    QMap<int, LaneSection *> laneSections_;
	QMap<int, ShapeSection *> shapeSections_;

    QMap<int, LaneSection *> laneSectionsAdd_; // lane sections to add //
};

#endif // TRACKCOMMANDS_HPP
