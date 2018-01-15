/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.06.2010
**
**************************************************************************/

#ifndef ELEVATIONSECTIONCOMMANDS_HPP
#define ELEVATIONSECTIONCOMMANDS_HPP

// 800

#include "datacommand.hpp"

class ElevationSection;
class RSystemElementRoad;
class RSystemElementJunction;

class SceneryMap;
class Heightmap;

#include <QList>
#include <QPointF>
#include <QMap>
#include <QMultiMap>

//################//
// Split          //
//################//

class SplitElevationSectionCommand : public DataCommand
{
public:
    explicit SplitElevationSectionCommand(ElevationSection *elevationSection, double splitPos, DataCommand *parent = NULL);
    virtual ~SplitElevationSectionCommand();

    virtual int id() const
    {
        return 0x801;
    }

    virtual void undo();
    virtual void redo();

private:
    SplitElevationSectionCommand(); /* not allowed */
    SplitElevationSectionCommand(const SplitElevationSectionCommand &); /* not allowed */
    SplitElevationSectionCommand &operator=(const SplitElevationSectionCommand &); /* not allowed */

private:
    ElevationSection *oldSection_;
    ElevationSection *newSection_;

    double splitPos_;
};

//################//
// Merge          //
//################//

class MergeElevationSectionCommand : public DataCommand
{
public:
    explicit MergeElevationSectionCommand(ElevationSection *elevationSectionLow, ElevationSection *elevationSectionHigh, DataCommand *parent = NULL);
    virtual ~MergeElevationSectionCommand();

    virtual int id() const
    {
        return 0x802;
    }

    virtual void undo();
    virtual void redo();

private:
    MergeElevationSectionCommand(); /* not allowed */
    MergeElevationSectionCommand(const MergeElevationSectionCommand &); /* not allowed */
    MergeElevationSectionCommand &operator=(const MergeElevationSectionCommand &); /* not allowed */

private:
    ElevationSection *oldSectionLow_;
    ElevationSection *oldSectionHigh_;
    ElevationSection *newSection_;

    RSystemElementRoad *parentRoad_;
};

//################//
// Remove          //
//################//

class RemoveElevationSectionCommand : public DataCommand
{
public:
    explicit RemoveElevationSectionCommand(ElevationSection *elevationSection, DataCommand *parent = NULL);
    virtual ~RemoveElevationSectionCommand();

    virtual int id() const
    {
        return 0x804;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveElevationSectionCommand(); /* not allowed */
    RemoveElevationSectionCommand(const RemoveElevationSectionCommand &); /* not allowed */
    RemoveElevationSectionCommand &operator=(const RemoveElevationSectionCommand &); /* not allowed */

private:
    ElevationSection *oldSectionLow_;
    ElevationSection *oldSectionMiddle_;
    ElevationSection *oldSectionHigh_;

    ElevationSection *newSectionHigh_;

    RSystemElementRoad *parentRoad_;
};

//################//
// Smooth          //
//################//

class SmoothElevationSectionCommand : public DataCommand
{
public:
    explicit SmoothElevationSectionCommand(ElevationSection *elevationSectionLow, ElevationSection *elevationSectionHigh, double radius, DataCommand *parent = NULL);
    virtual ~SmoothElevationSectionCommand();

    virtual int id() const
    {
        return 0x808;
    }

    virtual void undo();
    virtual void redo();

private:
    SmoothElevationSectionCommand(); /* not allowed */
    SmoothElevationSectionCommand(const SmoothElevationSectionCommand &); /* not allowed */
    SmoothElevationSectionCommand &operator=(const SmoothElevationSectionCommand &); /* not allowed */

private:
    ElevationSection *oldSectionLow_;
    ElevationSection *oldSectionHigh_;
    ElevationSection *newSection_;
    ElevationSection *newSectionHigh_;

    RSystemElementRoad *parentRoad_;

    double sLow_;
    double sHigh_;

    double radius_;
};

//################//
// Smooth          //
//################//

class SmoothElevationRoadsCommand : public DataCommand
{
    enum contactPoints
    {
        EndStart = 1,
        EndEnd = 2,
        StartStart = 3
    };

public:
    explicit SmoothElevationRoadsCommand(ElevationSection *elevationSectionLow, ElevationSection *elevationSectionHigh, double radius, DataCommand *parent = NULL);
    virtual ~SmoothElevationRoadsCommand();

    virtual int id() const
    {
        return 0x808;
    }

    virtual void undo();
    virtual void redo();

private:
    SmoothElevationRoadsCommand(); /* not allowed */
    SmoothElevationRoadsCommand(const SmoothElevationRoadsCommand &); /* not allowed */
    SmoothElevationRoadsCommand &operator=(const SmoothElevationRoadsCommand &); /* not allowed */

    bool createSmoothSections(ElevationSection *elevationSectionLow, RSystemElementRoad *parentLow, ElevationSection *elevationSectionHigh, RSystemElementRoad *parentHigh, contactPoints contact);

private:
    RSystemElementRoad *parentRoad1_;
    RSystemElementRoad *parentRoad2_;

    double radius_;

    QMultiMap<RSystemElementRoad *, ElevationSection *> oldSections_;
    QMultiMap<RSystemElementRoad *, ElevationSection *> newSections_;
};

//################//
// Move           //
//################//

class SetElevationCommand : public DataCommand
{
public:
    explicit SetElevationCommand(const QList<ElevationSection *> &endPointSections, const QList<ElevationSection *> &startPointSections, double deltaHeight, DataCommand *parent = NULL);
    virtual ~SetElevationCommand();

    virtual int id() const
    {
        return 0x810;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    SetElevationCommand(); /* not allowed */
    SetElevationCommand(const SetElevationCommand &); /* not allowed */
    SetElevationCommand &operator=(const SetElevationCommand &); /* not allowed */

private:
    QList<ElevationSection *> endPointSections_;
    QList<ElevationSection *> startPointSections_;

    QList<double> oldEndHeights_;
    QList<double> oldStartHeights_;

    double deltaHeight_;
};

//################//
// Move           //
//################//

class ElevationMovePointsCommand : public DataCommand
{
public:
    explicit ElevationMovePointsCommand(const QList<ElevationSection *> &endPointSections, const QList<ElevationSection *> &startPointSections, const QPointF &deltaPos, DataCommand *parent = NULL);
    virtual ~ElevationMovePointsCommand();

    virtual int id() const
    {
        return 0x820;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    ElevationMovePointsCommand(); /* not allowed */
    ElevationMovePointsCommand(const ElevationMovePointsCommand &); /* not allowed */
    ElevationMovePointsCommand &operator=(const ElevationMovePointsCommand &); /* not allowed */

private:
    QList<ElevationSection *> endPointSections_;
    QList<ElevationSection *> startPointSections_;

    bool elevationOnly_;

    QList<double> oldStartPointsSs_;
    QList<double> oldStartPointsAs_;
    QList<double> oldStartPointsBs_;

    QList<double> oldEndPointsBs_;

    QPointF deltaPos_;
};

//################//
// Move           //
//################//

class ElevationSetHeightCommand : public DataCommand
{
public:
    explicit ElevationSetHeightCommand(const QList<ElevationSection *> &endPointSections, const QList<ElevationSection *> &startPointSections, float height, bool absolute = true, DataCommand *parent = NULL);
    virtual ~ElevationSetHeightCommand();

    virtual int id() const
    {
        return 0x830;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    ElevationSetHeightCommand(); /* not allowed */
    ElevationSetHeightCommand(const ElevationMovePointsCommand &); /* not allowed */
    ElevationSetHeightCommand &operator=(const ElevationSetHeightCommand &); /* not allowed */

private:
    QList<ElevationSection *> endPointSections_;
    QList<ElevationSection *> startPointSections_;

    QList<double> oldStartPointsAs_;
    QList<double> oldStartPointsBs_;
    QList<double> oldStartPointsCs_;
    QList<double> oldStartPointsDs_;

    QList<double> oldEndPointsAs_;
    QList<double> oldEndPointsBs_;
    QList<double> oldEndPointsCs_;
    QList<double> oldEndPointsDs_;

    float newHeight;
    bool absoluteHeight;
};

//##################//
// Apply Heightmaps //
//##################//

class ApplyHeightMapElevationCommand : public DataCommand
{
public:
    explicit ApplyHeightMapElevationCommand(RSystemElementRoad *road, const QList<Heightmap *> &maps, double heightOffset, double sampleDistance, double maxDeviation, double lowPassFilter, bool useCubic, bool smoothLinear = false, double radius = 0.0, DataCommand *parent = NULL);
    virtual ~ApplyHeightMapElevationCommand();

    virtual int id() const
    {
        return 0x840;
    }

    virtual void undo();
    virtual void redo();

    double getHeight(double s);

    //	virtual bool			mergeWith(const QUndoCommand * other);

private:
    ApplyHeightMapElevationCommand(); /* not allowed */
    ApplyHeightMapElevationCommand(const ApplyHeightMapElevationCommand &); /* not allowed */
    ApplyHeightMapElevationCommand &operator=(const ApplyHeightMapElevationCommand &); /* not allowed */

private:
    RSystemElementRoad *road_;
    QList<Heightmap *> maps_;

    double heightOffset_;
    double sampleDistance_;
    double maxDeviation_;
    double lowPassFilter_;
    bool useCubic_;
    bool smoothLinear_;
    double radius_;

    QMap<double, ElevationSection *> newSections_;
    QMap<double, ElevationSection *> oldSections_;
};

//##################//
// SetEndElevationCommand //
//##################//

class SetEndElevationCommand : public DataCommand
{
public:
    explicit SetEndElevationCommand(RSystemElementRoad *road, bool isStart, double newHeight, double newSlope, double transitionLength, DataCommand *parent = NULL);
    virtual ~SetEndElevationCommand();

    virtual int id() const
    {
        return 0x811;
    }

    virtual void undo();
    virtual void redo();

private:
    SetEndElevationCommand(); /* not allowed */
    SetEndElevationCommand(const SetEndElevationCommand &); /* not allowed */
    SetEndElevationCommand &operator=(const SetEndElevationCommand &); /* not allowed */

private:
    RSystemElementRoad *road_;

    ElevationSection *newSection_;
    ElevationSection *newSectionB_;
    QList<ElevationSection *> oldSections_;
};

//##################//
// Apply Heightmaps //
//##################//

class FlatJunctionsElevationCommand : public DataCommand
{
public:
    explicit FlatJunctionsElevationCommand(RSystemElementJunction *junction, double transitionLength, DataCommand *parent = NULL);
    virtual ~FlatJunctionsElevationCommand();

    virtual int id() const
    {
        return 0x880;
    }

    virtual void undo();
    virtual void redo();

private:
    FlatJunctionsElevationCommand(); /* not allowed */
    FlatJunctionsElevationCommand(const FlatJunctionsElevationCommand &); /* not allowed */
    FlatJunctionsElevationCommand &operator=(const FlatJunctionsElevationCommand &); /* not allowed */

private:
    RSystemElementJunction *junction_;

    QList<RSystemElementRoad *> paths_;
    QList<RSystemElementRoad *> endRoads_;
    QList<RSystemElementRoad *> startRoads_;

    QMultiMap<QString, ElevationSection *> newSections_;
    QMultiMap<QString, ElevationSection *> oldSections_;

    QList<SetEndElevationCommand *> roadCommands_;
};

//##################################//
// Select ElevationSection         //
//################################//

class SelectElevationSectionCommand : public DataCommand
{
public:
    explicit SelectElevationSectionCommand(const QList<ElevationSection *> &endPointSections, const QList<ElevationSection *> &startPointSections, DataCommand *parent = NULL);
    virtual ~SelectElevationSectionCommand();

    virtual int id() const
    {
        return 0x810;
    }

    virtual void undo(){};
    virtual void redo();

private:
    SelectElevationSectionCommand(); /* not allowed */
    SelectElevationSectionCommand(const SelectElevationSectionCommand &); /* not allowed */
    SelectElevationSectionCommand &operator=(const SelectElevationSectionCommand &); /* not allowed */

private:
    QList<ElevationSection *> endPointSections_;
    QList<ElevationSection *> startPointSections_;
};

#endif // ELEVATIONSECTIONCOMMANDS_HPP
