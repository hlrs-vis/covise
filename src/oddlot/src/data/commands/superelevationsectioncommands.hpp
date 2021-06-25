/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   16.07.2010
**
**************************************************************************/

#ifndef SUPERELEVATIONSECTIONCOMMANDS_HPP
#define SUPERELEVATIONSECTIONCOMMANDS_HPP

// 1200

#include "datacommand.hpp"

#include "src/data/scenerysystem/heightmap.hpp"

class RSystemElementRoad;
class SuperelevationSection;
class CrossfallSection;

// Qt //
//
#include <QList>
#include <QPointF>

//################//
// Split          //
//################//

class SplitSuperelevationSectionCommand : public DataCommand
{
public:
    explicit SplitSuperelevationSectionCommand(SuperelevationSection *superelevationSection, double splitPos, DataCommand *parent = NULL);
    virtual ~SplitSuperelevationSectionCommand();

    virtual int id() const
    {
        return 0x1201;
    }

    virtual void undo();
    virtual void redo();

private:
    SplitSuperelevationSectionCommand(); /* not allowed */
    SplitSuperelevationSectionCommand(const SplitSuperelevationSectionCommand &); /* not allowed */
    SplitSuperelevationSectionCommand &operator=(const SplitSuperelevationSectionCommand &); /* not allowed */

private:
    SuperelevationSection *oldSection_;
    SuperelevationSection *newSection_;

    double splitPos_;
};

//################//
// Merge          //
//################//

class MergeSuperelevationSectionCommand : public DataCommand
{
public:
    explicit MergeSuperelevationSectionCommand(SuperelevationSection *superelevationSectionLow, SuperelevationSection *superelevationSectionHigh, DataCommand *parent = NULL);
    virtual ~MergeSuperelevationSectionCommand();

    virtual int id() const
    {
        return 0x1202;
    }

    virtual void undo();
    virtual void redo();

private:
    MergeSuperelevationSectionCommand(); /* not allowed */
    MergeSuperelevationSectionCommand(const MergeSuperelevationSectionCommand &); /* not allowed */
    MergeSuperelevationSectionCommand &operator=(const MergeSuperelevationSectionCommand &); /* not allowed */

private:
    SuperelevationSection *oldSectionLow_;
    SuperelevationSection *oldSectionHigh_;
    SuperelevationSection *newSection_;

    RSystemElementRoad *parentRoad_;
};

//################//
// Remove          //
//################//

class RemoveSuperelevationSectionCommand : public DataCommand
{
public:
    explicit RemoveSuperelevationSectionCommand(SuperelevationSection *superelevationSection, DataCommand *parent = NULL);
    virtual ~RemoveSuperelevationSectionCommand();

    virtual int id() const
    {
        return 0x1204;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveSuperelevationSectionCommand(); /* not allowed */
    RemoveSuperelevationSectionCommand(const RemoveSuperelevationSectionCommand &); /* not allowed */
    RemoveSuperelevationSectionCommand &operator=(const RemoveSuperelevationSectionCommand &); /* not allowed */

private:
    SuperelevationSection *oldSectionLow_;
    SuperelevationSection *oldSectionMiddle_;
    SuperelevationSection *oldSectionHigh_;

    SuperelevationSection *newSectionHigh_;

    RSystemElementRoad *parentRoad_;
};

//################//
// Smooth          //
//################//

class SmoothSuperelevationSectionCommand : public DataCommand
{
public:
    explicit SmoothSuperelevationSectionCommand(SuperelevationSection *superelevationSectionLow, SuperelevationSection *superelevationSectionHigh, double radius, DataCommand *parent = NULL);
    virtual ~SmoothSuperelevationSectionCommand();

    virtual int id() const
    {
        return 0x1208;
    }

    virtual void undo();
    virtual void redo();

private:
    SmoothSuperelevationSectionCommand(); /* not allowed */
    SmoothSuperelevationSectionCommand(const SmoothSuperelevationSectionCommand &); /* not allowed */
    SmoothSuperelevationSectionCommand &operator=(const SmoothSuperelevationSectionCommand &); /* not allowed */

private:
    SuperelevationSection *oldSectionLow_;
    SuperelevationSection *oldSectionHigh_;
    SuperelevationSection *newSection_;
    SuperelevationSection *newSectionHigh_;

    RSystemElementRoad *parentRoad_;

    double sLow_;
    double sHigh_;

    double radius_;
};

//################//
// Move           //
//################//

class SuperelevationMovePointsCommand : public DataCommand
{
public:
    explicit SuperelevationMovePointsCommand(const QList<SuperelevationSection *> &endPointSections, const QList<SuperelevationSection *> &startPointSections, const QPointF &deltaPos, DataCommand *parent = NULL);
    virtual ~SuperelevationMovePointsCommand();

    virtual int id() const
    {
        return 0x1210;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    SuperelevationMovePointsCommand(); /* not allowed */
    SuperelevationMovePointsCommand(const SuperelevationMovePointsCommand &); /* not allowed */
    SuperelevationMovePointsCommand &operator=(const SuperelevationMovePointsCommand &); /* not allowed */

private:
    QList<SuperelevationSection *> endPointSections_;
    QList<SuperelevationSection *> startPointSections_;

    bool superelevationOnly_;

    QList<double> oldStartPointsSs_;
    QList<double> oldStartPointsAs_;
    QList<double> oldStartPointsBs_;

    QList<double> oldEndPointsBs_;

    QPointF deltaPos_;
};

//##################//
// Apply Heightmaps //
//##################//

class ApplyHeightMapSuperelevationCommand : public DataCommand
{
public:
    explicit ApplyHeightMapSuperelevationCommand(RSystemElementRoad *road, const QList<Heightmap *> &maps, double heightOffset, double sampleDistance, double maxDeviation, double lowPassFilter, bool useCubic, bool smoothLinear = false, double radius = 0.0, DataCommand *parent = NULL);
    virtual ~ApplyHeightMapSuperelevationCommand();

    virtual int id() const
    {
        return 0x840;
    }

    virtual void undo();
    virtual void redo();

    double getSuperelevation(double s);
    double getCrossfall(double s);

    //	virtual bool			mergeWith(const QUndoCommand * other);

private:
    ApplyHeightMapSuperelevationCommand(); /* not allowed */
    ApplyHeightMapSuperelevationCommand(const ApplyHeightMapSuperelevationCommand &); /* not allowed */
    ApplyHeightMapSuperelevationCommand &operator=(const ApplyHeightMapSuperelevationCommand &); /* not allowed */

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
    double lastCrossfall_;

    QMap<double, SuperelevationSection *> newSections_;
    QMap<double, SuperelevationSection *> oldSections_;

    QMap<double, CrossfallSection *> newCSections_;
    QMap<double, CrossfallSection *> oldCSections_;
};

#endif // SUPERELEVATIONSECTIONCOMMANDS_HPP
