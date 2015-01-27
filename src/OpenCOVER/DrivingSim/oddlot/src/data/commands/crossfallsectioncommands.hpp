/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   14.07.2010
**
**************************************************************************/

#ifndef CROSSFALLSECTIONCOMMANDS_HPP
#define CROSSFALLSECTIONCOMMANDS_HPP

// 1100

#include "datacommand.hpp"

class RSystemElementRoad;
class CrossfallSection;

// Qt //
//
#include <QList>
#include <QPointF>

//################//
// Split          //
//################//

class SplitCrossfallSectionCommand : public DataCommand
{
public:
    explicit SplitCrossfallSectionCommand(CrossfallSection *crossfallSection, double splitPos, DataCommand *parent = NULL);
    virtual ~SplitCrossfallSectionCommand();

    virtual int id() const
    {
        return 0x1101;
    }

    virtual void undo();
    virtual void redo();

private:
    SplitCrossfallSectionCommand(); /* not allowed */
    SplitCrossfallSectionCommand(const SplitCrossfallSectionCommand &); /* not allowed */
    SplitCrossfallSectionCommand &operator=(const SplitCrossfallSectionCommand &); /* not allowed */

private:
    CrossfallSection *oldSection_;
    CrossfallSection *newSection_;

    double splitPos_;
};

//################//
// Merge          //
//################//

class MergeCrossfallSectionCommand : public DataCommand
{
public:
    explicit MergeCrossfallSectionCommand(CrossfallSection *crossfallSectionLow, CrossfallSection *crossfallSectionHigh, DataCommand *parent = NULL);
    virtual ~MergeCrossfallSectionCommand();

    virtual int id() const
    {
        return 0x1102;
    }

    virtual void undo();
    virtual void redo();

private:
    MergeCrossfallSectionCommand(); /* not allowed */
    MergeCrossfallSectionCommand(const MergeCrossfallSectionCommand &); /* not allowed */
    MergeCrossfallSectionCommand &operator=(const MergeCrossfallSectionCommand &); /* not allowed */

private:
    CrossfallSection *oldSectionLow_;
    CrossfallSection *oldSectionHigh_;
    CrossfallSection *newSection_;

    RSystemElementRoad *parentRoad_;
};

//################//
// Remove          //
//################//

class RemoveCrossfallSectionCommand : public DataCommand
{
public:
    explicit RemoveCrossfallSectionCommand(CrossfallSection *crossfallSection, DataCommand *parent = NULL);
    virtual ~RemoveCrossfallSectionCommand();

    virtual int id() const
    {
        return 0x1104;
    }

    virtual void undo();
    virtual void redo();

private:
    RemoveCrossfallSectionCommand(); /* not allowed */
    RemoveCrossfallSectionCommand(const RemoveCrossfallSectionCommand &); /* not allowed */
    RemoveCrossfallSectionCommand &operator=(const RemoveCrossfallSectionCommand &); /* not allowed */

private:
    CrossfallSection *oldSectionLow_;
    CrossfallSection *oldSectionMiddle_;
    CrossfallSection *oldSectionHigh_;

    CrossfallSection *newSectionHigh_;

    RSystemElementRoad *parentRoad_;
};

//################//
// Smooth          //
//################//

class SmoothCrossfallSectionCommand : public DataCommand
{
public:
    explicit SmoothCrossfallSectionCommand(CrossfallSection *crossfallSectionLow, CrossfallSection *crossfallSectionHigh, double radius, DataCommand *parent = NULL);
    virtual ~SmoothCrossfallSectionCommand();

    virtual int id() const
    {
        return 0x1108;
    }

    virtual void undo();
    virtual void redo();

private:
    SmoothCrossfallSectionCommand(); /* not allowed */
    SmoothCrossfallSectionCommand(const SmoothCrossfallSectionCommand &); /* not allowed */
    SmoothCrossfallSectionCommand &operator=(const SmoothCrossfallSectionCommand &); /* not allowed */

private:
    CrossfallSection *oldSectionLow_;
    CrossfallSection *oldSectionHigh_;
    CrossfallSection *newSection_;
    CrossfallSection *newSectionHigh_;

    RSystemElementRoad *parentRoad_;

    double sLow_;
    double sHigh_;

    double radius_;
};

//################//
// Move           //
//################//

class CrossfallMovePointsCommand : public DataCommand
{
public:
    explicit CrossfallMovePointsCommand(const QList<CrossfallSection *> &endPointSections, const QList<CrossfallSection *> &startPointSections, const QPointF &deltaPos, DataCommand *parent = NULL);
    virtual ~CrossfallMovePointsCommand();

    virtual int id() const
    {
        return 0x1110;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    CrossfallMovePointsCommand(); /* not allowed */
    CrossfallMovePointsCommand(const CrossfallMovePointsCommand &); /* not allowed */
    CrossfallMovePointsCommand &operator=(const CrossfallMovePointsCommand &); /* not allowed */

private:
    QList<CrossfallSection *> endPointSections_;
    QList<CrossfallSection *> startPointSections_;

    bool crossfallOnly_;

    QList<double> oldStartPointsSs_;
    QList<double> oldStartPointsAs_;
    QList<double> oldStartPointsBs_;

    QList<double> oldEndPointsBs_;

    QPointF deltaPos_;
};

#endif // CROSSFALLSECTIONCOMMANDS_HPP
