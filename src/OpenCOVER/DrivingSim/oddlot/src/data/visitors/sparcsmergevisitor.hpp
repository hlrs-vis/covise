/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   08.04.2010
**
**************************************************************************/

#ifndef SPARCSMERGEVISITOR_HPP
#define SPARCSMERGEVISITOR_HPP

#include "src/data/acceptor.hpp"

class TrackEditor;

/*! This class merges contiguous spiral-arc-spiral combinations.
*/
class SpArcSMergeVisitor : public Visitor
{

    //################//
    // STATIC         //
    //################//

private:
    enum State
    {
        STATE_INSPIRAL,
        STATE_ARC_POS,
        STATE_ARC_NEG,
        STATE_OUTSPIRAL_POS,
        STATE_OUTSPIRAL_NEG,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SpArcSMergeVisitor();
    virtual ~SpArcSMergeVisitor()
    { /* does nothing */
    }

    // Visitor Pattern //
    //
    virtual void visit(RoadSystem *roadSystem);
    virtual void visit(RSystemElementRoad *road);

    virtual void visit(TrackElementLine *);
    virtual void visit(TrackElementArc *);
    virtual void visit(TrackElementSpiral *);

private:
    //	SpArcSMergeVisitor(); /* not allowed */
    SpArcSMergeVisitor(const SpArcSMergeVisitor &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    SpArcSMergeVisitor::State state_;

    RSystemElementRoad *road_;

    TrackElementSpiral *inSpiral_;
    TrackElementArc *arc_;
    TrackElementSpiral *outSpiral_;
};

#endif // SPARCSMERGEVISITOR_HPP
