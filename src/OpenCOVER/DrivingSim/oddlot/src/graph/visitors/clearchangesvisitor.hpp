/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   29.03.2010
**
**************************************************************************/

#ifndef CLEARCHANGESVISITOR_HPP
#define CLEARCHANGESVISITOR_HPP

#include "src/data/acceptor.hpp"

/*! DEPRECATED.
*
* Once designed to reset all changes to 0x0. But now
* every changed Subject is registered by the ChangeManager
* and reset later!
*/
class ClearChangesVisitor : public Visitor
{
public:
    ClearChangesVisitor();
    virtual ~ClearChangesVisitor(){};

    virtual void visit(Acceptor * /*acceptor*/){ /* does nothing by default */ };

    virtual void visit(RoadSystem *a);

    virtual void visit(RSystemElementRoad *a);
    virtual void visit(RSystemElementController *a);
    virtual void visit(RSystemElementJunction *a);
    virtual void visit(RSystemElementFiddleyard *a);

    virtual void visit(TypeSection *a);

    //	virtual void visit(TrackElementLine * a);
    //	virtual void visit(TrackElementArc * a);
    //	virtual void visit(TrackElementSpiral * a);

    virtual void visit(LaneSection *a);
    //	virtual void visit(Lane * a);
    //	virtual void visit(LaneWidth * a);
    //	virtual void visit(LaneRoadMark * a);
    //	virtual void visit(LaneSpeed * a);
    //
    //	virtual void visit(FiddleyardSink * a);
    //	virtual void visit(FiddleyardSource * a);
};

#endif // CLEARCHANGESVISITOR_HPP
