/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   17.03.2010
**
**************************************************************************/

#ifndef ROADMARKVISITOR_HPP
#define ROADMARKVISITOR_HPP

#include "src/data/acceptor.hpp"

class QGraphicsItem;
class ProjectEditor;

class RoadMarkVisitor : public Visitor
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    RoadMarkVisitor(ProjectEditor *editor, QGraphicsItem *parentItem);

    // Visitor Pattern //
    //
    virtual void visit(RoadSystem *system);
    virtual void visit(RSystemElementRoad *road);

    virtual void visit(LaneSection *laneSection);
    virtual void visit(Lane *lane);
    virtual void visit(LaneRoadMark *mark);

private:
    RoadMarkVisitor(); /* not allowed */
    RoadMarkVisitor(const RoadMarkVisitor &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    ProjectEditor *editor_;

    QGraphicsItem *rootItem_;

    QGraphicsItem *currentSectionItem_;

    RSystemElementRoad *currentRoad_;
    LaneSection *currentLaneSection_;
    Lane *currentLane_;
};

#endif // ROADMARKVISITOR_HPP
