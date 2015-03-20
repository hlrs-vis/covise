/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11.03.2010
**
**************************************************************************/

#ifndef ROADLINKEDITOR_HPP
#define ROADLINKEDITOR_HPP

#include "projecteditor.hpp"

class ProjectData;
class TopviewGraph;

class RoadLinkRoadSystemItem;
class RSystemElementRoad;
class LaneSection;
class Lane;

class RoadLinkEditor : public ProjectEditor
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadLinkEditor(ProjectWidget *projectWidget, ProjectData *projectData, TopviewGraph *topviewGraph);
    virtual ~RoadLinkEditor();

    // Tool //
    //
    virtual void toolAction(ToolAction *);

protected:
    virtual void init();
    virtual void kill();

private:
    RoadLinkEditor(); /* not allowed */
    RoadLinkEditor(const RoadLinkEditor &); /* not allowed */
    RoadLinkEditor &operator=(const RoadLinkEditor &); /* not allowed */

    //################//
    // SLOTS          //
    //################//

public slots:

    //################//
    // PROPERTIES     //
    //################//

private:
    // RoadSystem //
    //
    RoadLinkRoadSystemItem *roadSystemItem_;
    double threshold_;

    // create lane links //
    //
    void createLaneLinks(RSystemElementRoad * road);
    double getTValue(LaneSection * laneSection, Lane * lane, double s, double laneWidth);
    void removeZeroWidthLanes(RSystemElementRoad * road); 
};

#endif // ROADLINKEDITOR_HPP
