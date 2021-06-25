/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/15/2010
**
**************************************************************************/

#ifndef JUNCTIONLANEROADSYSTEMITEM_HPP
#define JUNCTIONLANEROADSYSTEMITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class JunctionLaneRoadSystemItem : public RoadSystemItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit JunctionLaneRoadSystemItem(TopviewGraph *projectGraph, RoadSystem *roadSystem);
    virtual ~JunctionLaneRoadSystemItem();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    JunctionLaneRoadSystemItem(); /* not allowed */
    JunctionLaneRoadSystemItem(const JunctionLaneRoadSystemItem &); /* not allowed */
    JunctionLaneRoadSystemItem &operator=(const JunctionLaneRoadSystemItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
};

#endif // JUNCTIONLANEROADSYSTEMITEM_HPP
