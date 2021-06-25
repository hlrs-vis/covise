/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   05.07.2010
**
**************************************************************************/

#ifndef ROADTYPEROADSYSTEMITEM_HPP
#define ROADTYPEROADSYSTEMITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class RoadTypeRoadSystemItem : public RoadSystemItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadTypeRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem);
    virtual ~RoadTypeRoadSystemItem();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    RoadTypeRoadSystemItem(); /* not allowed */
    RoadTypeRoadSystemItem(const RoadTypeRoadSystemItem &); /* not allowed */
    RoadTypeRoadSystemItem &operator=(const RoadTypeRoadSystemItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
};

#endif // ROADTYPEROADSYSTEMITEM_HPP
