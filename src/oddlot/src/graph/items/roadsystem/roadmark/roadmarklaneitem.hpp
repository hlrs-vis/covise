/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/22/2010
**
**************************************************************************/

#ifndef ROADMARKLANEITEM_HPP
#define ROADMARKLANEITEM_HPP

#include "src/graph/items/graphelement.hpp"

class RoadMarkLaneSectionItem;

class RoadMarkLaneItem : public GraphElement
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadMarkLaneItem(RoadMarkLaneSectionItem *parentLaneSectionItem, Lane *lane);
    virtual ~RoadMarkLaneItem();

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest()
    {
        return false;
    };

private:
    RoadMarkLaneItem(); /* not allowed */
    RoadMarkLaneItem(const RoadMarkLaneItem &); /* not allowed */
    RoadMarkLaneItem &operator=(const RoadMarkLaneItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
    RoadMarkLaneSectionItem *parentLaneSectionItem_;

    Lane *lane_;

    RSystemElementRoad *grandparentRoad_;
};

#endif // ROADMARKLANEITEM_HPP
