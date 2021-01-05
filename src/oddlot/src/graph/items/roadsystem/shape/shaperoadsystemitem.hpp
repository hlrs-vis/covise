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

#ifndef SHAPEROADSYSTEMITEM_HPP
#define SHAPEROADSYSTEMITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class ShapeRoadSystemItem : public RoadSystemItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ShapeRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem);
    explicit ShapeRoadSystemItem(ProfileGraph *profileGraph, RoadSystem *roadSystem);
    virtual ~ShapeRoadSystemItem();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    ShapeRoadSystemItem(); /* not allowed */
    ShapeRoadSystemItem(const ShapeRoadSystemItem &); /* not allowed */
    ShapeRoadSystemItem &operator=(const ShapeRoadSystemItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
};

#endif // SHAPEROADSYSTEMITEM_HPP
