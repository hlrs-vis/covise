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

#ifndef SUPERELEVATIONROADSYSTEMITEM_HPP
#define SUPERELEVATIONROADSYSTEMITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class SuperelevationRoadSystemItem : public RoadSystemItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SuperelevationRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem);
    explicit SuperelevationRoadSystemItem(ProfileGraph *profileGraph, RoadSystem *roadSystem);
    virtual ~SuperelevationRoadSystemItem();

    // Obsever Pattern //
    //
    virtual void updateObserver();

    // delete this item
    virtual bool deleteRequest()
    {
        return false;
    };

private:
    SuperelevationRoadSystemItem(); /* not allowed */
    SuperelevationRoadSystemItem(const SuperelevationRoadSystemItem &); /* not allowed */
    SuperelevationRoadSystemItem &operator=(const SuperelevationRoadSystemItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
};

#endif // SUPERELEVATIONROADSYSTEMITEM_HPP
