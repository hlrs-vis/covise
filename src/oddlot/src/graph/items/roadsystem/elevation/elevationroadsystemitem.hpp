/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.07.2010
**
**************************************************************************/

#ifndef ELEVATIONROADSYSTEMITEM_HPP
#define ELEVATIONROADSYSTEMITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class ElevationRoadSystemItem : public RoadSystemItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ElevationRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem);
    explicit ElevationRoadSystemItem(ProfileGraph *profileGraph, RoadSystem *roadSystem);
    virtual ~ElevationRoadSystemItem();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    ElevationRoadSystemItem(); /* not allowed */
    ElevationRoadSystemItem(const ElevationRoadSystemItem &); /* not allowed */
    ElevationRoadSystemItem &operator=(const ElevationRoadSystemItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
};

#endif // ELEVATIONROADSYSTEMITEM_HPP
