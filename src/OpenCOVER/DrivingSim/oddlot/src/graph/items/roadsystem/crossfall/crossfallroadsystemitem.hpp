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

#ifndef CROSSFALLROADSYSTEMITEM_HPP
#define CROSSFALLROADSYSTEMITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class CrossfallRoadSystemItem : public RoadSystemItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit CrossfallRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem);
    explicit CrossfallRoadSystemItem(ProfileGraph *profileGraph, RoadSystem *roadSystem);
    virtual ~CrossfallRoadSystemItem();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    CrossfallRoadSystemItem(); /* not allowed */
    CrossfallRoadSystemItem(const CrossfallRoadSystemItem &); /* not allowed */
    CrossfallRoadSystemItem &operator=(const CrossfallRoadSystemItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
};

#endif // CROSSFALLROADSYSTEMITEM_HPP
