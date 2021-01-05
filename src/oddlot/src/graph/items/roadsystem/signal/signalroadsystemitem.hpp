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

#ifndef SIGNALROADSYSTEMITEM_HPP
#define SIGNALROADSYSTEMITEM_HPP

#include "src/graph/items/roadsystem/roadsystemitem.hpp"

class SignalRoadSystemItem : public RoadSystemItem
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SignalRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem);
    virtual ~SignalRoadSystemItem();

    // Obsever Pattern //
    //
    virtual void updateObserver();

private:
    SignalRoadSystemItem(); /* not allowed */
    SignalRoadSystemItem(const SignalRoadSystemItem &); /* not allowed */
    SignalRoadSystemItem &operator=(const SignalRoadSystemItem &); /* not allowed */

    void init();

    //################//
    // PROPERTIES     //
    //################//

private:
};

#endif // ROADTYPEROADSYSTEMITEM_HPP
