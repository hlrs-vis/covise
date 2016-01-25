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

#include "oscroadsystemitem.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
//
#include "src/graph/items/roadsystem/scenario/oscroaditem.hpp"
#include "src/graph/items/roadsystem/controlleritem.hpp"

//################//
// CONSTRUCTOR    //
//################//

OSCRoadSystemItem::OSCRoadSystemItem(TopviewGraph *topviewGraph, RoadSystem *roadSystem)
    : RoadSystemItem(topviewGraph, roadSystem)
{
    init();
}

OSCRoadSystemItem::~OSCRoadSystemItem()
{
}

void
OSCRoadSystemItem::init()
{
    foreach (RSystemElementRoad *road, getRoadSystem()->getRoads())
    {
        new OSCRoadItem(this, road);
    }

    // Controllers //
    //
    foreach (RSystemElementController *controller, getRoadSystem()->getControllers())
    {
        (new ControllerItem(this, controller))->setZValue(-1.0);
    }
}

