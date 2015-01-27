/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   29.03.2010
**
**************************************************************************/

#include "clearchangesvisitor.hpp"

#include "src/data/dataelement.hpp"

#include "src/data/roadsystem/roadsystem.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/rsystemelementcontroller.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/rsystemelementfiddleyard.hpp"

#include "src/data/roadsystem/sections/typesection.hpp"

#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementspiral.hpp"

#include "src/data/roadsystem/sections/lanesection.hpp"

ClearChangesVisitor::ClearChangesVisitor()
{
}

void
ClearChangesVisitor::visit(RoadSystem *a)
{
    a->notificationDone();
}

void
ClearChangesVisitor::visit(RSystemElementRoad *a)
{
    a->notificationDone();
}
void
ClearChangesVisitor::visit(RSystemElementController *a)
{
    a->notificationDone();
}
void
ClearChangesVisitor::visit(RSystemElementJunction *a)
{
    a->notificationDone();
}
void
ClearChangesVisitor::visit(RSystemElementFiddleyard *a)
{
    a->notificationDone();
}

void
ClearChangesVisitor::visit(TypeSection *a)
{
    a->notificationDone();
}

//void
//	ClearChangesVisitor
//	::visit(TrackElementLine * a)
//{
//	a->notificationDone();
//}
//
//void
//	ClearChangesVisitor
//	::visit(TrackElementArc * a)
//{
//	a->notificationDone();
//}
//
//void
//	ClearChangesVisitor
//	::visit(TrackElementSpiral * a)
//{
//	a->notificationDone();
//}

void
ClearChangesVisitor::visit(LaneSection *a)
{
    a->notificationDone();
}
//void
//	ClearChangesVisitor
//	::visit(Lane * a)
//{
//	a->notificationDone();
//}
//
//void
//	ClearChangesVisitor
//	::visit(LaneWidth * a)
//{
//	a->notificationDone();
//}
//
//void
//	ClearChangesVisitor
//	::visit(LaneRoadMark * a)
//{
//	a->notificationDone();
//}
//
//void
//	ClearChangesVisitor
//	::visit(LaneSpeed * a)
//{
//	a->notificationDone();
//}
//
//void
//	ClearChangesVisitor
//	::visit(FiddleyardSink * a)
//{
//	a->notificationDone();
//}
//
//void
//	ClearChangesVisitor
//	::visit(FiddleyardSource * a)
//{
//	a->notificationDone();
//}
