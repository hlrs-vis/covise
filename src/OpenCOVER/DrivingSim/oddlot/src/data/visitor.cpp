/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   04.02.2010
**
**************************************************************************/

#include "visitor.hpp"

// Data //
//
#include "src/data/roadsystem/track/trackcomponent.hpp"
#include "src/data/roadsystem/track/trackcomposite.hpp"
#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementspiral.hpp"
#include "src/data/roadsystem/track/trackelementpoly3.hpp"
#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"
#include "src/data/roadsystem/track/trackelementcubiccurve.hpp"

#include "src/data/roadsystem/sections/lanewidth.hpp"
#include "src/data/roadsystem/sections/laneborder.hpp"

#include "src/data/scenerysystem/heightmap.hpp"

Visitor::Visitor()
{
}

void
Visitor::visit(TrackComponent * /*acceptor*/)
{
    // nothing in here
}

void
Visitor::visit(TrackComposite *acceptor)
{
    visit(static_cast<TrackComponent *>(acceptor));
}

void
Visitor::visit(TrackElement *acceptor)
{
    visit(static_cast<TrackComponent *>(acceptor));
}

void
Visitor::visit(TrackElementLine *acceptor)
{
    visit(static_cast<TrackElement *>(acceptor));
}

void
Visitor::visit(TrackElementArc *acceptor)
{
    visit(static_cast<TrackElement *>(acceptor));
}

void
Visitor::visit(TrackElementSpiral *acceptor)
{
    visit(static_cast<TrackElement *>(acceptor));
}

void
Visitor::visit(TrackElementPoly3 *acceptor)
{
    visit(static_cast<TrackElement *>(acceptor));
}

void
Visitor::visit(TrackSpiralArcSpiral *acceptor)
{
    visit(static_cast<TrackComposite *>(acceptor));
}

void
Visitor::visit(TrackElementCubicCurve *acceptor)
{
	visit(static_cast<TrackElement *>(acceptor));
}

void
Visitor::visit(SceneryMap * /*acceptor*/)
{
    // nothing in here
}

void
Visitor::visit(Heightmap *acceptor)
{
    visit(static_cast<SceneryMap *>(acceptor));
}

void
Visitor::visit(LaneBorder *acceptor)
{
	visit(static_cast<LaneWidth *>(acceptor));
}
