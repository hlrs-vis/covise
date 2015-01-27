/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   08.04.2010
**
**************************************************************************/

#include "sparcsmergevisitor.hpp"

// Data //
//
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "src/data/roadsystem/track/trackelementline.hpp"
#include "src/data/roadsystem/track/trackelementarc.hpp"
#include "src/data/roadsystem/track/trackelementspiral.hpp"

#include "src/data/roadsystem/track/trackspiralarcspiral.hpp"

#include "src/util/odd.hpp"
#include "math.h"

/*!
*
*/
SpArcSMergeVisitor::SpArcSMergeVisitor()
    : state_(SpArcSMergeVisitor::STATE_INSPIRAL)
    , inSpiral_(NULL)
    , arc_(NULL)
    , outSpiral_(NULL)
{
}

/*!
*
*/
void
SpArcSMergeVisitor::visit(RoadSystem *roadSystem)
{
    // Run //
    //
    roadSystem->acceptForRoads(this);
}

/*!
*
*/
void
SpArcSMergeVisitor::visit(RSystemElementRoad *road)
{
    road_ = road;

    // New road, new luck //
    //
    state_ = SpArcSMergeVisitor::STATE_INSPIRAL; // back to start
    inSpiral_ = NULL;
    arc_ = NULL;
    outSpiral_ = NULL;

    // Track Elements //
    //
    road->acceptForTracks(this);
}

// TODO: ANDERE COMPOSITES ABFANGEN //

/*!
*
*/
void
SpArcSMergeVisitor::visit(TrackElementLine * /*trackElement*/)
{
    // Spoil it //
    //
    state_ = SpArcSMergeVisitor::STATE_INSPIRAL; // back to start
    inSpiral_ = NULL;
    arc_ = NULL;
    outSpiral_ = NULL;

    return;
}

/*!
*
*/
void
SpArcSMergeVisitor::visit(TrackElementArc *trackElement)
{
    if (inSpiral_)
    {
        arc_ = trackElement;
        state_ = SpArcSMergeVisitor::STATE_OUTSPIRAL_POS;
    }
    /*	if(state_ == SpArcSMergeVisitor::STATE_ARC_POS)
	{
		if(trackElement->getCurvature(trackElement->getSStart()) > 0.0)
		{
			// jackpot: next one should be a spiral with positive curvature
			state_ = SpArcSMergeVisitor::STATE_OUTSPIRAL_POS;
			arc_ = trackElement;
			return;
		}
	}
	else if(state_ == SpArcSMergeVisitor::STATE_ARC_NEG)
	{
		if(trackElement->getCurvature(trackElement->getSStart()) < 0.0)
		{
			// jackpot: next one should be a spiral with negative curvature
			state_ = SpArcSMergeVisitor::STATE_OUTSPIRAL_NEG;
			arc_ = trackElement;
			return;
		}
	}*/

    else
    {

        // Back to start //
        //
        state_ = SpArcSMergeVisitor::STATE_INSPIRAL; // back to start
        inSpiral_ = NULL;
        arc_ = NULL;
        outSpiral_ = NULL;
    }

    return;
}

/*!
*
*/
void
SpArcSMergeVisitor::visit(TrackElementSpiral *trackElement)
{
    if ((state_ == SpArcSMergeVisitor::STATE_INSPIRAL) || (state_ == SpArcSMergeVisitor::STATE_ARC_POS))
    {
        inSpiral_ = trackElement;
        state_ = SpArcSMergeVisitor::STATE_ARC_POS;
        return;
        // Condition: curvStart == 0 //
        //
        /*		if(fabs(trackElement->getCurvature(trackElement->getSStart())) <= NUMERICAL_ZERO)
		{
			if(trackElement->getCurvature(trackElement->getSEnd()) > 0.0)
			{
				// jackpot: next one should be an arc with positive curvature
				state_ = SpArcSMergeVisitor::STATE_ARC_POS;
				inSpiral_ = trackElement;
				return;
			}
			else if(trackElement->getCurvature(trackElement->getSEnd()) < 0.0)
			{
				// jackpot: next one should be an arc with negative curvature
				state_ = SpArcSMergeVisitor::STATE_ARC_NEG;
				inSpiral_ = trackElement;
				return;
			}
		}*/
    }

    else if (state_ == SpArcSMergeVisitor::STATE_OUTSPIRAL_POS)
    {
        road_->delTrackComponent(inSpiral_);
        road_->delTrackComponent(arc_);
        road_->delTrackComponent(trackElement);

        TrackSpiralArcSpiral *spArcS = new TrackSpiralArcSpiral(inSpiral_, arc_, trackElement);

        road_->addTrackComponent(spArcS);

        // Condition: curvEnd == 0 //
        //
        /*		if(fabs(trackElement->getCurvature(trackElement->getSEnd())) <= NUMERICAL_ZERO)
		{
			if(trackElement->getCurvature(trackElement->getSStart()) > 0.0)
			{
				// jackpot //
				//
				road_->delTrackComponent(inSpiral_);
				road_->delTrackComponent(arc_);
				road_->delTrackComponent(trackElement);

				TrackSpiralArcSpiral * spArcS = new TrackSpiralArcSpiral(inSpiral_, arc_, trackElement);

				road_->addTrackComponent(spArcS);
			}
		}
	}

	else if(state_ == SpArcSMergeVisitor::STATE_OUTSPIRAL_NEG)
	{
		// Condition: curvEnd == 0 //
		//
		if(fabs(trackElement->getCurvature(trackElement->getSEnd())) <= NUMERICAL_ZERO)
		{
			if(trackElement->getCurvature(trackElement->getSStart()) < 0.0)
			{
				// jackpot //
				//
				road_->delTrackComponent(inSpiral_);
				road_->delTrackComponent(arc_);
				road_->delTrackComponent(trackElement);

				TrackSpiralArcSpiral * spArcS = new TrackSpiralArcSpiral(inSpiral_, arc_, trackElement);

				road_->addTrackComponent(spArcS);
			}
		}*/
    }

    // Back to start //
    //
    state_ = SpArcSMergeVisitor::STATE_INSPIRAL;
    inSpiral_ = NULL;
    arc_ = NULL;
    outSpiral_ = NULL;

    return;
}
