/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   22.03.2010
**
**************************************************************************/

#include "signalreference.hpp"

// Data
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"


//####################//
// Constructors       //
//####################//

SignalReference::SignalReference(const odrID &id, Signal *signal, const odrID &refId, double s, double t, Signal::OrientationType orientation, QList<Signal::Validity> validity)
    : RoadSection(s)
	, id_(id)
	, refId_(refId)
	, signal_(signal)
	, refT_(t)
	, refOrientation_(orientation)
	, validity_(validity)
{

}

Signal *
SignalReference::getSignal()
{
	if (!signal_)
	{
		RoadSystem *roadSystem = getParentRoad()->getRoadSystem();
		foreach (RSystemElementRoad *road, roadSystem->getRoads())
		{
			Signal *signal = road->getSignal(refId_);
			if (signal)
			{
				signal_ = signal;
				break;
			}
		}
	}

	return signal_;
}
void 
SignalReference::setSignal(Signal *signal)
{
	signal_ = signal;
	addSignalReferenceChanges(SignalReference::SRC_SignalChange);
}

void 
SignalReference::setReferenceT(const double refT)
{
	refT_ = refT;
	addSignalReferenceChanges(SignalReference::SRC_ParameterChange);
}

void
SignalReference::setReferenceOrientation(Signal::OrientationType orientation)
{
	refOrientation_ = orientation;
	addSignalReferenceChanges(SignalReference::SRC_ParameterChange);
}

bool 
SignalReference::addValidity(int fromLane, int toLane)
{
	Signal::Validity validity{ fromLane, toLane };

    foreach (Signal::Validity entry, validity_)
	{
		if ((entry.fromLane == fromLane) && (entry.toLane == toLane))
		{
			return false;
		}
	}
	validity_.append(validity);

	return true;
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
SignalReference::notificationDone()
{
    signalReferenceChanges_ = 0x0;
    RoadSection::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
SignalReference::addSignalReferenceChanges(int changes)
{
    if (changes)
    {
        signalReferenceChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
SignalReference *
SignalReference::getClone()
{
    // SignalReference //
    //
	SignalReference *clone = new SignalReference(id_, signal_, refId_, getSStart(), refT_, refOrientation_, validity_);

    return clone;
}

//###################//
// Visitor Pattern   //
//###################//

/*!
* Accepts a visitor for this section.
*
* \param visitor The visitor that will be visited.
*/
void
SignalReference::accept(Visitor *visitor)
{
    visitor->visit(this);
}
