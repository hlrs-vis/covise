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

#include "objectreference.hpp"

// Data
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"


//####################//
// Constructors       //
//####################//

ObjectReference::ObjectReference(const QString &id, Object *object, const QString &refId, double s, double t, double zOffset, double validLength, Signal::OrientationType orientation, QList<Signal::Validity> validity)
    : RoadSection(s)
	, id_(id)
	, refId_(refId)
	, object_(object)
	, refT_(t)
	, refOrientation_(orientation)
	, refZOffset_(zOffset)
	, refValidLength_(validLength)
	, validity_(validity)
{

}

Object *
ObjectReference::getObject()
{
	if (!object_)
	{
		RoadSystem *roadSystem = getParentRoad()->getRoadSystem();
		foreach (RSystemElementRoad *road, roadSystem->getRoads())
		{
			Object *object = road->getObject(refId_);
			if (object)
			{
				object_ = object;
				break;
			}
		}
	}

	return object_;
}
void 
ObjectReference::setObject(Object *object)
{
	object_ = object;
	addObjectReferenceChanges(ObjectReference::ORC_ObjectChange);
}

void 
ObjectReference::setReferenceT(const double refT)
{
	refT_ = refT;
	addObjectReferenceChanges(ObjectReference::ORC_ParameterChange);
}

void
ObjectReference::setReferenceOrientation(Signal::OrientationType orientation)
{
	refOrientation_ = orientation;
	addObjectReferenceChanges(ObjectReference::ORC_ParameterChange);
}

bool 
ObjectReference::addValidity(int fromLane, int toLane)
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
ObjectReference::notificationDone()
{
    objectReferenceChanges_ = 0x0;
    RoadSection::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
ObjectReference::addObjectReferenceChanges(int changes)
{
    if (changes)
    {
        objectReferenceChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
ObjectReference *
ObjectReference::getClone()
{
    // ObjectReference //
    //
	ObjectReference *clone = new ObjectReference(id_, object_, refId_, getSStart(), refT_, refZOffset_, refValidLength_, refOrientation_, validity_);

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
ObjectReference::accept(Visitor *visitor)
{
    visitor->visit(this);
}
