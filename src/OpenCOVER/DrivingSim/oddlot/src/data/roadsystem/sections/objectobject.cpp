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

#include "objectobject.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/roadsystem/sections/parkingspaceobject.hpp"

//####################//
// Constructors       //
//####################//

Object::Object(const odrID &id, const QString &name, double s, ObjectProperties &objectProps, ObjectRepeatRecord &repeatRecord, const QString &textureFile)
	: RoadSection(s)
	, id_(id)
	, name_(name)
	, objectProps_(objectProps)
	, objectRepeat_(repeatRecord)
	, parkingSpace_(NULL)
	, outline_(NULL)
{
	userData_.modelFile = name;
	userData_.textureFile = textureFile;
}

/*!
* Returns the end coordinate of this section.
* In road coordinates [m].
*
*/
double
Object::getSEnd() const
{
    if (objectProps_.validLength < objectProps_.length)
    {
        return getSStart() + objectProps_.validLength;
    }
    else
    {
        return objectRepeat_.s + objectRepeat_.length;
    }
}

void 
Object::setParkingSpace(ParkingSpace *parkingSpace)
{
	objectProps_.type = "parkingSpace";
	parkingSpace_ = parkingSpace;
	parkingSpace->setParentObject(this);
	addObjectChanges(Object::CEL_TypeChange);
}

void 
Object::setOutline(Outline *outline)
{
	outline_ = outline;
	outline_->setParentObject(this);
	addObjectChanges(Object::CEL_OutlineChange);
}


//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
Object::notificationDone()
{
    objectChanges_ = 0x0;
    RoadSection::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
Object::addObjectChanges(int changes)
{
    if (changes)
    {
        objectChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
Object *
Object::getClone()
{
    // Object //
    //
	Object *clone = new Object(id_, name_, getSStart(), objectProps_, objectRepeat_, userData_.textureFile);

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
Object::accept(Visitor *visitor)
{
    visitor->visit(this);
}



//####################//
// Outline            //
//####################//

Outline::Outline(QList<ObjectCorner *> corners)
	: corners_(corners)
{

}

Outline::~Outline()
{
	foreach(ObjectCorner *corner, corners_)
		delete corner;
}

void 
Outline::setParentObject(Object *object)
{
	parentObject_ = object;
	object->addObjectChanges(Object::CEL_OutlineChange);
}

bool 
Outline::addCorner(ObjectCorner *corner)
{
	if (corners_.contains(corner))
	{
		return false;
	}

	corners_.append(corner);

	return true;
}


//####################//
// ObjectCorner       //
//####################//

