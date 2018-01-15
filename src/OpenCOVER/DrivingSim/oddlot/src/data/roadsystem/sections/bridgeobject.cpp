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

#include "bridgeobject.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

Bridge::BridgeType 
Bridge::parseBridgeType(const QString &type)
{
	if (type == "concrete")
	{
		return Bridge::BT_CONCRETE;
	}
	else if (type == "steel")
	{
		return Bridge::BT_STEEL;
	}
	else if (type == "brick")
	{
		return Bridge::BT_BRICK;
	}
	else if (type == "wood")
	{
		return Bridge::BT_WOOD;
	}
	else
	{
		qDebug("WARNING: unknown bridge type: %s", type.toUtf8().constData());
		return Bridge::BT_UNKNOWN;
	}

}

QString 
Bridge::parseBridgeTypeBack(int type)
{
	if (type == Bridge::BT_CONCRETE)
	{
		return  QString("concrete");
	}
	else if (type == Bridge::BT_STEEL)
	{
		return  QString("steel");
	}
	else if (type == Bridge::BT_BRICK)
	{
		return  QString("brick");
	}
	else if (type == Bridge::BT_WOOD)
	{
		return  QString("wood");
	}
	else
	{
		qDebug("WARNING: unknown bridge type");
		return  QString("none");
	}
}

//####################//
// Constructors       //
//####################//

Bridge::Bridge(const odrID &id, const QString &file, const QString &name, int type, double s, double length)
    : RoadSection(s)
    , id_(id)
    , name_(name)
    , type_(type)
    , length_(length)
{
    userData_.fileName = file;
}

/*!
* Returns the end coordinate of this section.
* In road coordinates [m].
*
*/
double
Bridge::getSEnd() const
{

    return getSStart() + length_;
}


//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
Bridge::notificationDone()
{
    bridgeChanges_ = 0x0;
    RoadSection::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
Bridge::addBridgeChanges(int changes)
{
    if (changes)
    {
        bridgeChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
Bridge *
Bridge::getClone()
{
    // Bridge //
    //
    Bridge *clone = new Bridge(id_, userData_.fileName, name_, type_, getSStart(), length_);

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
Bridge::accept(Visitor *visitor)
{
    visitor->visit(this);
}
