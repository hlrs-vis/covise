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

#include "tunnelobject.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

//####################//
// Constructors       //
//####################//

Tunnel::Tunnel(const QString &id, const QString &file, const QString &name, int type, double s, double length, double lighting, double daylight)
    : Bridge(id, file, name, type, s, length)
	, lighting_(lighting)
	, daylight_(daylight)
{

}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
Tunnel::notificationDone()
{
    tunnelChanges_ = 0x0;
    RoadSection::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
Tunnel::addTunnelChanges(int changes)
{
    if (changes)
    {
        tunnelChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
Tunnel *
Tunnel::getClone()
{
    // Tunnel //
    //
	Tunnel *clone = new Tunnel(getId(), getFileName(), getName(), getType(), getSStart(), getLength(), lighting_, daylight_);

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
Tunnel::accept(Visitor *visitor)
{
    visitor->visit(this);
}
