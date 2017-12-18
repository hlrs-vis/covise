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

#include "georeference.hpp"


//####################//
// Constructors       //
//####################//

GeoReference::GeoReference(const QString &geoReferenceParams)
{
	geoReferenceParams_ = geoReferenceParams;
}

GeoReference::~GeoReference()
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
GeoReference::notificationDone()
{
	geoReferenceChanges_ = 0x0;
}

/*! \brief Add one or more change flags.
*
*/
void
GeoReference::addGeoReferenceChanges(int changes)
{
    if (changes)
    {
        geoReferenceChanges_ |= changes;
        notifyObservers();
    }
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
GeoReference::accept(Visitor *visitor)
{
    visitor->visit(this);
}
