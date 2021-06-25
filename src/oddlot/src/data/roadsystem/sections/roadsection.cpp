/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   18.03.2010
**
**************************************************************************/

#include "roadsection.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

//################//
// CONSTRUCTOR    //
//################//

/*!
* Checks if a road has been passed as parent. Warns if not so.
*/
RoadSection::RoadSection(double s)
    : DataElement()
    , parentRoad_(NULL)
    , s_(s)
    , roadSectionChanges_(0x0)
{
}

RoadSection::~RoadSection()
{
}

void
RoadSection::setSStart(double s)
{
    s_ = s;
    addRoadSectionChanges(RoadSection::CRS_SChange);
}

void
RoadSection::setParentRoad(RSystemElementRoad *parentRoad)
{
    parentRoad_ = parentRoad;
    setParentElement(parentRoad);
    addRoadSectionChanges(RoadSection::CRS_ParentRoadChange);
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
RoadSection::notificationDone()
{
    roadSectionChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
RoadSection::addRoadSectionChanges(int changes)
{
    if (changes)
    {
        roadSectionChanges_ |= changes;
        notifyObservers();
    }
}
