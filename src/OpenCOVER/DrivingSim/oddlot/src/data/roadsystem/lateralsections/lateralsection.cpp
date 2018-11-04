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

#include "lateralsection.hpp"

#include "src/data/roadsystem/sections/shapesection.hpp"

//################//
// CONSTRUCTOR    //
//################//

/*!
* Checks if a road has been passed as parent. Warns if not so.
*/
LateralSection::LateralSection(double t)
    : DataElement()
    , parentSection_(NULL)
    , t_(t)
    , lateralSectionChanges_(0x0)
{
}

LateralSection::~LateralSection()
{
}

void
LateralSection::setTStart(double t)
{
    t_ = t;
    addLateralSectionChanges(LateralSection::CLS_TChange);
}

void
LateralSection::setParentSection(ShapeSection *shapeSection)
{
    parentSection_ = shapeSection;
    setParentElement(shapeSection);
    addLateralSectionChanges(LateralSection::CLS_ParentSectionChange);
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
LateralSection::notificationDone()
{
    lateralSectionChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
LateralSection::addLateralSectionChanges(int changes)
{
    if (changes)
    {
		lateralSectionChanges_ |= changes;
        notifyObservers();
    }
}
