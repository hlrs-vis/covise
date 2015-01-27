/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   15.03.2010
**
**************************************************************************/

#include "surfacesection.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

//####################//
// Constructors       //
//####################//

SurfaceSection::SurfaceSection()
{
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
SurfaceSection *
SurfaceSection::getClone()
{
    SurfaceSection *clone = new SurfaceSection();
    for (int i = 0; i < getNumCRG(); i++)
    {
        clone->addCRG(getFile(i),
                      getSStart(i),
                      getSEnd(i),
                      getOrientation(i),
                      getMode(i),
                      getSOffset(i),
                      getTOffset(i),
                      getZOffset(i),
                      getZScale(i),
                      getHOffset(i));
    }
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
SurfaceSection::accept(Visitor *visitor)
{
    visitor->visit(this);
}
