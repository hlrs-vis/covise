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

#include "elevationsection.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

//####################//
// Constructors       //
//####################//

ElevationSection::ElevationSection(double s, double a, double b, double c, double d)
    : RoadSection(s)
    , Polynomial(a, b, c, d)
{
}

void
ElevationSection::setParameters(double a, double b, double c, double d)
{
    protectedSetParameters(a, b, c, d);

    addElevationSectionChanges(ElevationSection::CEL_ParameterChange);
}

/*! \brief Checks if the sections are equal.
*
* Checks the parameters a, b, c and d
*/
bool
ElevationSection::isEqualTo(ElevationSection *otherSection) const
{
    if ((getA() - otherSection->getA() <= NUMERICAL_ZERO8)
        && (getB() - otherSection->getB() <= NUMERICAL_ZERO8)
        && (getC() - otherSection->getC() <= NUMERICAL_ZERO8)
        && (getD() - otherSection->getD() <= NUMERICAL_ZERO8))
    {
        return true;
    }
    else
    {
        return false;
    }
}

//####################//
// Section Functions  //
//####################//

/*!
* Returns the end coordinate of this section.
* In road coordinates [m].
*
*/
double
ElevationSection::getSEnd() const
{
    return getParentRoad()->getElevationSectionEnd(getSStart());
}

/*!
* Returns the length coordinate of this section.
* In [m].
*
*/
double
ElevationSection::getLength() const
{
    return getParentRoad()->getElevationSectionEnd(getSStart()) - getSStart();
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
ElevationSection::notificationDone()
{
    elevationSectionChanges_ = 0x0;
    RoadSection::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
ElevationSection::addElevationSectionChanges(int changes)
{
    if (changes)
    {
        elevationSectionChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
ElevationSection *
ElevationSection::getClone()
{
    // ElevationSection //
    //
    ElevationSection *clone = new ElevationSection(getSStart(), a_, b_, c_, d_);

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
ElevationSection::accept(Visitor *visitor)
{
    visitor->visit(this);
}
