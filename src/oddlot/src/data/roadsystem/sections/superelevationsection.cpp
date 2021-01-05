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

#include "superelevationsection.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "math.h"

//####################//
// Constructors       //
//####################//

SuperelevationSection::SuperelevationSection(double s, double a, double b, double c, double d)
    : RoadSection(s)
    , Polynomial(a, b, c, d)
{
}

double
SuperelevationSection::getSuperelevationRadians(double s /*, double t*/)
{
    return f(s - getSStart()) * 2.0 * M_PI / 360.0;
}

double
SuperelevationSection::getSuperelevationDegrees(double s /*, double t*/)
{
    return f(s - getSStart());
}

double
SuperelevationSection::getSuperelevationSlopeRadians(double s /*, double t*/)
{
    return df(s - getSStart()) * 2.0 * M_PI / 360.0;
}

double
SuperelevationSection::getSuperelevationSlopeDegrees(double s /*, double t*/)
{
    return df(s - getSStart());
}

double
SuperelevationSection::getSuperelevationCurvatureRadians(double s /*, double t*/)
{
    return ddf(s - getSStart()) * 2.0 * M_PI / 360.0;
}

double
SuperelevationSection::getSuperelevationCurvatureDegrees(double s /*, double t*/)
{
    return ddf(s - getSStart());
}

void
SuperelevationSection::setParametersDegrees(double a, double b, double c, double d)
{
    protectedSetParameters(a, b, c, d);

    addSuperelevationSectionChanges(SuperelevationSection::CSE_ParameterChange);
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
SuperelevationSection::getSEnd() const
{
    return getParentRoad()->getSuperelevationSectionEnd(getSStart());
}

/*!
* Returns the length coordinate of this section.
* In [m].
*
*/
double
SuperelevationSection::getLength() const
{
    return getParentRoad()->getSuperelevationSectionEnd(getSStart()) - getSStart();
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
SuperelevationSection::notificationDone()
{
    superelevationSectionChanges_ = 0x0;
    RoadSection::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
SuperelevationSection::addSuperelevationSectionChanges(int changes)
{
    if (changes)
    {
        superelevationSectionChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
SuperelevationSection *
SuperelevationSection::getClone()
{
    // SuperelevationSection //
    //
    SuperelevationSection *clone = new SuperelevationSection(getSStart(), a_, b_, c_, d_);

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
SuperelevationSection::accept(Visitor *visitor)
{
    visitor->visit(this);
}
