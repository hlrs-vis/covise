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

#include "crossfallsection.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

#include "math.h"

//###################//
// Static Functions  //
//###################//

CrossfallSection::DCrossfallSide
CrossfallSection::parseCrossfallSide(const QString &side)
{
    if (side == "left")
        return CrossfallSection::DCF_SIDE_LEFT;
    else if (side == "right")
        return CrossfallSection::DCF_SIDE_RIGHT;
    else if (side == "both")
        return CrossfallSection::DCF_SIDE_BOTH;
    else
    {
        qDebug(("WARNING: unknown crossfall side: " + side).toUtf8());
        return CrossfallSection::DCF_SIDE_NONE;
    }
};

QString
CrossfallSection::parseCrossfallSideBack(CrossfallSection::DCrossfallSide side)
{
    if (side == CrossfallSection::DCF_SIDE_LEFT)
        return QString("left");
    else if (side == CrossfallSection::DCF_SIDE_RIGHT)
        return QString("right");
    else if (side == CrossfallSection::DCF_SIDE_BOTH)
        return QString("both");
    else
    {
        qDebug("WARNING: unknown crossfall side.");
        return "both";
    }
};

//####################//
// Constructors       //
//####################//

CrossfallSection::CrossfallSection(CrossfallSection::DCrossfallSide side, double s, double a, double b, double c, double d)
    : RoadSection(s)
    , Polynomial(a, b, c, d)
    , side_(side)
{
}

double
CrossfallSection::getCrossfallRadians(double s /*, double t*/)
{
    return f(s - getSStart()) * 2.0 * M_PI / 360.0;
}

double
CrossfallSection::getCrossfallDegrees(double s /*, double t*/)
{
    return f(s - getSStart());

    //	if(t>0)
    //	{
    //		// Right side //
    //		//
    //		if(side_ != CrossfallSection::DCF_SIDE_LEFT)
    //		{
    //			return f(s - getSStart());
    //		}
    //	}
    //	else
    //	{
    //		// Left side //
    //		//
    //		if(side_ != CrossfallSection::DCF_SIDE_RIGHT)
    //		{
    //			return f(s - getSStart());
    //		}
    //	}
    //
    //	return 0.0;
}

double
CrossfallSection::getCrossfallSlopeRadians(double s)
{
    return df(s - getSStart()) * 2.0 * M_PI / 360.0;
}

double
CrossfallSection::getCrossfallSlopeDegrees(double s)
{
    return df(s - getSStart());
}

double
CrossfallSection::getCrossfallCurvatureRadians(double s)
{
    return ddf(s - getSStart()) * 2.0 * M_PI / 360.0;
}

double
CrossfallSection::getCrossfallCurvatureDegrees(double s)
{
    return ddf(s - getSStart());
}

void
CrossfallSection::setParametersDegrees(double a, double b, double c, double d)
{
    protectedSetParameters(a, b, c, d);

    addCrossfallSectionChanges(CrossfallSection::CCF_ParameterChange);
}

void
CrossfallSection::setSide(CrossfallSection::DCrossfallSide side)
{
    side_ = side;
    addCrossfallSectionChanges(CrossfallSection::CCF_SideChange);
}

/*! \brief Checks if the sections are equal.
*
* Checks the parameters a, b, c and d and the side.
*/
bool
CrossfallSection::isEqualTo(CrossfallSection *otherSection) const
{
    if ((getA() - otherSection->getA() <= NUMERICAL_ZERO8)
        && (getB() - otherSection->getB() <= NUMERICAL_ZERO8)
        && (getC() - otherSection->getC() <= NUMERICAL_ZERO8)
        && (getD() - otherSection->getD() <= NUMERICAL_ZERO8)
        && (side_ == otherSection->getSide()))
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
CrossfallSection::getSEnd() const
{
    return getParentRoad()->getCrossfallSectionEnd(getSStart());
}

/*!
* Returns the length coordinate of this section.
* In [m].
*
*/
double
CrossfallSection::getLength() const
{
    return getParentRoad()->getCrossfallSectionEnd(getSStart()) - getSStart();
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
CrossfallSection::notificationDone()
{
    crossfallSectionChanges_ = 0x0;
    RoadSection::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
CrossfallSection::addCrossfallSectionChanges(int changes)
{
    if (changes)
    {
        crossfallSectionChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
CrossfallSection *
CrossfallSection::getClone()
{
    // CrossfallSection //
    //
    CrossfallSection *clone = new CrossfallSection(side_, getSStart(), a_, b_, c_, d_);

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
CrossfallSection::accept(Visitor *visitor)
{
    visitor->visit(this);
}
