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

SurfaceSection::SurfaceOrientation 
SurfaceSection::parseSurfaceOrientation(const QString &orientation)
{
	if (orientation == "same")
	{
		return SurfaceSection::SurfaceOrientation::SSO_SAME;
	}
	else if (orientation == "opposite")
	{
		return SurfaceSection::SurfaceOrientation::SSO_OPPOSITE;
	}
	else
	{
		qDebug("WARNING: unknown surface orientation: %s", orientation.toUtf8().constData());
		return SurfaceSection::SurfaceOrientation::SSO_SAME;
	}
}

QString 
SurfaceSection::parseSurfaceOrientationBack(SurfaceSection::SurfaceOrientation orientation)
{
	if (orientation == SurfaceSection::SurfaceOrientation::SSO_SAME)
	{
		return "same";
	}
	else if (orientation == SurfaceSection::SurfaceOrientation::SSO_OPPOSITE)
	{
		return "opposite";
	}
	else
	{
		qDebug("WARNING: unknown surface orientation");
		return "none";
	}

}

SurfaceSection::ApplicationMode
SurfaceSection::parseApplicationMode(const QString &application)
{
	if (application == "attached")
	{
		return SurfaceSection::ApplicationMode::SSA_ATTACHED;
	}
	else if (application == "attached0")
	{
		return SurfaceSection::ApplicationMode::SSA_ATTACHED0;
	}
	else if (application == "genuine")
	{
		return SurfaceSection::ApplicationMode::SSA_GENUINE;
	}
	else
	{
		qDebug("WARNING: unknown surface application mode: %s", application.toUtf8().constData());
		return SurfaceSection::ApplicationMode::SSA_GENUINE;
	}
}

QString
SurfaceSection::parseApplicationModeBack(SurfaceSection::ApplicationMode application)
{
	if (application == SurfaceSection::ApplicationMode::SSA_ATTACHED)
	{
		return "attached";
	}
	else if (application == SurfaceSection::ApplicationMode::SSA_ATTACHED0)
	{
		return "attached0";
	}
	else if (application == SurfaceSection::ApplicationMode::SSA_GENUINE)
	{
		return "genuine";
	}
	else
	{
		qDebug("WARNING: unknown surface application mode");
		return "none";
	}

}

SurfaceSection::SurfacePurpose
SurfaceSection::parseSurfacePurpose(const QString &purpose)
{
	if (purpose == "elevation")
	{
		return SurfaceSection::SurfacePurpose::SSP_ELEVATION;
	}
	else if (purpose == "friction")
	{
		return SurfaceSection::SurfacePurpose::SSP_FRICTION;
	}
	else
	{
		qDebug("WARNING: unknown surface purpose: %s", purpose.toUtf8().constData());
		return SurfaceSection::SurfacePurpose::SSP_ELEVATION;
	}
}

QString
SurfaceSection::parseSurfacePurposeBack(SurfaceSection::SurfacePurpose purpose)
{
	if (purpose == SurfaceSection::SurfacePurpose::SSP_ELEVATION)
	{
		return "elevation";
	}
	else if (purpose == SurfaceSection::SurfacePurpose::SSP_FRICTION)
	{
		return "friction";
	}
	else
	{
		qDebug("WARNING: unknown surface orientation");
		return "none";
	}

}

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
				      getPurpose(i),
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
