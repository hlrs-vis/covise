/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/18/2010
**
**************************************************************************/

#include "scenerytesselation.hpp"

#include "scenerysystem.hpp"

SceneryTesselation::SceneryTesselation()
    : DataElement()
    , parentScenerySystem_(NULL)
    , sceneryTesselationChanges_(0x0)
    , tesselateRoads_(true)
{
}

SceneryTesselation::~SceneryTesselation()
{
}

//###################//
// SceneryTesselation //
//###################//

void
SceneryTesselation::setTesselateRoads(bool tesselate)
{
    tesselateRoads_ = tesselate;
    addSceneryTesselationChanges(SceneryTesselation::CST_TesselateRoadsChanged);
}

void
SceneryTesselation::setTesselatePaths(bool tesselate)
{
    tesselatePaths_ = tesselate;
    addSceneryTesselationChanges(SceneryTesselation::CST_TesselatePathsChanged);
}

void
SceneryTesselation::setParentScenerySystem(ScenerySystem *scenerySystem)
{
    parentScenerySystem_ = scenerySystem;
    setParentElement(scenerySystem);
    addSceneryTesselationChanges(SceneryTesselation::CST_ScenerySystemChanged);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
SceneryTesselation::notificationDone()
{
    sceneryTesselationChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
SceneryTesselation::addSceneryTesselationChanges(int changes)
{
    if (changes)
    {
        sceneryTesselationChanges_ |= changes;
        notifyObservers();
    }
}

//###################//
// Visitor Pattern   //
//###################//

/*!
* Accepts a visitor and passes it to all child
* nodes if autoTraverse is true.
*/
void
SceneryTesselation::accept(Visitor *visitor)
{
    visitor->visit(this);
}
