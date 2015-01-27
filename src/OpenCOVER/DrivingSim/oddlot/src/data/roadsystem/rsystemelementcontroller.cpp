/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.02.2010
**
**************************************************************************/

#include "rsystemelementcontroller.hpp"

/** CONSTRUCTOR.
*/
RSystemElementController::RSystemElementController(const QString &name, const QString &id, int sequence, const QString &script, double cycleTime, const QList<ControlEntry *> &controlEntries)
    : RSystemElement(name, id, RSystemElement::DRE_Controller)
    , name_(name)
    , id_(id)
    , sequence_(sequence)
    , script_(script)
    , cycleTime_(cycleTime)
    , controlEntries_(controlEntries) /*,
		controllerChanges_(0x0)*/
{
}

/** DESTRUCTOR.
*/
RSystemElementController::~RSystemElementController()
{
    controlEntries_.clear();
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
RSystemElementController *
RSystemElementController::getClone()
{
    return new RSystemElementController(getName(), getID(), getSequence(), getScript(), getCycleTime(), getControlEntries());
}

//###################//
// Visitor Pattern   //
//###################//

/*!
* Accepts a visitor and passes it to all child
* nodes if autoTraverse is true.
*/
void
RSystemElementController::accept(Visitor *visitor)
{
    visitor->visit(this);
}
