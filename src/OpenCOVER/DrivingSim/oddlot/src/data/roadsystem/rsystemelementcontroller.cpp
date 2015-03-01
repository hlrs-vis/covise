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

#include "src/data/roadsystem/sections/signalobject.hpp"

/** CONSTRUCTOR.
*/
RSystemElementController::RSystemElementController(const QString &name, const QString &id, int sequence, const QString &script, double cycleTime, const QList<ControlEntry *> &controlEntries)
    : RSystemElement(name, id, RSystemElement::DRE_Controller)
    , sequence_(sequence)
    , controlEntries_(controlEntries) /*,
		controllerChanges_(0x0)*/
{
    controllerUserData_.script = script;
    controllerUserData_.cycleTime = cycleTime;
}

RSystemElementController::RSystemElementController(const QString &name, const QString &id, int sequence, ControllerUserData &controllerUserData, const QList<ControlEntry *> &controlEntries)
    : RSystemElement(name, id, RSystemElement::DRE_Controller)
    , sequence_(sequence)
    , controllerUserData_(controllerUserData)
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

void
RSystemElementController::addControlEntry(ControlEntry *controlEntry, Signal * signal)
{

    // Append and Notify //
    //
    controlEntries_.append(controlEntry);
    signals_.insert(signal->getId(), signal);
    signal->attachObserver(this);
    addControllerChanges(RSystemElementController::CRC_EntryChange);
}

bool
RSystemElementController::delControlEntry(ControlEntry *controlEntry, Signal * signal)
{
    
    controlEntries_.removeOne(controlEntry);
    signals_.remove(controlEntry->getSignalId());
    addControllerChanges(RSystemElementController::CRC_EntryChange);

    return true;
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
RSystemElementController::notificationDone()
{
    controllerChanges_ = 0x0;
    RSystemElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
RSystemElementController::addControllerChanges(int changes)
{
    if (changes)
    {
        controllerChanges_ |= changes;
        notifyObservers();
    }
}

void
RSystemElementController::updateObserver()
{

    bool entryChanged = false;
    QMap<QString, Signal *>::const_iterator iter = signals_.constBegin();
    while (iter != signals_.constEnd())
    {
        Signal * signal = iter.value();
        int changes = signal->getSignalChanges();

        if (changes & Signal::CEL_ParameterChange)
        {
            foreach (ControlEntry *controlEntry, controlEntries_)
            {
                if (controlEntry->getSignalId() == iter.key())
                {
                    controlEntry->setType(QString::number(signal->getType()));
                    if (iter.key() != signal->getId())
                    {
                        delControlEntry(controlEntry, signal);
                        controlEntry->setSignalId(signal->getId());
                        addControlEntry(controlEntry, signal);
                    }
                    entryChanged = true;
                    break;
                }
            }
        }

        if (entryChanged)
        {
            break;
        }

        iter++;
    }

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
