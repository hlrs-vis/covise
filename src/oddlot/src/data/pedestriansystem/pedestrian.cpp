/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.06.2010
**
**************************************************************************/

#include "pedestrian.hpp"

#include "pedestriangroup.hpp"

Pedestrian::Pedestrian(
    bool defaultPed,
    bool templatePed,
    const QString &id,
    const QString &name,
    const QString &templateId,
    const QString &rangeLOD,
    const QString &debugLvl,
    const QString &modelFile,
    const QString &scale,
    const QString &heading,
    const QString &startRoadId,
    const QString &startLane,
    const QString &startDir,
    const QString &startSOffset,
    const QString &startVOffset,
    const QString &startVel,
    const QString &startAcc,
    const QString &idleIdx,
    const QString &idleVel,
    const QString &slowIdx,
    const QString &slowVel,
    const QString &walkIdx,
    const QString &walkVel,
    const QString &jogIdx,
    const QString &jogVel,
    const QString &lookIdx,
    const QString &waveIdx)
    : DataElement()
    , pedestrianChanges_(0x0)
    , parentPedestrianGroup_(NULL)
    , defaultPed_(defaultPed)
    , templatePed_(templatePed)
    , id_(id)
    , name_(name)
    , templateId_(templateId)
    , rangeLOD_(rangeLOD)
    , debugLvl_(debugLvl)
    , modelFile_(modelFile)
    , scale_(scale)
    , heading_(heading)
    , startRoadId_(startRoadId)
    , startLane_(startLane)
    , startDir_(startDir)
    , startSOffset_(startSOffset)
    , startVOffset_(startVOffset)
    , startVel_(startVel)
    , startAcc_(startAcc)
    , idleIdx_(idleIdx)
    , idleVel_(idleVel)
    , slowIdx_(slowIdx)
    , slowVel_(slowVel)
    , walkIdx_(walkIdx)
    , walkVel_(walkVel)
    , jogIdx_(jogIdx)
    , jogVel_(jogVel)
    , lookIdx_(lookIdx)
    , waveIdx_(waveIdx)
{
}

Pedestrian::~Pedestrian()
{
}

//##################//
// ProjectData      //
//##################//

void
Pedestrian::setParentPedestrianGroup(PedestrianGroup *pedestrianGroup)
{
    parentPedestrianGroup_ = pedestrianGroup;
    setParentElement(pedestrianGroup);
    addPedestrianChanges(Pedestrian::CVR_PedestrianGroupChanged);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
Pedestrian::notificationDone()
{
    pedestrianChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
Pedestrian::addPedestrianChanges(int changes)
{
    if (changes)
    {
        pedestrianChanges_ |= changes;
        notifyObservers();
    }
}

//##################//
// Visitor Pattern  //
//##################//

/*! \brief Accepts a visitor.
*
* With autotraverse: visitor will be send to roads, fiddleyards, etc.
* Without: accepts visitor as 'this'.
*/
void
Pedestrian::accept(Visitor *visitor)
{
    visitor->visit(this);
}
