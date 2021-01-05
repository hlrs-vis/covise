/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   07.06.2010
**
**************************************************************************/

#include "pedestriangroup.hpp"

// Data //
//
#include "pedestriansystem.hpp"
#include "pedestrian.hpp"

PedestrianGroup::PedestrianGroup()
    : DataElement()
    , parentPedestrianSystem_(NULL)
    , pedestrianGroupChanges_(0x0)
    , spawnRangeSet_(false)
    , maxPedsSet_(false)
    , reportIntervalSet_(false)
    , avoidCountSet_(false)
    , avoidTimeSet_(false)
    , autoFiddleSet_(false)
    , movingFiddleSet_(false)
{
}

PedestrianGroup::~PedestrianGroup()
{
    // Delete child nodes //
    //
    foreach (Pedestrian *child, pedestrians_)
    {
        delete child;
    }
}

//##################//
// PedestrianGroup  //
//##################//

void
PedestrianGroup::addPedestrian(Pedestrian *pedestrian)
{
    addPedestrianGroupChanges(PedestrianGroup::CVG_PedestrianChanged);
    pedestrians_.insert(parentPedestrianSystem_->getUniqueId(pedestrian->getId()), pedestrian);

    pedestrian->setParentPedestrianGroup(this);
}

//##################//
// PedestrianSystem //
//##################//

void
PedestrianGroup::setParentPedestrianSystem(PedestrianSystem *pedestrianSystem)
{
    parentPedestrianSystem_ = pedestrianSystem;
    setParentElement(pedestrianSystem);
    addPedestrianGroupChanges(PedestrianGroup::CVG_PedestrianSystemChanged);
}

//##################//
// Observer Pattern //
//##################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
PedestrianGroup::notificationDone()
{
    pedestrianGroupChanges_ = 0x0;
    DataElement::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
PedestrianGroup::addPedestrianGroupChanges(int changes)
{
    if (changes)
    {
        pedestrianGroupChanges_ |= changes;
        notifyObservers();
    }
}

//##################//
// Visitor Pattern  //
//##################//

/*! \brief Accepts a visitor.
*
*/
void
PedestrianGroup::accept(Visitor *visitor)
{
    visitor->visit(this);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
PedestrianGroup::acceptForChildNodes(Visitor *visitor)
{
    acceptForPedestrians(visitor);
}

/*! \brief Accepts a visitor and passes it to child nodes.
*/
void
PedestrianGroup::acceptForPedestrians(Visitor *visitor)
{
    foreach (Pedestrian *child, pedestrians_)
    {
        child->accept(visitor);
    }
}
