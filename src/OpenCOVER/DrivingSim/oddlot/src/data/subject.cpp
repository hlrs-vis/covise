/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.03.2010
**
**************************************************************************/

#include "subject.hpp"

#include "changemanager.hpp"

#include "src/util/odd.hpp"
#include "src/gui/projectwidget.hpp"
#include "src/data/projectdata.hpp"

/*! \brief The Contructor does nothing.
*
*/
Subject::Subject()
{
}

/*! \brief The Destructor does nothing.
*
*
*/
Subject::~Subject()
{
    ChangeManager *changeManager = ODD::getChangeManager();
    if (changeManager)
    {
        changeManager->unregisterSubject(this);
    }

    // Do not call virtual functions in destructors!
}

/*! \brief Attach an Observer to this Subject.
*
* After attaching, all notify events from this Subject
* will reach the Observer.
*/
void
Subject::attachObserver(Observer *observer)
{
    if (!observers_.contains(observer))
    {
        observers_.append(observer);
    }
}

/*! \brief Detach an Observer from this Subject.
*
* After detaching, no more notify events from this Subject
* will reach the Observer.
* If the observer is NULL, all pairs will be unregistered.
*/
void
Subject::detachObserver(Observer *observer)
{
    observers_.removeOne(observer);
}

/*! \brief Trigger a notify event.
*
* All Observers will be notified that this Subject has changed.
*/
void
Subject::notifyObservers()
{
    //	ChangeManager * changeManager = ODD::getChangeManager();
    ChangeManager *changeManager = getChangeManager();
    if (changeManager)
    {
        changeManager->notifyObserversOf(this, observers_);
    }
}
