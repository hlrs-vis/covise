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

#include "changemanager.hpp"

#include "subject.hpp"
#include "observer.hpp"

/*! \brief The Contructor (does nothing special).
*
*/
ChangeManager::ChangeManager(QObject *parent)
    : QObject(parent)
{
}

/*! \brief Tell the ChangeManager that a Subject has been deleted.
*
*/
void
ChangeManager::unregisterSubject(Subject *subject)
{
    changedSubjects_.removeOne(subject);
}

/*! \brief Mark all Observers of the Subject, so they will ne notified.
*
* This also registers the Subject in a list, so after the notification
* of all Observers, the Subjects change flags can be reset to 0x0.
*/
void
ChangeManager::notifyObserversOf(Subject *subject, QList<Observer *> observers)
{
    foreach (Observer *observer, observers)
    {
        // Append Observer to the list //
        //
        if (!markedObservers_.contains(observer)) // the list is not that big, so this should be ok
        {
            markedObservers_.append(observer);
        }
    }

    // Register the Subject //
    //
    // Register the Subject even if there are no Observers, because the changes must be reset to 0x0 anyway.
    if (!changedSubjects_.contains(subject)) // the list is not that big, so this should be ok
    {
        changedSubjects_.append(subject);
    }
}

/*! \brief Notify all marked Observers.
*
* Goes through the list of Observers that has been created by
* notifyObserversOf() and notifies all Observers in the list.
* Afterwards goes through the list of Subjects and tells them
* that their Observers have been notified, so the Subjects
* can reset their change flags, etc.
* Clears the two lists afterwards.
* Finally it emits a signal to tell the view that the model is
* once again in a consistent state.
*/
void
ChangeManager::notifyObservers()
{
    // Notify Observers //
    //
    foreach (Observer *observer, markedObservers_)
    {
        observer->updateObserver();
    }

    // Tell Subjects //
    //
    foreach (Subject *subject, changedSubjects_)
    {
        subject->notificationDone();
    }

    // Clean up //
    //
    markedObservers_.clear();
    changedSubjects_.clear();

    // Emit Signal //
    //
    emit notificationDone(); // (this is used e.g. for the garbage disposal of the view/controller)
}
