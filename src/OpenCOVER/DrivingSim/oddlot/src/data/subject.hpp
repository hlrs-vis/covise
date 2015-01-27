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

#ifndef SUBJECT_HPP
#define SUBJECT_HPP

class Observer;
class ChangeManager;

#include <QList>

/*! \brief Observer Pattern: The Subject can be observed by some Observers.
*
* An arbitrary number of Observers can be attachted to the Subject. When
* the notify() function is called, all Observers will be notified that
* something has changed.
* This implementation uses a ChangeManager that buffers all notifications
* and notifies each Observer only once.
*/
class Subject
{
public:
    explicit Subject();
    virtual ~Subject();

    virtual ChangeManager *getChangeManager() = 0; // implemented by subclasses (DataElement and ProjectData)

    void attachObserver(Observer *observer);
    void detachObserver(Observer *observer);

    void notifyObservers();
    virtual void notificationDone() = 0;

private:
    //	Subject(); /* not allowed */
    Subject(const Subject &); /* not allowed */
    Subject &operator=(const Subject &); /* not allowed */

private:
    QList<Observer *> observers_;
};

#endif // SUBJECT_HPP
