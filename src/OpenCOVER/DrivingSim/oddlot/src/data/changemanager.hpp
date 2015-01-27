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

#ifndef CHANGEMANAGER_HPP
#define CHANGEMANAGER_HPP

#include <QObject>

class Observer;
class Subject;

// Qt //
//
#include <QList>

/*! \brief Observer Pattern: The ChangeManager handles all notifications.
*
* The ChangeManager buffers all notifications and notifies each
* Observer only once.
*
* Then all Subjects are called to tell them that their Observers have
* been notified, so the Subjects can reset their change flags, etc.
*/

class ChangeManager : public QObject
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ChangeManager(QObject *parent);
    virtual ~ChangeManager()
    { /* does nothing */
    }

    void unregisterSubject(Subject *subject);
    void notifyObserversOf(Subject *subject, QList<Observer *> observers);

private:
    ChangeManager(); /* not allowed */
    ChangeManager(const ChangeManager &); /* not allowed */
    ChangeManager &operator=(const ChangeManager &); /* not allowed */

//################//
// SIGNALS        //
//################//

signals:
    void notificationDone();

    //################//
    // SLOTS          //
    //################//

public slots:
    void notifyObservers();

    //################//
    // PROPERTIES     //
    //################//

private:
    QList<Observer *> markedObservers_;
    QList<Subject *> changedSubjects_;
};

#endif // CHANGEMANAGER_HPP
