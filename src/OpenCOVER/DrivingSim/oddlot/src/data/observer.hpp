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

#ifndef OBSERVER_HPP
#define OBSERVER_HPP

class Subject;

/** \brief Observer Pattern: The Observer observes some Subjects.
*
* An Observer can be attached to an arbitrary number of Subjects. When
* the notify() function of a connected Subject is called, the Observer
* will be notified that something has changed.
* This implementation uses a ChangeManager that buffers all notifications
* and notifies each Observer only once.
*/
class Observer
{
public:
    Observer();
    virtual ~Observer()
    {
    }

    virtual void updateObserver();

private:
    Observer(const Observer &); /* not allowed */
    Observer &operator=(const Observer &); /* not allowed */
};

#endif // OBSERVER_HPP
