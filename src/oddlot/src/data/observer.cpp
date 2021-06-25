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

#include "observer.hpp"

#include "subject.hpp"

/** \brief The Contructor does nothing at all.
*
*
*/
Observer::Observer()
{
}

/** \brief Receives a notify event.
*
* The Observer will be notified that one of his Subjects has changed.
*/
void
Observer::updateObserver()
{
}
