/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   04.02.2010
**
**************************************************************************/

#ifndef ACCEPTOR_HPP
#define ACCEPTOR_HPP

#include "visitor.hpp"

class Acceptor
{
public:
    explicit Acceptor();
    virtual ~Acceptor()
    { /* does nothing */
    }

    virtual void accept(Visitor * /*visitor*/) = 0;

private:
    Acceptor(const Acceptor &); /* not allowed */
    Acceptor &operator=(const Acceptor &); /* not allowed */
};

#endif // ACCEPTOR_HPP
