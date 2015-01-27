/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coInteractor.h"

#include <cassert>

namespace opencover
{

coInteractor::coInteractor()
    : d_refCount(0)
{
}

// if you get an interactor and you want to keep it use ...
void coInteractor::incRefCount()
{
    d_refCount++;
}

// if you don't need the interactor any more use ...
void coInteractor::decRefCount()
{
    d_refCount--;
    if (d_refCount <= 0)
        delete this;
}

coInteractor::~coInteractor()
{
    assert(d_refCount == 0);
}
}
