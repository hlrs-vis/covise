/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvInteractor.h"

#include <cassert>

namespace vive
{

vvInteractor::vvInteractor()
    : d_refCount(0)
{
}



// if you get an interactor and you want to keep it use ...
void vvInteractor::incRefCount()
{
    d_refCount++;
}

// if you don't need the interactor any more use ...
void vvInteractor::decRefCount()
{
    d_refCount--;
    assert(d_refCount >= 0);
    if (d_refCount <= 0)
        delete this;
}

vvInteractor::~vvInteractor()
{
    assert(d_refCount == 0);
}

int vvInteractor::refCount() const
{
    assert(d_refCount >= 0);
    return d_refCount;
}

}
