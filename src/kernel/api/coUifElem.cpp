/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include "coUifElem.h"

/// no prevention of automatic functions : virtual base class only

using namespace covise;

coUifElem::~coUifElem() {}

int coUifElem::preCompute()
{
    return 0;
}

int coUifElem::postCompute()
{
    return 0;
}

int coUifElem::paramChange()
{
    return 0;
}

/// Hide everything below
void coUifElem::hide()
{
}

/// Show everything below
void coUifElem::show()
{
}

/// whether this may be a part of a switch group
int coUifElem::switchable() const
{
    return 0;
}
