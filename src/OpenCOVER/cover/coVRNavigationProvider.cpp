/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRNavigationProvider.h"

using namespace opencover;

coVRNavigationProvider::coVRNavigationProvider(const std::string n, coVRPlugin* p):
    name(n),
    plugin(p)
{
}

coVRNavigationProvider::~coVRNavigationProvider()
{
}

const std::string& coVRNavigationProvider::getName() const {
    return name;
}

bool coVRNavigationProvider::isEnabled() const {
    return enabled;
}

void coVRNavigationProvider::setEnabled(bool state)
{
    enabled = state;
}

