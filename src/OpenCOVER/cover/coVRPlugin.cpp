/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRPlugin.h"

using namespace opencover;

coVRPlugin::coVRPlugin()
    : handle(NULL)
{
}

coVRPlugin::~coVRPlugin()
{
}

void coVRPlugin::setName(const char *name)
{
    if (name)
        m_name = name;
    else
        m_name = "";
}
