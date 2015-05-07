/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRPlugin.h"
#include "coVRPluginList.h"
#include <cassert>

using namespace opencover;

coVRPlugin::coVRPlugin()
    : handle(NULL)
    , m_outstandingTimestep(-1)
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

void coVRPlugin::commitTimestep(int t)
{
    assert(m_outstandingTimestep == t);
    m_outstandingTimestep = -1;
    coVRPluginList::instance()->commitTimestep(t, this);
}

void coVRPlugin::requestTimestepWrapper(int t)
{
    assert(m_outstandingTimestep == -1);
    m_outstandingTimestep = t;

    requestTimestep(t);
}
