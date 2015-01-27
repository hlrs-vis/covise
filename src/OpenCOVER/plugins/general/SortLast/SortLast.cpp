/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SortLast.h"

#include "SortLastMaster.h"
#include "SortLastSlave.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>

#include <sstream>

SortLast::SortLast()
    : impl(0)
{
    std::stringstream s;
    s << opencover::coVRMSController::instance()->getID();
    this->nodename = s.str();
}

SortLast::~SortLast()
{
    delete this->impl;
}

bool SortLast::init()
{
    if (this->impl)
        return this->impl->init();
    else
        return true;
}

void SortLast::preFrame()
{
    if (this->impl)
        this->impl->preFrame();
}

void SortLast::postFrame()
{
    if (this->impl)
        this->impl->postFrame();
}

void SortLast::preSwapBuffers(int windowNumber)
{
    if (this->impl)
        this->impl->preSwapBuffers(windowNumber);
}

std::string SortLast::getHostID() const
{
    return this->nodename;
}

bool SortLast::initialiseAsMaster()
{
    assert(this->impl == 0);
    this->impl = new SortLastMaster(this->nodename, this->session);
    return this->impl->initialiseAsMaster();
}

bool SortLast::initialiseAsSlave()
{
    assert(this->impl == 0);
    this->impl = new SortLastSlave(this->nodename, this->session);
    return this->impl->initialiseAsSlave();
}

bool SortLast::createContext(const std::list<std::string> &hostlist, int gid)
{
    if (this->impl)
        return this->impl->createContext(hostlist, gid);
    else
        return false;
}

COVERPLUGIN(SortLast)
