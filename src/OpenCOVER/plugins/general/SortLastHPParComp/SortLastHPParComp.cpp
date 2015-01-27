/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SortLastHPParComp.h"

#include "SortLastMaster.h"
#include "SortLastSlave.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <config/coConfig.h>

SortLastHPParComp::SortLastHPParComp()
    : impl(0)
{
    //   this->isMaster = opencover::coVRMSController::instance()->isMaster();
    //   if (this->isMaster)
    //   {
    //      this->isMaster =
    //         covise::coConfig::getInstance()->isOn("internal",
    //                                               "COVER.Plugin.SortLastHPParComp.Master",
    //                                               this->isMaster);
    //   }

    //   if (this->isMaster)
    //   {
    //      this->impl = new SortLastMaster();
    //   }
    //   else
    //   {
    //      this->impl = new SortLastSlave();
    //   }

    this->session = covise::coConfig::getInstance()->getInt("COVER.Parallel.SortLastHPParComp.Session", 0);

    QString nodeNameSuffix = covise::coConfig::getInstance()->getString("COVER.Parallel.SortLastHPParComp.NodeNameSuffix");

    if (covise::coConfig::getInstance()->getHostname().endsWith(nodeNameSuffix))
        nodeNameSuffix = "";

    this->nodename = QString("%1%2:%3").arg(covise::coConfig::getInstance()->getHostname()).arg(nodeNameSuffix).arg(this->session).toStdString();
}

SortLastHPParComp::~SortLastHPParComp()
{
    delete this->impl;
}

bool SortLastHPParComp::init()
{
    if (this->impl)
        return this->impl->init();
    else
        return true;
}

void SortLastHPParComp::preFrame()
{
    if (this->impl)
        this->impl->preFrame();
}

void SortLastHPParComp::postFrame()
{
    if (this->impl)
        this->impl->postFrame();
}

void SortLastHPParComp::preSwapBuffers(int windowNumber)
{
    if (this->impl)
        this->impl->preSwapBuffers(windowNumber);
}

std::string SortLastHPParComp::getHostID() const
{
    return this->nodename;
}

bool SortLastHPParComp::initialiseAsMaster()
{
    assert(this->impl == 0);
    this->impl = new SortLastMaster(this->nodename, this->session);
    return this->impl->initialiseAsMaster();
}

bool SortLastHPParComp::initialiseAsSlave()
{
    assert(this->impl == 0);
    this->impl = new SortLastSlave(this->nodename, this->session);
    return this->impl->initialiseAsSlave();
}

bool SortLastHPParComp::createContext(const std::list<std::string> &hostlist, int gid)
{
    if (this->impl)
        return this->impl->createContext(hostlist, gid);
    else
        return false;
}

COVERPLUGIN(SortLastHPParComp)
