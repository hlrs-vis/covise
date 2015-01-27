/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class ModuleInfoSingleton             ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 20.12.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "ModuleInfo.h"

// initialize instance_
ModuleInfoSingleton *ModuleInfoSingleton::instance_ = NULL;

//
// Constructor
//
ModuleInfoSingleton::ModuleInfoSingleton()
    : initialized_(false)
    , doExit_(false)
{
}

//
// the one and only access
//
ModuleInfoSingleton *
ModuleInfoSingleton::instance()
{
    if (instance_ == NULL)
    {
        instance_ = new ModuleInfoSingleton;
    }
    return instance_;
}

void
ModuleInfoSingleton::initialize(const std::string &hostname, const int &inst)
{
    hostname_ = hostname;
    modInstance_ = inst;
    initialized_ = true;
}

std::string
ModuleInfoSingleton::getHostname()
{
    initCheck();
    return hostname_;
}

int
ModuleInfoSingleton::iGetInstance()
{
    initCheck();
    return modInstance_;
}

std::string
ModuleInfoSingleton::getInstance()
{
    initCheck();
    char tmp[32];
    sprintf(tmp, "%d", modInstance_);
    std::string ret(tmp);
    return ret;
}

std::string
ModuleInfoSingleton::getCoMsgHeader()
{
    initCheck();
    int tmpLen(hostname_.size() + 64); // surely overestimated

    char *tmp = new char[tmpLen];
    // we assume that this obj is only used in the renderer
    sprintf(tmp, "Renderer\n%d\n%s", modInstance_, hostname_.c_str());
    //  sprintf(ModuleHead,"Renderer\n%s\n%s", instance, appmod->get_hostname());

    std::string ret(tmp);
    delete[] tmp;

    return ret;
}

void
ModuleInfoSingleton::initCheck()
{
    if (doExit_)
    {
        if (!initialized_)
        {
            cerr << "FATAL: ModuleInfoSingleton::initCheck() object uninitialized";
            cerr << "       will exit now!!!";
            exit(EXIT_FAILURE);
        }
    }
    return;
}

//
// Destructor
//
ModuleInfoSingleton::~ModuleInfoSingleton()
{
}
