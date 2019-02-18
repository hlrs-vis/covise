/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SharedStateManager.h"
#include <vrbclient/VrbClientRegistry.h>
#include <vrbclient/VRBClient.h>


namespace vrb
{
SharedStateManager* SharedStateManager::s_instance = nullptr;

SharedStateManager::SharedStateManager(VrbClientRegistry * reg)
    :registry(reg)
{
    assert(!s_instance);
    s_instance = this;
}

SharedStateManager::~SharedStateManager()
{
}

VrbClientRegistry * SharedStateManager::getRegistry()


{
    return registry;
}

SharedStateManager *SharedStateManager::instance()
{
    return s_instance;
}

int SharedStateManager::add(SharedStateBase *base, SharedStateType mode)
{
    switch (mode)
    {
    case vrb::USE_COUPLING_MODE:
        useCouplingMode.insert(base);
        return m_useCouplingModeSessionID;
        break;
    case vrb::NEVER_SHARE:
        neverShare.insert(base);
        return m_privateSessionID;
        break;
    case vrb::ALWAYS_SHARE:
        alwaysShare.insert(base);
        return m_publicSessionID;
        break;
    default:
        break;
    }
}

void SharedStateManager::remove(SharedStateBase *base)
{
    useCouplingMode.erase(base);
    alwaysShare.erase(base);
    neverShare.erase(base);
}

void SharedStateManager::update(int privateSessionID, int publicSessionID, int useCouplingModeSessionID, int sesisonToSubscribe)
{
    m_privateSessionID = privateSessionID;
    m_publicSessionID = publicSessionID;
    m_useCouplingModeSessionID = useCouplingModeSessionID;

    for (const auto sharedState : neverShare)
    {
        sharedState->resubscribe(privateSessionID);
        sharedState->setID(m_privateSessionID);
    }

    for (const auto sharedState : alwaysShare)
    {
        sharedState->resubscribe(publicSessionID);
        sharedState->setID(m_publicSessionID);
    }

    for (const auto sharedState : useCouplingMode)
    {
        sharedState->resubscribe(sesisonToSubscribe);
        sharedState->setID(m_useCouplingModeSessionID);
    }
}
}
