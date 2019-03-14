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

SessionID &SharedStateManager::add(SharedStateBase *base, SharedStateType mode)
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
    case vrb::SHARE_WITH_ALL:
        shareWithAll.insert(base);
        std::string name = "all";
        static SessionID sid(0, name, false);
        return sid;
    }

    std::cerr << "SharedStateManager: invalid mode for " << base->getName() << std::endl;
    return m_privateSessionID;
}

void SharedStateManager::remove(SharedStateBase *base)
{
    useCouplingMode.erase(base);
    alwaysShare.erase(base);
    neverShare.erase(base);
}

void SharedStateManager::update(SessionID & privateSessionID, SessionID & publicSessionID, SessionID & useCouplingModeSessionID, SessionID & sessionToSubscribe, bool force)
{

    if (m_privateSessionID != privateSessionID ||force)
    {
        for (const auto sharedState : neverShare)
        {
            sharedState->resubscribe(privateSessionID);
            sharedState->setID(privateSessionID);
        }
    }

    if (m_publicSessionID != publicSessionID || force)
    {
        for (const auto sharedState : alwaysShare)
        {
            sharedState->resubscribe(publicSessionID);
            sharedState->setID(publicSessionID);
        }
    }

    if (m_useCouplingModeSessionID != useCouplingModeSessionID || force)
    {
        for (const auto sharedState : useCouplingMode)
        {
            sharedState->resubscribe(sessionToSubscribe);
            sharedState->setID(useCouplingModeSessionID);
        }
    }

    m_privateSessionID = privateSessionID;
    m_publicSessionID = publicSessionID;
    m_useCouplingModeSessionID = useCouplingModeSessionID;
}

void SharedStateManager::frame(double time)
{
    for (const auto sharedState : neverShare)
    {
        sharedState->frame(time);
    }
    for (const auto sharedState : alwaysShare)
    {
        sharedState->frame(time);
    }
    for (const auto sharedState : useCouplingMode)
    {
        sharedState->frame(time);
    }
}
}
