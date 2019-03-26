/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SharedStateManager.h"
#include <vrbclient/VrbClientRegistry.h>
#include <vrbclient/VRBClient.h>
#include <cassert>
#include <assert.h>

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

std::pair<SessionID, bool> SharedStateManager::add(SharedStateBase *base, SharedStateType mode)
{
    bool alreadyExists = false;
    switch (mode)
    {
    case vrb::USE_COUPLING_MODE:
        alreadyExists = !useCouplingMode.insert(base).second;
        return std::make_pair(m_publicSessionID, m_muted);
        break;
    case vrb::NEVER_SHARE:
        alreadyExists = !neverShare.insert(base).second;
        return std::make_pair(m_privateSessionID, false);
        break;
    case vrb::ALWAYS_SHARE:
        alreadyExists = !alwaysShare.insert(base).second;
        return std::make_pair(m_publicSessionID, false);
        break;
    case vrb::SHARE_WITH_ALL:
        alreadyExists = !shareWithAll.insert(base).second;
        std::string name = "all";
        static SessionID sid(0, name, false);
        return std::make_pair(sid, false);
    }
    assert(alreadyExists);
    std::cerr << "SharedStateManager: invalid mode for " << base->getName() << std::endl;
    return std::make_pair(m_privateSessionID, true);
}

void SharedStateManager::remove(SharedStateBase *base)
{
    useCouplingMode.erase(base);
    alwaysShare.erase(base);
    neverShare.erase(base);
}

void SharedStateManager::update(SessionID &privateSessionID, SessionID & publicSessionID, bool muted, bool force)
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
        for (const auto sharedState : useCouplingMode)
        {
            sharedState->resubscribe(publicSessionID);
            sharedState->setID(publicSessionID);
        }
    }

    if (m_muted != muted || force)
    {
        for (const auto sharedState : useCouplingMode)
        {
            sharedState->setMute(muted);
        }
    }

    m_privateSessionID = privateSessionID;
    m_publicSessionID = publicSessionID;
    m_muted = muted;
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
