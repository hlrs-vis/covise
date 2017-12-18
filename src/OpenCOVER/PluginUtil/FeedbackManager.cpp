#include "FeedbackManager.h"
#include "ModuleInteraction.h"
#include <cover/coInteractor.h>
#include <cassert>
#include <iostream>

namespace opencover
{

FeedbackManager *FeedbackManager::s_instance = nullptr;

FeedbackManager *FeedbackManager::instance()
{
    if (!s_instance)
    {
        s_instance = new FeedbackManager;
    }

    return s_instance;
}

FeedbackManager::FeedbackManager()
{
    assert(!s_instance);
}

FeedbackManager::~FeedbackManager()
{
    s_instance = nullptr;
}

void FeedbackManager::registerFeedback(ModuleInteraction *feedback, coInteractor *inter)
{
    m_moduleFeedback.emplace(inter, feedback);
    std::cerr << "FeedbackManager: reg " << feedback->ModuleName() << ", #reg: " << m_moduleFeedback.size() << std::endl;
}

void FeedbackManager::unregisterFeedback(ModuleInteraction *feedback)
{
    for (auto it = m_moduleFeedback.begin(), next=it; it != m_moduleFeedback.end(); it=next)
    {
        if (it->second == feedback)
        {
            next = m_moduleFeedback.erase(it);
        }
        else
        {
            next = it;
            ++next;
        }
    }
}

ModuleInteraction *FeedbackManager::findFeedback(coInteractor *inter) const
{
    auto it = m_moduleFeedback.find(inter);
    if (it != m_moduleFeedback.end())
        return it->second;

    for (auto mfm: m_moduleFeedback)
    {
        if (mfm.second->compare(inter))
        {
            return mfm.second;
        }
    }

    for (auto mfm: m_moduleFeedback)
    {
        if (inter->isSame(mfm.second->getInteractor()))
            return mfm.second;
    }

    for (auto mfm: m_moduleFeedback)
    {
        if (inter->isSame(mfm.first))
            return mfm.second;
    }

    std::cerr << "FeedbackManager: nothing found for " << inter->getObjName() << std::endl;

    return nullptr;
}

}
