#include "Deletable.h"
#include <cassert>

DeletionManager *DeletionManager::s_instance = nullptr;

Deletable::~Deletable()
{
}

void Deletable::deleteLater()
{
    DeletionManager::the()->add(this);
}


DeletionManager::DeletionManager()
{
    assert(!s_instance);
    s_instance = this;
    m_destroyList.resize(2);
}

DeletionManager::~DeletionManager()
{
    while (run())
        ;
    s_instance = nullptr;
}

void DeletionManager::add(Deletable *obj)
{
    m_destroyList.back().push_back(obj);
}

DeletionManager *DeletionManager::the()
{
    if (!s_instance)
        s_instance = new DeletionManager;
    return s_instance;
}

void DeletionManager::destroy()
{
    delete s_instance;
}

bool DeletionManager::run()
{
    auto l = m_destroyList.front();
    m_destroyList.pop_front();
    m_destroyList.emplace_back();

    bool deleted = false;
    for (auto obj: l)
    {
        deleted = true;
        delete obj;
    }
    return deleted;
}
