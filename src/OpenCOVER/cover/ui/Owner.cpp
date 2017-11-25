#include "Owner.h"
#include "Manager.h"

#include <cassert>
#include <iostream>

namespace opencover {
namespace ui {

namespace {

bool checkName(const std::string &name)
{
    for (auto c: name)
    {
        if (!isalnum(c) && c != '_')
        {
            std::cerr << "ui::Owner: invalid name " << name << std::endl;
            assert("Invalid owner name" == 0);
            return false;
        }
    }
    return true;
}

}

Owner::Owner(const std::string &name, Owner *owner)
: m_name(name)
, m_owner(owner)
, m_manager(m_owner->manager())
{
    if (!m_manager)
    {
        std::cerr << "ui::Owner: " << path() << " is unmanaged" << std::endl;
    }
    assert(m_manager);
    checkName(m_name);
    owner->addItem(this);
}

Owner::Owner(const std::string &name, Manager *manager)
: m_name(name)
, m_owner(nullptr)
, m_manager(manager)
{
    if (!m_manager)
    {
        std::cerr << "ui::Owner: " << path() << " is unmanaged" << std::endl;
    }
    assert(m_manager);
    checkName(name);
}

Owner::~Owner()
{
    if (m_owner)
        m_owner->removeItem(this);
    m_owner = nullptr;

    if (m_manager)
        m_manager->remove(this);

    clearItems();
}

void Owner::clearItems()
{
    while (!m_items.empty())
    {
        auto it =  m_items.begin();
        delete it->second; // item removes itself from m_items
    }
}

Owner *Owner::owner() const
{
    return m_owner;
}

Manager *Owner::manager() const
{
    return m_manager;
}

const std::string &Owner::name() const
{
    return m_name;
}

std::string Owner::path() const
{
    std::string p;
    if (owner())
    {
        //if (!dynamic_cast<Manager *>(owner()))
        p = owner()->path();
    }
    if (!p.empty())
        p += ".";
    p += name();
    return p;
}

bool Owner::addItem(Owner *item)
{
    auto it = m_items.find(item->name());
    assert(it == m_items.end());
    if (it == m_items.end())
    {
        m_items.emplace(item->name(), item);
        return true;
    }
    return false;
}

bool Owner::removeItem(Owner *item)
{
    auto it = m_items.find(item->name());
    if (it == m_items.end())
    {
        std::cerr << "ui::Owner::removeItem: item to remove not found: " << item->name() << std::endl;
    }
    //assert(it != m_items.end());
    if (it != m_items.end())
    {
        m_items.erase(it);
        return true;
    }
    return false;
}

}
}
