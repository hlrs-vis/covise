#include "Container.h"
#include "Element.h"

#include <algorithm>
#include <iostream>

namespace opencover {
namespace ui {

Container::~Container()
{
    clearChildren();
}

size_t Container::numChildren() const
{
    return m_children.size();
}

bool Container::add(Element *elem, int where)
{
    auto idx = index(elem);
    if (idx >= 0)
        return false;

    elem->m_containers.insert(this);

    if (where == KeepLast)
    {
        m_children.emplace_back(elem, where);
    }
    else if (where == KeepFirst)
    {
        auto it = std::find_if(m_children.begin(), m_children.end(), [](const Child &c){ return c.where!=KeepFirst; });
        m_children.emplace(it, elem, where);
    }
    else if (where == Append)
    {
        auto it = std::find_if(m_children.begin(), m_children.end(), [](const Child &c){ return c.where==KeepLast; });
        m_children.emplace(it, elem, it==m_children.begin() ? 0 : (it-1)->where+1);
    }
    else
    {
        auto it = std::find_if(m_children.begin(), m_children.end(), [where](const Child &c){ return c.where != KeepFirst && c.where != KeepLast && c.where > where; });
        m_children.emplace(it, elem, where);
    }

    return true;
}

bool Container::remove(Element *elem)
{
    auto it = std::find_if(m_children.begin(), m_children.end(), [elem](const Child &c){
        return c.elem == elem;
    });

    if (it == m_children.end())
        return false;

    m_children.erase(it);
    elem->m_containers.erase(this);
    return true;
}

Element *Container::child(size_t index) const
{
    return m_children[index].elem;
}

int Container::index(const Element *elem) const
{
    auto it = std::find_if(m_children.begin(), m_children.end(), [elem](const Child &c){
        return c.elem == elem;
    });

    if (it == m_children.end())
        return -1;

    return it - m_children.begin();
}

void Container::clearChildren()
{
    while (!m_children.empty())
    {
        remove(m_children.back().elem);
    }
}

}
}


