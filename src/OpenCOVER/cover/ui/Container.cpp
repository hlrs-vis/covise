#include "Container.h"
#include "Element.h"

#include <algorithm>

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
    auto it = std::find(m_children.begin(), m_children.end(), elem);
    if (it == m_children.end())
    {
        if (where == Front)
            where = 0;

        if (where >= 0 && size_t(where) < m_children.size())
        {
            m_children.insert(m_children.begin()+where, elem);
            elem->m_containers.insert(this);
        }
        else
        {
            m_children.push_back(elem);
            elem->m_containers.insert(this);
        }
        return true;
    }
    return false;
}

bool Container::remove(Element *elem)
{
    auto it = std::find(m_children.begin(), m_children.end(), elem);
    if (it != m_children.end())
    {
        m_children.erase(it);
        elem->m_containers.erase(this);
        return true;
    }
    return false;
}

Element *Container::child(size_t index) const
{
    return m_children[index];
}

void Container::clearChildren()
{
    while (!m_children.empty())
    {
        remove(m_children.back());
    }
}

}
}


