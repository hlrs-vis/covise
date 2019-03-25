#include "SelectionList.h"
#include "Manager.h"
#include <iostream>
#include <cassert>

#include <vrbclient/SharedState.h>
#include <net/tokenbuffer.h>

using namespace vrb;
namespace opencover {
namespace ui {

SelectionList::SelectionList(Group *parent, const std::string &name)
: Element(parent, name)
{
}

void SelectionList::update(Element::UpdateMaskType mask) const
{
    Element::update(mask);
    if (mask & UpdateChildren)
        manager()->updateChildren(this);
}

void SelectionList::setCallback(const std::function<void (int)> &f)
{
    m_callback = f;
}

void SelectionList::triggerImplementation() const
{
    if (m_callback)
        m_callback(selectedIndex());
}

void SelectionList::save(covise::TokenBuffer &buf) const
{
    Element::save(buf);
    int sz = m_selection.size();
    buf << sz;
    for (size_t i=0; i<m_selection.size(); ++i)
        buf << m_selection[i];
}

void SelectionList::load(covise::TokenBuffer &buf)
{
	Element::load(buf);
	int sz = 0;
	buf >> sz;
	if (sz > 0) // during startup we get an update with sz ++ 0 even though there is a list of four entries e.g. in draw style please check this Martin
	{
		assert(sz == m_selection.size());
		for (size_t i = 0; i < m_selection.size(); ++i)
		{
			bool s = false;
			buf >> s;
			m_selection[i] = s;
		}
        }
}

void SelectionList::setShared(bool shared)
{
    if (shared)
    {
        if (!m_sharedState)
        {
            m_sharedState.reset(new SharedValue("ui."+path(), selectedIndex()));
            m_sharedState->setUpdateFunction([this](){
                select(*static_cast<SharedValue *>(m_sharedState.get()));
            });
        }
    }
    else
    {
        m_sharedState.reset();
    }

}

void SelectionList::updateSharedState()
{
    if (auto st = static_cast<SharedValue *>(m_sharedState.get()))
    {
        *st = selectedIndex();
    }
}

void SelectionList::shortcutTriggered()
{
    int idx = selectedIndex();
    if (idx >= 0)
    {
        ++idx;
        idx %= m_items.size();
        select(idx);
        trigger();
    }
}

SelectionList::SelectionList(const std::string &name, Owner *owner)
: Element(name, owner)
{
}

SelectionList::~SelectionList()
{
    manager()->remove(this);
}

void SelectionList::setList(const std::vector<std::string> &items)
{
    m_items = items;
    m_selection.resize(m_items.size(), false);
    if (selectedIndex() < 0)
        select(0, false);
    updateSharedState();
    manager()->updateChildren(this);
}

const std::vector<std::string> &SelectionList::items() const
{
    return m_items;
}

void SelectionList::append(const std::string &item)
{
    m_items.push_back(item);
    m_selection.resize(m_items.size(), false);
    if (m_items.size() == 1)
        m_selection[0] = true;
    updateSharedState();
    manager()->updateChildren(this);
}

void SelectionList::setSelection(const std::vector<bool> &selection)
{
    assert(m_selection.size() == m_items.size());
    m_selection = selection;
    updateSharedState();
    manager()->queueUpdate(this, UpdateChildren);
}

const std::vector<bool> &SelectionList::selection() const
{
    return m_selection;
}

void SelectionList::select(int index, bool update)
{
    for (size_t i=0; i<m_items.size(); ++i)
        m_selection[i] = i==index;
    updateSharedState();
    if (update)
        manager()->queueUpdate(this, UpdateChildren);
}

int SelectionList::selectedIndex() const
{
    int idx = -1;
    for (int i=0; i<m_selection.size(); ++i)
    {
        if (m_selection[i])
        {
            if (idx >= 0)
            {
                std::cerr << "ui::SelectionList " << path() << ": multiple selection: " << idx << " and " << i << std::endl;
            }
            else
            {
                idx = i;
            }
        }
    }
    return idx;
}

std::string SelectionList::selectedItem() const
{
    auto idx = selectedIndex();
    if (idx < 0)
        return std::string();
    return m_items[idx];
}

}
}
