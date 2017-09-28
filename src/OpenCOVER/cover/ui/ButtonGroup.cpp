#include "ButtonGroup.h"
#include "Button.h"
#include <cassert>

namespace opencover {
namespace ui {

ButtonGroup::ButtonGroup(const std::string &name, Owner *owner)
: Element(name, owner)
{
}

ButtonGroup::ButtonGroup(Group *parent, const std::string &name)
: Element(parent, name)
{
}

void ButtonGroup::enableDeselect(bool flag)
{
    m_allowDeselect = flag;
    if (!m_allowDeselect)
    {
        if (numChildren() > 0)
            toggle(dynamic_cast<Button *>(m_children[0]));
    }
}

int ButtonGroup::value() const
{
    int id = 0;
    if (numChildren() == 0)
        return id;

    int numSet = 0;
    for (auto e: m_children)
    {
        auto b = dynamic_cast<Button *>(e);
        assert(b);
        if (b->state())
        {
            ++numSet;
            id = b->buttonId();
        }
    }
    assert(numSet == 1);
    return id;
}

Button *ButtonGroup::activeButton() const
{
    Button *ret = nullptr;
    int numSet = 0;
    for (auto e: m_children)
    {
        auto b = dynamic_cast<Button *>(e);
        assert(b);
        if (b->state())
        {
            ++numSet;
            ret = b;
        }
    }
    assert(numSet == 1);
    return ret;
}

bool ButtonGroup::add(Element *elem)
{
    auto rb = dynamic_cast<Button *>(elem);
    assert(rb);
    if (Container::add(elem))
    {
        bool prevState = rb->state();
        // ensure that exactly one button is set
        if (numChildren() == 1)
        {
            if (!m_allowDeselect)
                rb->setState(true, false);
        }
        else
            rb->setState(false, false);
        if (rb->state() != prevState)
            rb->trigger();
        return true;
    }
    return false;
}

bool ButtonGroup::remove(Element *elem)
{
    auto rb = dynamic_cast<Button *>(elem);
    assert(rb);
    if (Container::remove(elem))
    {
        if (rb->state() && numChildren()>0 && !m_allowDeselect)
        {
            auto rb0 = dynamic_cast<Button *>(child(0));
            assert(rb0);
            rb0->setState(true, false);
            rb0->trigger();
        }
        return true;
    }
    return false;
}

void ButtonGroup::setCallback(const std::function<void (int)> &f)
{
    m_callback = f;
}

std::function<void (int)> ButtonGroup::callback() const
{
    return m_callback;
}

void ButtonGroup::toggle(const Button *b)
{
    bool change = false;

    Button *bset = nullptr;
    for (auto e: m_children)
    {
        auto bb = dynamic_cast<Button *>(e);
        assert(bb);
        if (b == bb)
        {

        }
        else
        {
            if (bb->state())
                bset = bb;
        }
    }
    if (bset)
    {
        if (b->state())
        {
            change = true;
            bset->setState(false, false);
            bset->radioTrigger();
        }
    }
    else if (!m_allowDeselect)
    {
        if (!b->state())
        {
#if 0
            b->setState(true, false);
            b->trigger();
#endif
        }
    }

    if (change)
        trigger();
}

void ButtonGroup::triggerImplementation() const
{
    if (m_callback)
        m_callback(value());
}

}
}
