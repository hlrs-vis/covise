#include "ButtonGroup.h"
#include "Button.h"
#include <cassert>
#include <iostream>

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

ButtonGroup::~ButtonGroup()
{
    clearItems();
    clearChildren();
}

void ButtonGroup::setDefaultValue(int val)
{
    m_defaultValue = val;
}

int ButtonGroup::defaultValue() const
{
    return m_defaultValue;
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
    int id = defaultValue();

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
    assert(numSet <= 1);
    assert(numChildren()==0 || m_allowDeselect || numSet == 1);
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
    assert(numSet <= 1);
    assert(numChildren()==0 || m_allowDeselect || numSet == 1);
    return ret;
}

void ButtonGroup::setActiveButton(Button *button)
{
    for (auto e: m_children)
    {
        auto b = dynamic_cast<Button *>(e);
        assert(b);
        if (b == button)
        {
            bool oldState = button->state();
            button->setState(true, true /* update group */);
            if (!oldState)
                button->trigger();
            return;
        }
    }
}

bool ButtonGroup::add(Element *elem, int where)
{
    auto rb = dynamic_cast<Button *>(elem);
    assert(rb);
    if (Container::add(elem, where))
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
        rb->m_radioGroup = nullptr;
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
    //std::cerr << "TOGGLE: group=" << path() << ", toggle=" << b->path() << ", state=" << b->state() << std::endl;
    Button *bset = nullptr, *bthis = nullptr;;
    for (auto e: m_children)
    {
        auto bb = dynamic_cast<Button *>(e);
        assert(bb);
        if (b == bb)
            bthis = bb;
        if (b != bb && bb->state())
            bset = bb;
    }

    bool change = false;
    if (bset)
    {
        if (b->state() || m_allowDeselect)
        {
            change = true;
            bset->setState(false, false);
            bset->radioTrigger();
        }
    }
    else if (!bthis->state() && !m_allowDeselect)
    {
        change = true;
        bthis->setState(true, false);
        bthis->radioTrigger();
    }
    if (change)
        trigger();
}

void ButtonGroup::triggerImplementation() const
{
    //std::cerr << "ButtonGroup: trigger: value=" << value() << ", " << path() << std::endl;
    if (m_callback)
        m_callback(value());
}

}
}
