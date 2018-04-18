#include "View.h"
#include "Manager.h"

#include "Menu.h"
#include "ButtonGroup.h"
#include "Label.h"
#include "Action.h"
#include "Button.h"
#include "Slider.h"
#include "SelectionList.h"
#include "EditField.h"

#include <iostream>
#include <cassert>

namespace opencover {
namespace ui {

View::View(const std::string &name)
: m_name(name)
{
}

View::~View()
{
    if (manager())
        manager()->removeView(this);
}

const std::string &View::name() const
{
    return m_name;
}

Manager *View::manager() const
{
    return m_manager;
}

bool View::update()
{
    // don't request a re-render
    return false;
}

View::ViewElement *View::elementFactory(Element *elem)
{
    ViewElement *ve = nullptr;
    //std::cerr << "ui: creating " << elem->path() << std::endl;

    if (auto menu = dynamic_cast<Menu *>(elem))
    {
        ve = elementFactoryImplementation(menu);
    }
    else if (auto group = dynamic_cast<Group *>(elem))
    {
        ve = elementFactoryImplementation(group);
    }
    else if (auto label = dynamic_cast<Label *>(elem))
    {
        ve = elementFactoryImplementation(label);
    }
    else if (auto action = dynamic_cast<Action *>(elem))
    {
        ve = elementFactoryImplementation(action);
    }
    else if (auto button = dynamic_cast<Button *>(elem))
    {
        ve = elementFactoryImplementation(button);
    }
    else if (auto slider = dynamic_cast<Slider *>(elem))
    {
        ve = elementFactoryImplementation(slider);
    }
    else if (auto sl = dynamic_cast<SelectionList *>(elem))
    {
        ve = elementFactoryImplementation(sl);
    }
    else if (auto input = dynamic_cast<EditField *>(elem))
    {
        ve = elementFactoryImplementation(input);
    }

    if (ve)
    {
        assert(ve->element == elem);
        ve->view = this;
        m_viewElements[elem] = ve;
    }

    return ve;
}

bool View::removeElement(Element *elem)
{
    for (auto item: elem->m_items)
    {
        if (auto e = dynamic_cast<Element *>(item.second))
            removeElement(e);
    }

    //std::cerr << "REMOVE: " << elem->path() << std::endl;
    auto it = m_viewElements.find(elem);
    if (it == m_viewElements.end())
        return false;
    auto ve = it->second;
    m_viewElements.erase(it);
    if (!ve)
        return false;
    delete ve;
    return true;
}

View::ViewElement *View::viewElement(const std::string &path) const
{
    auto elem = manager()->getByPath(path);
    return viewElement(elem);
}

View::ViewElement *View::viewElement(const Element *elem) const
{
    auto it = m_viewElements.find(elem);
    if (it == m_viewElements.end())
    {
        return nullptr;
    }

    return it->second;
}

View::ViewElement *View::viewParent(const Element *elem) const
{
    if (elem && elem->parent())
        return viewElement(elem->parent());
    return nullptr;
}

}
}
