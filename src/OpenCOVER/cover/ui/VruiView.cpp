#include "VruiView.h"

#include <cassert>

#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>

#include <OpenVRUI/coToolboxMenu.h>

#include "Element.h"
#include "Menu.h"
#include "ButtonGroup.h"
#include "Label.h"
#include "Action.h"
#include "Button.h"
#include "Slider.h"
#include "SelectionList.h"
#include "Input.h"

#include <cover/coVRPluginSupport.h>
#include <cover/VRVruiRenderInterface.h>
#include <config/CoviseConfig.h>
#include <util/unixcompat.h>

using namespace vrui;

namespace opencover {
namespace ui {

using covise::coCoviseConfig;

bool toolbar = false;

VruiView::VruiView()
: View("vrui")
{
    m_root = new VruiViewElement(nullptr);
    m_rootMenu = cover->getMenu();
    m_root->m_menu = m_rootMenu;

    if (toolbar && !cover->getToolBar())
    {
        auto tb = new coToolboxMenu("Toolbar");

        //////////////////////////////////////////////////////////
        // position AK-Toolbar and make it visible
        float x = coCoviseConfig::getFloat("x", "COVER.Plugin.AKToolbar.Position", -100);
        float y = coCoviseConfig::getFloat("y", "COVER.Plugin.AKToolbar.Position", 20);
        float z = coCoviseConfig::getFloat("z", "COVER.Plugin.AKToolbar.Position", -50);

        float h = coCoviseConfig::getFloat("h", "COVER.Plugin.AKToolbar.Orientation", 0);
        float p = coCoviseConfig::getFloat("p", "COVER.Plugin.AKToolbar.Orientation", 0);
        float r = coCoviseConfig::getFloat("r", "COVER.Plugin.AKToolbar.Orientation", 0);

        float scale = coCoviseConfig::getFloat("COVER.Plugin.AKToolbar.Scale", 0.2);

        int attachment = coUIElement::TOP;
        std::string att = coCoviseConfig::getEntry("COVER.Plugin.AKToolbar.Attachment");
        if (att != "")
        {
            if (!strcasecmp(att.c_str(), "BOTTOM"))
            {
                attachment = coUIElement::BOTTOM;
            }
            else if (!strcasecmp(att.c_str(), "LEFT"))
            {
                attachment = coUIElement::LEFT;
            }
            else if (!strcasecmp(att.c_str(), "RIGHT"))
            {
                attachment = coUIElement::RIGHT;
            }
        }

        //float sceneSize = cover->getSceneSize();

        vruiMatrix *mat = vruiRendererInterface::the()->createMatrix();
        vruiMatrix *rot = vruiRendererInterface::the()->createMatrix();
        vruiMatrix *trans = vruiRendererInterface::the()->createMatrix();

        rot->makeEuler(h, p, r);
        trans->makeTranslate(x, y, z);
        mat->makeIdentity();
        mat->mult(rot);
        mat->mult(trans);
        tb->setTransformMatrix(mat);
        tb->setScale(scale);
        tb->setVisible(true);
        tb->fixPos(true);
        tb->setAttachment(attachment);

        cover->setToolBar(tb);
    }
}

VruiView::~VruiView()
{
    delete m_root;
}

View::ViewType VruiView::typeBit() const
{
    return View::VR;
}

coMenu *VruiView::getMenu(const Element *elem) const
{
    manager()->update();

    auto ve = vruiElement(elem);
    if (ve)
        return ve->m_menu;
    return nullptr;
}

coMenuItem *VruiView::getItem(const Element *elem) const
{
    manager()->update();

    auto ve = vruiElement(elem);
    if (ve)
        return ve->m_menuItem;
    return nullptr;
}

VruiViewElement *VruiView::vruiElement(const std::string &path) const
{
    auto e = viewElement(path);
    auto ve = dynamic_cast<VruiViewElement *>(e);
    assert(!e || ve);
    return ve;
}

VruiViewElement *VruiView::vruiElement(const Element *elem) const
{
    if (!elem)
        return nullptr;
    auto e = viewElement(elem);
    auto ve = dynamic_cast<VruiViewElement *>(e);
    assert(!e || ve);
    return ve;
}

VruiViewElement *VruiView::vruiParent(const Element *elem) const
{
    if (!elem)
        return nullptr;

    std::string configPath = "COVER.UI." + elem->path();
    bool exists = false;
    std::string parentPath = covise::coCoviseConfig::getEntry("parent", configPath, &exists);
    //std::cerr << "config: " << configPath << " parent: " << parentPath << std::endl;
    if (exists)
    {
        if (parentPath.empty())
        {
            return m_root;
        }
        if (auto parent = vruiElement(parentPath))
            return parent;

        std::cerr << "ui::Vrui: did not find configured parent '" << parentPath << "' for '" << elem->path() << "'" << std::endl;
    }

    return vruiElement(elem->parent());
}


VruiViewElement *VruiView::vruiContainer(const Element *elem) const
{
    auto p = vruiParent(elem);
    if (p)
    {
        if (p->m_menu)
            return p;
        if (p->element)
            return vruiParent(p->element);
    }
    return nullptr;
}

void VruiView::add(VruiViewElement *ve, const Element *elem)
{
    if (!ve)
        std::cerr << "VruiView::add: Warning: no VruiViewElement" << std::endl;
    if (!elem)
        std::cerr << "VruiView::add: Warning: no Element" << std::endl;

    if (ve->m_menuItem)
    {
        ve->m_menuItem->setMenuListener(ve);
    }

    updateParent(elem);
}

void VruiView::updateEnabled(const Element *elem)
{

}

void VruiView::updateVisible(const Element *elem)
{
    //std::cerr << "Vrui: updateVisible(" << elem->path() << ")" << std::endl;
    auto ve = vruiElement(elem);
    if (!ve)
        return;

    bool inMenu = elem->priority() >= ui::Element::Default;

    if (auto m = dynamic_cast<const Menu *>(elem))
    {
        if (auto smi = dynamic_cast<coSubMenuItem *>(ve->m_menuItem))
        {
            if (!m->visible(this) || !inMenu)
            {
                smi->closeSubmenu();
                delete smi;
                ve->m_menuItem = nullptr;
            }
        } else if (!ve->m_menuItem) {
            if (m->visible(this) && inMenu)
            {
                auto smi = new coSubMenuItem(m->text()+"...");
                smi->setMenu(ve->m_menu);
                ve->m_menuItem = smi;
                add(ve, m);
            }
        }
        if (ve->m_menu)
        {
            if (!elem->visible(this) || !inMenu)
                ve->m_menu->setVisible(elem->visible(this));
        }
    }
    else if (ve->m_menuItem)
    {
        auto container = vruiContainer(elem);
        if (container)
        {
            //std::cerr << "changing visible to " << elem->visible(this) << ": elem=" << elem->path() << ", container=" << (container&&container->element ? container->element->path() : "(null)") << std::endl;
            //auto m = dynamic_cast<const Menu *>(container->element);
            //auto mve = vruiElement(m);
            //if (mve)
            //{
                auto menu = container->m_menu;
                if (menu)
                {
                auto idx = menu->index(ve->m_menuItem);
                if ((inMenu && elem->visible(this)) && idx < 0)
                {
                    if (menu)
                        menu->add(ve->m_menuItem);
                }
                else if ((!elem->visible(this) || !inMenu) && idx >= 0)
                {
                    if (menu)
                        menu->remove(ve->m_menuItem);
                }
                }
            //}
        }
    }
}

void VruiView::updateText(const Element *elem)
{
    auto ve = vruiElement(elem);
    if (ve)
    {
        ve->m_text = elem->text();
        std::string itemText = elem->text();
        if (auto m = dynamic_cast<const Menu *>(elem))
        {
            itemText += "...";
        }
        else if (auto i = dynamic_cast<const Input *>(elem))
        {
            itemText += ": " + i->value();
        }
        if (ve->m_menuItem)
            ve->m_menuItem->setName(itemText);
        if (auto re = dynamic_cast<coRowMenu *>(ve->m_menu))
            re->updateTitle(ve->m_text.c_str());
    }
}

void VruiView::updateState(const Button *button)
{
    auto ve = vruiElement(button);
    if (ve)
    {
        if (auto cb = dynamic_cast<coCheckboxMenuItem *>(ve->m_menuItem))
            cb->setState(button->state(), false);
    }
}

void VruiView::updateParent(const Element *elem)
{
    auto ve = vruiElement(elem);
    if (!ve)
        return;
    if (ve->m_menuItem)
    {
        auto oldMenu = ve->m_menuItem->getParentMenu();
        if (oldMenu)
            oldMenu->remove(ve->m_menuItem);

        bool inMenu = elem->priority() >= ui::Element::Default;
        if (inMenu)
        {
            auto parent = vruiContainer(elem);
            if (parent && parent->m_menu)
            {
                parent->m_menu->add(ve->m_menuItem);
            }
            else
            {
                if (parent)
                {
                    std::cerr << "ui::Vrui: parent " << parent->element->path() << " for " << elem->path() << " is not a menu" << std::endl;
                }
                if (m_rootMenu && ve->m_menu)
                    m_rootMenu->add(ve->m_menuItem);
            }
        }
    }
    updateVisible(elem);
}

void VruiView::updateChildren(const SelectionList *sl)
{
    auto ve = vruiElement(sl);
    if (!ve)
        return;

    auto m = dynamic_cast<coRowMenu *>(ve->m_menu);
    if (!m)
        return;

    const auto items = sl->items();
    while (m->getItemCount() < items.size())
    {
        int i = m->getItemCount();
        coCheckboxMenuItem *item = new coCheckboxMenuItem(items[i], false);
        item->setMenuListener(ve);
        m->add(item);
    }
    auto all = m->getAllItems();
    while (m->getItemCount() > items.size())
    {
        int i = m->getItemCount()-1;
        m->remove(all[i]);
    }
    for (size_t i=0; i<items.size(); ++i)
    {
        auto cb = dynamic_cast<coCheckboxMenuItem *>(all[i]);
        if (!cb)
        {
            std::cerr << "VruiView::updateChildren(SelectionList): menu entry is not a checkbox" << std::endl;
            continue;
        }
        cb->setName(items[i]);
        cb->setState(sl->selection()[i]);
    }

    std::string t = sl->text();
    int s = sl->selectedIndex();
    if (s >= 0)
    {
        t += ": ";
        t += sl->items()[s];
    }
    if (ve->m_menuItem)
        ve->m_menuItem->setName(t);
}

void VruiView::updateIntegral(const Slider *slider)
{
    auto ve = vruiElement(slider);
    if (!ve)
        return;
    if (auto vp = dynamic_cast<coPotiMenuItem *>(ve->m_menuItem))
    {
        vp->setInteger(slider->integral());
    }
    else if (auto vs = dynamic_cast<coSliderMenuItem *>(ve->m_menuItem))
    {
        vs->setInteger(slider->integral());
    }
}

void VruiView::updateScale(const Slider *slider)
{
    updateBounds(slider);
    updateValue(slider);
}

void VruiView::updateValue(const Slider *slider)
{
    auto ve = vruiElement(slider);
    if (!ve)
        return;
    if (auto vp = dynamic_cast<coPotiMenuItem *>(ve->m_menuItem))
    {
        vp->setValue(slider->value());
    }
    else if (auto vs = dynamic_cast<coSliderMenuItem *>(ve->m_menuItem))
    {
        vs->setValue(slider->value());
    }
}

void VruiView::updateBounds(const Slider *slider)
{
    auto ve = vruiElement(slider);
    if (!ve)
        return;

    if (auto vp = dynamic_cast<coPotiMenuItem *>(ve->m_menuItem))
    {
        vp->setMin(slider->min());
        vp->setMax(slider->max());
    }
    else if (auto vs = dynamic_cast<coSliderMenuItem *>(ve->m_menuItem))
    {
        vs->setMin(slider->min());
        vs->setMax(slider->max());
    }
}

void VruiView::updateValue(const Input *input)
{
    updateText(input);
}

VruiViewElement *VruiView::elementFactoryImplementation(Label *label)
{
    auto ve = new VruiViewElement(label);
    ve->m_menuItem = new coLabelMenuItem(label->text());
    add(ve, label);
    return ve;
}

VruiViewElement *VruiView::elementFactoryImplementation(Button *button)
{
    auto ve = new VruiViewElement(button);
    auto parent = vruiParent(button);
    vrui::coCheckboxGroup *vrg = nullptr;
    if (parent)
        vrg = parent->m_group;
    ve->m_menuItem = new coCheckboxMenuItem(button->text(), button->state(), vrg);
    add(ve, button);
    return ve;
}

VruiViewElement *VruiView::elementFactoryImplementation(Group *group)
{
    auto ve = new VruiViewElement(group);
    //ve->m_group = new vrui::coCheckboxGroup;
    //ve->m_menuItem = new coCheckboxMenuItem(rg->text(), rg->state());
    //add(ve, rg);
#if 0
    auto parent = vruiParent(rg);
    if (parent)
    {
        ve->m_menu = parent->m_menu;
    }
    else
    {
        ve->m_menu = m_rootMenu;
    }
#endif
    return ve;
}


VruiViewElement *VruiView::elementFactoryImplementation(Slider *slider)
{
    auto ve = new VruiViewElement(slider);
    switch (slider->presentation())
    {
    case Slider::AsSlider:
        ve->m_menuItem = new coSliderMenuItem(slider->text(), slider->min(), slider->max(), slider->value());
        break;
    case Slider::AsDial:
        ve->m_menuItem = new coPotiMenuItem(slider->text(), slider->min(), slider->max(), slider->value());
        break;
    }
    add(ve, slider);
    return ve;

}

VruiViewElement *VruiView::elementFactoryImplementation(SelectionList *sl)
{
    auto parent = vruiContainer(sl);
    auto ve = new VruiViewElement(sl);
    auto smi  = new coSubMenuItem(sl->name());
    ve->m_menuItem = smi;
    ve->m_menu = new coRowMenu(ve->m_text.c_str(), parent ? parent->m_menu : m_rootMenu);
    smi->setMenu(ve->m_menu);
    smi->closeSubmenu();
    add(ve, sl);
    return ve;
}

VruiViewElement *VruiView::elementFactoryImplementation(Input *input)
{
    auto ve = new VruiViewElement(input);
    ve->m_menuItem = new coLabelMenuItem(input->text());
    add(ve, input);
    return ve;
}

VruiViewElement *VruiView::elementFactoryImplementation(Menu *menu)
{
    auto parent = vruiContainer(menu);
#if 0
    std::cerr <<"Vrui: creating menu: text=" << menu->text() << std::endl;
    if (parent)
    {
        std::cerr <<"   parent: text=" << parent->element->text() << std::endl;
        std::cerr <<"   parent: menu=" << parent->m_menu << std::endl;
    }
#endif
    auto ve = new VruiViewElement(menu);
    auto smi  = new coSubMenuItem(menu->name()+"...");
    ve->m_menuItem = smi;
    add(ve, menu);
    ve->m_menu = new coRowMenu(ve->m_text.c_str(), parent ? parent->m_menu : m_rootMenu);
    smi->setMenu(ve->m_menu);
    smi->closeSubmenu();
    //ve->m_menu->closeMenu();
    //ve->m_menu->show();
    //ve->m_menu->setVisible(false);
    return ve;
}

VruiViewElement *VruiView::elementFactoryImplementation(Action *action)
{
    auto ve = new VruiViewElement(action);
    ve->m_menuItem = new coButtonMenuItem(action->text());
    add(ve, action);
    return ve;
}

VruiViewElement::VruiViewElement(Element *elem)
: View::ViewElement(elem)
, m_text(elem ? elem->text() : "")
{
}

VruiViewElement::~VruiViewElement()
{
    if (m_menu)
        m_menu->closeMenu();

    delete m_menu;
    m_menu = nullptr;

    delete m_menuItem;
    m_menuItem = nullptr;

    delete m_group;
    m_group = nullptr;
}

namespace {

void updateSlider(Slider *s, coMenuItem *item, bool moving)
{
    auto vd = dynamic_cast<coPotiMenuItem *>(item);
    auto vs = dynamic_cast<coSliderMenuItem *>(item);
    if (vd)
        s->setValue(vd->getValue());
    if (vs)
        s->setValue(vs->getValue());
    s->setMoving(moving);
    s->trigger();
}

}

void VruiViewElement::menuEvent(coMenuItem *menuItem)
{
    if (auto b = dynamic_cast<Button *>(element))
    {
        auto cb = dynamic_cast<coCheckboxMenuItem *>(menuItem);
        if (cb)
        {
            b->setState(cb->getState());
            b->trigger();
        }
    }
    else if (auto a = dynamic_cast<Action *>(element))
    {
        auto b = dynamic_cast<coButtonMenuItem *>(menuItem);
        if (b)
            a->trigger();
    }
    else if (auto s = dynamic_cast<Slider *>(element))
    {
        updateSlider(s, menuItem, true);
    }
    else if (auto sl = dynamic_cast<SelectionList *>(element))
    {
        auto m = dynamic_cast<coRowMenu *>(m_menu);
        if (dynamic_cast<coCheckboxMenuItem *>(menuItem))
        {
            if (auto smi = dynamic_cast<coSubMenuItem *>(m_menuItem))
                smi->closeSubmenu();
            auto idx = m_menu->index(menuItem);
            sl->select(idx);
            sl->trigger();
        }
    }
}

void VruiViewElement::menuReleaseEvent(coMenuItem *menuItem)
{
    if (auto s = dynamic_cast<Slider *>(element))
    {
        updateSlider(s, menuItem, false);
    }
}

}
}
