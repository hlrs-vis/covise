#include "VruiView.h"

#include <cassert>
#include <map>

#include <OpenVRUI/osg/mathUtils.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>

#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>

#include <OpenVRUI/coLabelSubMenuToolboxItem.h>
#include <OpenVRUI/coIconButtonToolboxItem.h>
#include <OpenVRUI/coIconToggleButtonToolboxItem.h>
#include <OpenVRUI/coSliderToolboxItem.h>

#include <OpenVRUI/coToolboxMenu.h>
#include <OpenVRUI/coToolboxMenuItem.h>

#include "Element.h"
#include "Menu.h"
#include "ButtonGroup.h"
#include "Label.h"
#include "Action.h"
#include "Button.h"
#include "Slider.h"
#include "SelectionList.h"
#include "EditField.h"

#include <cover/coVRPluginSupport.h>
#include <cover/VRVruiRenderInterface.h>
#include <config/CoviseConfig.h>
#include <util/unixcompat.h>

using namespace vrui;

namespace opencover {
namespace ui {

using covise::coCoviseConfig;

namespace
{

std::map<std::string, std::string> nameMap;

void initPathMap()
{
    if (nameMap.empty())
    {
        nameMap["AnimationManager.Animation.Animate"] = "toggleAnimation";
        nameMap["AnimationManager.Animation.TimestepGroup.StepBackward"] = "media-seek-backward";
        nameMap["AnimationManager.Animation.TimestepGroup.StepForward"] = "media-seek-forward";

        nameMap["NavigationManager.Navigation.ViewAll"] = "viewall";

        nameMap["NavigationManager.Navigation.Modes.Drive"] = "drive";
        nameMap["NavigationManager.Navigation.Modes.MoveWorld"] = "xform";
        nameMap["NavigationManager.Navigation.Modes.Fly"] = "fly";
        nameMap["NavigationManager.Navigation.Modes.Walk"] = "walk";
        nameMap["NavigationManager.Navigation.Modes.Scale"] = "scale";
        nameMap["NavigationManager.Navigation.Modes.TraverseInteractors"] = "traverseInteractors";
        nameMap["NavigationManager.Navigation.Modes.ShowName"] = "showName";
        nameMap["NavigationManager.Navigation.ScaleUp"] = "scalePlus";
        nameMap["NavigationManager.Navigation.ScaleDown"] = "scaleMinus";

        nameMap["Manager.ViewOptions.Lighting.Headlight"] = "headlight";
        nameMap["Manager.ViewOptions.Lighting.SpecularLight"] = "specularlight";
        nameMap["Manager.ViewOptions.Lighting.Spotlight"] = "spotlight";
        nameMap["Manager.ViewOptions.StereoSep"] = "stereoSeparation";
        nameMap["Manager.ViewOptions.Orthographic"] = "orthographicProjection";

        nameMap["Manager.File.QuitGroup.Quit"] = "quit";
    }
}

std::string mapPath(const std::string &path)
{
    initPathMap();

    auto it = nameMap.find(path);
    if (it == nameMap.end())
    {
        //std::cerr << "no mapping: " << path << std::endl;
        return std::string();
    }

    return "AKToolbar/" + it->second;
}

std::string configPath(const std::string &path)
{
    if (path.empty())
        return "COVER.UI";

    return "COVER.UI." + path;
}

}

VruiView::VruiView()
: View("vrui")
{
    m_root = new VruiViewElement(nullptr);
    m_root->view = this;
    m_toolbarStack.push_back(m_root);

    m_rootMenu = cover->getMenu();
    m_root->m_menu = m_rootMenu;

    m_useToolbar = covise::coCoviseConfig::isOn("toolbar", configPath("VRUI"), m_useToolbar);
    m_allowTearOff = false;
    if (m_useToolbar)
    {
        m_allowTearOff = covise::coCoviseConfig::isOn("tearOffMenus", configPath("VRUI"), m_useToolbar);
    }

    if (auto tb = cover->getToolBar(m_useToolbar))
    {
        std::cerr << "ui::VruiView: toolbar mode" << std::endl;
        auto menuButton = new coLabelSubMenuToolboxItem("COVER");
        menuButton->setMenu(cover->getMenu());
        menuButton->setMenuListener(m_root);
        m_root->m_toolboxItem = menuButton;
        tb->add(menuButton);

        auto smi = new coSubMenuItem("COVER...");
        smi->setMenuListener(m_root);
        smi->setMenu(cover->getMenu());
        m_root->m_menuItem = smi;
        m_root->showMenu(false);
    }
}

VruiView::~VruiView()
{
    delete m_root;
}

bool VruiView::update()
{
    bool tornOff = false;
    for (auto ve: m_toolbarStack)
    {
        auto tmi = dynamic_cast<coSubMenuToolboxItem *>(ve->m_toolboxItem);
        if (!tornOff)
        {
            tornOff = m_allowTearOff && ve->m_menu && ve->m_menu->wasMoved();
        }
        if (auto smi = dynamic_cast<coSubMenuItem *>(ve->m_menuItem))
        {
            if (tmi)
            {
                if (smi->isOpen() && !tornOff)
                    tmi->openSubmenu();
                else
                    tmi->closeSubmenu();
            }
        }

        if (tornOff)
        {

            ve->clearStackToTop();
            break;
        }
    }

    return false;
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

    bool exists = false;
    std::string parentPath = covise::coCoviseConfig::getEntry("parent", configPath(elem->path()), &exists);
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

    if (ve->m_toolboxItem)
    {
        ve->m_toolboxItem->setMenuListener(ve);
        auto tb = cover->getToolBar();
        if (tb && isInToolbar(elem))
        {
            ve->m_toolboxItem->setParentMenu(tb);
            int n = tb->index(m_root->m_toolboxItem);
            if (n >= 0)
            {
                tb->insert(ve->m_toolboxItem, n);
            }
            else
            {
                tb->add(ve->m_toolboxItem);
            }
        }
    }

    updateParent(elem);
}

void VruiView::updateEnabled(const Element *elem)
{
    auto ve = vruiElement(elem);
    if (!ve)
        return;

    if (ve->m_menuItem)
        ve->m_menuItem->setActive(elem->enabled());
    if (ve->m_toolboxItem)
        ve->m_toolboxItem->setActive(elem->enabled());
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
        else if (auto i = dynamic_cast<const EditField *>(elem))
        {
            itemText += ": " + i->value();
        }
        if (ve->m_menuItem)
            ve->m_menuItem->setName(itemText);
        if (ve->m_toolboxItem)
            ve->m_toolboxItem->setName(elem->text());
        if (auto re = dynamic_cast<coRowMenu *>(ve->m_menu))
            re->updateTitle(ve->m_text.c_str());
    }
}

void VruiView::updateState(const Button *button)
{
    auto ve = vruiElement(button);
    if (!ve)
        return;

    if (auto cb = dynamic_cast<coCheckboxMenuItem *>(ve->m_menuItem))
        cb->setState(button->state(), false);
    if (auto tb = dynamic_cast<coIconToggleButtonToolboxItem *>(ve->m_toolboxItem))
    {
        if (tb->getState() != button->state())
            tb->setState(button->state(), false);
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
    if (auto ts = dynamic_cast<coSliderToolboxItem *>(ve->m_toolboxItem))
    {
        ts->setInteger(slider->integral());
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
    if (auto ts = dynamic_cast<coSliderToolboxItem *>(ve->m_toolboxItem))
    {
        ts->setValue(slider->value());
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
    if (auto ts = dynamic_cast<coSliderToolboxItem *>(ve->m_toolboxItem))
    {
        ts->setMin(slider->min());
        ts->setMax(slider->max());
    }
}

void VruiView::updateValue(const EditField *input)
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

    if (isInToolbar(button))
    {
        auto bi = new coIconToggleButtonToolboxItem(mapPath(button->path()));
        ve->m_toolboxItem = bi;
    }

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

    ve->m_toolboxItem = new coSliderToolboxItem(slider->text(), slider->min(), slider->max(), slider->value());
    ve->m_toolboxItem->setMenuListener(ve);

    ve->m_mappableToToolbar = true;

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

    auto t = new coLabelSubMenuToolboxItem(sl->name());
    t->setMenu(ve->m_menu);
    ve->m_toolboxItem = t;
    t->setMenuListener(ve);

    ve->m_mappableToToolbar = true;

    return ve;
}

VruiViewElement *VruiView::elementFactoryImplementation(EditField *input)
{
    auto ve = new VruiViewElement(input);
    ve->m_menuItem = new coLabelMenuItem(input->text());
    add(ve, input);
    return ve;
}

bool VruiView::useToolbar() const
{
    return m_useToolbar;
}

bool VruiView::allowTearOff() const
{
    return m_allowTearOff;
}

bool VruiView::isInToolbar(const Element *elem) const
{
    const std::string navmode("NavigationManager.Navigation.Modes.");
    const std::string anim("AnimationManager.Animation.");

    if (!cover->getToolBar())
        return false;

    bool exists = false;
    bool inToolbar = covise::coCoviseConfig::isOn("toolbar", configPath(elem->path()), elem->priority()>=ui::Element::Toolbar, &exists);
    if (exists)
        return inToolbar;

    if (elem->path().substr(0, anim.length()) == anim)
        inToolbar = false;

    if (elem->path().substr(0, navmode.length()) == navmode)
        inToolbar = true;

    if (!inToolbar)
        return false;

    if (dynamic_cast<const Action *>(elem) || dynamic_cast<const Button *>(elem))
    {
        std::string name = mapPath(elem->path());
        return !name.empty() && name != "AKToolbar/showName" && name != "AKToolbar/traverseInteractors";
    }

    return true;
}

VruiViewElement *VruiView::elementFactoryImplementation(Menu *menu)
{
    auto parent = vruiContainer(menu);
    auto ve = new VruiViewElement(menu);
    auto smi = new coSubMenuItem(menu->name()+"...");
    ve->m_menuItem = smi;
    ve->m_menu = new coRowMenu(ve->m_text.c_str(), parent ? parent->m_menu : m_rootMenu);
    smi->setMenu(ve->m_menu);
    smi->closeSubmenu();

    auto t = new coLabelSubMenuToolboxItem(menu->name());
    t->setMenu(ve->m_menu);
    ve->m_toolboxItem = t;
    t->setMenuListener(ve);

    add(ve, menu);

    ve->m_mappableToToolbar = true;
    return ve;
}

VruiViewElement *VruiView::elementFactoryImplementation(Action *action)
{
    auto ve = new VruiViewElement(action);
    ve->m_menuItem = new coButtonMenuItem(action->text());

    if (isInToolbar(action))
    {
        auto tb = new coIconButtonToolboxItem(mapPath(action->path()));
        ve->m_toolboxItem = tb;
    }

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

    if (auto tmi = dynamic_cast<coGenericSubMenuItem *>(m_toolboxItem))
    {
        tmi->setMenu(nullptr);
    }

    delete m_toolboxItem;
    m_toolboxItem = nullptr;

    delete m_menuItem;
    m_menuItem = nullptr;

    delete m_menu;
    m_menu = nullptr;

    delete m_group;
    m_group = nullptr;
}

namespace {

void updateSlider(Slider *s, coMenuItem *item, bool moving)
{
    auto vd = dynamic_cast<coPotiMenuItem *>(item);
    auto vs = dynamic_cast<coSliderMenuItem *>(item);
    auto ts = dynamic_cast<coSliderToolboxItem *>(item);
    if (vd)
        s->setValue(vd->getValue());
    if (vs)
        s->setValue(vs->getValue());
    if (ts)
        s->setValue(ts->getValue());
    s->setMoving(moving);
    s->trigger();
}

}

void VruiViewElement::menuEvent(coMenuItem *menuItem)
{
    auto vv = static_cast<VruiView *>(view);
    auto container = vv->vruiContainer(element);
    auto menu = dynamic_cast<Menu *>(element);
    auto sl = dynamic_cast<SelectionList *>(element);
    bool containerTornOff = vv->allowTearOff() && container && container->m_menu && container->m_menu->wasMoved();
    if (menu || isRoot() || sl)
    {
        if (sl && dynamic_cast<coCheckboxMenuItem *>(menuItem))
        {
            showMenu(false);
            auto idx = m_menu->index(menuItem);
            sl->select(idx);
            sl->trigger();
        }
        auto tmi = dynamic_cast<coLabelSubMenuToolboxItem *>(menuItem);
        if (vv->useToolbar() && (!containerTornOff || tmi))
        {
            bool open = false;
            if (tmi)
            {
                open = tmi->isOpen();
            }

            if (auto smi = dynamic_cast<coSubMenuItem *>(menuItem))
            {
                open = smi->isOpen();
                if (container)
                {
                    container->clearStackToTop();
                }
                else
                {
                    root()->clearStackToTop();
                }
            }

            showMenu(open);
            if (open)
                hideOthers();

            return;
        }
    }
    else if (auto s = dynamic_cast<Slider *>(element))
    {
        if (vv->useToolbar() && !containerTornOff && !vv->isInToolbar(element) && m_toolboxItem && menuItem != m_toolboxItem)
        {
            if (container)
            {
                container->clearStackToTop();
                container->showMenu(false);
            }
            else if (isInStack())
            {
                root()->clearStackToTop();
            }
            addToStack();
            if (cover->getToolBar())
                cover->getToolBar()->add(m_toolboxItem);
        }
        else
        {
            updateSlider(s, menuItem, true);
        }
        return;
    }
    else if (auto b = dynamic_cast<Button *>(element))
    {
        if (auto cb = dynamic_cast<coCheckboxMenuItem *>(menuItem))
        {
            b->setState(cb->getState());
            if (auto tb = dynamic_cast<coIconToggleButtonToolboxItem *>(m_toolboxItem))
            {
                if (tb->getState() != b->state())
                    tb->setState(b->state());
            }
            b->trigger();
        }
        if (auto tb = dynamic_cast<coIconToggleButtonToolboxItem *>(menuItem))
        {
            b->setState(tb->getState());
            if (auto cb = dynamic_cast<coCheckboxMenuItem *>(m_menuItem))
                cb->setState(b->state());
            b->trigger();
        }
    }
    else if (auto a = dynamic_cast<Action *>(element))
    {
        auto b = dynamic_cast<coButtonMenuItem *>(menuItem);
        auto tb = dynamic_cast<coIconButtonToolboxItem *>(menuItem);
        if (b || tb)
            a->trigger();
    }

    if (vv->useToolbar() && !containerTornOff)
    {
        if (menuItem != m_toolboxItem)
        {
            if (container)
            {
                container->showMenu(false);
                container->clearStackToTop();
            }
            else
            {
                root()->clearStackToTop();
            }
        }
    }
}

void VruiViewElement::menuReleaseEvent(coMenuItem *menuItem)
{
    if (auto s = dynamic_cast<Slider *>(element))
    {
        if (!m_toolboxItem || menuItem == m_toolboxItem)
            updateSlider(s, menuItem, false);
    }
}

VruiViewElement *VruiViewElement::root() const
{
    if (!view)
        return nullptr;

    auto vv = static_cast<VruiView *>(view);
    return vv->m_root;
}

void VruiViewElement::hideOthers()
{
    if (!view)
        return;

    auto vv = static_cast<VruiView *>(view);

    for (auto ve: vv->m_toolbarStack)
    {
        if (ve != this)
        {
            if (!ve->m_menu || !ve->m_menu->wasMoved() || !vv->allowTearOff())
            {
                ve->showMenu(false);
            }
        }
    }
}

void VruiViewElement::showMenu(bool state)
{
    auto smi = dynamic_cast<coSubMenuItem *>(m_menuItem);
    auto tmi = dynamic_cast<coSubMenuToolboxItem *>(m_toolboxItem);

    if (m_menu)
        m_menu->setMoved(false);

    if (state)
    {
        if (smi)
            smi->openSubmenu();

        if (!isInStack())
        {
            addToStack();
            if (cover->getToolBar())
                cover->getToolBar()->add(tmi);
        }

        if (tmi)
        {
            tmi->openSubmenu();
            tmi->positionSubmenu();
        }
    }
    else
    {
        if (smi)
            smi->closeSubmenu();
        if (tmi)
            tmi->closeSubmenu();
    }
}

bool VruiViewElement::isRoot() const
{
    if (!view)
        return false;

    auto vv = static_cast<VruiView *>(view);

    if (vv->m_toolbarStack.empty())
        return false;

    return vv->m_toolbarStack.front() == this;
}

bool VruiViewElement::isTopOfStack() const
{
    if (!view)
        return false;

    auto vv = static_cast<VruiView *>(view);

    if (vv->m_toolbarStack.empty())
        return false;

    return vv->m_toolbarStack.back() == this;
}

void VruiViewElement::addToStack()
{
    assert(!isInStack());

    if (!view)
        return;

    auto vv = static_cast<VruiView *>(view);
    vv->m_toolbarStack.push_back(this);
}

void VruiViewElement::popStack()
{
    if (!view)
        return;

    auto vv = static_cast<VruiView *>(view);
    if (vv->m_toolbarStack.empty())
        return;
    vv->m_toolbarStack.pop_back();
}

void VruiViewElement::clearStackToTop()
{
    if (!isInStack())
    {
        std::cerr << "ui::VruiView: refusing to clear toolbar stack: " << element->path() << " not in stack" << std::endl;
        return;
    }

    if (!view)
        return;
    auto vv = static_cast<VruiView *>(view);

    while (!isTopOfStack())
    {
        if (auto tb = cover->getToolBar())
            tb->remove(vv->m_toolbarStack.back()->m_toolboxItem);
        popStack();
    }
}

bool VruiViewElement::isInStack() const
{
    if (!view)
        return false;

    auto vv = static_cast<VruiView *>(view);

    for (auto ve: vv->m_toolbarStack)
        if (ve == this)
            return true;

    return false;
}

}
}
