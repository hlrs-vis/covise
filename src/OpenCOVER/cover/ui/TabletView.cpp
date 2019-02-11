#include "TabletView.h"

#include <cassert>
#include <iostream>

#include "Element.h"
#include "Menu.h"
#include "ButtonGroup.h"
#include "Label.h"
#include "Action.h"
#include "Button.h"
#include "Slider.h"
#include "SelectionList.h"
#include "EditField.h"
#include "FileBrowser.h"

#include <cover/coVRPluginSupport.h>

namespace opencover {
namespace ui {

const bool GroupAsGroupbox = false;
const bool GroupWithTitle = true;
const bool SlidersOnOwnRow = true;
const int SliderLength = 2;

TabletView::TabletView(const std::string &name, coTabletUI *tui)
: View(name)
{
    assert(tui);

    m_tui = tui;
    m_root = new TabletViewElement(nullptr);
    m_root->m_tabs = new coTUITabFolder(m_tui, "TabletViewRoot");
    m_root->m_id = m_root->m_tabs->getID();

}

TabletView::TabletView(const std::string &name, coTUITabFolder *root)
: View(name)
{
    assert(root);

    m_tui = root->tui();
    m_root = new TabletViewElement(nullptr);
    m_root->m_tabs = root;
    m_root->m_id = m_root->m_tabs->getID();
}

TabletView::~TabletView()
{
    delete m_root->m_elem;
    m_root->m_elem = nullptr;
    delete m_root;
}

View::ViewType TabletView::typeBit() const
{
    return View::Tablet;
}

TabletViewElement *TabletView::tuiElement(const std::string &path) const
{
    auto e = viewElement(path);
    auto ve = dynamic_cast<TabletViewElement *>(e);
    assert(!e || ve);
    return ve;
}

TabletViewElement *TabletView::tuiElement(const Element *elem) const
{
    if (!elem)
        return nullptr;
    auto e = viewElement(elem);
    auto ve = dynamic_cast<TabletViewElement *>(e);
    assert(!e || ve);
    return ve;
}

TabletViewElement *TabletView::tuiParent(const Element *elem) const
{
    if (!elem)
        return nullptr;

    if (elem->parent())
        return tuiElement(elem->parent());

    return m_root;
}

int TabletView::tuiParentId(const Element *elem) const
{
    auto parent = tuiParent(elem);
    if (parent && parent->m_id >= 0)
        return parent->m_id;

    return m_root->m_id;
}

TabletViewElement *TabletView::tuiContainer(const Element *elem) const
{
    auto p = tuiParent(elem);
    if (p)
    {
        if (p->m_elem && !dynamic_cast<coTUILabel *>(p->m_elem))
            return p;
        if (p->element)
            return tuiParent(p->element);
    }
    return m_root;
}

int TabletView::tuiContainerId(const Element *elem) const
{
    auto parent = tuiContainer(elem);
    if (parent && parent->m_id >= 0)
    {
        return parent->m_id;
    }

    return m_root->m_id;
}

void TabletView::add(TabletViewElement *ve, const Element *elem)
{
    if (!ve)
    {
        std::cerr << "TabletView::add: Warning: no TabletViewElement" << std::endl;
        return;
    }
    if (!elem)
    {
        std::cerr << "TabletView::add: Warning: no Element" << std::endl;
        return;
    }
    auto container = tuiContainer(elem);
    auto parent = tuiParent(elem);

    if (parent)
    {
        ve->m_parentLevel = parent->m_parentLevel+1;
    }
    if (container)
    {
        ve->m_containerLevel = container->m_containerLevel+1;
    }

    auto slider = dynamic_cast<const Slider *>(elem);
    auto group = dynamic_cast<const Group *>(elem);
    auto menu = dynamic_cast<const Menu *>(elem);
    const bool beginGroup = (m_lastParentLevel < ve->m_parentLevel) && !group;
    const bool groupWasEnded = m_lastParentLevel > ve->m_parentLevel;
    m_lastParentLevel = ve->m_parentLevel;
    if (menu || GroupAsGroupbox)
        group = nullptr;
    if (container)
    {
        if (group && GroupWithTitle)
            container->newRow();
        else if (beginGroup)
            container->newRow();
        else if (groupWasEnded)
            container->newRow();
        else if (slider && SlidersOnOwnRow)
            container->newRow();
    }
    if (auto te = ve->m_elem)
    {
        ve->m_id = te->getID();
        if (container)
            te->setPos(container->m_column, container->m_row);
        else
            te->setPos(0, 0);
        te->setLabel(elem->text());
        te->setEventListener(ve);
    }

    if (container)
    {
        ++container->m_column;
        if (group && GroupWithTitle)
        {
            container->newRow();
        }
        else if (slider)
        {
            container->m_column += SliderLength;
            if (SlidersOnOwnRow)
                container->newRow();
        }
    }

    updateParent(elem);
}

void TabletView::updateEnabled(const Element *elem)
{
    auto ve = tuiElement(elem);
    if (!ve)
        return;
    auto te = ve->m_elem;
    if (!te)
        return;
    te->setEnabled(elem->enabled());
}

void TabletView::updateVisible(const Element *elem)
{
    //std::cerr << "ui::Tablet: updateVisible(" << elem->path() << "): visible=" << elem->visible() << std::endl;
    auto ve = tuiElement(elem);
    if (!ve)
    {
        //std::cerr << "ui::Tablet: updateVisible(" << elem->path() << "): visible=" << elem->visible() << ": NO view element" << std::endl;
        return;
    }
    auto te = ve->m_elem;
    if (!te)
    {
        //std::cerr << "ui::Tablet: updateVisible(" << elem->path() << "): visible=" << elem->visible() << ": NO tablet element" << std::endl;
        return;
    }
    te->setHidden(!elem->visible(this));
}

void TabletView::updateText(const Element *elem)
{
    auto ve = tuiElement(elem);
    if (ve)
    {
        if (ve->m_elem)
        {
            ve->m_elem->setLabel(elem->text());
        }
    }
}

void TabletView::updateState(const Button *button)
{
    auto ve = tuiElement(button);
    if (ve)
    {
        if (auto b = dynamic_cast<coTUIToggleButton *>(ve->m_elem))
            b->setState(button->state());
    }
}

void TabletView::updateParent(const Element *elem)
{
    auto ve = tuiElement(elem);
    if (!ve)
        return;
    updateVisible(elem);
}

void TabletView::updateChildren(const SelectionList *sl)
{
    auto ve = tuiElement(sl);
    if (!ve)
        return;
    auto tcb = dynamic_cast<coTUIComboBox *>(ve->m_elem);
    if (!tcb)
        return;
    tcb->clear();
    auto items = sl->items();
    for (auto c: items)
        tcb->addEntry(c);
    tcb->setSelectedEntry(sl->selectedIndex());
}

void TabletView::updateIntegral(const Slider *slider)
{
    auto ve = tuiElement(slider);
    if (!ve)
        return;
    if (slider->integral())
    {
        if (!dynamic_cast<coTUISlider *>(ve->m_elem))
        {
            delete ve->m_elem;
            auto ts = new coTUISlider(m_tui, slider->path(), tuiContainerId(slider));
            ve->m_elem = ts;
            updateScale(slider);
        }
    }
    else
    {
        if (!dynamic_cast<coTUIFloatSlider *>(ve->m_elem))
        {
            delete ve->m_elem;
            auto ts = new coTUIFloatSlider(m_tui, slider->path(), tuiContainerId(slider));
            ve->m_elem = ts;
            updateScale(slider);
        }
    }
    ve->m_elem->setSize(SliderLength+1, 1);
}

void TabletView::updateScale(const Slider *slider)
{
    auto ve = tuiElement(slider);
    if (!ve)
        return;
    if (!slider->integral())
    {
        if (auto ts = dynamic_cast<coTUIFloatSlider *>(ve->m_elem))
        {
            ts->setLogarithmic(slider->scale() == Slider::Logarithmic);
        }
        else
        {
            std::cerr << "TabletView::updateScale: " << slider->path() << " not a coTUIFloatSlider" << std::endl;

        }
    }
    updateBounds(slider);
    updateValue(slider);
}

void TabletView::updateValue(const Slider *slider)
{
    auto ve = tuiElement(slider);
    if (!ve)
        return;
    if (auto vs = dynamic_cast<coTUIFloatSlider *>(ve->m_elem))
    {
        vs->setValue(slider->value());
    }
    else if (auto vs = dynamic_cast<coTUISlider *>(ve->m_elem))
    {
        vs->setValue(slider->value());
    }
}

void TabletView::updateBounds(const Slider *slider)
{
    auto ve = tuiElement(slider);
    if (!ve)
        return;
    if (auto vp = dynamic_cast<coTUIFloatSlider *>(ve->m_elem))
    {
        vp->setRange(slider->min(), slider->max());
    }
    else if (auto vs = dynamic_cast<coTUISlider *>(ve->m_elem))
    {
        vs->setRange(slider->min(), slider->max());
    }
}

void TabletView::updateValue(const TextField *input)
{
    auto ve = tuiElement(input);
    if (!ve)
        return;
    if (auto vs = dynamic_cast<coTUIEditField *>(ve->m_elem))
    {
        vs->setText(input->value());
    }
    else if (auto vs = dynamic_cast<coTUIEditTextField *>(ve->m_elem))
    {
        vs->setText(input->value());
    }
    else if (auto te = dynamic_cast<coTUIFileBrowserButton *>(ve->m_elem))
    {
        te->setCurDir(input->value().c_str());
    }
}

void TabletView::updateFilter(const FileBrowser *fb)
{
    auto ve = tuiElement(fb);
    if (!ve)
        return;
    if (auto te = dynamic_cast<coTUIFileBrowserButton *>(ve->m_elem))
    {
        te->setFilterList(fb->filter());
    }
}

TabletViewElement *TabletView::elementFactoryImplementation(Label *label)
{
    auto parent = tuiContainer(label);
    auto ve = new TabletViewElement(label);

    auto te = new coTUILabel(m_tui, label->path(), tuiContainerId(label));
    ve->m_elem = te;

    add(ve, label);
    return ve;
}

TabletViewElement *TabletView::elementFactoryImplementation(Button *button)
{
    auto ve = new TabletViewElement(button);
    auto parent = tuiContainer(button);

    auto te = new coTUIToggleButton(m_tui, button->path(), tuiContainerId(button));
    ve->m_elem = te;

    add(ve, button);
    return ve;
}

TabletViewElement *TabletView::elementFactoryImplementation(Group *group)
{
    auto ve = new TabletViewElement(group);
    auto parent = tuiContainer(group);
    if (GroupAsGroupbox)
    {
        ve->m_elem = new coTUIGroupBox(m_tui, group->path(), tuiContainerId(group));
    }
    else if (GroupWithTitle)
    {
        ve->m_elem = new coTUILabel(m_tui, group->path(), tuiContainerId(group));
    }
    add(ve, group);

    return ve;
}


TabletViewElement *TabletView::elementFactoryImplementation(Slider *slider)
{
    auto ve = new TabletViewElement(slider);
    auto parent = tuiContainer(slider);

    if (slider->integral())
    {
        auto te = new coTUISlider(m_tui, slider->path(), tuiContainerId(slider));
        ve->m_elem = te;
    }
    else
    {
        auto te = new coTUIFloatSlider(m_tui, slider->path(), tuiContainerId(slider));
        ve->m_elem = te;
    }
    ve->m_elem->setSize(SliderLength+1, 1);
    ve->m_elem->setLabel(slider->text());

    add(ve, slider);
    return ve;

}

TabletViewElement *TabletView::elementFactoryImplementation(SelectionList *sl)
{
    auto parent = tuiContainer(sl);
    auto ve = new TabletViewElement(sl);
    auto tcb = new coTUIComboBox(m_tui, sl->path(), tuiContainerId(sl));
    ve->m_elem = tcb;
    add(ve, sl);
    return ve;
}

TabletViewElement *TabletView::elementFactoryImplementation(EditField *input)
{
    auto ve = new TabletViewElement(input);
    auto parent = tuiContainer(input);

    auto te = new coTUIEditField(m_tui, input->path(), tuiContainerId(input));
    ve->m_elem = te;
    te->setText(input->value());

    ve->m_elem->setLabel(input->text());

    add(ve, input);
    return ve;

}

TabletViewElement *TabletView::elementFactoryImplementation(FileBrowser *fb)
{
    auto ve = new TabletViewElement(fb);
    auto parent = tuiContainer(fb);

    auto te = new coTUIFileBrowserButton(m_tui, fb->path().c_str(), tuiContainerId(fb));
    ve->m_elem = te;
    if (fb->forSaving())
        te->setMode(coTUIFileBrowserButton::SAVE);
    else
        te->setMode(coTUIFileBrowserButton::OPEN);

    ve->m_elem->setLabel(fb->text());

    add(ve, fb);
    return ve;
}

TabletViewElement *TabletView::elementFactoryImplementation(Menu *menu)
{
    auto parent = tuiContainer(menu);
    if (!parent)
        return nullptr;

    auto ve = new TabletViewElement(menu);
    if (!parent->m_tabs)
    {
        parent->m_tabs = new coTUITabFolder(m_tui, parent->element->path()+"_Folder", tuiContainerId(menu));
        parent->m_tabs->setSize(-1, 1);
        parent->m_tabs->setPos(0,50);
    }

#if 0
    if (parent == m_root)
    {
        auto te = new coTUIParagraphLayout(m_tui, menu->path(), parent->m_tabs->getID());
        ve->m_elem = te;
        add(ve, menu);
    }
    else
#endif
    {
        auto tab = new coTUITab(m_tui, menu->path(), parent->m_tabs->getID());
        ve->m_elem = tab;
        add(ve, menu);
    }
    return ve;
}

TabletViewElement *TabletView::elementFactoryImplementation(Action *action)
{
    auto parent = tuiContainer(action);
    auto ve = new TabletViewElement(action);

    auto te = new coTUIButton(m_tui, action->path(), tuiContainerId(action));
    ve->m_elem = te;

    add(ve, action);
    return ve;
}

TabletViewElement::TabletViewElement(Element *elem)
: View::ViewElement(elem)
{
}

TabletViewElement::~TabletViewElement()
{
    delete m_tab;
    m_tab = nullptr;
    delete m_elem;
    m_elem = nullptr;
}

void TabletViewElement::newRow()
{
    ++m_row;
    m_column = 0;
}

void TabletViewElement::tabletEvent(coTUIElement *elem)
{
    //std::cerr << "tabletEvent: " << element->path() << std::endl;
    if (auto b = dynamic_cast<Button *>(element))
    {
        if (auto tb = dynamic_cast<coTUIToggleButton *>(elem))
        {
            b->setState(tb->getState());
            b->trigger();
        }
    }
    else if (auto s = dynamic_cast<Slider *>(element))
    {
        if (auto ts = dynamic_cast<coTUIFloatSlider *>(elem))
        {
            s->setValue(ts->getValue());
            s->setMoving(true);
            s->trigger();
        }
        else if (auto ts = dynamic_cast<coTUISlider *>(elem))
        {
            s->setValue(ts->getValue());
            s->setMoving(true);
            s->trigger();
        }
    }
    else if (auto sl = dynamic_cast<SelectionList *>(element))
    {
        if (auto tcb = dynamic_cast<coTUIComboBox *>(elem))
        {
            sl->select(tcb->getSelectedEntry());
            sl->trigger();
        }
    }
    else if (auto in = dynamic_cast<EditField *>(element))
    {
        if (auto te = dynamic_cast<coTUIEditTextField *>(elem))
        {
            in->setValue(te->getText());
            in->trigger();
        }
        else if (auto te = dynamic_cast<coTUIEditField *>(elem))
        {
            in->setValue(te->getText());
            in->trigger();
        }
    }
    else if (auto fb = dynamic_cast<FileBrowser *>(element))
    {
        if (auto te = dynamic_cast<coTUIFileBrowserButton *>(elem))
        {
            fb->setValue(te->getSelectedPath());
            fb->trigger();
        }
    }
}

void TabletViewElement::tabletPressEvent(coTUIElement *elem)
{
    //std::cerr << "tabletPressEvent: " << element->path() << std::endl;
}

void TabletViewElement::tabletReleaseEvent(coTUIElement *elem)
{
    //std::cerr << "tabletReleaseEvent: " << element->path() << std::endl;

    if (auto a = dynamic_cast<Action *>(element))
    {
        if (auto tb = dynamic_cast<coTUIButton *>(elem))
        {
            a->trigger();
        }
    }
    else if (auto s = dynamic_cast<Slider *>(element))
    {
        if (auto ts = dynamic_cast<coTUIFloatSlider *>(elem))
        {
            s->setMoving(false);
            s->trigger();
        }
        else if (auto ts = dynamic_cast<coTUISlider *>(elem))
        {
            s->setMoving(false);
            s->trigger();
        }
    }
}


}
}
