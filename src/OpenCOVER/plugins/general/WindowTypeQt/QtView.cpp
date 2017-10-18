#include "QtView.h"

#include <cover/ui/Menu.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Label.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/SelectionList.h>

#include <QMenuBar>
#include <QAction>
#include <QActionGroup>
#include <QLabel>
#include <QWidgetAction>
#include <QSlider>

#include <cassert>
#include <iostream>

namespace opencover {
namespace ui {

namespace {
const int SliderIntMax = 1000000000;

QString sliderText(const Slider *slider)
{
    QString text = QString::fromStdString(slider->text());
    text += ": ";
    text += QString::number(slider->value());
    return text;
}

}


QtView::QtView(QMenuBar *menubar)
: View("Qt")
, m_menubar(menubar)
{
}

void QtView::add(QtViewElement *ve)
{
    if (!ve)
        return;
    auto elem = ve->element;
    auto parent = qtViewParent(elem);
    auto container = qtContainerWidget(elem);

    if (auto m = dynamic_cast<QMenu *>(ve->object))
    {
        if (auto pmb = dynamic_cast<QMenuBar *>(container))
        {
            pmb->addMenu(m);
        }
        else if (auto pm = dynamic_cast<QMenu *>(container))
        {
            pm->addMenu(m);
        }
        return;
    }

    auto a = ve->action;
    if (!a)
        a = dynamic_cast<QAction *>(ve->object);
    if (!a)
        return;

    if (auto ag = dynamic_cast<QActionGroup *>(qtObject(parent)))
    {
        //std::cerr << "ui: adding button " << button->path() << " to action group" << std::endl;
        if (ve->action)
            ag->addAction(ve->action);
        else if (auto a = dynamic_cast<QAction *>(ve->object))
            ag->addAction(a);
        if (container)
            container->addActions(ag->actions());
    }
    else if (container)
    {
        //std::cerr << "ui: adding button " << button->path() << " to widget" << std::endl;
        container->addAction(a);
    }
}

QtViewElement *QtView::qtViewParent(const Element *elem) const
{
    auto ve = viewParent(elem);
    return dynamic_cast<QtViewElement *>(ve);
}

QtViewElement *QtView::qtViewContainer(const Element *elem) const
{
    auto parent = qtViewParent(elem);
    if (parent)
    {
        if (dynamic_cast<QWidget *>(parent->object))
            return parent;
        return qtViewParent(parent->element);
    }
    return nullptr;
}

QtViewElement *QtView::qtViewElement(const Element *elem) const
{
    auto ve = viewElement(elem);
    return dynamic_cast<QtViewElement *>(ve);
}

QObject *QtView::qtObject(const Element *elem) const
{
    auto qve = qtViewElement(elem);
    if (qve)
        return qve->object;
    return nullptr;
}

QWidget *QtView::qtWidget(const Element *elem) const
{
    return dynamic_cast<QWidget *>(qtObject(elem));
}

QObject *QtView::qtObject(const QtViewElement *elem) const
{
    if (elem)
        return elem->object;
    return nullptr;
}

QWidget *QtView::qtWidget(const QtViewElement *elem) const
{
    return dynamic_cast<QWidget *>(qtObject(elem));
}

QWidget *QtView::qtContainerWidget(const Element *elem) const
{
    auto parent = viewParent(elem);
    auto ve = qtViewContainer(elem);
    auto w = qtWidget(ve);
    if (!parent && !w)
        return m_menubar;
    return w;
}

QtViewElement *QtView::elementFactoryImplementation(Menu *menu)
{
    auto parent = qtContainerWidget(menu);
    auto m = new QMenu(parent);
    auto ve = new QtViewElement(menu, m);
    ve->action = m->menuAction();
    add(ve);
    ve->markForDeletion(m);
    return ve;
}

QtViewElement *QtView::elementFactoryImplementation(Group *group)
{
    auto parent = qtViewParent(group);
    if (!parent)
        return nullptr;

    auto ag = new QActionGroup(qtObject(parent));
    ag->setExclusive(false);

    auto sep = new QAction(ag);
    sep->setShortcutContext(Qt::WidgetShortcut);
    sep->setSeparator(true);
    sep->setText(QString::fromStdString(group->text()));
    ag->addAction(sep);

    auto sep2 = new QAction(ag);
    sep2->setShortcutContext(Qt::WidgetShortcut);
    sep2->setSeparator(true);
    ag->addAction(sep2);

    auto ve = new QtViewElement(group, ag);
    ve->group = ag;
    if (auto w = qtWidget(parent))
        w->addActions(ag->actions());
    ve->markForDeletion(ag);
#if 0
    connect(a, &QAction::triggered, [rg](bool state){rg->setState(state); rg->trigger();});
#endif
    return ve;
}

QtViewElement *QtView::elementFactoryImplementation(Label *label)
{
    auto parent = qtViewParent(label);
    if (!parent)
        return nullptr;

    auto la = new QWidgetAction(qtObject(parent));
    auto l = new QLabel(qtWidget(parent));
    la->setDefaultWidget(l);

    auto ve = new QtViewElement(label, la);
    ve->action = la;
    ve->label = l;
    ve->markForDeletion(la);
    add(ve);
    return ve;
}

QtViewElement *QtView::elementFactoryImplementation(Action *action)
{
    auto parent = qtViewParent(action);
    if (!parent)
        return nullptr;

    auto a = new QAction(qtObject(parent));
    a->setShortcutContext(Qt::WidgetShortcut);
    a->setCheckable(false);
    auto ve = new QtViewElement(action, a);
    ve->action = a;
    add(ve);
    ve->markForDeletion(a);
#if 0
    if (auto w = qtWidget(parent))
        w->addAction(a);
#endif
    connect(a, &QAction::triggered, [action](bool){action->trigger();});
    return ve;
}

QtViewElement *QtView::elementFactoryImplementation(Button *button)
{
    auto parent = qtViewParent(button);
    if (!parent)
        return nullptr;

    auto a = new QAction(qtObject(parent));
    a->setShortcutContext(Qt::WidgetShortcut);
    a->setCheckable(true);
    auto ve = new QtViewElement(button, a);
    ve->action = a;
    add(ve);
    ve->markForDeletion(a);
    connect(a, &QAction::triggered, [button](bool state){
        button->setState(state);
        button->trigger();
    });
    return ve;
}

QtViewElement *QtView::elementFactoryImplementation(Slider *slider)
{
    auto parent = qtViewParent(slider);
    if (!parent)
        return nullptr;

    auto po = qtObject(parent);
    auto pw = qtWidget(parent);

    auto s = new QSlider(Qt::Horizontal, pw);
    auto a = new QWidgetAction(po);
    a->setDefaultWidget(s);
    pw->addAction(a);
    auto ve = new QtViewElement(slider, s);
    ve->action = a;

    auto l = new QLabel(pw);
    auto la = new QWidgetAction(po);
    la->setDefaultWidget(l);
    pw->addAction(la);
    ve->label = l;
    add(ve);
    ve->markForDeletion(la);
    ve->markForDeletion(a);

    connect(s, &QSlider::sliderMoved, [slider](int value){
        if (slider->integer())
        {
            slider->setValue(value);
        }
        else
        {
            auto r = slider->max() - slider->min();
            auto a = (double)value/SliderIntMax;
            double val = slider->min()+r*a;
            slider->setValue(val);
        }
        slider->setMoving(true);
        slider->trigger();
    });
    connect(s, &QSlider::sliderReleased, [slider](){
        slider->setMoving(false);
        slider->trigger();
    });

    return ve;
}

QtViewElement *QtView::elementFactoryImplementation(SelectionList *sl)
{
    auto parent = qtContainerWidget(sl);
    auto m = new QMenu(parent);
    auto ve = new QtViewElement(sl, m);
    ve->action = m->menuAction();
    add(ve);
    ve->markForDeletion(m);
    return ve;
}

void QtView::updateEnabled(const Element *elem)
{
    auto w = qtWidget(elem);
    if (w)
        w->setEnabled(elem->enabled());
}

void QtView::updateVisible(const Element *elem)
{
    if (auto w = qtWidget(elem))
    {
        auto m = dynamic_cast<QMenu *>(w);
        if (!m)
        {
            w->setVisible(elem->visible());
        }
    }

    if (auto ve = qtViewElement(elem))
    {
        if (auto a = ve->action)
            a->setVisible(elem->visible());
    }
}

void QtView::updateText(const Element *elem)
{
    auto ve = qtViewElement(elem);
    if (!ve)
        return;
    auto o = qtObject(elem);
    auto t = QString::fromStdString(elem->text());
    if (auto l = dynamic_cast<QLabel *>(o))
    {
        l->setText(t);
    }
    else if (auto a = dynamic_cast<QAction *>(o))
    {
        a->setText(t);
    }
    else if (auto m = dynamic_cast<QMenu *>(o))
    {
        m->setTitle(t);
    }
    else if (auto qs = dynamic_cast<QSlider *>(o))
    {
        auto s = dynamic_cast<const Slider *>(elem);
        if (ve->label)
            ve->label->setText(sliderText(s));
    }
}

void QtView::updateState(const Button *button)
{
    auto o = qtObject(button);
    auto a = dynamic_cast<QAction *>(o);
    if (a)
        a->setChecked(button->state());
}

void QtView::updateChildren(const Menu *menu)
{
#if 0
    auto o = qtObject(menu);
    auto m = dynamic_cast<QMenu *>(o);
    if (!m)
        return;

    auto actions = m->actions();
    for (size_t i=0; i<menu->numChildren(); ++i)
    {
        auto ve = qtViewElement(menu->child(i));
        if (!ve)
            continue;
        auto obj = ve->object;
        auto act = ve->action;
        if (!act)
            act = dynamic_cast<QAction *>(obj);
        if (!act)
            continue;

        if (auto a = dynamic_cast<QAction *>(obj))
            m->addAction(a);
        else if (auto mm = dynamic_cast<QMenu *>(obj))
            m->addMenu(mm);
        else if (auto ag = dynamic_cast<QActionGroup *>(obj))
            m->addActions(ag->actions());
    }
#endif
}

void QtView::updateChildren(const SelectionList *sl)
{
    auto ve = qtViewElement(sl);
    if (!ve)
        return;
    auto m = dynamic_cast<QMenu *>(ve->object);
    if (!m)
        return;
    auto ag = ve->group;
    if (ag)
    {
        for (const auto &a: ag->actions())
        {
            m->removeAction(a);
        }
        delete ag;
    }

    ve->group = new QActionGroup(m);
    ag = ve->group;
    ve->markForDeletion(ag);
    connect(ve->group, &QActionGroup::triggered, [ve, sl](QAction *a){
        auto al = ve->group->actions();
        int idx = -1;
        for (int i=0; i<al.size(); ++i)
        {
            if (a == al[i])
                idx = i;
        }
        const_cast<SelectionList *>(sl)->select(idx);
        sl->trigger();
    });
    const auto items = sl->items();
    for (size_t i=0; i<items.size(); ++i)
    {
        auto a = new QAction(ag);
        a->setShortcutContext(Qt::WidgetShortcut);
        a->setText(QString::fromStdString(items[i]));
        a->setCheckable(true);
        a->setChecked(sl->selection()[i]);
    }
    if (auto m = dynamic_cast<QMenu *>(ve->object))
    {
        std::string t = sl->text();
        int s = sl->selectedIndex();
        if (s >= 0)
        {
            t += ": ";
            t += sl->items()[s];
        }
        m->setTitle(QString::fromStdString(t));
    }
    m->addActions(ag->actions());
}

void QtView::updateInteger(const Slider *slider)
{
    updateBounds(slider);
    updateValue(slider);
}

void QtView::updateValue(const Slider *slider)
{
    auto ve = qtViewElement(slider);
    if (!ve)
        return;
    auto o = qtObject(slider);
    auto s = dynamic_cast<QSlider *>(o);
    if (s)
    {
        if (slider->integer())
        {
            if (s->value() != slider->value())
                s->setValue(slider->value());
        }
        else
        {
            auto min = slider->min();
            auto r = slider->max() - min;
            auto v = int((slider->value()-min)/r*SliderIntMax);
            if (v != s->value())
            {
                //std::cerr << "update: v=" << v << ", slider=" << s->value() << std::endl;
                s->setValue(v);
            }
        }
    }
    if (ve->label)
        ve->label->setText(sliderText(slider));
}

void QtView::updateBounds(const Slider *slider)
{
    auto o = qtObject(slider);
    auto s = dynamic_cast<QSlider *>(o);
    if (s)
    {
        if (slider->integer())
        {
            if (s->minimum() != slider->min() || s->maximum() != slider->max())
                s->setRange(slider->min(), slider->max());
        }
        else
        {
            if (s->minimum() != 0 || s->maximum() != SliderIntMax)
            {
                s->setRange(0, SliderIntMax);
                updateValue(slider);
            }
        }
    }
}

QtViewElement::QtViewElement(Element *elem, QObject *obj)
: View::ViewElement (elem)
, object(obj)
{
    QString n = QString::fromStdString(elem->name());
    if (object)
        object->setObjectName(n);
}

QtViewElement::~QtViewElement()
{
    while (!toDelete.empty())
    {
        delete toDelete.back();
        // automatically removed from deletion list through QObject connection established in markForDeletion
    }

    action = nullptr;
    object = nullptr;
    group = nullptr;
    label = nullptr;
}

void QtViewElement::markForDeletion(QObject *obj)
{
    obj->connect(obj, &QObject::destroyed, [this](QObject *obj){
        auto it = std::find(toDelete.begin(), toDelete.end(), obj);
        if (it != toDelete.end())
        {
            //std::cerr << "not deleting something for " << element->path() << std::endl;
            toDelete.erase(it);
        }
    });
    toDelete.push_back(obj);
}

}
}

#include "moc_QtView.cpp"
