#include "QtView.h"

#include <config/CoviseConfig.h>

#include <cover/ui/Menu.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Label.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/SelectionList.h>

#include <QMenuBar>
#include <QToolBar>
#include <QAction>
#include <QActionGroup>
#include <QLabel>
#include <QWidgetAction>
#include <QSlider>
#include <QVBoxLayout>
#include <QTextStream>
#include <QFontMetrics>

#include <cassert>
#include <iostream>

namespace opencover {
namespace ui {

namespace {
const int SliderIntMax = 1000000000;

QString sliderText(const Slider *slider, double value, int digits=0)
{
    QString text;
    QTextStream(&text) << QString::fromStdString(slider->text()) << ": " << qSetPadChar('0') << qSetFieldWidth(digits) << value;
    return text;
}

QString sliderText(const Slider *slider)
{
    return sliderText(slider, slider->value());
}

QString sliderWidthText(const Slider *slider)
{
    int digits = std::max(QString::number(slider->min()).size(), QString::number(slider->max()).size());
    return sliderText(slider, 0.0, digits);
}

}

QtView::QtView(QMenuBar *menubar, QToolBar *toolbar)
: View("Qt")
, m_menubar(menubar)
, m_toolbar(toolbar)
{
}

QtView::QtView(QToolBar *toolbar)
: View("QtToolbar")
, m_toolbar(toolbar)
{
}

void QtView::add(QtViewElement *ve)
{
    if (!ve)
        return;
    auto elem = ve->element;
    auto parent = qtViewParent(elem);
    auto container = qtContainerWidget(elem);

    if (m_menubar)
    {
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
    }

    std::string configPath = "COVER.UI." + elem->path();
    bool exists = false;
    bool inToolbar = covise::coCoviseConfig::isOn("toolbar", configPath, elem->priority()>=ui::Element::Toolbar, &exists);

    auto a = ve->action;
    if (!a)
        a = dynamic_cast<QAction *>(ve->object);
    if (!a)
        return;

    Group *group = nullptr;
    if (ve->element)
        group = ve->element->parent();
    if (auto ag = dynamic_cast<QActionGroup *>(qtObject(parent)))
    {
        //std::cerr << "ui: adding button " << button->path() << " to action group" << std::endl;
        if (ve->action)
            ag->addAction(ve->action);
        else if (auto a = dynamic_cast<QAction *>(ve->object))
            ag->addAction(a);
        if (container)
            container->addActions(ag->actions());
        if (m_toolbar && inToolbar)
        {
            if (m_lastToolbarGroup && m_lastToolbarGroup != group)
                m_toolbar->addSeparator();
            m_toolbar->addActions(ag->actions());
            m_lastToolbarGroup = group;
        }
    }
    else if (container)
    {
        //std::cerr << "ui: adding button " << button->path() << " to widget" << std::endl;
        container->addAction(a);
        if (m_toolbar && inToolbar)
        {
            if (m_lastToolbarGroup && m_lastToolbarGroup != group)
                m_toolbar->addSeparator();
            //std::cerr << "ui: adding action for " << ve->element->path() << " to toolbar" << std::endl;
            m_toolbar->addAction(a);
            m_lastToolbarGroup = group;
        }
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
    auto ve = qtViewContainer(elem);
    auto w = qtWidget(ve);
    if (!w)
        w = qtWidget(qtViewElement(elem->parent()));
    if (w)
        return w;
    if (m_menubar)
        return m_menubar;
    return m_toolbar;
}

QtViewElement *QtView::elementFactoryImplementation(Menu *menu)
{
    auto parent = qtContainerWidget(menu);
    auto m = new QMenu(parent);
    m->setTearOffEnabled(true);
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

    auto la = new QtLabelAction(qtObject(parent));
    auto ve = new QtViewElement(label, la);
    ve->action = la;
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
    if (!action->iconName().empty())
    {
        a->setIcon(QIcon::fromTheme(QString::fromStdString(action->iconName())));
    }
    auto ve = new QtViewElement(action, a);
    ve->action = a;
    add(ve);
    ve->markForDeletion(a);
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
    if (!button->iconName().empty())
    {
        a->setIcon(QIcon::fromTheme(QString::fromStdString(button->iconName())));
    }
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

    auto s = new QtSliderAction(pw);
    auto ve = new QtViewElement(slider, s);
    ve->action = s;
    ve->markForDeletion(s);
    add(ve);
    connect(s, &QtSliderAction::sliderMoved, [slider](int value){
        if (slider->integral() && slider->scale()==ui::Slider::Linear)
        {
            slider->setValue(value);
        }
        else
        {
            auto r = slider->linMax() - slider->linMin();
            auto a = (double)value/SliderIntMax;
            double val = slider->linMin()+r*a;
            slider->setLinValue(val);
        }
        slider->setMoving(true);
        slider->trigger();
    });
    connect(s, &QtSliderAction::sliderReleased, [slider](){
        slider->setMoving(false);
        slider->trigger();
    });

#if 0
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
        if (slider->integral() && slider->scale()==ui::Slider::Linear)
        {
            slider->setValue(value);
        }
        else
        {
            auto r = slider->linMax() - slider->linMin();
            auto a = (double)value/SliderIntMax;
            double val = slider->linMin()+r*a;
            slider->setLinValue(val);
        }
        slider->setMoving(true);
        slider->trigger();
    });
    connect(s, &QSlider::sliderReleased, [slider](){
        slider->setMoving(false);
        slider->trigger();
    });
#endif

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
    if (auto w = qtWidget(elem))
        w->setEnabled(elem->enabled());
    auto ve = qtViewElement(elem);
    if (ve && ve->action)
        ve->action->setEnabled(elem->enabled());
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
    t.replace('&', "&&");
    if (auto sa = dynamic_cast<QtSliderAction *>(o))
    {
        auto s = dynamic_cast<const Slider *>(elem);
        sa->setText(sliderText(s));
        sa->setWidthText(sliderWidthText(s));
    }
    else if (auto la = dynamic_cast<QtLabelAction *>(o))
    {
        la->setText(t);
    }
    else if (auto l = dynamic_cast<QLabel *>(o))
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

void QtView::updateParent(const Element *elem)
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

void QtView::updateIntegral(const Slider *slider)
{
    updateBounds(slider);
    updateValue(slider);
}

void QtView::updateScale(const Slider *slider)
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
    auto s = dynamic_cast<QtSliderAction *>(o);
    if (s)
    {
        if (slider->integral() && slider->scale()==Slider::Linear)
        {
            if (s->value() != slider->value())
                s->setValue(slider->value());
        }
        else
        {
            auto min = slider->linMin();
            auto r = slider->linMax() - min;
            auto v = int((slider->linValue()-min)/r*SliderIntMax);
            if (v != s->value())
            {
                //std::cerr << "update: v=" << v << ", slider=" << s->value() << std::endl;
                s->setValue(v);
            }
        }
        s->setText(sliderText(slider));
    }
    if (ve->label)
        ve->label->setText(sliderText(slider));
}

void QtView::updateBounds(const Slider *slider)
{
    auto o = qtObject(slider);
    auto s = dynamic_cast<QtSliderAction *>(o);
    if (s)
    {
        s->setToolTip(QString("%1 - %2").arg(slider->min()).arg(slider->max()));
        s->setWidthText(sliderWidthText(slider));
        if (slider->integral() && slider->scale()==Slider::Linear)
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

QtLabelAction::QtLabelAction(QObject *parent)
: QWidgetAction(parent)
{
}

void QtLabelAction::setText(const QString &text)
{
    for (auto w: createdWidgets())
    {
        auto l = dynamic_cast<QLabel *>(w);
        assert(l);
        l->setText(text);
    }
}

QWidget *QtLabelAction::createWidget(QWidget *parent)
{
    return new QLabel(parent);
}

QtSliderWidget::QtSliderWidget(QBoxLayout::Direction dir, QWidget *parent)
: QWidget (parent)
, m_layout(new QBoxLayout(dir))
, m_label(new QLabel(this))
, m_slider(new QSlider(Qt::Horizontal, this))
{
    m_slider->setFocusPolicy(Qt::NoFocus);

    m_layout->setSizeConstraint(QLayout::SetMinAndMaxSize);
    m_layout->setSpacing(0);
    m_layout->setMargin(1);
    //m_layout->setContentsMargins(1, 1, 1, 1);
    m_label->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    m_slider->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
    m_slider->setMaximumWidth(400);
    m_layout->addWidget(m_label);
    m_layout->addWidget(m_slider);
    setLayout(m_layout);
    //connect(m_slider, &QSlider::sliderMoved, [this](int value){ emit sliderMoved(value); });
    connect(m_slider, &QSlider::actionTriggered, [this](int action){
        switch (action) {
        case QSlider::SliderSingleStepAdd:
        case QSlider::SliderSingleStepSub:
        case QSlider::SliderPageStepAdd:
        case QSlider::SliderPageStepSub:
        case QSlider::SliderToMinimum:
        case QSlider::SliderToMaximum:
        case QSlider::SliderMove:
            emit sliderMoved(m_slider->sliderPosition());
            break;
        }
    });

    connect(m_slider, &QSlider::sliderReleased, [this](){ emit sliderReleased(); });
}

void QtSliderWidget::setDirection(QBoxLayout::Direction dir)
{
    m_layout->setDirection(dir);
}

void QtSliderWidget::setText(const QString &text)
{
    m_label->setText(text);
}

void QtSliderWidget::setWidthText(const QString &text)
{
    QFontMetrics fm(m_label->font());
    int w = fm.width(text);
    m_label->setMinimumWidth(w);
    //std::cerr << "Slider label width " << w << " for " << text.toStdString() << std::endl;
}

void QtSliderWidget::setRange(int min, int max)
{
    m_slider->setRange(min, max);
}

void QtSliderWidget::setValue(int value)
{
    m_slider->setValue(value);
}

int QtSliderWidget::value() const
{
    return m_slider->value();
}

int QtSliderWidget::minimum() const
{
    return m_slider->minimum();
}

int QtSliderWidget::maximum() const
{
    return m_slider->maximum();
}

QtSliderAction::QtSliderAction(QObject *parent)
: QWidgetAction(parent)
{
}

void QtSliderAction::setToolTip(const QString &tip)
{
    m_tip = tip;
    QWidgetAction::setToolTip(tip);
    for (auto w: createdWidgets())
    {
        auto s = dynamic_cast<QtSliderWidget *>(w);
        assert(s);
        s->setToolTip(tip);
    }
}

void QtSliderAction::setText(const QString &text)
{
    m_text = text;
    for (auto w: createdWidgets())
    {
        auto s = dynamic_cast<QtSliderWidget *>(w);
        assert(s);
        s->setText(text);
    }
}

void QtSliderAction::setWidthText(const QString &text)
{
    m_widthText = text;
    for (auto w: createdWidgets())
    {
        auto s = dynamic_cast<QtSliderWidget *>(w);
        assert(s);
        s->setWidthText(text);
    }
}

void QtSliderAction::setRange(int min, int max)
{
    m_min = min;
    m_max = max;
    for (auto w: createdWidgets())
    {
        auto s = dynamic_cast<QtSliderWidget *>(w);
        assert(s);
        s->setRange(min, max);
    }
}

void QtSliderAction::setValue(long value)
{
    m_value = value;
    for (auto w: createdWidgets())
    {
        auto s = dynamic_cast<QtSliderWidget *>(w);
        assert(s);
        s->setValue(value);
    }
}

int QtSliderAction::value() const
{
    return m_value;
}

int QtSliderAction::minimum() const
{
    return m_min;
}

int QtSliderAction::maximum() const
{
    return m_max;
}

QWidget *QtSliderAction::createWidget(QWidget *parent)
{
    QBoxLayout::Direction dir = QBoxLayout::TopToBottom;
    auto toolbar = dynamic_cast<QToolBar *>(parent);
    auto s =  new QtSliderWidget(dir, parent);
    if (toolbar)
    {
        auto changeOrient = [this, s](Qt::Orientation ori){
            if (ori == Qt::Horizontal)
                s->setDirection(QBoxLayout::LeftToRight);
            else
                s->setDirection(QBoxLayout::BottomToTop);
        };
        connect(toolbar, &QToolBar::orientationChanged, changeOrient);
        changeOrient(toolbar->orientation());
    }
    s->setToolTip(m_tip);
    s->setRange(m_min, m_max);
    s->setValue(m_value);
    s->setText(m_text);
    s->setWidthText(m_widthText);
    connect(s, &QtSliderWidget::sliderMoved, [this, s](int value){
        m_value = value;
        for (auto w: createdWidgets())
        {
            auto sw = dynamic_cast<QtSliderWidget *>(w);
            assert(sw);
            if (sw != s)
                sw->setValue(value);
        }
        emit sliderMoved(value);
    });
    connect(s, &QtSliderWidget::sliderReleased, [this](){ emit sliderReleased(); });
    return s;
}

}
}

#include "moc_QtView.cpp"
