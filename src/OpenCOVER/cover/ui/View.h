#ifndef UI_VIEW_H
#define UI_VIEW_H

#include <string>
#include <map>

#include "Export.h"

namespace opencover {
namespace ui {

class Manager;
class Element;
class Label;
class Menu;
class Group;
class RadioGroup;
class Button;
class Action;
class Slider;

class COVER_UI_EXPORT View {
    friend class Manager;

 public:
    struct ViewElement
    {
        ViewElement() = delete;
        ViewElement(Element *elem): element(elem) {}
        virtual ~ViewElement() { /* just for polymorphism */ }

        View *view = nullptr;
        Element *element = nullptr;
    };

    View(const std::string &name);
    virtual ~View();

    const std::string &name() const;
    Manager *manager() const;

    ViewElement *elementFactory(Element *elem);

    virtual void updateEnabled(const Element *elem) = 0;
    virtual void updateVisible(const Element *elem) = 0;
    virtual void updateText(const Element *elem) = 0;
    virtual void updateState(const Button *button) = 0;
    virtual void updateChildren(const Menu *menu) = 0;
    virtual void updateInteger(const Slider *slider) = 0;
    virtual void updateValue(const Slider *slider) = 0;
    virtual void updateBounds(const Slider *slider) = 0;

    ViewElement *viewElement(const std::string &path) const;
    ViewElement *viewElement(const Element *elem) const;
    ViewElement *viewParent(const Element *elem) const;

 protected:
    virtual ViewElement *elementFactoryImplementation(Menu *Menu) = 0;
    virtual ViewElement *elementFactoryImplementation(Group *group) = 0;
    virtual ViewElement *elementFactoryImplementation(RadioGroup *rg) = 0;
    virtual ViewElement *elementFactoryImplementation(Label *label) = 0;
    virtual ViewElement *elementFactoryImplementation(Action *action) = 0;
    virtual ViewElement *elementFactoryImplementation(Button *button) = 0;
    virtual ViewElement *elementFactoryImplementation(Slider *slider) = 0;

 private:
    const std::string m_name;
    std::map<const Element *, ViewElement *> m_viewElements;
    std::map<std::string, ViewElement *> m_viewElementsByPath;
    Manager *m_manager = nullptr;
};

}
}
#endif
