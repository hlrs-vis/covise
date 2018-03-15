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
class Button;
class Action;
class Slider;
class SelectionList;
class EditField;

//! abstract base class for all views onto the user interface elements handled by a Manager
class COVER_UI_EXPORT View {
    friend class Manager;

 public:
    enum ViewType
    {
        Other = 1,
        WindowMenu = 2,
        VR = 4,
        Tablet = 8,
        Phone = 16,
    };
    //! base class for objects where a view stores all the data for the graphical representation of an Element
    struct ViewElement
    {
        ViewElement() = delete;
        ViewElement(Element *elem): element(elem) {}
        virtual ~ViewElement() { /* just for polymorphism */ }

        View *view = nullptr;
        Element *element = nullptr;
    };

    //! create view with a name
    View(const std::string &name);
    //! destroy view and all data from its ViewElements
    virtual ~View();

    //! return a bit identifying this View type
    virtual ViewType typeBit() const = 0;
    //! return name of this view
    const std::string &name() const;
    //! return manager this view is attached to
    Manager *manager() const;

    //! update View, return true if rendering is required
    virtual bool update();

    //! establish graphical representation of a user interface element
    ViewElement *elementFactory(Element *elem);

    //! reflect changed enabled state in graphical representation
    virtual void updateEnabled(const Element *elem) = 0;
    //! reflect changed visibility in graphical representation
    virtual void updateVisible(const Element *elem) = 0;
    //! reflect changed text in graphical representation
    virtual void updateText(const Element *elem) = 0;
    //! reflect change of parent item in graphical representation
    virtual void updateParent(const Element *elem) = 0;
    //! reflect changed button state in graphical representation
    virtual void updateState(const Button *button) = 0;
    //! reflect change of child items in graphical representation
    virtual void updateChildren(const SelectionList *sl) = 0;
    //! reflect change of slider type in graphical representation
    virtual void updateIntegral(const Slider *slider) = 0;
    //! reflect change of slider scale type in graphical representation
    virtual void updateScale(const Slider *slider) = 0;
    //! reflect change of slider value in graphical representation
    virtual void updateValue(const Slider *slider) = 0;
    //! reflect change of slider range in graphical representation
    virtual void updateBounds(const Slider *slider) = 0;
    //! reflect change of input field value in graphical representation
    virtual void updateValue(const EditField *input) = 0;

    //! remove elem from View and delete associated data
    bool removeElement(Element *elem);

    //! find corresponding ViewElement by element path
    ViewElement *viewElement(const std::string &path) const;
    //! find corresponding ViewElement by pointer to abstract element
    ViewElement *viewElement(const Element *elem) const;
    //! get parent of element
    ViewElement *viewParent(const Element *elem) const;

 protected:
    //! implement to create graphical representation of a menu
    virtual ViewElement *elementFactoryImplementation(Menu *Menu) = 0;
    //! implement to create graphical representation of an item group (e.g. a frame, possible with a label)
    virtual ViewElement *elementFactoryImplementation(Group *group) = 0;
    //! implement to create graphical representation of a text label
    virtual ViewElement *elementFactoryImplementation(Label *label) = 0;
    //! implement to create graphical representation of a stateless button
    virtual ViewElement *elementFactoryImplementation(Action *action) = 0;
    //! implement to create graphical representation of a button with binary state
    virtual ViewElement *elementFactoryImplementation(Button *button) = 0;
    //! implement to create graphical representation of a slider
    virtual ViewElement *elementFactoryImplementation(Slider *slider) = 0;
    //! implement to create graphical representation of a selection list
    virtual ViewElement *elementFactoryImplementation(SelectionList *sl) = 0;
    //! implement to create graphical representation of an input field
    virtual ViewElement *elementFactoryImplementation(EditField *input) = 0;

 private:
    const std::string m_name;
    std::map<const Element *, ViewElement *> m_viewElements;
    Manager *m_manager = nullptr;
};

}
}
#endif
