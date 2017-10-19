#ifndef UI_VRUI_VIEW_H
#define UI_VRUI_VIEW_H

#include "View.h"
#include "Menu.h"

#include <OpenVRUI/coMenuItem.h>

namespace vrui {

class coMenuItem;
class coMenu;
class coCheckboxGroup;

}

namespace opencover {
namespace ui {

struct VruiViewElement: public View::ViewElement, public vrui::coMenuListener
{
   VruiViewElement(Element *elem);
   ~VruiViewElement();

   std::string m_text;
   vrui::coMenuItem *m_menuItem = nullptr;
   vrui::coMenu *m_menu = nullptr;
   vrui::coCheckboxGroup *m_group = nullptr;

   void menuEvent(vrui::coMenuItem *menuItem) override;
   void menuReleaseEvent(vrui::coMenuItem *menuItem) override;
};

class VruiView: public View
{
 public:
   VruiView();
   COVER_UI_EXPORT vrui::coMenu *getMenu(const Element *elem) const;
   COVER_UI_EXPORT vrui::coMenuItem *getItem(const Element *elem) const;

 private:
   VruiViewElement *vruiElement(const std::string &path) const;
   VruiViewElement *vruiElement(const Element *elem) const;
   VruiViewElement *vruiParent(const Element *elem) const;
   VruiViewElement *vruiContainer(const Element *elem) const;

   void add(VruiViewElement *ve, const Element *elem);

   void updateEnabled(const Element *elem) override;
   void updateVisible(const Element *elem) override;
   void updateText(const Element *elem) override;
   void updateParent(const Element *elem) override;
   void updateState(const Button *) override;
   void updateChildren(const SelectionList *sl) override;
   void updateInteger(const Slider *slider) override;
   void updateValue(const Slider *slider) override;
   void updateBounds(const Slider *slider) override;

   VruiViewElement *elementFactoryImplementation(Menu *menu) override;
   VruiViewElement *elementFactoryImplementation(Group *group) override;
   VruiViewElement *elementFactoryImplementation(Label *label) override;
   VruiViewElement *elementFactoryImplementation(Action *action) override;
   VruiViewElement *elementFactoryImplementation(Button *button) override;
   VruiViewElement *elementFactoryImplementation(Slider *slider) override;
   VruiViewElement *elementFactoryImplementation(SelectionList *sl) override;

   vrui::coMenu *m_rootMenu = nullptr;
};

}
}
#endif
