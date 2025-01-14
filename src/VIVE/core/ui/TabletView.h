#pragma once

#include "View.h"
#include "Menu.h"
#include "Manager.h"

#include "../vvTabletUI.h"

namespace vive {
namespace ui {

//! store the data for the representation of a UI Element within a TabletView
struct VIVE_UI_EXPORT TabletViewElement: public View::ViewElement, public vvTUIListener
{
   TabletViewElement(Element *elem);
   ~TabletViewElement();

   int m_parentLevel = 0;
   int m_containerLevel = 0;
   int m_id = -1;
   int m_row = 1, m_column = 0;
   vvTUIElement *m_elem = nullptr;
   vvTUITabFolder *m_tabs = nullptr;
   vvTUITab *m_tab = nullptr;

   void tabletEvent(vvTUIElement *elem) override;
   void tabletPressEvent(vvTUIElement *elem) override;
   void tabletReleaseEvent(vvTUIElement *elem) override;

   void newRow();
};

//! concrete implementation of View for showing user interface \ref Element "elements" on a tablet PC running tabletUI
class VIVE_UI_EXPORT TabletView: public View
{
 public:
   TabletView(const std::string &name, vvTabletUI *tui);
   TabletView(const std::string &name, vvTUITabFolder *root);
   ~TabletView();
   ViewType typeBit() const override;

 private:
   TabletViewElement *tuiElement(const std::string &path) const;
   TabletViewElement *tuiElement(const Element *elem) const;
   TabletViewElement *tuiParent(const Element *elem) const;
   int tuiParentId(const Element *elem) const;
   TabletViewElement *tuiContainer(const Element *elem) const;
   int tuiContainerId(const Element *elem) const;

   void add(TabletViewElement *ve, const Element *elem);

   void updateEnabled(const Element *elem) override;
   void updateVisible(const Element *elem) override;
   void updateText(const Element *elem) override;
   void updateState(const Button *) override;
   void updateChildren(const Group *elem) override;
   void updateChildren(const SelectionList *sl) override;
   void updateIntegral(const Slider *slider) override;
   void updateScale(const Slider *slider) override;
   void updateValue(const Slider *slider) override;
   void updateBounds(const Slider *slider) override;
   void updateValue(const TextField *input) override;
   void updateFilter(const FileBrowser* fb) override;
   void updateRelayout(const Group* co) override;

   TabletViewElement *elementFactoryImplementation(Menu *menu) override;
   TabletViewElement *elementFactoryImplementation(Group *group) override;
   TabletViewElement *elementFactoryImplementation(Label *label) override;
   TabletViewElement *elementFactoryImplementation(Action *action) override;
   TabletViewElement *elementFactoryImplementation(Button *button) override;
   TabletViewElement *elementFactoryImplementation(Slider *slider) override;
   TabletViewElement *elementFactoryImplementation(SelectionList *sl) override;
   TabletViewElement *elementFactoryImplementation(EditField *input) override;
   TabletViewElement *elementFactoryImplementation(FileBrowser *fb) override;

   TabletViewElement *m_root = nullptr;

   vvTabletUI *m_tui = nullptr;
   int m_lastParentLevel = -1;
};

}
}
#endif
