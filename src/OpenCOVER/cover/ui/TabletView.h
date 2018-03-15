#ifndef UI_TABLET_VIEW_H
#define UI_TABLET_VIEW_H

#include "View.h"
#include "Menu.h"

#include <cover/coTabletUI.h>

namespace opencover {
namespace ui {

//! store the data for the representation of a UI Element within a TabletView
struct COVER_UI_EXPORT TabletViewElement: public View::ViewElement, public coTUIListener
{
   TabletViewElement(Element *elem);
   ~TabletViewElement();

   int m_parentLevel = 0;
   int m_containerLevel = 0;
   int m_id = -1;
   int m_row = 1, m_column = 0;
   coTUIElement *m_elem = nullptr;
   coTUITabFolder *m_tabs = nullptr;
   coTUITab *m_tab = nullptr;

   void tabletEvent(coTUIElement *elem) override;
   void tabletPressEvent(coTUIElement *elem) override;
   void tabletReleaseEvent(coTUIElement *elem) override;

   void newRow();
};

//! concrete implementation of View for showing user interface \ref Element "elements" on a tablet PC running tabletUI
class COVER_UI_EXPORT TabletView: public View
{
 public:
   TabletView(const std::string &name, coTabletUI *tui);
   TabletView(coTUITabFolder *root);
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
   void updateParent(const Element *elem) override;
   void updateState(const Button *) override;
   void updateChildren(const SelectionList *sl) override;
   void updateIntegral(const Slider *slider) override;
   void updateScale(const Slider *slider) override;
   void updateValue(const Slider *slider) override;
   void updateBounds(const Slider *slider) override;
   void updateValue(const EditField *input) override;

   TabletViewElement *elementFactoryImplementation(Menu *menu) override;
   TabletViewElement *elementFactoryImplementation(Group *group) override;
   TabletViewElement *elementFactoryImplementation(Label *label) override;
   TabletViewElement *elementFactoryImplementation(Action *action) override;
   TabletViewElement *elementFactoryImplementation(Button *button) override;
   TabletViewElement *elementFactoryImplementation(Slider *slider) override;
   TabletViewElement *elementFactoryImplementation(SelectionList *sl) override;
   TabletViewElement *elementFactoryImplementation(EditField *input) override;

   TabletViewElement *m_root = nullptr;

   coTabletUI *m_tui = nullptr;
   int m_lastParentLevel = -1;
};

}
}
#endif
