#ifndef UI_MANAGER_H
#define UI_MANAGER_H

#include <set>
#include <map>
#include <deque>
#include <string>

#include "View.h"
#include "Owner.h"

namespace opencover {

namespace ui {

class View;

//! responsible for handling all \ref Element "elements" of a user interface
class COVER_UI_EXPORT Manager: public Owner {
 public:
   Manager();

   //! attach a view (creating a graphical representation) of the user interface
   bool addView(View *view);
   //! remove a view by pointer
   bool removeView(const View *view);
   //! remove a view by name
   bool removeView(const std::string &name);

   //! render updated text of elem on all attached views
   void updateText(const Element *elem) const;
   //! update enabled state of elem on all attached views
   void updateEnabled(const Element *elem) const;
   //! update visibility of elem on all attached views
   void updateVisible(const Element *elem) const;
   //! update pressed/released state of button on all views
   void updateState(const Button *button) const;
   //! update list of child elements on all views
   void updateChildren(const Menu *menu) const;
   //! update whether slider is integral on all views
   void updateInteger(const Slider *slider) const;
   //! update slidre value on all attached views
   void updateValue(const Slider *slider) const;
   //! update allowed value range for slider on all attached views
   void updateBounds(const Slider *slider) const;

   //! add elem to list of managed user interface items
   void add(Element *elem);
   //! some managed element was changed
   void setChanged();
   //! trigger internal book-keeping and updates
   /** return true if any change occurred */
   bool update();
   //! trigger short-cuts configured for user interface elements
   bool keyEvent(int type, int keySym, int mod) const;

 private:
   bool m_changed = false;
   std::set<Element *> m_elements;
   std::deque<Element *> m_newElements;
   std::map<const std::string, View *> m_views;
};

}
}
#endif
