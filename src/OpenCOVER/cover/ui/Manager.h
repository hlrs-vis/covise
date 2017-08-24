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

class COVER_UI_EXPORT Manager: public Owner {
 public:
   Manager();

   bool addView(View *view);
   bool removeView(const View *view);
   bool removeView(const std::string &name);

   void updateText(const Element *elem) const;
   void updateEnabled(const Element *elem) const;
   void updateVisible(const Element *elem) const;
   void updateState(const Button *button) const;
   void updateChildren(const Menu *menu) const;
   void updateInteger(const Slider *slider) const;
   void updateValue(const Slider *slider) const;
   void updateBounds(const Slider *slider) const;

   void add(Element *elem);
   void setChanged();
   bool update();
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
