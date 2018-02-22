#ifndef UI_MANAGER_H
#define UI_MANAGER_H

#include <set>
#include <map>
#include <deque>
#include <string>
#include <memory>

#include "View.h"
#include "Owner.h"
#include "Element.h"

namespace covise {
class TokenBuffer;
}

namespace vrui {
class coMouseButtonInteraction;
class coButtonInteraction;
}

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
   //! update parent of element on all views
   void updateParent(const Element *elem) const;
   //! update pressed/released state of button on all views
   void updateState(const Button *button) const;
   //! update list of child elements on all views
   void updateChildren(const SelectionList *sl) const;
   //! update whether slider is logarithmic on all views
   void updateScale(const Slider *slider) const;
   //! update whether slider is integral on all views
   void updateIntegral(const Slider *slider) const;
   //! update slider value on all attached views
   void updateValue(const Slider *slider) const;
   //! update allowed value range for slider on all attached views
   void updateBounds(const Slider *slider) const;
   //! update input value on all attached views
   void updateValue(const EditField *input) const;

   //! add elem to list of managed user interface items
   void add(Element *elem);
   //! remove elem from list of managed user interface items
   void remove(Element *elem);
   //! remove an owner from manager
   void remove(Owner *owner);
   //! search for an Element by unique id, returs nullptr if not found
   Element *getById(int id) const;
   //! search for an Element by path, returs nullptr if not found
   Element *getByPath(const std::string &path) const;
   //! some managed element was changed
   void setChanged();
   //! trigger internal book-keeping and updates
   /** return true if any change occurred */
   bool update();
   //! trigger keyboard short-cuts configured for user interface elements
   bool keyEvent(int type, int mod, int keySym);
   //! trigger short-cuts from input button presses
   bool buttonEvent(int buttons) const;

   //! mark an element for syncing its state to slaves, optionally triggering its action
   void queueUpdate(const Element *elem, Element::UpdateMaskType mask, bool trigger=false);
   //! sync state and events from master to cluster slaves
   bool sync();
   //! get all elements
   std::vector<const Element *> getAllElements() const;

 private:
   bool m_updateAllElements = false;
   bool m_changed = false;
   int m_numCreated = 0, m_elemOrder = 0;
   std::map<int, Element *> m_elements;
   std::map<int, Element *> m_elementsById;
   std::map<const std::string, Element *> m_elementsByPath;
   std::deque<Element *> m_newElements;
   std::map<const std::string, View *> m_views;

   int m_numUpdates = 0;
   std::map<int, std::pair<Element::UpdateMaskType, std::shared_ptr<covise::TokenBuffer>>> m_elemState;
   std::shared_ptr<covise::TokenBuffer> m_updates;
   void flushUpdates();
   void processUpdates(std::shared_ptr<covise::TokenBuffer> updates, int numUpdates, bool runTriggers);

   int m_modifiers = 0;
   std::vector<vrui::coMouseButtonInteraction *> m_wheelInteraction;
   std::vector<vrui::coButtonInteraction *> m_buttonInteraction;
};

}
}
#endif
