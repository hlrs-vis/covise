#ifndef UI_BUTTON_H
#define UI_BUTTON_H

#include "Element.h"
#include "ShortcutListener.h"

#include <functional>

namespace opencover {
namespace ui {

class ButtonGroup;

//! a graphical item showing a text label that can be pressed to toggle between two states

/** \note QToggleButton
    \note vrui::coCheckboxMenuItem
    \note coTUIToggleButton */
class COVER_UI_EXPORT Button: public Element {
   friend class ButtonGroup;
 public:
   enum UpdateMask: UpdateMaskType
   {
       UpdateState = 0x100
   };
   Button(const std::string &name, Owner *owner, ButtonGroup *bg=nullptr, int id=0);
   Button(Group *parent, const std::string &name, ButtonGroup *bg=nullptr, int id=0);
   ~Button();
   //Button(ButtonGroup *parent, const std::string &name, int id=0);

   int buttonId() const;
   ButtonGroup *group() const;
   void setGroup(ButtonGroup *rg, int id=0);

   bool state() const;
   void setState(bool flag, bool updateGroup=true);

    void setCallback(const std::function<void(bool)> &f);
    std::function<void(bool)> callback() const;

    void triggerImplementation() const override;
    void radioTrigger() const;
    void shortcutTriggered() override;

    void update(UpdateMaskType mask) const override;
    void save(covise::TokenBuffer &buf) const override;
    void load(covise::TokenBuffer &buf) override;

 private:
    ButtonGroup *m_radioGroup = nullptr;
    int m_buttonId = 0;
    bool m_state = false;
    std::function<void(bool)> m_callback;
};

}
}
#endif
