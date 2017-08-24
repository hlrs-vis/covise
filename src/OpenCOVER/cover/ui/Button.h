#ifndef UI_BUTTON_H
#define UI_BUTTON_H

#include "Element.h"
#include "ShortcutListener.h"

#include <functional>

namespace opencover {
namespace ui {

class RadioGroup;

class COVER_UI_EXPORT Button: public Element {
 public:
   Button(const std::string &name, Owner *owner);
   Button(Group *parent, const std::string &name);
   Button(RadioGroup *parent, const std::string &name, int id=0);

   int id() const;
   RadioGroup *radioGroup() const;
   void setRadioGroup(RadioGroup *rg, int id=0);

   bool state() const;
   void setState(bool flag);

    void setCallback(const std::function<void(bool)> &f);
    std::function<void(bool)> callback() const;

    void triggerImplementation() const override;
    void radioTrigger() const;
    void shortcutTriggered() override;

    void update() const override;

 private:
    RadioGroup *m_radioGroup = nullptr;
    int m_id = 0;
    bool m_state = false;
    std::function<void(bool)> m_callback;
};

}
}
#endif
