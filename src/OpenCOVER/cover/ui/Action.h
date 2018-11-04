#ifndef UI_ACTION_H
#define UI_ACTION_H

#include "Element.h"
#include "ShortcutListener.h"

#include <functional>

namespace opencover {
namespace ui {

//! a graphical item showing a text label that can be pressed to trigger an action

/** \note QPushButton
    \note vrui::coButtonMenuItem
    \note coTUIButton
    */
class COVER_UI_EXPORT Action: public Element {
 public:
    Action(const std::string &name, Owner *owner);
    Action(Group *parent, const std::string &name);
    ~Action();

    void setCallback(const std::function<void()> &f);
    std::function<void()> callback() const;

    void triggerImplementation() const override;
    void shortcutTriggered() override;

private:
    std::function<void()> m_callback;
};

}
}
#endif
