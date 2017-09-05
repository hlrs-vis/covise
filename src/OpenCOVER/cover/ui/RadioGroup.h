#ifndef UI_RADIOGROUP_H
#define UI_RADIOGROUP_H

#include "Group.h"
#include <string>
#include <functional>

namespace opencover {
namespace ui {

class Button;

//! Group of Buttons where just a single one can be in pressed state

/** \note QActionGroup */
class COVER_UI_EXPORT RadioGroup: public Group {

 public:
    RadioGroup(const std::string &name, Owner *owner);
    RadioGroup(Group *parent, const std::string &name);

    //! value assigned to active (=pressed) Button
    int value() const;
    //! pointer to active Button
    Button *activeButton() const;

    //! add Button to this RadioGroup, toggling its state to fulfill constraint that exactly one button be active
    virtual bool add(Element *elem) override;
    //! remove Button from this RadioGroup, toggling other Button's state to fulfill constraint that exactly one button be active
    virtual bool remove(Element *elem) override;

    void setCallback(const std::function<void(int)> &f);
    std::function<void(int)> callback() const;

    //! if required, toggle state of @param b so that exactly one button is active
    void toggle(const Button *b);

    void triggerImplementation() const override;

private:
    std::function<void(int)> m_callback;
};

}
}
#endif
