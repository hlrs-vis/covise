#ifndef UI_RADIOGROUP_H
#define UI_RADIOGROUP_H

#include "Group.h"
#include <string>
#include <functional>

namespace opencover {
namespace ui {

class Button;

class COVER_UI_EXPORT RadioGroup: public Group {

 public:
    RadioGroup(const std::string &name, Owner *owner);
    RadioGroup(Group *parent, const std::string &name);

    int value() const;
    Button *activeButton() const;

    virtual bool add(Element *elem) override;
    virtual bool remove(Element *elem) override;

    void setCallback(const std::function<void(int)> &f);
    std::function<void(int)> callback() const;

    void toggle(const Button *b);
    void triggerImplementation() const override;

private:
    std::function<void(int)> m_callback;
};

}
}
#endif
