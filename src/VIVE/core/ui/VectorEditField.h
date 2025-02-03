#pragma once


#include "Element.h"
#include "EditField.h"
#include "Group.h"

#include <vsg/maths/vec3.h> 

#include <array>
namespace vive {
namespace ui {


class VIVE_UI_EXPORT VectorEditField {

public:
    VectorEditField(Group *parent, const std::string &name);
    VectorEditField(const std::string &name, Owner *owner);
    ~VectorEditField();

    //Element interface
    void setPriority(Element::Priority prio);
    Element::Priority priority() const;
    void setShared(bool state);
    bool isShared() const;
    void setIcon(const std::string &iconName);
    const std::string &iconName() const;
    Group *parent() const;
    void update(Element::UpdateMaskType updateMask=Element::UpdateAll) const;
    const std::string &text() const;
    void setText(const std::string &text);
    bool visible(const View *view=nullptr) const;
    void setVisible(bool flag, int viewBits=~0);
    bool enabled() const;
    void setEnabled(bool flag);
    void trigger() const;

    typedef vsg::vec3 ValueType;
    void setValue(const ValueType &val);
    ValueType value() const;
    void setCallback(const std::function<void(const ValueType &vec)> &f);
private:
    ui::Group * m_group;
    std::array<ui::EditField*, 3> m_edits;
    std::function<void(const ValueType &vec)> m_callback;
    
    void initEditFields(const std::string &name);
};

}
}
