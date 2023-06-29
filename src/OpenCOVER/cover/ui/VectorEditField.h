#ifndef UI_VECTOR_EDIT_FIELD_H
#define UI_VECTOR_EDIT_FIELD_H

#include "Element.h"
#include "EditField.h"
#include "Group.h"

#include <osg/Vec3> 

#include <array>
namespace opencover {
namespace ui {


class COVER_UI_EXPORT VectorEditField {

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

    typedef osg::Vec3 ValueType;
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

#endif // UI_VECTOR_EDIT_FIELD_H