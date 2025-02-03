#include "VectorEditField.h"
#include "Manager.h"
namespace vive {
namespace ui {

VectorEditField::VectorEditField(Group *parent, const std::string &name)
{
    m_group = new ui::Group(parent, name);
    initEditFields(name);
    
}

VectorEditField::VectorEditField(const std::string &name, Owner *owner)
{
    m_group = new ui::Group(name, owner);
    initEditFields(name);
}

void VectorEditField::initEditFields(const std::string &name)
{
    constexpr std::array<const char*, 3> vecNames{"x", "y", "z"};
    for (size_t i = 0; i < m_edits.size(); i++)
    {
        m_edits[i] = new ui::EditField(m_group, name + std::to_string(i));
        m_edits[i]->setText(vecNames[i]);
    }
}

void VectorEditField::setValue(const VectorEditField::ValueType &val)
{
    for (size_t i = 0; i < m_edits.size(); i++)
        m_edits[i]->setValue(val[i]);
}

VectorEditField::ValueType VectorEditField::value() const
{
    ValueType v;
    for (size_t i = 0; i < m_edits.size(); i++)
        v [i] = (float)(m_edits[i]->number());
    return v;
}

void VectorEditField::setCallback(const std::function<void(const VectorEditField::ValueType &vec)> &f)
{
    m_callback = f;
    for (auto edit : m_edits)   
    {
        edit->setCallback([this](const std::string &){
            m_callback(value());
        });
    }
}


#define COMMA ,

#define ELEMENT_INTERFACE_IMPL(returnvalue, functionName, arguments, argumentNames, modifier)\
returnvalue VectorEditField::functionName(arguments) modifier\
{\
    for(auto edit : m_edits)\
        edit->functionName(argumentNames); \
    return m_group->functionName(argumentNames); \
}

ELEMENT_INTERFACE_IMPL(void, setPriority, Element::Priority prio, prio,)
ELEMENT_INTERFACE_IMPL(Element::Priority, priority, , , const)
ELEMENT_INTERFACE_IMPL(void,  setShared, bool state, state,)
ELEMENT_INTERFACE_IMPL(bool,  isShared, , , const);
ELEMENT_INTERFACE_IMPL(void,  setIcon, const std::string &iconName, iconName,)
ELEMENT_INTERFACE_IMPL(const std::string &, iconName, , , const)
ELEMENT_INTERFACE_IMPL(Group *, parent, , , const)
ELEMENT_INTERFACE_IMPL(void,  update, Element::UpdateMaskType updateMask, updateMask, const)
ELEMENT_INTERFACE_IMPL(const std::string &, text, , , const);
ELEMENT_INTERFACE_IMPL(bool, visible, const View *view, view, const)
ELEMENT_INTERFACE_IMPL(void, setVisible, bool flag COMMA int viewBits, flag COMMA viewBits,)
ELEMENT_INTERFACE_IMPL(bool, enabled, , , const)
ELEMENT_INTERFACE_IMPL(void, setEnabled, bool flag, flag,)
ELEMENT_INTERFACE_IMPL(void, trigger, , , const)

void VectorEditField::setText(const std::string &text)
{
    m_group->setText(text);
}


}
}