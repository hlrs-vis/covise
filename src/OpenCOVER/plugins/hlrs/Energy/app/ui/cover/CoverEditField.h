#pragma once
#include <cover/ui/EditField.h>
#include <lib/core/EditFieldBase.h>

class CoverEditField : public core::EditFieldBase
{
public:
    CoverEditField(core::interface::ui::IComponent *parent, const std::string &name);
    void setCallback(const std::function<void(std::string)> &func) override { m_field->setCallback(func); }
    void setValue(double val) override { m_field->setValue(val); }
    double getValue() override { return m_field->number(); }

private:
    opencover::ui::EditField *m_field;
};
