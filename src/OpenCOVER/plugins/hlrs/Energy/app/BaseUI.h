#pragma once
#include <string>
#include <functional>
#include <lib/core/interfaces/ui/IComponent.h>
#include <lib/core/interfaces/ui/IButton.h>
#include <lib/core/interfaces/ui/IEditField.h>

typedef std::function<void(bool)> BtnCallback;
typedef std::function<void(const std::string &)> EditCallback;

class BaseUI
{
public:
    BaseUI(const std::string &name, core::interface::ui::IComponent *parent)
    {
    }
protected:
    void setBtnCallback(core::interface::ui::IButton* btn, const BtnCallback &func) { btn->setCallback(func); }
    void setTxtFieldCallback(core::interface::ui::IEditDoubleField *edit, const EditCallback &func) { edit->setCallback(func); }
};
