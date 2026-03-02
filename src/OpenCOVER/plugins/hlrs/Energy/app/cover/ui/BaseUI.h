#pragma once
#include <cover/ui/Button.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Owner.h>
#include <cover/coTabletUI.h>
#include <PluginUtil/colors/coColorBar.h>
#include <functional>

typedef std::function<void(bool)> BtnCallback;
typedef std::function<void(const std::string &)> EditCallback;

class BaseUI : public opencover::ui::Owner,
               public opencover::coTUIListener
{
public:
    BaseUI(const std::string &name, opencover::ui::Owner *owner)
        : opencover::ui::Owner(name, owner)
    {
    }
protected:
    void setBtnCallback(opencover::ui::Button *btn, BtnCallback func) { btn->setCallback(func); }
    void setTxtFieldCallback(opencover::ui::EditField *edit, EditCallback func) { edit->setCallback(func); }
};
