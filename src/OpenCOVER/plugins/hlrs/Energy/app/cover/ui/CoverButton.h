#pragma once
#include <lib/core/ButtonBase.h>
#include <cover/ui/Button.h>
#include <string>

class CoverButton final : public core::ButtonBase
{
public:
    CoverButton(core::interface::ui::IComponent *parent, const std::string &name);
    void setCallback(const std::function<void(bool)> &func) override;
    void setState(bool state) override;
    void setText(const std::string &txt) override;
    bool state() const override { return m_button->state(); }
    auto getButton() const
    {
        return m_button;
    }

private:
    opencover::ui::Button *m_button;
};
