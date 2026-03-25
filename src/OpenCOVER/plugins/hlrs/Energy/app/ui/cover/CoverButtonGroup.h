#pragma once

#include <lib/core/interfaces/ui/IButton.h>
#include <lib/core/ButtonGroupBase.h>
#include <cover/ui/ButtonGroup.h>
#include <memory>
#include <string>
#include <vector>

class CoverButtonGroup final : public core::ButtonGroupBase
{
public:
    CoverButtonGroup(core::interface::ui::IComponent *parent, const std::string &name, std::vector<std::unique_ptr<core::interface::ui::IButton>> &&buttons);
    void setCallback(const std::function<void(int)> &func) override;
    void add(std::unique_ptr<core::interface::ui::IButton> button) override;
    core::interface::ui::IButton* getChild(int position) override;

private:
    opencover::ui::ButtonGroup *m_buttonGroup;
    std::vector<std::unique_ptr<core::interface::ui::IButton>> m_childs;
};
