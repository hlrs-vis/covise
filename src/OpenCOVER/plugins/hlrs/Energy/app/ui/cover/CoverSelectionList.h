#pragma once
#include <cover/ui/SelectionList.h>
#include <lib/core/SelectionListBase.h>

class CoverSelectionList final : public core::SelectionListBase
{
public:
    CoverSelectionList(core::interface::ui::IComponent *parent, const std::string &name);
    void setCallback(const std::function<void(int)> &func) override { m_selection->setCallback(func); }
    int selectedIndex() const override;
    std::string selectedItem() const override;
    void setList(const std::vector<std::string> &names) override;
    void select(int i) override;

private:
    opencover::ui::SelectionList *m_selection;
};
