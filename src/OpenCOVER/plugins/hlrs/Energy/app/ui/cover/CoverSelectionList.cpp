#include "CoverSelectionList.h"
#include "CoverMenu.h"
#include <stdexcept>

CoverSelectionList::CoverSelectionList(core::interface::ui::IComponent *parent, const std::string &name)
    : core::SelectionListBase(parent, name)
{
    auto menu = dynamic_cast<CoverMenu *>(parent);
    if (menu)
        m_selection = new opencover::ui::SelectionList(menu->getMenu(), name);
    else
        std::runtime_error("Parent invalid for SelectionList " + name);
}

int CoverSelectionList::selectedIndex() const
{
    return m_selection->selectedIndex();
}

std::string CoverSelectionList::selectedItem() const
{
    return m_selection->selectedItem();
}

void CoverSelectionList::setList(const std::vector<std::string> &names)
{
    m_selection->setList(names);
}

void CoverSelectionList::select(int i)
{
    m_selection->select(i);
}
