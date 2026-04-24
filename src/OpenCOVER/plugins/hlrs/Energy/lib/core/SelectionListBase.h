#pragma once

#include "interfaces/ui/ISelectionList.h"
#include "ComponentBase.h"

namespace core
{
class SelectionListBase : public ComponentBase, public interface::ui::ISelectionList
{
public:
    using ComponentBase::ComponentBase;
    virtual ~SelectionListBase() = default;
};
}
