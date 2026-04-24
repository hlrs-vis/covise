#pragma once

#include "interfaces/ui/IEditField.h"
#include "ComponentBase.h"

namespace core
{
class EditFieldBase : public ComponentBase, public interface::ui::IEditDoubleField
{
public:
    using ComponentBase::ComponentBase;
    virtual ~EditFieldBase() = default;
};
}
