#pragma once

#include "interfaces/ui/IEditField.h"
#include "ComponentBase.h"

namespace core
{
class EditFieldBase : public ComponentBase, public interface::ui::IEditDoubleField
{
public:
    EditFieldBase(interface::ui::IComponent *parent, const std::string &name)
        : ComponentBase(parent, name)
    {
    }
    virtual ~EditFieldBase() = default;
};
}
