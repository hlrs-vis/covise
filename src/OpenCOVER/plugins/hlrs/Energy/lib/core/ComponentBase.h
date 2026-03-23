#pragma once

#include "interfaces/ui/IComponent.h"

namespace core
{
class ComponentBase : virtual public interface::ui::IComponent
{
protected:
    std::string m_name;
    interface::ui::IComponent *m_parent;

public:
    ComponentBase(interface::ui::IComponent *parent, const std::string &name)
        : m_parent(parent)
        , m_name(std::move(name))
    {
    }
    virtual ~ComponentBase() = default;
    const std::string &getName() override { return m_name; }
    auto getParent() { return m_parent; }
};
}
