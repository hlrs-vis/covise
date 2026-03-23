#pragma once
#include <string>

namespace core::interface::ui
{
class IComponent
{
public:
    virtual ~IComponent() = default;
    virtual const std::string &getName() = 0;
};
}
