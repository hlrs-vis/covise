#pragma once
#include <string_view>

namespace core::interface
{
class ILogger
{
public:
    virtual void info(std::string_view msg) = 0;
    virtual void error(std::string_view msg) = 0;
    virtual void warn(std::string_view msg) = 0;
    virtual ~ILogger() = default;
};
}
