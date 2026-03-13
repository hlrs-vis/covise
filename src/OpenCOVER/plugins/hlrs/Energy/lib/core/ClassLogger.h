#pragma once
#include "interfaces/ILogger.h"
#include <string_view>
#include <string>

namespace core
{
class ClassLogger
{
public:
    ClassLogger(interface::ILogger &logger, std::string_view className)
        : m_logger(logger)
        , m_className(className)
    {
    }

    void info(std::string_view msg)
    {
        m_logger.info(prefixMsg(msg));
    }
    void error(std::string_view msg)
    {
        m_logger.error(prefixMsg(msg));
    }
    void warn(std::string_view msg)
    {
        m_logger.warn(prefixMsg(msg));
    }

    interface::ILogger &getLogger() { return m_logger; }

private:
    std::string prefixMsg(std::string_view msg)
    {
        std::string prefix { "[" + m_className + "] " };
        return prefix += msg;
    }

    interface::ILogger &m_logger;
    std::string m_className;
};
}
