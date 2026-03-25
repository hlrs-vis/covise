#pragma once
#include <utils/logging/log.h>
#include <lib/core/interfaces/ILogger.h>
#include <string_view>

class EnergyLogger : public core::interface::ILogger
{
    enum class Level
    {
        info,
        error,
        warn
    };

public:
    EnergyLogger(const std::string &name, int rank = 0);
    void info(std::string_view msg) override;
    void error(std::string_view msg) override;
    void warn(std::string_view msg) override;

private:
    void log(Level level, std::string_view msg);

    std::shared_ptr<spdlog::logger> m_logger;
    int m_rank;
};
