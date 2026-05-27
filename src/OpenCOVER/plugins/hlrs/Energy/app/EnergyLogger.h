#pragma once
#include <utils/logging/log.h>
#include <lib/core/Logger.h>

class EnergyLogger
{
public:
    explicit EnergyLogger(const std::string &name, int rank = 0)
        : spdlogger_(opencover::utils::logging::create_logger(name))
        , logger_(
            [&](std::string_view msg) { spdlogger_->info(msg);}, 
            [&](std::string_view msg) { spdlogger_->error(msg);}, 
            [&](std::string_view msg) { spdlogger_->warn(msg);},
            name)
    {
    }
    
    const Logger& getLogger() { return logger_; }

private:
    std::shared_ptr<spdlog::logger> spdlogger_;
    Logger logger_;
};
