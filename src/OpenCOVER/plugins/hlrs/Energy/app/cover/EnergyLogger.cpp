#include "EnergyLogger.h"

EnergyLogger::EnergyLogger(const std::string &name, int rank)
    : m_rank(rank)
    , m_logger(opencover::utils::logging::create_logger(name))
{
}

void EnergyLogger::info(std::string_view msg)
{
    log(Level::info, msg);
}

void EnergyLogger::error(std::string_view msg)
{
    log(Level::error, msg);
}

void EnergyLogger::warn(std::string_view msg)
{
    log(Level::warn, msg);
}

void EnergyLogger::log(Level level, std::string_view msg) {
    // display only for rank 0
    if (m_rank != 0)
        return;
   switch(level) {
        case Level::error:
            m_logger->error(msg);
        case Level::warn:
            m_logger->warn(msg);
        case Level::info:
        default:
            m_logger->info(msg);
   }
}
