#include "PowerParser.h"
#include <lib/core/simulation/powerresult.h>

result_ptr PowerParser::operator()(CSVDataMap &map)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    for (auto &[name, stream] : map)
    {
        // TODO: put data into result
    }
    return result;
}

result_ptr PowerParser::operator()(const ArrowDataMap &map)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    for (auto &[name, table] : map)
    {
        // TODO: put data into result
    }
    return result;
}

result_ptr PowerParser::operator()(const ArrowData &data)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    // TODO: put data into result
    return result;
}

result_ptr PowerParser::operator()(CSVData &data)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    // TODO: put data into result
    return result;
}
