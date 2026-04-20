#include "HeatingParser.h"
#include <lib/core/simulation/heatingresult.h>

result_ptr HeatingParser::operator()(CSVDataMap &map)
{
    auto result = std::make_shared<cs::heating::HeatingSimulationResult>();
    for (auto &[name, stream] : map)
    {
        // TODO: put data into result
    }
    return result;
}

result_ptr HeatingParser::operator()(const ArrowDataMap &map)
{
    auto result = std::make_shared<cs::heating::HeatingSimulationResult>();
    for (auto &[name, table] : map)
    {
        // TODO: put data into result
    }
    return result;
}

result_ptr HeatingParser::operator()(const ArrowData &data)
{
    auto result = std::make_shared<cs::heating::HeatingSimulationResult>();
    // TODO: put data into result
    return result;
}

result_ptr HeatingParser::operator()(CSVData &data)
{
    auto result = std::make_shared<cs::heating::HeatingSimulationResult>();
    // TODO: put data into result
    return result;
}
