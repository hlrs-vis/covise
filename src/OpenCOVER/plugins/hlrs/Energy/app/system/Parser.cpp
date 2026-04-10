#include "Parser.h"

std::shared_ptr<cs::SimulationResult> PowerParser::operator()(const CSVDataMap &map)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    for (auto &[name, stream] : map)
    {
        // TODO: put data into result
    }
    return result;
}

std::shared_ptr<cs::SimulationResult> PowerParser::operator()(const ArrowDataMap &map)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    for (auto &[name, table] : map)
    {
        // TODO: put data into result
    }
    return result;
}

std::shared_ptr<cs::SimulationResult> PowerParser::operator()(const ArrowData &data)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    // TODO: put data into result
    return result;
}

std::shared_ptr<cs::SimulationResult> PowerParser::operator()(const CSVData &data)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    // TODO: put data into result
    return result;
}

std::shared_ptr<cs::SimulationResult> HeatingParser::operator()(const CSVDataMap &map)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    for (auto &[name, stream] : map)
    {
        // TODO: put data into result
    }
    return result;
}

std::shared_ptr<cs::SimulationResult> HeatingParser::operator()(const ArrowDataMap &map)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    for (auto &[name, table] : map)
    {
        // TODO: put data into result
    }
    return result;
}

std::shared_ptr<cs::SimulationResult> HeatingParser::operator()(const ArrowData &data)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    // TODO: put data into result
    return result;
}

std::shared_ptr<cs::SimulationResult> HeatingParser::operator()(const CSVData &data)
{
    auto result = std::make_shared<cs::power::PowerSimulationResult>();
    // TODO: put data into result
    return result;
}
