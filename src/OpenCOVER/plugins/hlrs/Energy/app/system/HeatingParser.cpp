#include "HeatingParser.h"
#include <lib/core/simulation/heatingresult.h>
#include <utils/read/csv/csv.h>
#include <regex>

using namespace opencover::utils::read;

result_ptr HeatingParser::operator()(CSVDataMap &map)
{
    auto result = std::make_shared<cs::heating::HeatingSimulationResult>();
    if (map.find("results") == map.end())
        return result;
    auto stream = map["results"];

    std::regex consumer_value_split_regex("Consumer_(\\d+)_(.+)");
    std::regex producer_value_split_regex("Producer_(\\d+)_(.+)");
    std::smatch match;

    CSVStream::CSVRow row;
    const auto &header = stream->getHeader();
    double val = 0.0f;
    std::string name(""), valName("");

    while (stream->readNextRow(row))
    {
        for (const auto &col : header)
        {
            ACCESS_CSV_ROW(row, col, val);
            if (std::regex_search(col, match, consumer_value_split_regex))
            {
                name = match[1];
                valName = match[2];
                addDataToMap(result, core::simulation::ObjectType::Consumer, name, valName, val);
            }
            else if (std::regex_search(col, match, producer_value_split_regex))
            {
                name = match[1];
                valName = match[2];
                addDataToMap(result, core::simulation::ObjectType::Producer, name, valName, val);
            }
            else
            {
                if (val == 0)
                    continue;
                result->getDataStorage().addData(col, val);
            }
        }
    }

    result->init();
    return std::move(result);
}

auto HeatingParser::getObjMapByType(heating_ptr result, core::simulation::ObjectType type) -> ObjectMap *
{
    if (type == core::simulation::ObjectType::Consumer)
        return &result->Consumers();
    else if (type == core::simulation::ObjectType::Producer)
        return &result->Producers();
    return nullptr;
};

void HeatingParser::createObjAndAddToMap(heating_ptr result, core::simulation::ObjectType type, const std::string &name)
{
    Object obj{name, { { std::string("value"), {} } }};
    auto map = getObjMapByType(result, type);
    if (map == nullptr)
        return;
    map->emplace(name, std::move(obj));
}

auto HeatingParser::getObjPtr(heating_ptr result, core::simulation::ObjectType type, const std::string &name) -> Object *
{
    auto map = getObjMapByType(result, type);
    auto it = map->find(name);
    if (it != map->end())
        return &it->second;
    createObjAndAddToMap(result, type, name);
    return &map->at(name);
}

void HeatingParser::addDataToMap(heating_ptr result, ObjectType type, const std::string &name, const std::string &valName, double value)
{
    auto objPtr = getObjPtr(result, type, name);
    objPtr->addData(valName, value);
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
