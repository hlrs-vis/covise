#include "PowerParser.h"
#include <lib/core/simulation/powerresult.h>

namespace
{
const std::array<std::string, 13> skipInfluxTables {
    "timestamp", "district", "hkw", "new-buildings",
    "pv-penetration", "loc_emob", "n_emob", "awz_scaling",
    "loc_ev", "n_ev", "new_buildings", "operation_mode",
    "pv_scaling"
};

auto isSkippedInfluxTable(const std::string &name)
{
    return std::any_of(skipInfluxTables.begin(), skipInfluxTables.end(),
        [&](const auto &s)
        { return s == name; });
}
}

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
    auto &cables = result->Cables();
    auto &buses = result->Buses();
    auto &buildings = result->Buildings();
    auto *objectMap = &cables;
    std::string key { "loading_percent" };

    for (auto &[name, table] : map)
    {
        if (name == "electrical_grid.res_bus.vm_pu")
        {
            key = "vm_pu";
            objectMap = &buses;
        }
        else if (name == "electrical_prosumer.res_mw")
        {
            key = "res_mw";
            objectMap = &buildings;
        }
        else if (name == "electrical_grid.res_line.loading_percent")
        {
            key = "loading_percent";
            objectMap = &cables;
        }
        processArrowTableColumns(table, *objectMap, key);
    }
    result->init();
    return std::move(result);
}

void PowerParser::processArrowTableColumns(const std::shared_ptr<arrow::Table> &tbl, ObjectMap &objMap,
    const std::string &dataKey)
{
    auto columnNames = tbl->schema()->fields();
    for (int j = 0; j < tbl->num_columns(); ++j)
    {
        auto columnName = columnNames[j]->name();
        std::replace(columnName.begin(), columnName.end(), ' ', '_');
        std::replace(columnName.begin(), columnName.end(), '/', '-');
        if (isSkippedInfluxTable(columnName))
            continue;
        auto column = tbl->column(j);
        int64_t chunk_offset = 0;
        for (int i = 0; i < column->num_chunks(); ++i)
        {
            auto chunk = column->chunk(i);
            if (chunk->type_id() == arrow::Type::DOUBLE)
            {
                auto darr = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                auto rawValues = darr->raw_values();
                if (objMap.find(columnName) == objMap.end())
                {
                    Data data{ { dataKey, std::vector<double>(column->length()) } };
                    objMap.emplace(columnName, Object{columnName, std::move(data)});
                }

                auto &vec = objMap.at(columnName).getData()[dataKey];
                std::copy(rawValues, rawValues + darr->length(),
                    vec.begin() + chunk_offset);
                chunk_offset += darr->length();
            }
        }
    }
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
