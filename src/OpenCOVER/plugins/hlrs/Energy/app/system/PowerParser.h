#pragma once
#include "SimulationParser.h"
#include <lib/core/simulation/object.h>

using namespace core::simulation;

struct PowerParser final : public SimulationParser
{
    using SimulationParser::SimulationParser;
    result_ptr operator()(CSVDataMap &map) override;
    result_ptr operator()(const ArrowDataMap &map) override;
    result_ptr operator()(const ArrowData &data) override;
    result_ptr operator()(CSVData &data) override;

private:
    void processArrowTableColumns(const std::shared_ptr<arrow::Table> &tbl, ObjectMap &map,
        const std::string &dataKey);
};
