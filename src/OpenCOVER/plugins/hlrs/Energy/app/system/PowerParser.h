#pragma once
#include "SimulationParser.h"

struct PowerParser final : public SimulationParser
{
    using SimulationParser::SimulationParser;
    result_ptr operator()(CSVDataMap &map) override;
    result_ptr operator()(const ArrowDataMap &map) override;
    result_ptr operator()(const ArrowData &data) override;
    result_ptr operator()(CSVData &data) override;
};
