#pragma once
#include "GridParser.h"

struct HeatingGridParser final : GridParser
{
    using GridParser::GridParser;
    grid_ptr operator()(CSVDataMap &map) override;
    grid_ptr operator()(const ArrowDataMap &map) override;
    grid_ptr operator()(const ArrowData &data) override;
    grid_ptr operator()(CSVData &data) override;
};
