#include "HeatingGridParser.h"

grid_ptr HeatingGridParser::operator()(CSVDataMap &map)
{
    return nullptr;
}

grid_ptr HeatingGridParser::operator()(const ArrowDataMap &map)
{
    return nullptr;
}

grid_ptr HeatingGridParser::operator()(const ArrowData &data)
{
    return nullptr;
}

grid_ptr HeatingGridParser::operator()(CSVData &data)
{
    return nullptr;
}
