#pragma once
#include "SimulationParser.h"
#include "lib/core/simulation/heatingresult.h"
#include <lib/core/simulation/object_type.h>
#include <lib/core/simulation/object_factory.h>
#include <lib/core/simulation/object.h>
#include <memory>

using namespace core::simulation;
typedef std::shared_ptr<heating::HeatingSimulationResult> heating_ptr;

struct HeatingParser final : public SimulationParser
{
    using SimulationParser::SimulationParser;
    result_ptr operator()(CSVDataMap &map) override;
    result_ptr operator()(const ArrowDataMap &map) override;
    result_ptr operator()(const ArrowData &data) override;
    result_ptr operator()(CSVData &data) override;

private:
    auto getObjMapByType(heating_ptr result, ObjectType type) -> ObjectMap *;
    void createObjAndAddToMap(heating_ptr result, ObjectType type, const std::string &name);
    auto getObjPtr(heating_ptr result, ObjectType type, const std::string &name) -> Object*;
    void addDataToMap(heating_ptr result, ObjectType type, const std::string &name, const std::string &valName, double value);
};
