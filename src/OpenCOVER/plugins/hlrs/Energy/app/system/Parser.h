#pragma once
#include "EnergyType.h"
#include "DataPackage.h"
#include "Scenario.h"
#include "app/osg/presentation/grid.h"
#include <lib/core/simulation/powerresult.h>
#include <lib/core/simulation/heatingresult.h>
#include <lib/core/interfaces/IEnergyGrid.h>
#include <lib/core/ClassLogger.h>
#include <utils/read/csv/csv.h>
#include <utility>

namespace cs = core::simulation;
using result_ptr = std::shared_ptr<cs::SimulationResult>;
using grid_ptr = std::unique_ptr<core::interface::IEnergyGrid>;

template <typename T>
struct DataPackageParser
{
    typedef T Type;
    virtual T operator()(CSVDataMap &map) = 0;
    virtual T operator()(const ArrowDataMap &map) = 0;
    virtual T operator()(const ArrowData &data) = 0;
    virtual T operator()(CSVData &data) = 0;
};

struct GridParser : DataPackageParser<grid_ptr>
{
};

struct PowerGridParser final : GridParser, core::ClassLogger
{
    PowerGridParser(core::interface::ILogger &logger) : core::ClassLogger(logger, "PowerGridParser"){}
    grid_ptr operator()(CSVDataMap &map) override;
    grid_ptr operator()(const ArrowDataMap &map) override;
    grid_ptr operator()(const ArrowData &data) override;
    grid_ptr operator()(CSVData &data) override;

private:
    typedef std::unordered_map<int, std::string> IDLookupTable;

    std::vector<IDLookupTable> retrieveBusNameIdMapping(opencover::utils::read::CSVStream &stream);
    std::vector<grid::PointsMap> createPowerGridPoints(
        opencover::utils::read::CSVStream &stream, size_t &numPoints,
        const float &sphereRadius, const std::vector<IDLookupTable> &busNames, CSVDataMap& map);
    std::pair<std::vector<grid::Lines>, std::vector<grid::ConnectionDataList>>
    getPowerGridLines(opencover::utils::read::CSVStream &stream, const std::vector<grid::PointsMap> &points);
    void helper_getAdditionalPowerGridPointData_addData(
        int busId, grid::PointDataList &additionalData, const grid::Data &data);
    void helper_getAdditionalPowerGridPointData_handleDuplicate(
        std::string &name, std::map<std::string, uint> &duplicateMap);
    std::unique_ptr<grid::PointDataList> getAdditionalPowerGridPointData(
        const std::size_t &numOfBus, CSVDataMap& map);
};

struct HeatingGridParser final : GridParser
{
    grid_ptr operator()(CSVDataMap &map) override;
    grid_ptr operator()(const ArrowDataMap &map) override;
    grid_ptr operator()(const ArrowData &data) override;
    grid_ptr operator()(CSVData &data) override;
};

struct SimulationParser : DataPackageParser<result_ptr>
{
    Scenario scenario;
    SimulationParser(const Scenario &s)
        : scenario(s)
    {
    }
};

struct PowerParser final : public SimulationParser
{
    using SimulationParser::SimulationParser;
    result_ptr operator()(CSVDataMap &map) override;
    result_ptr operator()(const ArrowDataMap &map) override;
    result_ptr operator()(const ArrowData &data) override;
    result_ptr operator()(CSVData &data) override;
};

struct HeatingParser final : public SimulationParser
{
    using SimulationParser::SimulationParser;
    result_ptr operator()(CSVDataMap &map) override;
    result_ptr operator()(const ArrowDataMap &map) override;
    result_ptr operator()(const ArrowData &data) override;
    result_ptr operator()(CSVData &data) override;
};

template <EnergyType E>
struct ParserMapping;

template <>
struct ParserMapping<EnergyType::POWER>
{
    using Sim = PowerParser;
    using Grid = PowerGridParser;
};

template <>
struct ParserMapping<EnergyType::HEATING>
{
    using Sim = HeatingParser;
    using Grid = HeatingGridParser;
};

struct ParseManager
{
    template <typename T, typename... Args>
    auto operator()(EnergyType type, T &&data, Args &&...args) const -> decltype(auto)
    {
        switch (type)
        {
        case EnergyType::POWER:
            return parse<EnergyType::POWER>(std::forward<T>(data), std::forward<Args>(args)...);
        case EnergyType::HEATING:
            return parse<EnergyType::HEATING>(std::forward<T>(data), std::forward<Args>(args)...);
        default:
            // Determine a safe null return type based on what we expected to get
            using ExpectedRet = decltype(parse<EnergyType::POWER>(std::forward<T>(data), std::forward<Args>(args)...));
            return ExpectedRet {};
        }
    }

private:
    template <EnergyType E, typename T, typename... Args>
    auto parse(T &&data, Args &&...args) const -> decltype(auto)
    {
        using Map = ParserMapping<E>;

        // Use SFINAE/Compile-time checks to pick the right parser
        if constexpr (std::is_constructible_v<typename Map::Sim, Args...>)
        {
            return typename Map::Sim(std::forward<Args>(args)...)(std::forward<T>(data));
        }
        if constexpr (std::is_constructible_v<typename Map::Grid, Args...>)
        {
            return typename Map::Grid(std::forward<Args>(args)...)(std::forward<T>(data));
        }
    }
};
