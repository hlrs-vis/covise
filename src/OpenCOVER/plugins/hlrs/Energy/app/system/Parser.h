#pragma once
#include "EnergyType.h"
#include "DataPackage.h"
#include "Scenario.h"
#include <lib/core/simulation/powerresult.h>
#include <lib/core/simulation/heatingresult.h>
#include <lib/core/interfaces/IEnergyGrid.h>
#include <utility>

namespace cs = core::simulation;
using Result = std::shared_ptr<cs::SimulationResult>;
using EnergyGrid = std::unique_ptr<core::interface::IEnergyGrid>;

template <typename T>
struct DataPackageParser
{
    typedef T Type;
    // template<typename...Args>
    // DataPackageParser<T>(Args&&... args){}
    virtual T operator()(const CSVDataMap &map) = 0;
    virtual T operator()(const ArrowDataMap &map) = 0;
    virtual T operator()(const ArrowData &data) = 0;
    virtual T operator()(const CSVData &data) = 0;
};

struct GridParser : DataPackageParser<EnergyGrid>
{
};

struct PowerGridParser final : GridParser
{
    EnergyGrid operator()(const CSVDataMap &map) override;
    EnergyGrid operator()(const ArrowDataMap &map) override;
    EnergyGrid operator()(const ArrowData &data) override;
    EnergyGrid operator()(const CSVData &data) override;
};

struct HeatingGridParser final : GridParser
{
    EnergyGrid operator()(const CSVDataMap &map) override;
    EnergyGrid operator()(const ArrowDataMap &map) override;
    EnergyGrid operator()(const ArrowData &data) override;
    EnergyGrid operator()(const CSVData &data) override;
};

struct SimulationParser : DataPackageParser<Result>
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
    Result operator()(const CSVDataMap &map) override;
    Result operator()(const ArrowDataMap &map) override;
    Result operator()(const ArrowData &data) override;
    Result operator()(const CSVData &data) override;
};

struct HeatingParser final : public SimulationParser
{
    using SimulationParser::SimulationParser;
    Result operator()(const CSVDataMap &map) override;
    Result operator()(const ArrowDataMap &map) override;
    Result operator()(const ArrowData &data) override;
    Result operator()(const CSVData &data) override;
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
        else
        {
            return typename Map::Grid(std::forward<Args>(args)...)(std::forward<T>(data));
        }
    }
};
