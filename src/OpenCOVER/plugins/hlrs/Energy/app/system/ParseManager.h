#pragma once
#include "EnergyType.h"
#include "PowerGridParser.h"
#include "HeatingGridParser.h"
#include "PowerParser.h"
#include "HeatingParser.h"
#include <utility>

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
