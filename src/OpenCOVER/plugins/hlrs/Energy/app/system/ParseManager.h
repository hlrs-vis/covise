#pragma once
#include "EnergyType.h"
#include "DataPackage.h"
#include "PowerGridParser.h"
#include "HeatingGridParser.h"
#include "PowerParser.h"
#include "HeatingParser.h"
#include <utility>
#include <type_traits>

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

// template <EnergyType E>
// struct ParserCapabilities
// {
//     static constexpr bool supports_csv_data_map = false;
//     static constexpr bool supports_csv_data = false;
//     static constexpr bool supports_arrow_data_map = false;
//     static constexpr bool supports_arrow_data = false;
// };

// template <EnergyType E>
// struct GridParserCapabilities : ParserCapabilities<E>
// {
// };

// template <EnergyType E>
// struct SimulationParserCapabilities : ParserCapabilities<E>
// {
// };

// template <>
// struct GridParserCapabilities<EnergyType::HEATING>
// {
//     static constexpr bool supports_csv_data_map = false;
//     static constexpr bool supports_csv_data = false;
//     static constexpr bool supports_arrow_data_map = false;
//     static constexpr bool supports_arrow_data = false;
// };

// template <>
// struct GridParserCapabilities<EnergyType::POWER>
// {
//     static constexpr bool supports_csv_data_map = false;
//     static constexpr bool supports_csv_data = false;
//     static constexpr bool supports_arrow_data_map = false;
//     static constexpr bool supports_arrow_data = false;
// };

// template <>
// struct SimulationParserCapabilities<EnergyType::HEATING>
// {
//     static constexpr bool supports_csv_data_map =  true;
//     static constexpr bool supports_csv_data = false;
//     static constexpr bool supports_arrow_data_map = false;
//     static constexpr bool supports_arrow_data = false;
// };

// template <>
// struct SimulationParserCapabilities<EnergyType::POWER>
// {
//     static constexpr bool supports_csv_data_map = false;
//     static constexpr bool supports_csv_data = false;
//     static constexpr bool supports_arrow_data_map = false;
//     static constexpr bool supports_arrow_data = false;
// };

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

    // NOTE: Trying to check for implemented operator Parser functions.
    // template <typename Capabilities, typename T>
    // struct is_supported {
    //     // remove references and cv-qualifiers
    //     using RawT = std::decay_t<T>;
    
    //     static constexpr bool value = 
    //         (std::is_same_v<RawT, CSVData> && Capabilities::supports_csv_data) ||
    //         (std::is_same_v<RawT, CSVDataMap> && Capabilities::supports_csv_data_map) ||
    //         (std::is_same_v<RawT, ArrowData> && Capabilities::supports_arrow_data) ||
    //         (std::is_same_v<RawT, ArrowDataMap> && Capabilities::supports_arrow_data_map);
    // };
    //
    // template <EnergyType E, typename T, typename... Args>
    // auto parse(T &&data, Args &&...args) const -> decltype(auto)
    // {
    //     using Map = ParserMapping<E>;

    //     // Use SFINAE/Compile-time checks to pick the right parser
    //     if constexpr (std::is_constructible_v<typename Map::Sim, Args...>)
    //     {
    //         using Caps = SimulationParserCapabilities<E>;
    //         // Check if parser is capabable of parsing this type of data
    //         if constexpr (is_supported<Caps, T>::value){
    //             return typename Map::Sim(std::forward<Args>(args)...)(std::forward<T>(data));
    //         } else {
    //             return result_ptr(nullptr);
    //         }
    //     }
    //     if constexpr (std::is_constructible_v<typename Map::Grid, Args...>)
    //     {
    //         using Caps = GridParserCapabilities<E>;
    //         if constexpr (is_supported<Caps, T>::value){
    //             return typename Map::Grid(std::forward<Args>(args)...)(std::forward<T>(data));
    //         } else {
    //             return grid_ptr(nullptr);
    //         }
    //     }
    // }
};
