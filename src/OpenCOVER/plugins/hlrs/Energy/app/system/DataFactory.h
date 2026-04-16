#pragma once
#include "EnergyType.h"
#include "Parser.h"
#include <lib/core/simulation/simulationresult.h>
#include <lib/core/simulation/powerresult.h>
#include <lib/core/simulation/heatingresult.h>
#include <tuple>
#include <utility>

template <typename... Args>
struct DataPackageVisitor
{
    EnergyType energyType;
    std::tuple<Args...> extraArgs;

    DataPackageVisitor(EnergyType t, Args &&...args)
        : energyType(t)
        , extraArgs(std::forward<Args>(args)...)
    {
    }

    template <typename T>
    auto operator()(T &&data) const -> decltype(auto)
    {
        return std::apply([&](auto &&...args)
            { return ParseManager {}(energyType, std::forward<T>(data), std::forward<decltype(args)>(args)...); }, extraArgs);
    }
};

struct DataFactory
{
    template <typename T, typename... Args>
    static auto create(T &&package, EnergyType type, Args &&...args) -> decltype(auto) 
    {
        return std::visit(DataPackageVisitor<Args...> { type, std::forward<Args>(args)... }, std::forward<T>(package));
    }
};
