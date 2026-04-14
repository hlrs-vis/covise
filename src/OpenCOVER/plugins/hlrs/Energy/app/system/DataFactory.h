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
    decltype(auto) operator()(T &&data) const
    {
        return std::apply([&](auto &&...args)
            { return ParseManager {}(energyType, std::forward<T>(data), std::forward<decltype(args)>(args)...); }, extraArgs);
    }
};

struct DataFactory
{
    template <typename T, typename... Args>
    static decltype(auto) create(T &&package, EnergyType type, Args &&...args)
    {
        return std::visit(DataPackageVisitor<Args...> { type, std::forward<Args>(args)... }, std::forward<T>(package));
    }
};
