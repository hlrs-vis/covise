#pragma once
#include "EnergyType.h"
#include "Scenario.h"

template <typename T>
class Provider
{
public:
    Provider() = default;
    virtual ~Provider() = default;
    Provider(const Provider&) = delete;
    Provider(Provider&&) = delete;
    Provider& operator=(const Provider&) = delete;
    Provider& operator=(Provider&&) = delete;
    
    virtual T load(const Scenario &scenario, EnergyType type) const = 0;
};
