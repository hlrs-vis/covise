#pragma once
#include "EnergyType.h"

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
    
    virtual T load(int scenarioId, EnergyType type) = 0;
};
