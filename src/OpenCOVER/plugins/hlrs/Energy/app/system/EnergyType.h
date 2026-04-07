#pragma once
#include <initializer_list>

enum class EnergyType
{
    HEATING,
    POWER,
    COOLING,
};

constexpr auto ENERGYTYPE_RANGE = { EnergyType::HEATING, EnergyType::POWER };

inline auto EnergyTypeToString(EnergyType type) {
    switch(type) {
        case EnergyType::HEATING:
            return "heating";
        case EnergyType::COOLING:
            return "cooling";
        case EnergyType::POWER:
            return "power";
    }
}
