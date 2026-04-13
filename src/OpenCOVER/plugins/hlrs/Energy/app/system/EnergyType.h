#pragma once
#include <array>

enum class EnergyType
{
    HEATING,
    POWER,
    COOLING,
};

inline constexpr std::array<EnergyType, 2> ENERGYTYPE_RANGE = { EnergyType::HEATING, EnergyType::POWER };

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
