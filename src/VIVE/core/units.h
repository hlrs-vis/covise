#pragma once
#include <array>
#include <string>
#include <util/coExport.h>

namespace vive
{
enum class LengthUnit{
    Kilometer, Meter, Centimeter, Millimeter, Micrometer, Mile, Yard, Foot, Inch, LAST
};
constexpr std::array<const char*, (int)LengthUnit::LAST> LengthUnitNames{"Kilometer", "Meter", "Centimeter", "Millimeter", "Micrometer", "Mile", "Yard", "Foot", "Inch"};
constexpr std::array<const char*, (int)LengthUnit::LAST> LengthUnitAbbreviation{"km", "m", "cm", "mm", "\xC2\xB5m", "mi", "yd", "ft", "in"};

bool VVCORE_EXPORT isValid(LengthUnit unit);
VVCORE_EXPORT const char*  getName(LengthUnit unit);
VVCORE_EXPORT const char* getAbbreviation(LengthUnit unit);

LengthUnit VVCORE_EXPORT getUnitFromName(const std::string &name); //returns LAST on failure

bool VVCORE_EXPORT isMetric(LengthUnit unit);
bool VVCORE_EXPORT isImperial(LengthUnit unit);
std::string VVCORE_EXPORT displayValueWithUnit(double value, LengthUnit unit);
}
