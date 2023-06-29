#ifndef COVER_UNITS_H
#define COVER_UNITS_H

#include <array>
#include <string>
#include <util/coExport.h>

namespace opencover
{
enum class LengthUnit{
    Kilometer, Meter, CentiMeter, Millimeter, Mikrometer, Yard, Foot, Inch, LAST
};
constexpr std::array<const char*, (int)LengthUnit::LAST> LengthUnitNames{"Kilometer", "Meter", "CentiMeter", "Millimeter", "Mikrometer", "Yard", "Foot", "Inch"};
constexpr std::array<const char*, (int)LengthUnit::LAST> LengthUnitAbbreviation{"km", "m", "cm", "mm", "\xC2\xB5m", "yd", "ft", "\""};

bool COVEREXPORT isValid(LengthUnit unit);
COVEREXPORT const char*  getName(LengthUnit unit); 
COVEREXPORT const char* getAbbreviation(LengthUnit unit); 

LengthUnit COVEREXPORT getUnitFromName(const std::string &name); //returns LAST on failure

bool COVEREXPORT isMetric(LengthUnit unit);
bool COVEREXPORT isImperial(LengthUnit unit);
std::string COVEREXPORT displayValueWithUnit(double value, LengthUnit unit);
}

#endif // COVER_UNITS_H