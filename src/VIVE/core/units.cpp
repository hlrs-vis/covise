#include "units.h"

#include <algorithm>
#include <sstream>
using namespace vive;

constexpr std::array<double, (int)LengthUnit::LAST> LengthUnitConversionInKm{1, 1000, 100000, 1000000, 1000000000, 0.621371, 1093.61, 3280.84, 39370.1};
constexpr std::array<bool, (int)LengthUnit::LAST> LengthUnitIsImperial{0,0,0,0,0,1,1,1, 1};

bool vive::isValid(LengthUnit unit)
{
    return unit != LengthUnit::LAST;
}

const char* vive::getName(LengthUnit unit)
{
    return LengthUnitNames[(int)unit];
}

const char* vive::getAbbreviation (LengthUnit unit)
{
    return LengthUnitAbbreviation[(int)unit];

}

LengthUnit vive::getUnitFromName(const std::string &unitName)
{
    auto it = std::find(LengthUnitNames.begin(), LengthUnitNames.end(), unitName);
    if(it != LengthUnitNames.end())
        return (LengthUnit) std::distance(it ,LengthUnitNames.begin());
    it = std::find(LengthUnitAbbreviation.begin(), LengthUnitAbbreviation.end(), unitName);
    if(it != LengthUnitAbbreviation.end())
        return (LengthUnit) std::distance(it ,LengthUnitAbbreviation.begin());
    return LengthUnit::LAST;
}


bool vive::isMetric(LengthUnit unit)
{
    return !isImperial(unit);
}

bool vive::isImperial(LengthUnit unit)
{
    return LengthUnitIsImperial[(int)unit];
}

std::string vive::displayValueWithUnit(double value, LengthUnit unit)
{
    auto metric = isMetric(unit);
    auto nativeConversionFactor = LengthUnitConversionInKm[int(unit)];
    auto outUnit = unit;
    if(value >= 1)
    {
        for (int i = (int)unit -1; i >= 0; --i)
        {
            if(isMetric((LengthUnit)i) == metric)
            {
                auto conversionFactor = LengthUnitConversionInKm[i] / LengthUnitConversionInKm[i+1];
                auto v = value * conversionFactor;
                if(v < 1)
                    break;
                value = v;
                outUnit = (LengthUnit)i;
            }
        }
    } else   
    {
        for (int i = (int)unit +1; i < (int)LengthUnit::LAST; ++i)
        {
            if(isMetric((LengthUnit)i) == metric)
            {
                auto conversionFactor = LengthUnitConversionInKm[i] / LengthUnitConversionInKm[i-1];
                auto v = value * conversionFactor;
                value = v;
                outUnit = (LengthUnit)i;
                if(v >= 1)
                {
                    break;
                }
            } 
        }
    }
    std::stringstream ss;
    ss.precision(3);
    ss << value << " " << getAbbreviation(outUnit);
    return ss.str();
}
