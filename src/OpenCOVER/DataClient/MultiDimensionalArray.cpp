#include "MultiDimensionalArray.h"

using namespace opencover::dataclient;

bool detail::MultiDimensionalArrayBase::isScalar() const {return dimensions.size() == 1 && dimensions[0] == 1;}
size_t detail::MultiDimensionalArrayBase::numEntries() const { return std::accumulate(dimensions.begin(), dimensions.end(), size_t(1), std::multiplies<size_t>());}
bool detail::MultiDimensionalArrayBase::isValid() const {return dimensions.size() > 0;}

