#ifndef COVER_DATACLIENT_MULTIDIMENSIONALARRAY_H
#define COVER_DATACLIENT_MULTIDIMENSIONALARRAY_H

#include "export.h"
#include <numeric>
#include <cstring>
#include <vector>
#include <array>

namespace opencover{namespace dataclient{

namespace detail{
class DATACLIENTEXPORT MultiDimensionalArrayBase{
public:
    virtual ~MultiDimensionalArrayBase() = default;    
    bool isScalar() const;
    size_t numEntries() const;
    bool isValid() const;

    std::vector<size_t> dimensions;
    double timestamp = 0;

};
}

template<typename T>
struct MultiDimensionalArray : public detail::MultiDimensionalArrayBase
{
    std::vector<T> data;
};


}}

#endif // COVER_DATACLIENT_MULTIDIMENSIONALARRAY_H