#ifndef COVER_OPCUA_VARAIANT_ACCESS_H
#define COVER_OPCUA_VARAIANT_ACCESS_H

#include "types.h"
#include "export.h"
#include <open62541/types.h>
#include <numeric>
#include <cstring>
#include <vector>
#include <array>

namespace opencover{namespace opcua{

int OPCUACLIENTEXPORT toTypeId(const UA_DataType *type);

namespace detail{
class MultiDimensionalArrayBase{
public:
    bool isScalar() const;
    size_t numEntries() const;
    bool isValid() const;

    std::vector<size_t> dimensions;

};
}

template<typename T>
class OPCUACLIENTEXPORT MultiDimensionalArray : public detail::MultiDimensionalArrayBase
{
public:
    MultiDimensionalArray(UA_Variant *variant)
    {
        if(!variant)
            return;
        if(toTypeId(variant->type) != detail::getTypeId<T>())
            return;

        auto size = std::max(size_t(1), variant->arrayLength);
        dimensions.push_back(size);
        data.resize(size);
        std::memcpy(data.data(), variant->data, size * sizeof(T));
        if(variant->arrayDimensionsSize > 0)
        {
            dimensions.resize(variant->arrayDimensionsSize);
            for (size_t i = 0; i < variant->arrayDimensionsSize; i++)
            {
                dimensions.push_back(variant->arrayDimensions[i]);
            }
        }
    }
    std::vector<T> data;
};

MultiDimensionalArray<double> OPCUACLIENTEXPORT toNumericalArray(UA_Variant *variant);

}}

#endif // COVER_OPCUA_VARAIANT_ACCESS_H