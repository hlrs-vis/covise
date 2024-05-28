#include "variantAccess.h"

using namespace opencover::opcua;

bool detail::MultiDimensionalArrayBase::isScalar() const {return dimensions.size() == 1 && dimensions[0] == 1;}
size_t detail::MultiDimensionalArrayBase::numEntries() const { return std::accumulate(dimensions.begin(), dimensions.end(), size_t(1), std::multiplies<size_t>());}
bool detail::MultiDimensionalArrayBase::isValid() const {return dimensions.size() > 0;}

int opencover::opcua::toTypeId(const UA_DataType *type)
{
    auto begin = UA_TYPES;
    auto end = begin + UA_TYPES_COUNT;
    auto it = std::find_if(begin, end, [type](const UA_DataType &t)
    {
        return&t == type;
    });
    return (int)(it - UA_TYPES);
}

MultiDimensionalArray<double> opencover::opcua::toNumericalArray(UA_Variant *variant)
{
    MultiDimensionalArray<double> retval(variant);
    detail::for_<8>([variant, &retval] (auto i) {      
        typedef typename detail::Type<detail::numericalTypes[i.value]>::type T;
        MultiDimensionalArray<T> array(variant);
        if (array.isValid())
        {
            
            retval.dimensions = std::move(array.dimensions);
            retval.data.resize(array.data.size());
            std::transform(array.data.begin(), array.data.end(), retval.data.begin(), [](T t){return (double) t;});
        }
    });
    return retval;

}