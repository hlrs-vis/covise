#include "types.h"

int opencover::opcua::detail::toTypeId(const UA_DataType *type)
{
    auto begin = UA_TYPES;
    auto end = begin + UA_TYPES_COUNT;
    auto it = std::find_if(begin, end, [type](const UA_DataType &t)
    {
        return&t == type;
    });
    return (int)(it - UA_TYPES);
}