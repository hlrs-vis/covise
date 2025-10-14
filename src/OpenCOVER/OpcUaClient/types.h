#ifndef COVER_OPCUA_TYPES_H
#define COVER_OPCUA_TYPES_H

#include "export.h"

#include <algorithm>
#include <array>
#include <open62541/types_generated.h>
#include <open62541/types.h>

namespace opencover{namespace opcua{namespace detail{

struct InvalidType
{};

template<typename T>
constexpr int getTypeId()
{
    return -1;
}

template<int I>
struct Type
{
    typedef InvalidType type;
};

#define SPEZIALIZE_GET_TYPE_ID(type_, typename)\
template<>\
constexpr int getTypeId<type_>()\
{\
    return typename;\
}\
\
template<>\
struct Type<typename>\
{\
    typedef type_ type;\
};

SPEZIALIZE_GET_TYPE_ID(UA_Boolean, UA_TYPES_BOOLEAN)
SPEZIALIZE_GET_TYPE_ID(UA_SByte, UA_TYPES_SBYTE)
SPEZIALIZE_GET_TYPE_ID(UA_Byte, UA_TYPES_BYTE)
SPEZIALIZE_GET_TYPE_ID(UA_Int16, UA_TYPES_INT16)
SPEZIALIZE_GET_TYPE_ID(UA_UInt16, UA_TYPES_UINT16)
SPEZIALIZE_GET_TYPE_ID(UA_Int32, UA_TYPES_INT32)
SPEZIALIZE_GET_TYPE_ID(UA_UInt32, UA_TYPES_UINT32)
SPEZIALIZE_GET_TYPE_ID(UA_Int64, UA_TYPES_INT64)
SPEZIALIZE_GET_TYPE_ID(UA_UInt64, UA_TYPES_UINT64)
SPEZIALIZE_GET_TYPE_ID(UA_Float, UA_TYPES_FLOAT)
SPEZIALIZE_GET_TYPE_ID(UA_Double, UA_TYPES_DOUBLE)
SPEZIALIZE_GET_TYPE_ID(UA_String, UA_TYPES_STRING)
SPEZIALIZE_GET_TYPE_ID(UA_Guid, UA_TYPES_GUID)
SPEZIALIZE_GET_TYPE_ID(UA_NodeId, UA_TYPES_NODEID)
SPEZIALIZE_GET_TYPE_ID(UA_ExpandedNodeId, UA_TYPES_EXPANDEDNODEID)
SPEZIALIZE_GET_TYPE_ID(UA_QualifiedName, UA_TYPES_QUALIFIEDNAME)
SPEZIALIZE_GET_TYPE_ID(UA_LocalizedText, UA_TYPES_LOCALIZEDTEXT)
SPEZIALIZE_GET_TYPE_ID(UA_ExtensionObject, UA_TYPES_EXTENSIONOBJECT)
SPEZIALIZE_GET_TYPE_ID(UA_DataValue, UA_TYPES_DATAVALUE)
SPEZIALIZE_GET_TYPE_ID(UA_Variant, UA_TYPES_VARIANT)
SPEZIALIZE_GET_TYPE_ID(UA_DiagnosticInfo, UA_TYPES_DIAGNOSTICINFO)


constexpr std::array<int, 8> numericalTypes{UA_TYPES_INT16, UA_TYPES_UINT16, UA_TYPES_INT32, UA_TYPES_UINT32, UA_TYPES_INT64, UA_TYPES_UINT64, UA_TYPES_FLOAT, UA_TYPES_DOUBLE};

// SPEZIALIZE_GET_TYPE_ID(UA_DateTime, UA_TYPES_DATETIME)
// SPEZIALIZE_GET_TYPE_ID(UA_ByteString, UA_TYPES_BYTESTRING)
// SPEZIALIZE_GET_TYPE_ID(UA_XmlElement, UA_TYPES_XMLELEMENT)
// SPEZIALIZE_GET_TYPE_ID(UA_StatusCode, UA_TYPES_STATUSCODE)

//constexpr for loop
template<std::size_t N>
struct num { static const constexpr auto value = N; };

template <class F, std::size_t... Is>
void for_(F func, std::index_sequence<Is...>)
{
  (func(num<Is>{}), ...);
}

template <std::size_t N, typename F>
void for_(F func)
{
  for_(func, std::make_index_sequence<N>());
}

int toTypeId(const UA_DataType *type);

}}}
#endif // COVER_OPCUA_TYPES_H
