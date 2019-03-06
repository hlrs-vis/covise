/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "ShareStateSerializer.h"

namespace vrb
{
template <>
SharedStateDataType getSharedStateType<bool>(const bool &type) {
    return BOOL;
}
template <>
SharedStateDataType getSharedStateType<int>(const int &type) {
    return INT;
}
template <>
SharedStateDataType getSharedStateType<float>(const float &type) {
    return FLOAT;
}
template <>
SharedStateDataType getSharedStateType<std::string>(const std::string &type) {
    return STRING;
}
template <>
SharedStateDataType getSharedStateType<char >(const char &type) {
    return STRING;
}
template <>
SharedStateDataType getSharedStateType<double>(const double &type) {
    return DOUBLE;
}

std::string tokenBufferToString(covise::TokenBuffer &&tb) {
    int typeID;
    tb >> typeID;
    std::string valueString;
    switch (typeID)
    {
    case vrb::UNDEFINED:
        valueString = "data of length: " + std::to_string(tb.get_length());
        break;
    case vrb::BOOL:
        bool b;
        tb >> b;
        valueString = std::to_string(b);
        break;
    case vrb::INT:
        int v;
        tb >> v;
        valueString = std::to_string(v);
        break;
    case vrb::FLOAT:
        float f;
        tb >> f;
        valueString = std::to_string(f);
        break;
    case vrb::STRING:
        tb >> valueString;
        break;
    case DOUBLE:
        double d;
        tb >> d;
        valueString = std::to_string(d);
        break;
    }
    return valueString;
}
////////////////////VECTOR<STRING>////////////////////////////
template <>
void serialize<std::vector<std::string>>(covise::TokenBuffer &tb, const std::vector<std::string> &value)
{
    int typeID = 0;
    tb << typeID;
    uint32_t size = value.size();
    tb << size;
    for (size_t i = 0; i < size; i++)
    {
        tb << value[i];
    }
}
template <>
void deserialize<std::vector<std::string>>(covise::TokenBuffer &tb, std::vector<std::string> &value)
{
    int typeID;
    uint32_t size;
    tb >> typeID;
    tb >> size;
    value.clear();
    value.resize(size);
    for (size_t i = 0; i < size; i++)
    {
        std::string path;
        tb >> path;
        value[i] = path;
    }
}
}