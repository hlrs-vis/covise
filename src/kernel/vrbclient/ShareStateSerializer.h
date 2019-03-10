/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include <net/tokenbuffer.h>
#include <util/coExport.h>
#include <string>
#include <vector>
#ifndef SHARED_STATE_SERIALIZER_H
#define SHARED_STATE_SERIALIZER_H

namespace vrb {

///////////////////////DATA TYPE FUNCTIONS //////////////////////////

enum SharedStateDataType
{
    UNDEFINED = 0,
    BOOL,   //1
    INT,    //2
    FLOAT,  //3
    STRING, //4
    DOUBLE    //5
};
template<class T>
SharedStateDataType getSharedStateType(const T &type) {
    return UNDEFINED;
}
template <>
VRBEXPORT SharedStateDataType getSharedStateType<bool>(const bool &type); 

template <>
VRBEXPORT SharedStateDataType getSharedStateType<int>(const int &type);
template <>
VRBEXPORT SharedStateDataType getSharedStateType<float>(const float &type);
template <>
VRBEXPORT SharedStateDataType getSharedStateType<std::string>(const std::string &type);
template <>
VRBEXPORT SharedStateDataType getSharedStateType<char >(const char &type);
template <>
VRBEXPORT SharedStateDataType getSharedStateType<double>(const double &type);
//tries to convert the serialized tokenbuffer to a string
VRBEXPORT std::string tokenBufferToString(covise::TokenBuffer &&tb);

///////////////////////SERIALIZE //////////////////////////


///convert the value to a TokenBuffer
template<class T>
void serialize(covise::TokenBuffer &tb, const T &value)
{
    tb << value;
}

template <>
VRBEXPORT void serialize<std::vector<std::string>>(covise::TokenBuffer &tb, const std::vector<std::string> &value);

/////////////////////DESERIALIZE///////////////////////////////////
///converts the TokenBuffer back to the value


template<class T>
void deserialize(covise::TokenBuffer &tb, T &value)
{
    tb >> value;
}

template <>
VRBEXPORT void deserialize<std::vector<std::string>>(covise::TokenBuffer &tb, std::vector<std::string> &value);

template<class T>
void serializeWithType(covise::TokenBuffer &tb, const T &value)
{
    int typeID = getSharedStateType(value);
    tb << typeID;
    serialize(tb, value);
}

template<class T>
void deserializeWithType(covise::TokenBuffer &tb, T &value)
{
    int typeID;
    tb >> typeID;
    deserialize(tb, value);
}

} // namespace vrb
#endif
