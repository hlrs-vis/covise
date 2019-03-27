/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include <net/tokenbuffer.h>
#include <util/coExport.h>
#include <string>
#include <vector>
#include <set>

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
    DOUBLE, //5
    VECTOR //6
};
template<class T>
SharedStateDataType getSharedStateType(const T &type) {
    return UNDEFINED;
}
template<class T>
SharedStateDataType getSharedStateType(const std::vector<T> &type) {
    return VECTOR;
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
VRBEXPORT std::string tokenBufferToString(covise::TokenBuffer &&tb, int typeID = -1);

///////////////////////SERIALIZE //////////////////////////


///convert the value to a TokenBuffer
template<class T>
void serialize(covise::TokenBuffer &tb, const T &value)
{
    tb << value;
}

template <class T>
void serialize(covise::TokenBuffer &tb, const std::vector<T> &value) {
    int size = value.size();
    if (size == 0)
    {
        tb << UNDEFINED;
    }
    else
    {
        tb << getSharedStateType(value.front());
    }
    tb << size;
    for (const T entry: value)
    {
        serialize(tb, entry);
    }
}

template <class T>
void serialize(covise::TokenBuffer &tb, const std::set<T> &value) {
    int size = value.size();
    if (size == 0)
    {
        tb << UNDEFINED;
    }
    else
    {
        tb << getSharedStateType(*value.begin());
    }
    tb << size;
    for (const T entry : value)
    {
        serialize(tb, entry);
    }
}
/////////////////////DESERIALIZE///////////////////////////////////
///converts the TokenBuffer back to the value
template<class T>
void deserialize(covise::TokenBuffer &tb, T &value)
{
    tb >> value;
}
template <class T>
void deserialize(covise::TokenBuffer &tb, std::vector<T> &value) 
{
    int size, typeID;
    tb >> typeID;
    tb >> size;
    value.clear();
    value.resize(size);
    for (int i = 0; i < size; i++)
    {
        T entry;
        deserialize(tb, entry);
        value[i] = entry;
    }
}

template <class T>
void deserialize(covise::TokenBuffer &tb, std::set<T> &value)
{
    int size, typeID;
    tb >> typeID;
    tb >> size;
    value.clear();
    for (int i = 0; i < size; i++)
    {
        T entry;
        deserialize(tb, entry);
        value.insert(entry);
    }
}
///////////////////TYPE SERIALIZATION/////////////////////////
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
}
#endif
