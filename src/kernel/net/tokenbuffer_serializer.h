/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "tokenbuffer.h"

#include <util/coExport.h>

#include <string>
#include <vector>
#include <set>
#include <map>
#include <utility>


#ifndef TOKEN_BUFFER_SERIALIZER_H
#define TOKEN_BUFFER_SERIALIZER_H
namespace covise
{
class DataHandle;

///////////////////////DATA TYPE FUNCTIONS //////////////////////////

enum class TokenBufferDataType
{
    TODETERMINE=-1,
    UNDEFINED = 0,
    BOOL,   //1
    INT,    //2
    FLOAT,  //3
    STRING, //4
    DOUBLE, //5
    VECTOR, //6
    SET,    //7
    MAP,	//8
    PAIR,	//9
    TRANSFERFUNCTION	//10
};

NETEXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, TokenBufferDataType t);
NETEXPORT covise::TokenBuffer &operator>>(covise::TokenBuffer &tb, TokenBufferDataType &t);


//how a sharedMap has changed
enum MapChangeType
{
    WHOLE, //every entry -> sent whole map (with types)
    ENTRY_CHANGE, // send position and new value

};
template<class T>
TokenBufferDataType getTokenBufferDataType(const T& type)
{
    return TokenBufferDataType::UNDEFINED;
}
template<class T>
TokenBufferDataType getTokenBufferDataType(const std::vector<T>& type)
{
    return TokenBufferDataType::VECTOR;
}
template<class T>
TokenBufferDataType getTokenBufferDataType(const std::set<T>& type)
{
    return TokenBufferDataType::SET;
}
template<class K, class V>
TokenBufferDataType getTokenBufferDataType(const std::map<K, V>& type)
{
    return TokenBufferDataType::MAP;
}
template<class K, class V>
TokenBufferDataType getTokenBufferDataType(const std::pair<K, V>& type)
{
    return TokenBufferDataType::PAIR;
}
template <>
NETEXPORT TokenBufferDataType getTokenBufferDataType<bool>(const bool& type);

template <>
NETEXPORT TokenBufferDataType getTokenBufferDataType<int>(const int& type);
template <>
NETEXPORT TokenBufferDataType getTokenBufferDataType<float>(const float& type);
template <>
NETEXPORT TokenBufferDataType getTokenBufferDataType<std::string>(const std::string& type);
template <>
NETEXPORT TokenBufferDataType getTokenBufferDataType<char >(const char& type);
template <>
NETEXPORT TokenBufferDataType getTokenBufferDataType<double>(const double& type);
//tries to convert the serializedWithType tokenbuffer to a string
NETEXPORT std::string tokenBufferToString(covise::TokenBuffer&& tb, TokenBufferDataType typeID = TokenBufferDataType::TODETERMINE);

///////////////////////SERIALIZE //////////////////////////
//pay attention to order or it may not compile!

///convert the value to a TokenBuffer

template<class T>
void serialize(covise::TokenBuffer& tb, const T& value)
{
    tb << value;
}

template <class K, class V>
covise::TokenBuffer &operator<<(covise::TokenBuffer& tb, const std::pair<K, V>& value)
{
    tb << value.first;
    tb << value.second;
    return tb;
}

template <class K, class V>
void serialize(covise::TokenBuffer& tb, const std::pair<K, V>& value)
{
    tb << getTokenBufferDataType(value.first);
    tb << getTokenBufferDataType(value.second);
    serialize(tb, value.first);
    serialize(tb, value.second);
}


template <class T>
covise::TokenBuffer &operator<<(covise::TokenBuffer& tb, const std::vector<T>& value)
{
    tb << static_cast<int>(value.size());
    for (const T &entry: value)
    {
        tb << entry;
    }
    return tb;
}

template <class T>
void serialize(covise::TokenBuffer& tb, const std::vector<T>& value)
{
    int size = value.size();
    if (size == 0)
    {
        tb << TokenBufferDataType::UNDEFINED;
    } else
    {
        tb << getTokenBufferDataType(value.front());
    }
    tb << size;
    for (const T &entry: value)
    {
        serialize(tb, entry);
    }
}

template <class T>
covise::TokenBuffer &operator<<(covise::TokenBuffer& tb, const std::set<T>& value)
{
    tb << static_cast<int>(value.size());
    for (const T &entry: value)
    {
        tb << entry;
    }
    return tb;
}

template <class T>
void serialize(covise::TokenBuffer& tb, const std::set<T>& value)
{
    int size = value.size();
    if (size == 0)
    {
        tb << TokenBufferDataType::UNDEFINED;
    } else
    {
        tb << getTokenBufferDataType(*value.begin());
    }
    tb << size;
    for (const T &entry : value)
    {
        serialize(tb, entry);
    }
}

template <class K, class V>
covise::TokenBuffer &operator<<(covise::TokenBuffer& tb, const std::map<K, V>& value)
{
    tb << static_cast<int>(value.size());
    for (const auto &entry: value)
    {
        tb << entry.first << entry.second;
    }
    return tb;
}

template <class K, class V>
void serialize(covise::TokenBuffer& tb, const std::map<K, V>& value)
{
    int size = value.size();
    if (size == 0)
    {
        tb << TokenBufferDataType::UNDEFINED;
        tb << TokenBufferDataType::UNDEFINED;
    } else
    {
        tb << getTokenBufferDataType(value.begin()->first);
        tb << getTokenBufferDataType(value.begin()->second);
    }
    tb << size;
    auto entry = value.begin();
    while (entry != value.end())
    {
        serialize(tb, entry->first);
        serialize(tb, entry->second);
        ++entry;
    }
}

/////////////////////DESERIALIZE///////////////////////////////////
///converts the TokenBuffer back to the value
template<class T>
void deserialize(covise::TokenBuffer& tb, T& value)
{
    tb >> value;
}

template <class K, class V>
covise::TokenBuffer &operator>>(covise::TokenBuffer& tb, std::pair<K, V>& value)
{
    tb >> value.first >> value.second;
    return tb;
}

template <class K, class V>
void deserialize(covise::TokenBuffer& tb, std::pair<K, V>& value)
{
    int type;
    tb >> type;
    tb >> type;
    deserialize(tb, value.first);
    deserialize(tb, value.second);
}

template <class T>
covise::TokenBuffer operator>>(covise::TokenBuffer& tb, std::vector<T>& value)
{
    int size;
    tb >> size;
    value.resize(size);
    for (int i = 0; i < size; i++)
    {
        tb >> value[i];
    }
    return tb;
}

template <class T>
void deserialize(covise::TokenBuffer& tb, std::vector<T>& value)
{
    int size, typeID;
    tb >> typeID;
    tb >> size;
    value.clear();
    value.resize(size);
    for (int i = 0; i < size; i++)
    {
        deserialize(tb, value[i]);
    }
}

template <class T>
covise::TokenBuffer operator>>(covise::TokenBuffer& tb, std::set<T>& value)
{
    int size;
    tb >> size;
    value.clear();
    for (int i = 0; i < size; i++)
    {
        auto it = value.insert(T{});
        tb >> *it.first;
    }
    return tb;
}

template <class T>
void deserialize(covise::TokenBuffer& tb, std::set<T>& value)
{
    int size, typeID;
    tb >> typeID;
    tb >> size;
    value.clear();
    for (int i = 0; i < size; i++)
    {
        auto it = value.insert(T{});
        tb >> *it.first;
    }
}


template <class K, class V>
covise::TokenBuffer operator>>(covise::TokenBuffer& tb, std::map<K, V>& value)
{
    int size;
    tb >> size;
    value.clear();
    for (int i = 0; i < size; i++)
    {
        auto it = value.emplace(std::pair<K, V>{K{}, V{}});
        tb >> it.first->first >> it.first->second;
    }
    return tb;
}

template <class K, class V>
void deserialize(covise::TokenBuffer& tb, std::map<K, V>& value)
{
    int size, typeID;
    tb >> typeID;
    tb >> typeID;
    tb >> size;
    value.clear();
    for (int i = 0; i < size; i++)
    {
        K key;
        V val;
        deserialize(tb, key);
        deserialize(tb, val);
        value[key] = val;
    }
}


///////////////////TYPE SERIALIZATION/////////////////////////
template<class T>
void serializeWithType(covise::TokenBuffer& tb, const T& value)
{
    auto typeID = getTokenBufferDataType(value);
    tb << typeID;
    serialize(tb, value);
}
template<class T>
void deserializeWithType(covise::TokenBuffer& tb, T& value)
{
    TokenBufferDataType typeID;
    tb >> typeID;
    deserialize(tb, value);
}

}//covise
#endif
