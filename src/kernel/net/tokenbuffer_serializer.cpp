/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "tokenbuffer_serializer.h"

#include <cassert>

using namespace covise;

TokenBuffer &covise::operator<<(TokenBuffer &tb, TokenBufferDataType t){
    tb << static_cast<int>(t);
    return tb;
}
TokenBuffer &covise::operator>>(TokenBuffer &tb, TokenBufferDataType &t){
    int tt = 0;
    tb >> tt;
    t = static_cast<TokenBufferDataType>(tt);
    return tb;
}

namespace covise
{
template <>
TokenBufferDataType getTokenBufferDataType<bool>(const bool &type) {
    return TokenBufferDataType::BOOL;
}
template <>
TokenBufferDataType getTokenBufferDataType<int>(const int &type) {
    return TokenBufferDataType::INT;
}
template <>
TokenBufferDataType getTokenBufferDataType<float>(const float &type) {
    return TokenBufferDataType::FLOAT;
}
template <>
TokenBufferDataType getTokenBufferDataType<std::string>(const std::string &type) {
    return TokenBufferDataType::STRING;
}
template <>
TokenBufferDataType getTokenBufferDataType<char >(const char &type) {
    return TokenBufferDataType::STRING;
}
template <>
TokenBufferDataType getTokenBufferDataType<double>(const double &type) {
    return TokenBufferDataType::DOUBLE;
}


std::string tokenBufferToString(covise::TokenBuffer &&tb, TokenBufferDataType typeID) {
    static bool abbort = true;
    if (typeID == TokenBufferDataType::TODETERMINE)
    {
        tb >> typeID;
        abbort = false;
    }
    std::string valueString;
    switch (typeID)
    {
    case TokenBufferDataType::TODETERMINE:
        std::cerr << "tokenBufferToString: TokenBufferDataType::TODETERMINE twice" << std::endl;
        assert("TokenBufferDataType::TODETERMINE twice" == nullptr);
        break;
    case TokenBufferDataType::UNDEFINED:
        valueString = "data of length: " + std::to_string(tb.getData().length());
        abbort = true;
        break;
    case TokenBufferDataType::BOOL:
        bool b;
        tb >> b;
        valueString = std::to_string(b);
        break;
    case TokenBufferDataType::INT:
        int v;
        tb >> v;
        valueString = std::to_string(v);
        break;
    case TokenBufferDataType::FLOAT:
        float f;
        tb >> f;
        valueString = std::to_string(f);
        break;
    case TokenBufferDataType::STRING:
        tb >> valueString;
        break;
    case TokenBufferDataType::DOUBLE:
        double d;
        tb >> d;
        valueString = std::to_string(d);
        break;
    case TokenBufferDataType::VECTOR:
    case TokenBufferDataType::SET:
        TokenBufferDataType tID;
        int size;
        tb >> tID;
        tb >> size;
        valueString = "Vector of size: " + std::to_string(size);
        for (int i = 0; i < size; i++)
        {
            valueString += "\n [" + std::to_string(i) + "] ";
            valueString += tokenBufferToString(std::move(tb), tID);
            if (abbort)
                break;
            }
		break;
	case TokenBufferDataType::MAP:
	{
        TokenBufferDataType keyType, valueType;
        int size;
        tb >> keyType;
		tb >> valueType;
		tb >> size;
		valueString = "Map of size: " + std::to_string(size); 
		for (int i = 0; i < size; i++)
		{
			valueString += "\n [" + std::to_string(i) + "] ";
			valueString += tokenBufferToString(std::move(tb), keyType);
            if (abbort)
			valueString += " | ";
			valueString += tokenBufferToString(std::move(tb), valueType);
            if (abbort)
                break;
		}
	}
	break;
	case TokenBufferDataType::PAIR:
	{
		TokenBufferDataType firstType, secondType;
		tb >> firstType;
		tb >> secondType;
		valueString = "Pair: ";
		valueString += tokenBufferToString(std::move(tb), firstType);
		valueString += " | ";
        if(abbort)
            break;
        valueString += tokenBufferToString(std::move(tb), secondType);
    }
	break;
    case TokenBufferDataType::TRANSFERFUNCTION:
    {
        valueString = "Transfer function";
        abbort = true;
    }
    break;
    case TokenBufferDataType::Enum:
    {
        int i;
        tb >> i;
        valueString += std::to_string(i);
    }
    break;
    }
    return valueString;
}

}
