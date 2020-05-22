/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "SharedStateSerializer.h"

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

std::string tokenBufferToString(covise::TokenBuffer &&tb, int typeID) {
    if (typeID == -1)
    {
        tb >> typeID;
    }
    std::string valueString;
    switch (typeID)
    {
    case vrb::UNDEFINED:
        valueString = "data of length: " + std::to_string(tb.getData().length());
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
    case VECTOR:
    case SET:
        int tID, size;
        tb >> tID;
        tb >> size;
        valueString = "Vector of size: " + std::to_string(size);
        for (int i = 0; i < size; i++)
        {
            valueString += "\n [" + std::to_string(i) + "] ";
            valueString += tokenBufferToString(std::move(tb), tID);
        }
		break;
	case MAP:
	{
		int keyType, valueType, size;
		tb >> keyType;
		tb >> valueType;
		tb >> size;
		valueString = "Map of size: " + std::to_string(size); 
		for (int i = 0; i < size; i++)
		{
			valueString += "\n [" + std::to_string(i) + "] ";
			valueString += tokenBufferToString(std::move(tb), keyType);
			valueString += " | ";
			valueString += tokenBufferToString(std::move(tb), valueType);
		}
	}
	break;
	case PAIR:
	{
		int firstType, secondType;
		tb >> firstType;
		tb >> secondType;
		valueString = "Pair: ";
		valueString += tokenBufferToString(std::move(tb), firstType);
		valueString += " | ";
		valueString += tokenBufferToString(std::move(tb), secondType);
	}
	break;
    }
    return valueString;
}

}
