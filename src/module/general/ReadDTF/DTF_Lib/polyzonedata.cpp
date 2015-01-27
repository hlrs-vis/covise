/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF_Lib/polyzonedata.cpp
 * @brief contains definition of methods of class DTF_Lib::PolyZoneData.
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 10.10.2003
 * created
 */

#include "polyzonedata.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibPolyZoneData, PolyZoneData, "DTF_Lib::PolyZoneData", INT_MAX);

PolyZoneData::PolyZoneData()
{
}

PolyZoneData::PolyZoneData(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

PolyZoneData::~PolyZoneData()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool PolyZoneData::fill(map<string, int> values, string name)
{
    this->valIterator = this->values.find(name);

    if (this->valIterator != values.end())
        this->values[name] = this->valIterator->second;
    else
        return false;

    return true;
}

bool PolyZoneData::getValue(string key, int &value)
{
    this->valIterator = values.find(key);

    if (this->valIterator != values.end())
        value = this->valIterator->second;
    else
        return false;

    return true;
}

void PolyZoneData::clear()
{
    this->values.clear();
}
