/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF_Lib/evaldata.cpp
 * @brief contains definition of methods of class DTF_Lib::EvalData.
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 09.10.2003
 * created
 */

#include "evaldata.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibEvalData, EvalData, "DTF_Lib::EvalData", INT_MAX);

EvalData::EvalData(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

EvalData::EvalData()
{
}

EvalData::~EvalData()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool EvalData::addValue(string name,
                        int value)
{
    intNames.push_back(name);
    intValues.push_back(value);

    return true;
}

bool EvalData::addValue(string name,
                        double value)
{
    realNames.push_back(name);
    realValues.push_back(value);

    return true;
}

bool EvalData::addValue(string name,
                        string value)
{
    stringNames.push_back(name);
    stringValues.push_back(value);

    return true;
}

bool EvalData::getInts(vector<string> &names,
                       vector<int> &values)
{
    names = this->intNames;
    values = this->intValues;

    return true;
}

bool EvalData::getReals(vector<string> &names,
                        vector<double> &values)
{
    names.assign(this->realNames.begin(), this->realNames.end());
    values.assign(this->realValues.begin(), this->realValues.end());

    return true;
}

bool EvalData::getStrings(vector<string> &names,
                          vector<string> &values)
{
    names.assign(this->stringNames.begin(), this->stringNames.end());

    return true;
}

void EvalData::clear()
{
    intNames.clear();
    intValues.clear();

    realNames.clear();
    realValues.clear();

    stringNames.clear();
    stringValues.clear();
}
