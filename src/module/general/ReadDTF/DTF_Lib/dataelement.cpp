/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF_Lib/dataelement.cpp
 * @brief contains definition of methods of class DTF_Lib::DataElement
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 10.10.2003
 * created
 */

#include "dataelement.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibDataElement, DataElement, "DTF_Lib::DataElement", INT_MAX);

DataElement::DataElement(){};

DataElement::DataElement(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
};

DataElement::~DataElement()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
};

bool DataElement::setValues(string name,
                            int numElements,
                            int dataType,
                            string units,
                            int topoType)
{
    this->name = name;
    this->numElements = numElements;
    this->dataType = dataType;
    this->units = units;
    this->topoType = topoType;

    return true;
}

string DataElement::getName()
{
    return this->name;
}

int DataElement::getNumElements()
{
    return this->numElements;
}

int DataElement::getDataType()
{
    return this->dataType;
}

string DataElement::getUnits()
{
    return this->units;
}

int DataElement::getTopoType()
{
    return this->topoType;
}

void DataElement::clear()
{
    name = "";
    numElements = 0;
    dataType = 0;
    units = "";
    topoType = 0;
}
