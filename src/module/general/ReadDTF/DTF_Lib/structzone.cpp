/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/structzone.cpp
 * @brief contains definition of methods for class DTF_Lib::StructZone
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "structzone.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibStructZone, StructZone, "DTF_Lib::StructZone", 1);

StructZone::StructZone()
    : LibObject(){};
StructZone::StructZone(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

StructZone::~StructZone()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool StructZone::queryDims(int simNum,
                           int zoneNum,
                           vector<int> &dims)
{
    dtf_int dim[3];
    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;

    if (dtf_query_dims(&handle, &simNumber, &zoneNumber, dim) != DTF_ERROR)
    {
        dims.resize(3, 0);

        dims[0] = dim[0];
        dims[1] = dim[1];
        dims[2] = dim[2];

        return true;
    }

    return false;
}

bool StructZone::init()
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();

    addChildFunc("Block", cm->getObject("DTF_Lib::Block"));
    addChildFunc("Patch", cm->getObject("DTF_Lib::Patch"));

    return true;
}

Tools::BaseObject *StructZone::operator()(string className)
{
    return getChildFunc(className);
}

bool StructZone::setFileHandle(int handle)
{
    if (handle >= 0)
    {
        this->fileHandle = handle;

        map<string, Tools::BaseObject *>::iterator funcIterator
            = this->childFuncs.begin();

        while (funcIterator != childFuncs.end())
        {
            LibObject *obj = (LibObject *)funcIterator->second;

            obj->setFileHandle(this->fileHandle);

            ++funcIterator;
        }

        return true;
    }

    return false;
}

/** EOC */
