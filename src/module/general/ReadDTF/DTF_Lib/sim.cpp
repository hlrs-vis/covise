/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/sim.cpp
 * @brief contains definition of methods for class DTF_Lib::Sim
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "sim.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibSim, Sim, "DTF_Lib::Sim", 1);

Sim::Sim()
    : LibObject(){};
Sim::Sim(string className, int objectID)
    : LibObject(className, objectID)
{
    this->fileHandle = 0;
    INC_OBJ_COUNT(getClassName());
}

Sim::~Sim()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool Sim::queryNumSDs(int simNum,
                      int &numSDs)
{
    int handle = this->fileHandle;
    int simNumber = simNum;

    if ((numSDs = dtf_query_nsds(&handle, &simNumber)) != DTF_ERROR)
        return true;

    return false;
}

bool Sim::queryNumZones(int simNum,
                        int &numZones)
{
    int handle = this->fileHandle;
    int simNumber = simNum;

    if ((numZones = dtf_query_nzones(&handle, &simNumber)) != DTF_ERROR)
        return true;

    return false;
}

bool Sim::queryMinMax(int simNum,
                      vector<double> &minMax)
{
    return implementMe();
}

bool Sim::querySimDescr(int simNum,
                        string &description)
{
    dtf_string descr;
    int handle = this->fileHandle;
    int simNumber = simNum;

    if (dtf_query_simdescr(&handle, &simNumber, descr) != DTF_ERROR)
    {
        description = descr;

        return true;
    }

    return false;
}

bool Sim::init()
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();

    addChildFunc("SimData", cm->getObject("DTF_Lib::SimData"));

    return true;
}

bool Sim::setFileHandle(int handle)
{
    if (handle >= 0)
    {
        this->fileHandle = handle;

        map<string, Tools::BaseObject *>::iterator funcIterator
            = this->childFuncs.begin();

        while (funcIterator != this->childFuncs.end())
        {
            LibObject *obj = (LibObject *)funcIterator->second;

            obj->setFileHandle(this->fileHandle);

            ++funcIterator;
        }

        return true;
    }

    return false;
}

Tools::BaseObject *Sim::operator()(string className)
{
    return getChildFunc(className);
}

/** EOC */
