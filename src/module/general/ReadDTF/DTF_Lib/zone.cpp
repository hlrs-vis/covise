/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/zone.cpp
 * @brief contains definition of methods for class DTF_Lib::Zone
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "zone.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibZone, Zone, "DTF_Lib::Zone", 1);

Zone::Zone()
    : LibObject(){};

Zone::Zone(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

Zone::~Zone()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool Zone::isCartesian(int simNum,
                       int zoneNum,
                       bool &result)
{
    return implementMe();
}

bool Zone::isPoint(int simNum,
                   int zoneNum,
                   bool &result)
{
    return implementMe();
}

bool Zone::isPoly(int simNum,
                  int zoneNum,
                  bool &result)
{
    result = false;
    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;

    int status = dtf_query_ispoly(&handle, &simNumber, &zoneNumber);

    if (status != DTF_ERROR)
    {
        if (status == DTF_TRUE)
            result = true;
        else
            result = false;

        return true;
    }

    return false;
}

bool Zone::isStruct(int simNum,
                    int zoneNum,
                    bool &result)
{
    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;

    int status = dtf_query_isstruct(&handle, &simNumber, &zoneNumber);

    if (status != DTF_ERROR)
    {
        if (status == DTF_TRUE)
            result = true;
        else
            result = false;

        return true;
    }

    return false;
}

bool Zone::isUnstruct(int simNum,
                      int zoneNum,
                      bool &result)
{
    result = false;
    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;

    int status = dtf_query_isunstruct(&handle, &simNumber, &zoneNumber);

    if (status != DTF_ERROR)
    {
        if (status == DTF_TRUE)
            result = true;
        else
            result = false;

        return true;
    }

    return false;
}

bool Zone::hasBlankingData(int simNum,
                           int zoneNum,
                           bool &present)
{
    return implementMe();
}

bool Zone::queryCellGroup(int simNum,
                          int zoneNum,
                          int groupNum,
                          int &key)
{
    return implementMe();
}

bool Zone::queryCells(int simNum,
                      int zoneNum,
                      vector<int> &cellsOfType,
                      int &numCells)
{
    dtf_int nCellsOfType[DTF_NCELLTYPES];
    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;

    if ((numCells = dtf_query_cells(&handle,
                                    &simNumber,
                                    &zoneNumber,
                                    nCellsOfType))
        != DTF_ERROR)
    {
        for (int i = 0; i < DTF_NCELLTYPES; i++)
            cellsOfType.push_back(nCellsOfType[i]);

        return true;
    }

    return false;
}

bool Zone::queryCellType(int simNum,
                         int zoneNum,
                         int cellNum,
                         int &cellType)
{
    int type = 0;
    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;
    int cellNumber = cellNum;

    if ((type = dtf_query_celltype(&handle,
                                   &simNumber,
                                   &zoneNumber,
                                   &cellNumber)) != DTF_ERROR)
    {
        cellType = type;
        return true;
    }
    return false;
}

bool Zone::queryMinMax(int simNum,
                       int zoneNum,
                       vector<double> &minMax)
{
    return implementMe();
}

bool Zone::queryNumCells(int simNum,
                         int zoneNum,
                         int &numCells)
{
    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;

    if ((numCells = dtf_query_ncells(&handle,
                                     &simNumber,
                                     &zoneNumber)) != DTF_ERROR)
        return true;

    return false;
}

bool Zone::queryNumCellGroups(int simNum,
                              int zoneNum,
                              int &numCellGroups)
{
    return implementMe();
}

bool Zone::queryNumNodes(int simNum,
                         int zoneNum,
                         int &numNodes)
{
    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;

    if ((numNodes = dtf_query_nnodes(&handle, &simNumber, &zoneNumber))
        != DTF_ERROR)
        return true;

    return false;
}

bool Zone::queryZoneType(int simNum,
                         int zoneNum,
                         int &zoneType)
{
    return implementMe();
}

bool Zone::readBlanking(int simNum,
                        int zoneNum,
                        int nodeNum,
                        vector<int> &blanking)
{
    return implementMe();
}

bool Zone::readCellGroup(int simNum,
                         int zoneNum,
                         int groupNum,
                         vector<int> &cellNums)
{
    return implementMe();
}

bool Zone::readGrid(int simNum,
                    int zoneNum,
                    int nodeNum,
                    vector<double> &xCoords,
                    vector<double> &yCoords,
                    vector<double> &zCoords,
                    vector<int> &blanking)
{
    dtf_double *x, *y, *z;
    dtf_int *blank;
    int numNodes = 0;
    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;
    int nodeNumber = nodeNum;

    if (nodeNumber >= 1)
    {
        x = new double;
        y = new double;
        z = new double;
        blank = new int;

        numNodes = 1;
    }
    else
    {
        if (!queryNumNodes(simNumber, zoneNumber, numNodes))
        {
            delete x;
            delete y;
            delete z;
            delete blank;
            return false;
        }

        x = new double[numNodes];
        y = new double[numNodes];
        z = new double[numNodes];
        blank = new int[numNodes];
    }

    if (numNodes > 0)
    {
        if (dtf_read_grid_d(&handle, &simNumber, &zoneNumber, &nodeNumber,
                            x, y, z, blank) != DTF_ERROR)
        {
            xCoords.clear();
            yCoords.clear();
            zCoords.clear();
            blanking.clear();
            xCoords.resize(numNodes);
            yCoords.resize(numNodes);
            zCoords.resize(numNodes);
            blanking.resize(numNodes);

            for (int i = 0; i < numNodes; i++)
            {
                xCoords[i] = x[i];
                yCoords[i] = y[i];
                zCoords[i] = z[i];
                blanking[i] = blank[i];
            }
        }

        if (x != NULL)
            delete[] x;
        if (y != NULL)
            delete[] y;
        if (z != NULL)
            delete[] z;
        if (blank != NULL)
            delete[] blank;

        return true;
    }

    return false;
}

bool Zone::readVirtualCellNums(int simNum,
                               int zoneNum,
                               vector<int> &cellNums)
{
    return implementMe();
}

bool Zone::readVirtualNodeNums(int simNum,
                               int zoneNum,
                               vector<int> &nodeNums)
{
    return implementMe();
}

bool Zone::init()
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();

    addChildFunc("BcRecords", cm->getObject("DTF_Lib::BcRecords"));
    addChildFunc("BcVc", cm->getObject("DTF_Lib::BcVc"));
    addChildFunc("Mesh", cm->getObject("DTF_Lib::Mesh"));
    addChildFunc("PolyZone", cm->getObject("DTF_Lib::PolyZone"));
    addChildFunc("StructZone", cm->getObject("DTF_Lib::StructZone"));
    addChildFunc("Surface", cm->getObject("DTF_Lib::Surface"));
    addChildFunc("UnstrZone", cm->getObject("DTF_Lib::UnstrZone"));
    addChildFunc("VcRecords", cm->getObject("DTF_Lib::VcRecords"));
    addChildFunc("VirtualZone", cm->getObject("DTF_Lib::VirtualZone"));
    addChildFunc("Volume", cm->getObject("DTF_Lib::Volume"));
    addChildFunc("ZoneData", cm->getObject("DTF_Lib::ZoneData"));
    addChildFunc("ZoneInterface", cm->getObject("DTF_Lib::ZoneInterface"));

    return true;
}

Tools::BaseObject *Zone::operator()(string className)
{
    return getChildFunc(className);
}

bool Zone::setFileHandle(int handle)
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
