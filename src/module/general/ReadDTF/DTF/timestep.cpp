/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "timestep.h"

using namespace DTF;

CLASSINFO_OBJ(ClassInfo_DTFTimeStep, TimeStep, "DTF::TimeStep", INT_MAX);

TimeStep::TimeStep()
    : Tools::BaseObject()
{
    grid = NULL;
    typeWrapper.clear();
    INC_OBJ_COUNT(getClassName());
}

TimeStep::TimeStep(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    grid = NULL;
    typeWrapper.clear();

    for (int i = 1; i <= 6; i++)
        typeWrapper.insert(make_pair(i, i + 1));

    INC_OBJ_COUNT(getClassName());
}

TimeStep::~TimeStep()
{
    grid = NULL;
    typeWrapper.clear();

    DEC_OBJ_COUNT(getClassName());
}

bool TimeStep::setData(Sim *sim, int zoneNr)
{
    Zone *zone = NULL;
    int numZones = 0;
    Grid *tempGrid = NULL;

    clear();
    if (sim != NULL)
    {
        zone = sim->getZone(zoneNr);

        if (zone != NULL)
            if (extractGrid(zone))
                if (extractData(zone))
                    return true;
    }

    return false;
}

bool TimeStep::extractGrid(Zone *zone)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DTF::Nodes *nodes = NULL;

    if (grid == NULL)
        grid = (DTF::Grid *)cm->getObject("DTF::Grid");

    if (zone != NULL)
    {
        nodes = zone->getNodeCoords();

        if (nodes != NULL)
            if (fillCoordList(nodes, grid))
                if (fillCornerList(zone, grid))
                    if (fillElementList(zone, grid))
                        if (fillTypeList(zone, grid))
                            return true;
    }

    return false;
}

bool TimeStep::extractData(Zone *zone)
{
    if (zone != NULL)
    {
        DTF::ZD *zd = zone->getZD();

        if (zd != NULL)
        {
            dblData = zd->getData();
        }

        return true;
    }

    return false;
}

bool TimeStep::fillCoordList(Nodes *nodes, Grid *gridObj)
{
    vector<float> x, y, z;
    bool retVal = false;
    int numCoords = 0;

    if ((nodes != NULL) && (gridObj != NULL))
    {
        numCoords = nodes->getNumCoords();

        x.resize(numCoords, 0.0);
        y.resize(numCoords, 0.0);
        z.resize(numCoords, 0.0);

        for (int i = 1; i <= numCoords; i++)
        {
            Coords coords;

            if (nodes->getCoords(i, coords))
            {
                x[i - 1] = coords.x;
                y[i - 1] = coords.y;
                z[i - 1] = coords.z;
            }
            else
            {
                x.clear();
                y.clear();
                z.clear();

                return false;
            }
        }

        if (gridObj->setCoordList(x, y, z))
            retVal = true;

        x.clear();
        y.clear();
        z.clear();
    }

    return retVal;
}

bool TimeStep::fillCornerList(Zone *zone, Grid *gridObj)
{
    DTF::Cell *cell = NULL;
    int numCells = 0;
    vector<int> corners;
    vector<int> nodeNumbers;
    bool retVal = false;

    if ((zone != NULL) && (gridObj != NULL))
    {
        numCells = zone->getNumCells();

        for (int i = 1; i <= numCells; i++)
        {
            cell = zone->getCell(i);
            nodeNumbers = cell->getNodeNumbers();

            for (int j = 0; j < nodeNumbers.size(); j++)
                nodeNumbers[j] -= 1;

            corners.insert(corners.end(), nodeNumbers.begin(), nodeNumbers.end());
        }

        if (!corners.empty())
        {
            if (gridObj->setCornerList(corners))
                retVal = true;
        }

        corners.clear();
        nodeNumbers.clear();
    }

    return retVal;
}

bool TimeStep::fillElementList(Zone *zone, Grid *gridObj)
{
    int numCells = 0;
    int offset = 0;
    DTF::Cell *cell = NULL;
    vector<int> elementList;
    bool retVal = false;
    int numElements = 0;

    if ((zone != NULL) && (gridObj != NULL))
    {
        numCells = zone->getNumCells();
        elementList.resize(numCells, 0);

        if (numCells > 0)
        {
            for (int i = 0; i < numCells; i++)
            {
                cell = zone->getCell(i + 1);

                if (cell != NULL)
                {
                    elementList[i] = offset;

                    offset += cell->getNumNodes();
                }
                else
                {
                    elementList.clear();

                    return false;
                }
            }

            if (gridObj->setElementList(elementList))
                retVal = true;

            elementList.clear();
        }
    }

    return retVal;
}

bool TimeStep::fillTypeList(Zone *zone, Grid *gridObj)
{
    if (zone == NULL)
        return false;

    bool retVal = false;
    vector<int> cellTypes = zone->getCellTypes();

    if (!cellTypes.empty())
    {
        vector<int> typeList;
        typeList.resize(cellTypes.size(), 0);

        for (int i = 0; i < cellTypes.size(); i++)
            convertType(cellTypes[i], typeList[i]);

        if (gridObj->setTypeList(typeList))
            retVal = true;

        typeList.clear();
    }

    return retVal;
}

coDoUnstructuredGrid *TimeStep::getGrid(string name)
{
    if (this->grid == NULL)
        return NULL;

    coDoUnstructuredGrid *coGrid = new coDoUnstructuredGrid(name.c_str(),
                                                            grid->getNumElements(),
                                                            grid->getNumCorners(),
                                                            grid->getNumCoords(),
                                                            grid->getElementList(),
                                                            grid->getCornerList(),
                                                            grid->getCoordX(),
                                                            grid->getCoordY(),
                                                            grid->getCoordZ(),
                                                            grid->getTypeList());

    return coGrid;
}

coDoFloat *TimeStep::getData(string name, string dataName)
{
    if (this->dblData.empty())
        return NULL;

    map<string, vector<double> >::iterator dataIterator = dblData.find(dataName);

    if (dataIterator == dblData.end())
        return NULL;

    vector<double> zoneData = dataIterator->second;

    float *data = new float[zoneData.size()];

    for (int i = 0; i < zoneData.size(); i++)
        data[i] = (float)zoneData[i];

    coDoFloat *coData = new coDoFloat(name.c_str(),
                                      zoneData.size(),
                                      data);

    if (data != NULL)
    {
        delete[] data;
        data = NULL;
    }

    return coData;
}

bool TimeStep::convertType(int dtfType, int &coType)
{
    map<int, int>::iterator typeIterator = typeWrapper.find(dtfType);

    if (typeIterator == typeWrapper.end())
        return false;

    coType = typeIterator->second;

    return true;
}

bool TimeStep::hasData()
{
    if (!this->dblData.empty())
        return true;
    else
        return false;
}

void TimeStep::clear()
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();

    if (grid != NULL)
    {
        grid->clear();
        cm->deleteObject(grid->getID());

        grid = NULL;
    }

    this->dblData.clear();
}

void TimeStep::print()
{
    cout << "Grid:" << endl;

    if (grid != NULL)
        grid->print();

    int length = dblData.size();

    cout << "Data: length = " << length << endl;

    map<string, vector<double> >::iterator mapIterator = dblData.begin();
    vector<double> data;

    while (mapIterator != dblData.end())
    {
        data = mapIterator->second;

        cout << mapIterator->first << ": " << endl;
        cout << "------------" << endl;

        for (int i = 0; i < data.size(); i++)
            cout << data[i] << endl;

        data.clear();
        ++mapIterator;
    }
}
