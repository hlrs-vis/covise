/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF/Zone.cpp
 * @brief contains implementation of class DTF::Zone
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 28.10.2003
 * created
 */
#include "Zone.h"

using namespace DTF;

CLASSINFO_OBJ(ClassInfo_DTFZone, Zone, "DTF::Zone", INT_MAX);

Zone::Zone()
{
    INC_OBJ_COUNT(getClassName());
}

Zone::Zone(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    this->structZone = false;
    this->unstructZone = false;
    this->polyZone = false;

    this->dims.clear();
    this->cells.clear();
    this->numCellsOfType.clear();

    INC_OBJ_COUNT(getClassName());
}

Zone::~Zone()
{
    cells.clear();
    numCellsOfType.clear();
    dims.clear();

    DEC_OBJ_COUNT(getClassName());
}

bool Zone::init()
{
    this->structZone = false;
    this->unstructZone = false;
    this->polyZone = false;

    this->dims.clear();
    this->cells.clear();
    this->numCellsOfType.clear();

    this->nodes = NULL;
    this->zd = NULL;

    return true;
}

bool Zone::read(string fileName, int simNum, int zoneNum)
{
    clear();
    bool retValue = true;

    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DTF_Lib::LibIF *libIF = (DTF_Lib::LibIF *)cm->getObject("DTF_Lib::LibIF");
    DTF_Lib::Zone *zone = (DTF_Lib::Zone *)(*libIF)("Zone");
    DTF_Lib::VirtualZone *vZone = (DTF_Lib::VirtualZone *)(*zone)("VirtualZone");

    vector<int> cellsOfType;
    int numCells;

    if (zone->queryCells(simNum, zoneNum, this->numCellsOfType, numCells))
        for (int i = 1; i <= numCells; i++)
        {
            Cell *cell = (DTF::Cell *)cm->getObject("DTF::Cell");

            if (cell->read(fileName, simNum, zoneNum, i))
                this->cells.insert(pair<int, Cell *>(i, cell));
            else
                cm->deleteObject(cell->getID());

            if (cell != NULL)
                cell = NULL;
        }
    else
        retValue = false;

    if (!readNodes(fileName, simNum, zoneNum, zone))
        retValue = false;

    if (!readZD(simNum, vZone))
        retValue = false;

    return retValue;
}

int Zone::getNumCells()
{
    return this->cells.size();
}

int Zone::getNumNodes()
{
    if (nodes != NULL)
        return nodes->getNumCoords();

    return 0;
}

Cell *Zone::getCell(int cellNum)
{
    Cell *cell = NULL;
    map<int, Cell *>::iterator cellIterator = cells.find(cellNum);

    if (cellIterator != cells.end())
        cell = cellIterator->second;

    return cell;
}

vector<int> Zone::getCellTypes()
{
    vector<int> retValue;

    for (unsigned int cellType = 0; cellType < numCellsOfType.size(); cellType++)
        if (numCellsOfType[cellType] > 0)
            for (int i = 0; i < numCellsOfType[cellType]; i++)
                retValue.push_back(cellType + 1);

    return retValue;
}

vector<int> Zone::getNumCellsOfType()
{
    return this->numCellsOfType;
}

bool Zone::isStruct()
{
    return this->structZone;
}

bool Zone::isUnstruct()
{
    return this->unstructZone;
}

bool Zone::isPoly()
{
    return this->polyZone;
}

vector<int> Zone::getDims()
{
    return this->dims;
}

void Zone::print()
{
    string cellType;
    map<int, Cell *>::iterator cellIterator;

    if (this->isStruct())
    {
        cout << "structured" << endl;
    }
    else if (this->isUnstruct())
    {
        cout << "unstructured" << endl;
    }
    else if (this->isPoly())
    {
        cout << "poly" << endl;
    }

    if (!dims.empty())
        cout << "\t\t" << dims[0] * dims[1] * dims[2] << " vertices ("
             << dims[0] << "x" << dims[1] << "x" << dims[2] << ")" << endl;

    cout << "\t\t" << cells.size() << " cells" << endl;

    DTF_Lib::CellTypes *ct = Tools::Singleton<DTF_Lib::CellTypes>::getInstance();

    cout << "\t\tcells: " << cells.size() << endl;

    for (unsigned int i = 0; i < numCellsOfType.size(); i++)
        if (numCellsOfType[i] > 0)
        {
            ct->toString(i + 1, cellType);
            cout << "\t\t\t" << numCellsOfType[i] << " cells of type "
                 << cellType << endl;
        }

    cellIterator = cells.begin();

    while (cellIterator != cells.end())
    {
        Cell *cell = cellIterator->second;

        if (cell != NULL)
        {
            cout << "\t\t\tcell #" << cellIterator->first << ": ";

            cell->print();

            vector<int> nodeNumbers = cell->getNodeNumbers();

            for (unsigned int i = 0; i < nodeNumbers.size(); i++)
            {
                Coords coords;
                nodes->getCoords(nodeNumbers[i], coords);

                cout << "\t\t\t\tnode #" << nodeNumbers[i] << ": (" << coords.x << ", " << coords.y << ", " << coords.z << ")"
                     << endl;
            }
        }

        ++cellIterator;
    }

    if (zd != NULL)
        zd->print();
}

void Zone::clear()
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    map<int, Cell *>::iterator cellIterator = cells.begin();

    while (cellIterator != cells.end())
    {
        Cell *cell = cellIterator->second;

        if (cell != NULL)
        {
            cell->clear();
            cm->deleteObject(cell->getID());
            cell = NULL;
        }

        ++cellIterator;
    }

    if (nodes != NULL)
    {
        nodes->clear();
        cm->deleteObject(nodes->getID());
        nodes = NULL;
    }

    if (zd != NULL)
    {
        zd->clear();
        cm->deleteObject(zd->getID());
        zd = NULL;
    }
}

Nodes *Zone::getNodeCoords()
{
    return this->nodes;
}

bool Zone::readNodes(string fileName, int simNum, int zoneNum, DTF_Lib::Zone *zone)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();

    int numNodes = 0;

    if (zone->queryNumNodes(simNum, zoneNum, numNodes))
    {
        Nodes *nodeStore = (DTF::Nodes *)cm->getObject("DTF::Nodes");

        if (nodeStore->read(fileName, simNum, zoneNum))
            nodes = nodeStore;
        else
        {
            cm->deleteObject(nodeStore->getID());
            return false;
        }

        return true;
    }

    return false;
}

bool Zone::readZD(int simNum, DTF_Lib::VirtualZone *vZone)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();

    int numZD = 0;

    if (vZone->queryNumVZDs(simNum, numZD))
    {
        cout << "numZDs: " << numZD << endl;
        ZD *tempZD = (DTF::ZD *)cm->getObject("DTF::ZD");

        if (tempZD->read(simNum))
            zd = tempZD;
        else
            cm->deleteObject(tempZD->getID());
    }

    return true;
}

ZD *Zone::getZD()
{
    return this->zd;
}
