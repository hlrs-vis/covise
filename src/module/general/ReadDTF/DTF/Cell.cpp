/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF/Cell.cpp
 * @brief contains implementation of class DTF::Cell
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 28.10.2003
 * created
 */

#include "Cell.h"

using namespace DTF;

CLASSINFO_OBJ(ClassInfo_DTFCell, Cell, "DTF::Cell", INT_MAX);

Cell::Cell()
    : Tools::BaseObject()
{
    INC_OBJ_COUNT(getClassName());
}

Cell::Cell(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    nodes.clear();

    INC_OBJ_COUNT(getClassName());
}

Cell::~Cell()
{
    nodes.clear();
    DEC_OBJ_COUNT(getClassName());
}

bool Cell::init()
{
    nodes.clear();

    return true;
}

bool Cell::read(string fileName, int simNum, int zoneNum, int cellNum)
{
    clear();

    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DTF_Lib::Mesh *mesh = (DTF_Lib::Mesh *)cm->getObject("DTF_Lib::Mesh");
    DTF_Lib::Zone *zone = (DTF_Lib::Zone *)cm->getObject("DTF_Lib::Zone");

    vector<int> c2n;
    int type = 0;
    nodes.clear();

    if (mesh->readC2N(simNum, zoneNum, cellNum, c2n))
        nodes = c2n;

    if (zone->queryCellType(simNum, zoneNum, cellNum, type))
        this->cellType = type;

    return true;
}

int Cell::getNumNodes()
{
    return this->nodes.size();
}

void Cell::clear()
{
    this->nodes.clear();
    this->cellType = 0;
}

vector<int> Cell::getNodeNumbers()
{
    return this->nodes;
}

void Cell::print()
{
    cout << " size cell->node = " << this->nodes.size() << endl;

    /*   for ( int i = 0; i < nodes.size(); i++ )
        cout << " \t\t\t\tnode #" << nodes[i] << endl;*/
}
