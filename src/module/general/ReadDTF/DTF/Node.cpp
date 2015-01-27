/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF/Node.cpp
 * @brief contains implementation of class DTF::Node
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 28.10.2003
 * created
 */
#include "Node.h"

using namespace DTF;

CLASSINFO_OBJ(ClassInfo_DTFNodes, Nodes, "DTF::Nodes", INT_MAX);

Nodes::Nodes()
{
    INC_OBJ_COUNT(getClassName());
}

Nodes::Nodes(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

Nodes::~Nodes()
{
    DEC_OBJ_COUNT(getClassName());
}

bool Nodes::init()
{
    coords.clear();
    return true;
}

bool Nodes::read(string fileName, int simNum, int zoneNum)
{
    clear();
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DTF_Lib::LibIF *libIF = (DTF_Lib::LibIF *)cm->getObject("DTF_Lib::LibIF");
    DTF_Lib::Zone *zone = (DTF_Lib::Zone *)(*libIF)("Zone");

    vector<double> xCoords, yCoords, zCoords;
    vector<int> blanking;

    if (zone->readGrid(simNum, zoneNum, 0, xCoords, yCoords, zCoords,
                       blanking))
    {
        Coords *coord = NULL;
        for (unsigned int i = 0; i < xCoords.size(); i++)
        {
            coord = new Coords();

            if (coord != NULL)
            {
                coord->x = xCoords[i];
                coord->y = yCoords[i];
                coord->z = zCoords[i];

                coords.insert(pair<int, Coords *>(i + 1, coord));
            }
            else
                return false;

            coord = NULL;
        }

        return true;
    }

    return false;
}

bool Nodes::getCoords(int nodeNum, Coords &value)
{
    map<int, Coords *>::iterator coordsIterator;

    coordsIterator = coords.find(nodeNum);

    if (coordsIterator != coords.end())
    {
        value = *coordsIterator->second;
        return true;
    }

    return false;
}

void Nodes::print()
{
    /*   cout << "(" << this->coords.x << ", " << this->coords.y << ", " <<
        this-> coords.z << ")" << endl;*/
}

int Nodes::getNumCoords()
{
    return coords.size();
}

void Nodes::clear()
{
    map<int, Coords *>::iterator coordsIterator;

    coordsIterator = coords.begin();

    Coords *coord;
    while (coordsIterator != coords.end())
    {
        coord = coordsIterator->second;

        delete coord;
        coord = NULL;

        ++coordsIterator;
    }

    coords.clear();
    coords.swap(coords);
}
