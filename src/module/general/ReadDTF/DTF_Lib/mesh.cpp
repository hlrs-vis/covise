/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/mesh.cpp
 * @brief contains definition of methods for class DTF_Lib::Mesh
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "mesh.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibMesh, Mesh, "DTF_Lib::Mesh", 1);

Mesh::Mesh()
    : LibObject(){};

Mesh::Mesh(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

Mesh::~Mesh()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool Mesh::queryC2F(int simNum,
                    int zoneNum,
                    int cellNum,
                    vector<int> &facesPerCell)
{
    return implementMe();
}

bool Mesh::queryC2N(int simNum,
                    int zoneNum,
                    int cellNum,
                    vector<int> &nodesPerCell)
{
    dtf_int *c2n = new int;
    int numCell2Node = 0;
    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;
    int cellNumber = cellNum;

    nodesPerCell.clear();

    if ((numCell2Node = dtf_query_c2n(&handle, &simNumber, &zoneNumber,
                                      &cellNumber, c2n))
        != DTF_ERROR)
    {
        nodesPerCell.resize(numCell2Node, 0);

        for (int i = 0; i < numCell2Node; i++)
            nodesPerCell[i] = c2n[i];

        if (c2n != NULL)
            delete[] c2n;

        return true;
    }

    if (c2n != NULL)
        delete[] c2n;

    return false;
}

bool Mesh::queryC2Npos(int simNum,
                       int zoneNum,
                       int cellNum,
                       int &offset)
{
    return implementMe();
}

bool Mesh::queryF2C(int simNum,
                    int zoneNum,
                    int faceNum,
                    int &numF2C)
{
    return implementMe();
}

bool Mesh::queryF2N(int simNum,
                    int zoneNum,
                    int faceNum,
                    vector<int> &nodesPerFace)
{
    return implementMe();
}

bool Mesh::queryF2Npos(int simNum,
                       int zoneNum,
                       int faceNum,
                       int &offset)
{
    return implementMe();
}

bool Mesh::queryN2C(int simNum,
                    int zoneNum,
                    int nodeNum,
                    vector<int> &cellsPerNode)
{
    return implementMe();
}

bool Mesh::readC2F(int simNum,
                   int zoneNum,
                   int cellNum,
                   vector<int> &c2f)
{
    return implementMe();
}

bool Mesh::readC2N(int simNum,
                   int zoneNum,
                   int cellNum,
                   vector<int> &c2n)
{
    dtf_int numCell2Node = 0;
    dtf_int *cell2Node;
    bool retVal = false;
    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;
    int cellNumber = cellNum;

    if (queryC2N(simNumber, zoneNumber, cellNumber, c2n))
    {
        numCell2Node = c2n.size();

        if (numCell2Node > 0)
        {
            c2n.resize(numCell2Node, 0);
            cell2Node = new int[numCell2Node];

            if (dtf_read_c2n(&handle, &simNumber, &zoneNumber, &cellNumber,
                             cell2Node) != DTF_ERROR)
            {
                for (int i = 0; i < numCell2Node; i++)
                    c2n[i] = cell2Node[i];

                retVal = true;
            }

            if (cell2Node != NULL)
                delete[] cell2Node;
        }
    }

    return retVal;
}

bool Mesh::readF2C(int simNum,
                   int zoneNum,
                   int faceNum,
                   vector<int> &f2c)
{
    return implementMe();
}

bool Mesh::readF2N(int simNum,
                   int zoneNum,
                   int faceNum,
                   vector<int> &f2n)
{
    return implementMe();
}

bool Mesh::readN2C(int simNum,
                   int zoneNum,
                   int nodeNum,
                   vector<int> &n2c)
{
    return implementMe();
}

/** EOC */
