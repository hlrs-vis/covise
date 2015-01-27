/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ReadUnstructured
// Filip Sadlo 2008
// Computer Graphics Laboratory, ETH Zurich

#include "stdlib.h"
#include "stdio.h"

#include "ReadUnstructured.h"

#include "linalg.h"

#include "unstructured.h"
#include "unisys.h"

UniSys us = UniSys(NULL);

int main(int argc, char *argv[])
{
    myModule *application = new myModule(argc, argv);
    application->start(argc, argv);
    return 0;
}

void myModule::postInst()
{
}

void myModule::param(const char *, bool)
{
    // force min/max
    adaptMyParams();
}

int myModule::compute(const char *)
{
    // force min/max
    adaptMyParams();

    // system wrapper
    us = UniSys(this);

    // read Unstructured from file
    Unstructured *unst_in = new Unstructured(fileName->getValue());
    if (!unst_in)
    {
        return FAIL;
    }

    // adapt component selectors
    int nodeCompScal = -1;
    int nodeCompVec = -1;
    {
        int scalCnt = 0;
        int vecCnt = 0;
        char *nodeScalarLabels[1024];
        char *nodeVectorLabels[1024];
        int selectedScal = -1;
        int selectedVec = -1;
        for (int c = 0; c < unst_in->getNodeCompNb(); c++)
        {
            if (unst_in->getNodeCompVecLen(c) == 1)
            {

                if (scalarComponent->getValue() == scalCnt)
                {
                    unst_in->selectScalarNodeData(c);
                    selectedScal = scalCnt;
                    nodeCompScal = c;
                }

                nodeScalarLabels[scalCnt] = new char[256];
                strcpy(nodeScalarLabels[scalCnt], unst_in->getNodeCompLabel(c));

                scalCnt++;
            }
            else if (unst_in->getNodeCompVecLen(c) == 3)
            {

                if (vectorComponent->getValue() == vecCnt)
                {
                    unst_in->selectVectorNodeData(c);
                    selectedVec = vecCnt;
                    nodeCompVec = c;
                }

                nodeVectorLabels[vecCnt] = new char[256];
                strcpy(nodeVectorLabels[vecCnt], unst_in->getNodeCompLabel(c));

                vecCnt++;
            }
            else
            {
                us.error("unsupported vector length");
            }
        }

        if (scalCnt > 0)
        {
            scalarComponent->updateValue(scalCnt, nodeScalarLabels, selectedScal);
        }
        else
        {
            scalarComponent->setValue(1, defaultChoice, 0);
        }

        if (vecCnt > 0)
        {
            vectorComponent->updateValue(vecCnt, nodeVectorLabels, selectedVec);
        }
        else
        {
            vectorComponent->setValue(1, defaultChoice, 0);
        }

        for (int i = 0; i < scalCnt; i++)
        {
            delete[] nodeScalarLabels[i];
        }
        for (int i = 0; i < vecCnt; i++)
        {
            delete[] nodeVectorLabels[i];
        }
    }

    // generate output
    {
        coDoUnstructuredGrid *gridData = NULL;

        // ### TODO: check
        float *coordX = new float[unst_in->nNodes];
        float *coordY = new float[unst_in->nNodes];
        float *coordZ = new float[unst_in->nNodes];
        int *elemList = new int[unst_in->nCells];
        int *typeList = new int[unst_in->nCells];
        int *cornerList = new int[unst_in->getNodeListSize()];

        // coords
        for (int n = 0; n < unst_in->nNodes; n++)
        {
            vec3 p;
            unst_in->getCoords(n, p);
            coordX[n] = p[0];
            coordY[n] = p[1];
            coordZ[n] = p[2];
        }

        // cells
        int cornerListCnt = 0;
        for (int c = 0; c < unst_in->nCells; c++)
        {

            // nodes
            int *cellNodesAVS = unst_in->getCellNodesAVS(c);
            int cellNodes[8];
            Unstructured::nodeOrderAVStoCovise(unst_in->getCellType(c), cellNodesAVS, cellNodes);
            int nvert = nVertices[unst_in->getCellType(c)];
            memcpy(cornerList + cornerListCnt, cellNodes, nvert * sizeof(int));
            elemList[c] = cornerListCnt;
            cornerListCnt += nvert;

            // type
            switch (unst_in->getCellType(c))
            {
            case Unstructured::CELL_TET:
            {
                typeList[c] = TYPE_TETRAHEDER;
            }
            break;
            case Unstructured::CELL_PYR:
            {
                typeList[c] = TYPE_PYRAMID;
            }
            break;
            case Unstructured::CELL_PRISM:
            {
                typeList[c] = TYPE_PRISM;
            }
            break;
            case Unstructured::CELL_HEX:
            {
                typeList[c] = TYPE_HEXAEDER;
            }
            break;
            }
        }

        // data
        coDoFloat *scalarData = NULL;
        coDoVec3 *vectorData = NULL;
        {
            if (nodeCompScal >= 0)
            {
                scalarData = new coDoFloat(scalar->getObjName(), unst_in->nNodes);

                float *wp;
                scalarData->getAddress(&wp);
                for (int n = 0; n < unst_in->nNodes; n++)
                {
                    wp[n] = unst_in->getScalar(n, nodeCompScal);
                }
            }
            if (nodeCompVec >= 0)
            {
                vectorData = new coDoVec3(vector->getObjName(), unst_in->nNodes);

                float *up, *vp, *wp;
                vectorData->getAddresses(&up, &vp, &wp);
                for (int n = 0; n < unst_in->nNodes; n++)
                {
                    vec3 v;
                    unst_in->getVector3(n, nodeCompVec, v);
                    up[n] = v[0];
                    vp[n] = v[1];
                    wp[n] = v[2];
                }
            }
        }

        gridData = new coDoUnstructuredGrid(grid->getObjName(),
                                            unst_in->nCells,
                                            unst_in->getNodeListSize(),
                                            unst_in->nNodes,
                                            elemList,
                                            cornerList,
                                            coordX, coordY, coordZ,
                                            typeList);

        delete[] coordX;
        delete[] coordY;
        delete[] coordZ;
        delete[] elemList;
        delete[] typeList;
        delete[] cornerList;

        // assign data to ports
        grid->setCurrentObject(gridData);
        if (nodeCompScal >= 0)
            scalar->setCurrentObject(scalarData);
        if (nodeCompVec >= 0)
            vector->setCurrentObject(vectorData);
    }

    delete unst_in;

    return SUCCESS;
}
