/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: Filter program in COVISE API                           ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

// this includes our own class's headers
#include "SelectIdx.h"
#include <util/coRestraint.h>
#include <values.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
SelectIdx::SelectIdx()
    : coModule("SelectIdx")
{
    // Parameters

    // create the parameter
    int i;
    char buffer[64];
    for (i = 0; i < NUM_PORTS; i++)
    {
        sprintf(buffer, "In_%d", i);
        p_in[i] = addInputPort(buffer, "DO_Rectilinar_Grid|coDoUnstructuredGrid|coDoFloat|coDoVec3", "Input fields");
        sprintf(buffer, "OUT_%d", i);
        p_out[i] = addOutputPort(buffer, "coDoUnstructuredGrid|coDoFloat|coDoVec3", "Output fields");
        p_in[i]->setRequired(0);
        p_out[i]->setDependency(p_in[i]);
    }
    p_index = addInputPort("index", "coDoIntArr", "Selection index field");

    // parameter
    p_selection = addStringParam("select", "Value selection");
    sprintf(buffer, "%d-%d", -MAXINT, MAXINT);
    p_selection->setValue(buffer);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int SelectIdx::compute()
{
    int i;
    char buffer[128];

    /// get the index length and field
    int numElem;
    char *selArr;
    if (getIndexField(numElem, selArr) == FAIL)
        return STOP_PIPELINE;

    /// show the results of the selection
    int numSelected = 0;
    for (i = 0; i < numElem; i++)
        if (selArr[i])
            numSelected++;
    sprintf(buffer, "Selected %d elements of %d", numSelected, numElem);
    sendInfo(buffer);

    /// now select at all points
    for (i = 0; i < NUM_PORTS; i++)
    {
        coDistributedObject *obj = p_in[i]->getCurrentObject();
        if (obj)
        {
            const char *outName = p_out[i]->getObjName();
            coDistributedObject *resObj
                = selectObj(obj, outName, numElem, numSelected, selArr);
            p_out[i]->setCurrentObject(resObj);
        }
    }

    delete[] selArr;

    return CONTINUE_PIPELINE;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  get the index field
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int SelectIdx::getIndexField(int &numElem, char *&selArr)
{
    int i;

    /// create selection array from index and selection
    coDoIntArr *idx = dynamic_cast<coDoIntArr *>(p_index->getCurrentObject());
    if (!idx)
    {
        sendError("Could not open Object at index field port");
        return STOP_PIPELINE;
    }
    numElem = idx->get_dim(0); // get 1st field size
    selArr = new char[numElem];
    int *idxField = idx->getAddress();

    // create a restarint construct
    coRestraint restraint;
    restraint.add(p_selection->getValue());

    // mark all selected elements true, all others false
    for (i = 0; i < numElem; i++)
        if (restraint(idxField[i]))
            selArr[i] = 1;
        else
            selArr[i] = 0;

    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  What's left to do for the Main program:
// ++++                                    create the module and start it
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coDistributedObject *SelectIdx::selectObj(coDistributedObject *inObj,
                                          const char *outName,
                                          int numElem, int numSelected,
                                          const char *selArr)
{

    int i;
    char buffer[128];

    /////////////////// SET ?
    {
        coDoSet *in_set = dynamic_cast<coDoSet *>(inObj);
        if (in_set)
        {
            int i, numSetElem;
            coDistributedObject *const *inObject = in_set->getAllElements(&numSetElem);
            coDistributedObject **outObject = new coDistributedObject *[numSetElem + 1];
            for (i = 0; i < numSetElem; i++)
            {
                sprintf(buffer, "%s_%d", outName, i);
                outObject[i] = selectObj(inObject[i], buffer, numElem, numSelected, selArr);
            }
            outObject[numSetElem] = NULL;
            coDoSet *resObj = new coDoSet(outName, outObject);
            resObj->copyAllAttributes(inObj);
            return resObj;
        }
    }

    /////////////////// USG Scalar Data ?
    {
        coDoFloat *usg_scalar = dynamic_cast<coDoFloat *>(inObj);

        if (usg_scalar)
        {
            int size = usg_scalar->getNumPoints();
            if (size != numElem)
            {
                sendError("Size mismatch");
                return NULL;
            }
            float *inData, *outData;
            usg_scalar->getAddress(&inData);
            coDoFloat *resObj = new coDoFloat(outName, numSelected);
            resObj->copyAllAttributes(inObj);
            resObj->getAddress(&outData);
            for (i = 0; i < numElem; i++)
            {
                if (selArr[i])
                {
                    *outData = inData[i];
                    outData++;
                }
            }
            return resObj;
        }
    }

    /////////////////// Struct Scalar Data ? (use scope to hide variables)
    {
        coDoFloat *str_scalar = dynamic_cast<coDoFloat *>(inObj);

        if (str_scalar)
        {
            int sx, sy, sz, size;
            str_scalar->getGridSize(&sx, &sy, &sz);
            size = sx * sy * sz;
            if (size != numElem)
            {
                sendError("Size mismatch");
                return NULL;
            }
            float *inData, *outData;
            str_scalar->getAddress(&inData);
            coDoFloat *resObj = new coDoFloat(outName, numSelected);
            resObj->copyAllAttributes(inObj);
            resObj->getAddress(&outData);
            for (i = 0; i < numElem; i++)
            {
                if (selArr[i])
                {
                    *outData = inData[i];
                    outData++;
                }
            }
            return resObj;
        }
    }

    /////////////////// Struct Vector Data ?
    {
        coDoVec3 *str_vector = dynamic_cast<coDoVec3 *>(inObj);

        if (str_vector)
        {
            int sx, sy, sz, size;
            str_vector->getGridSize(&sx, &sy, &sz);
            size = sx * sy * sz;
            if (size != numElem)
            {
                sendError("Size mismatch");
                return NULL;
            }
            float *inU, *inV, *inW, *outU, *outV, *outW;
            str_vector->getAddresses(&inU, &inV, &inW);
            coDoVec3 *resObj = new coDoVec3(outName, numSelected);
            resObj->copyAllAttributes(inObj);
            resObj->getAddresses(&outU, &outV, &outW);
            for (i = 0; i < numElem; i++)
            {
                if (selArr[i])
                {
                    *outU = inU[i];
                    outU++;
                    *outV = inV[i];
                    outV++;
                    *outW = inW[i];
                    outW++;
                }
            }
            return resObj;
        }
    }

    /////////////////// Rectilinear Grid : index PER CELL so far
    {
        coDoRectilinearGrid *rectGrd = dynamic_cast<coDoRectilinearGrid *>(inObj);

        if (rectGrd)
        {
            int sx, sy, sz, size;
            rectGrd->getGridSize(&sx, &sy, &sz);

            float *rx, *ry, *rz;
            rectGrd->getAddresses(&rx, &ry, &rz);

            size = (sx - 1) * (sy - 1) * (sz - 1); // idx per cell
            if (size != numElem)
            {
                sendError("Size mismatch");
                return NULL;
            }

            // vertex coordinate translation table
            int *vertTrans = new int[sx * sy * sz];
            for (i = 0; i < sx * sy * sz; i++)
                vertTrans[i] = -1;
            int numVert = 0;

            int ix, iy, iz;
            int cellNo = 0;

// #define VERT(i,j,k) ( (i) + (j)*sx + (k)*sx*sy )
#define VERT(k, j, i) ((i) + (j)*sx + (k)*sx * sy)

            for (iz = 0; iz < sz - 1; iz++)
                for (iy = 0; iy < sy - 1; iy++)
                    for (ix = 0; ix < sx - 1; ix++)
                    {
                        int vx, vy, vz;
                        if (selArr[cellNo])
                        {
                            for (vz = 0; vz < 2; vz++)
                                for (vy = 0; vy < 2; vy++)
                                    for (vx = 0; vx < 2; vx++)
                                    {
                                        int vertexID = VERT(ix + vx, iy + vy, iz + vz);
                                        if (vertTrans[vertexID] == -1)
                                        {
                                            vertTrans[vertexID] = numVert;
                                            numVert++;
                                        }
                                    }
                        }
                        cellNo++;
                    }

            coDoUnstructuredGrid *resObj = new coDoUnstructuredGrid(outName, numSelected, 8 * numSelected, numVert, 1);
            resObj->copyAllAttributes(inObj);

            int *elemList, *connList, *typeList;
            float *xc, *yc, *zc;
            resObj->getAddresses(&elemList, &connList, &xc, &yc, &zc);
            resObj->getTypeList(&typeList);

            cellNo = 0;
            int actCell = 0;

            for (iz = 0; iz < sz - 1; iz++)
                for (iy = 0; iy < sy - 1; iy++)
                    for (ix = 0; ix < sx - 1; ix++)
                    {
                        if (selArr[cellNo])
                        {
                            *elemList = actCell * 8;
                            elemList++;
                            *typeList = TYPE_HEXAEDER;
                            typeList++;
                            actCell++;

                            *connList = vertTrans[VERT(ix, iy, iz)];
                            connList++;
                            *connList = vertTrans[VERT(ix + 1, iy, iz)];
                            connList++;
                            *connList = vertTrans[VERT(ix + 1, iy + 1, iz)];
                            connList++;
                            *connList = vertTrans[VERT(ix, iy + 1, iz)];
                            connList++;

                            *connList = vertTrans[VERT(ix, iy, iz + 1)];
                            connList++;
                            *connList = vertTrans[VERT(ix + 1, iy, iz + 1)];
                            connList++;
                            *connList = vertTrans[VERT(ix + 1, iy + 1, iz + 1)];
                            connList++;
                            *connList = vertTrans[VERT(ix, iy + 1, iz + 1)];
                            connList++;
                        }
                        cellNo++;
                    }

            cellNo = 0;
            for (iz = 0; iz < sz; iz++)
                for (iy = 0; iy < sy; iy++)
                    for (ix = 0; ix < sx; ix++)
                    {
                        int vertexID = VERT(ix, iy, iz);
                        int trafo = vertTrans[vertexID];
                        if (trafo > -1)
                        {
                            xc[trafo] = rx[ix];
                            yc[trafo] = ry[iy];
                            zc[trafo] = rz[iz];
                        }
                        cellNo++;
                    }
            return resObj;
        }
    }
#undef VERT

    /////////////////// unknown so far
    sprintf(buffer, "Type `%s` not yet implemented", inObj->getType());
    sendError(buffer);

    return NULL;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  What's left to do for the Main program:
// ++++                                    create the module and start it
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int main(int argc, char *argv[])

{
    // create the module
    SelectIdx *application = new SelectIdx;

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
