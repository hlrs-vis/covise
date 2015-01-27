/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>

#include "PartitionHexaUSG.h"

#ifdef WIN32
using stdext::hash_set;
using stdext::hash_map;
#else
using __gnu_cxx::hash_set;
using __gnu_cxx::hash_map;
#endif

extern "C" {
#include "../../ihs/VISiT/lib/metis/Lib/metis.h"
}

PartitionHexaUSG::PartitionHexaUSG(int argc, char *argv[])
    : coSimpleModule(argc, argv, "USG Partitioning")
{
    p_gridIn = addInputPort("GridIn0", "UnstructuredGrid", "input mesh");
    p_gridOut = addOutputPort("GridOut0", "UnstructuredGrid", "output mesh");

    p_scalarIn[0] = addInputPort("DataIn0", "Float", "scalar input data");
    p_scalarOut[0] = addOutputPort("DataOut0", "Float", "scalar output data");

    p_scalarIn[1] = addInputPort("DataIn1", "Float", "scalar input data");
    p_scalarOut[1] = addOutputPort("DataOut1", "Float", "scalar output data");

    p_scalarIn[2] = addInputPort("DataIn2", "Float", "scalar input data");
    p_scalarOut[2] = addOutputPort("DataOut2", "Float", "scalar output data");

    p_scalarIn[3] = addInputPort("DataIn3", "Float", "scalar input data");
    p_scalarOut[3] = addOutputPort("DataOut3", "Float", "scalar output data");

    p_vectorIn[0] = addInputPort("DataIn4", "Vec3", "vector input data");
    p_vectorOut[0] = addOutputPort("DataOut4", "Vec3", "vector output data");

    p_vectorIn[1] = addInputPort("DataIn5", "Vec3", "vector input data");
    p_vectorOut[1] = addOutputPort("DataOut5", "Vec3", "vector output data");

    p_vectorIn[2] = addInputPort("DataIn6", "Vec3", "vector input data");
    p_vectorOut[2] = addOutputPort("DataOut6", "Vec3", "vector output data");

    p_vectorIn[3] = addInputPort("DataIn7", "Vec3", "vector input data");
    p_vectorOut[3] = addOutputPort("DataOut7", "Vec3", "vector output data");

    for (int index = 0; index < DATA_PORTS; index++)
    {
        p_scalarIn[index]->setRequired(false);
        p_vectorIn[index]->setRequired(false);
    }

    p_num = addInt32Param("numPart", "Number of Partitions");
    p_num->setValue(2);
}

int PartitionHexaUSG::compute(const char *)
{
    int numElem, numConn, numCoord;
    int *elem = NULL, *conn = NULL;
    float *x = NULL, *y = NULL, *z = NULL;

    // METIS elements
    int nn, numflag = 0, edgecut;
    int etype = 3; // hexahedra
    idxtype *elements = NULL, *epart = NULL, *npart = NULL;

    int nparts = p_num->getValue();

    if (nparts <= 1)
    {
        sendError("Too few partitions: %d", nparts);
        return FAIL;
    }

    coDoUnstructuredGrid *gridObj = dynamic_cast<coDoUnstructuredGrid *>(p_gridIn->getCurrentObject());

    if (!gridObj)
    {
        sendError("Received illegal type at port '%s'", p_gridIn->getName());
        return FAIL;
    }

    gridObj->getGridSize(&numElem, &numConn, &numCoord);
    gridObj->getAddresses(&elem, &conn, &x, &y, &z);

    // METIS only supports meshes with identical element types
    for (int index = 0; index < numElem - 1; index++)
        if (elem[index + 1] - elem[index] != 8)
        {
            sendError("Unstructured grid contains non-hexahedronal elements");
            return FAIL;
        }

    // create and partition METIS mesh
    elements = idxmalloc(8 * numElem, (char *)"elements");
    for (int index = 0; index < numElem; index++)
        for (int c = 0; c < 8; c++)
            elements[index * 8 + c] = conn[elem[index] + c];

    epart = idxmalloc(numElem, (char *)"epart");
    npart = idxmalloc(numConn, (char *)"npart");

    METIS_PartMeshDual(&numElem, &numCoord, elements, &etype, &numflag, &nparts,
                       &edgecut, epart, npart);

    for (int index = 0; index < nparts; index++)
        partitions.push_back(new Partition());

    // count elements in partitions
    // count unique coordinates in partitions
    for (int index = 0; index < numElem; index++)
    {
        partitions[epart[index]]->numElems++;
        for (int c = 0; c < 8; c++)
            partitions[epart[index]]->numCoords.insert(conn[elem[index] + c]);
    }

    int scalarDataBinding[DATA_PORTS];
    coDoFloat *scalarDataObject[DATA_PORTS];
    float *scalarData[DATA_PORTS];

    int vectorDataBinding[DATA_PORTS];
    coDoVec3 *vectorDataObject[DATA_PORTS];
    float *vectorDataU[DATA_PORTS];
    float *vectorDataV[DATA_PORTS];
    float *vectorDataW[DATA_PORTS];

    // determine data binding for input objects
    for (int index = 0; index < DATA_PORTS; index++)
    {
        scalarDataObject[index] = dynamic_cast<coDoFloat *>(p_scalarIn[index]->getCurrentObject());
        if (scalarDataObject[index])
        {
            scalarDataObject[index]->getAddress(&scalarData[index]);
            if (numElem == scalarDataObject[index]->getNumPoints())
                scalarDataBinding[index] = PER_ELEMENT;
            else if (scalarDataObject[index]->getNumPoints() == numCoord)
                scalarDataBinding[index] = PER_VERTEX;
            else
                scalarDataBinding[index] = NONE;
        }
        vectorDataObject[index] = dynamic_cast<coDoVec3 *>(p_vectorIn[index]->getCurrentObject());
        if (vectorDataObject[index])
        {
            vectorDataObject[index]->getAddresses(&vectorDataU[index],
                                                  &vectorDataV[index],
                                                  &vectorDataW[index]);
            if (numElem == vectorDataObject[index]->getNumPoints())
                vectorDataBinding[index] = PER_ELEMENT;
            else if (vectorDataObject[index]->getNumPoints() == numCoord)
                vectorDataBinding[index] = PER_VERTEX;
            else
                vectorDataBinding[index] = NONE;
        }
    }

    const char *gridName = p_gridOut->getObjName();
    const char *scalarDataName[DATA_PORTS];
    const char *vectorDataName[DATA_PORTS];

    for (int numData = 0; numData < DATA_PORTS; numData++)
    {
        scalarDataName[numData] = p_scalarOut[numData]->getObjName();
        vectorDataName[numData] = p_vectorOut[numData]->getObjName();
    }

    // create USG and data partitions
    char buf[128];
    for (int index = 0; index < nparts; index++)
    {

        Partition *p = partitions[index];
        snprintf(buf, 128, "%s_%d", gridName, index);
        p->createUSG(buf);
        for (int numData = 0; numData < DATA_PORTS; numData++)
        {
            snprintf(buf, 128, "%s_%d", scalarDataName[numData], index);
            p->createScalarData(numData, buf, scalarDataBinding[numData]);
            snprintf(buf, 128, "%s_%d", vectorDataName[numData], index);
            p->createVectorData(numData, buf, vectorDataBinding[numData]);
        }
    }

    // set elements in the COVISE objects
    // according to the partitions of the METIS mesh (epart)
    int *curElem = new int[nparts]; // current USG element per part
    int *curCoords = new int[nparts]; // current USG coord per part
    memset(curElem, 0, nparts * sizeof(int));
    memset(curCoords, 0, nparts * sizeof(int));

    for (int index = 0; index < numElem; index++)
    {
        int part = epart[index];
        Partition *p = partitions[part];
        p->elem[curElem[part]] = curElem[part] * 8;
        p->type[curElem[part]] = TYPE_HEXAEDER;

        // if there are per-element data elements, insert them
        for (int numData = 0; numData < DATA_PORTS; numData++)
        {
            if (scalarDataBinding[numData] == PER_ELEMENT)
                p->setScalarData(numData, curElem[part],
                                 scalarData[numData][index]);
            if (vectorDataBinding[numData] == PER_ELEMENT)
                p->setVectorData(numData, curElem[part],
                                 vectorDataU[numData][index],
                                 vectorDataV[numData][index],
                                 vectorDataW[numData][index]);
        }
        for (int c = 0; c < 8; c++)
        {
            int coordIndex = conn[elem[index] + c];

            int mapping = p->getPointMapping(coordIndex);
            if (mapping != -1)
            {
                // reuse the already existing coordinate
                coordIndex = mapping;
            }
            else
            {
                // coordinate is not yet in the USG partition, insert the point
                // (and optional per-vertex data)
                p->setPointMapping(coordIndex, curCoords[part]);
                p->setXYZ(curCoords[part], x[coordIndex], y[coordIndex], z[coordIndex]);
                for (int numData = 0; numData < DATA_PORTS; numData++)
                {
                    if (scalarDataBinding[numData] == PER_VERTEX)
                        p->setScalarData(numData, curCoords[part],
                                         scalarData[numData][coordIndex]);
                    if (vectorDataBinding[numData] == PER_VERTEX)
                        p->setVectorData(numData, curCoords[part],
                                         vectorDataU[numData][coordIndex],
                                         vectorDataV[numData][coordIndex],
                                         vectorDataW[numData][coordIndex]);
                }
                coordIndex = curCoords[part];
                curCoords[part]++;
            }
            p->conn[curElem[part] * 8 + c] = coordIndex;
        }
        curElem[part]++;
    }

    delete[] curElem;
    delete[] curCoords;

    coDistributedObject **gridObjects = new coDistributedObject *[nparts + 1];
    for (int index = 0; index < nparts; index++)
        gridObjects[index] = partitions[index]->gridObject;

    gridObjects[nparts] = NULL;
    p_gridOut->setCurrentObject(new coDoSet(gridName, gridObjects));
    delete[] gridObjects;

    for (int numData = 0; numData < DATA_PORTS; numData++)
    {
        coDistributedObject **dataObjects = new coDistributedObject *[nparts + 1];
        for (int index = 0; index < nparts; index++)
            dataObjects[index] = partitions[index]->scalarDataObject[numData];
        dataObjects[nparts] = NULL;

        p_scalarOut[numData]->setCurrentObject(new coDoSet(scalarDataName[numData], dataObjects));
        delete[] dataObjects;

        dataObjects = new coDistributedObject *[nparts + 1];
        for (int index = 0; index < nparts; index++)
            dataObjects[index] = partitions[index]->vectorDataObject[numData];
        dataObjects[nparts] = NULL;

        p_vectorOut[numData]->setCurrentObject(new coDoSet(vectorDataName[numData], dataObjects));
        delete[] dataObjects;
    }

    for (int index = 0; index < nparts; index++)
        delete partitions[index];
    partitions.clear();

    GKfree((void **)&elements, &epart, &npart, LTERM);

    return SUCCESS;
}

Partition::Partition()
    : numElems(0)
    , elem(NULL)
    , conn(NULL)
    , type(NULL)
    , gridObject(NULL)
{
    for (int index = 0; index < DATA_PORTS; index++)
    {
        scalarDataObject[index] = NULL;
        vectorDataObject[index] = NULL;

        scalarData[index] = NULL;
        vectorDataU[index] = NULL;
        vectorDataV[index] = NULL;
        vectorDataW[index] = NULL;
    }
}

void Partition::createUSG(const char *name)
{
    gridObject = new coDoUnstructuredGrid(name, numElems, numElems * 8,
                                          numCoords.size(), true);

    gridObject->getAddresses(&elem, &conn, &x, &y, &z);
    gridObject->getTypeList(&type);
}

void Partition::createScalarData(int index, const char *name, int dataBinding)
{
    if (dataBinding == PartitionHexaUSG::PER_VERTEX)
        scalarDataObject[index] = new coDoFloat(name, numCoords.size());
    else if (dataBinding == PartitionHexaUSG::PER_ELEMENT)
        scalarDataObject[index] = new coDoFloat(name, numElems);
    else
        scalarDataObject[index] = NULL;

    if (scalarDataObject[index])
        scalarDataObject[index]->getAddress(&scalarData[index]);
}

void Partition::createVectorData(int index, const char *name, int dataBinding)
{
    if (dataBinding == PartitionHexaUSG::PER_VERTEX)
        vectorDataObject[index] = new coDoVec3(name, numCoords.size());
    else if (dataBinding == PartitionHexaUSG::PER_ELEMENT)
        vectorDataObject[index] = new coDoVec3(name, numElems);
    else
        vectorDataObject[index] = NULL;

    if (vectorDataObject[index])
        vectorDataObject[index]->getAddresses(&vectorDataU[index],
                                              &vectorDataV[index],
                                              &vectorDataW[index]);
}

int Partition::getPointMapping(int point)
{
    hash_map<int, int>::iterator i = pointMap.find(point);
    if (i == pointMap.end())
        return -1;
    else
        return i->second;
}

void Partition::setPointMapping(int point, int map)
{
    pointMap[point] = map;
}

void Partition::setXYZ(int offset, float xp, float yp, float zp)
{
    x[offset] = xp;
    y[offset] = yp;
    z[offset] = zp;
}

void Partition::setScalarData(int index, int offset, float data)
{
    scalarData[index][offset] = data;
}

void Partition::setVectorData(int index, int offset, float u, float v, float w)
{
    vectorDataU[index][offset] = u;
    vectorDataV[index][offset] = v;
    vectorDataW[index][offset] = w;
}

MODULE_MAIN(Tools, PartitionHexaUSG)
