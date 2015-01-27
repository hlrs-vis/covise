/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "readcomsol.h"

#include "do/coDoUnstructuredGrid.h"
#include "do/coDoIntArr.h"
#include "do/coDoSet.h"
#include "do/coDoData.h"

void ReadCOMSOL::createOutputPorts()
{
    _portOutMesh = addOutputPort("mesh", "UnstructuredGrid", "finite element mesh");
    _portOutMeshDomain = addOutputPort("meshdomain", "IntArr", "domain numbers of elements");
    if (_portOutVec != 0)
        delete[] _portOutVec;
    _noPortsVec = 3;
    _portOutVec = new coOutputPort *[_noPortsVec];
    for (unsigned char i = 0; i < _noPortsVec; i++)
    {
        const std::string portName = getListName("VectorData", i, false);
        _portOutVec[i] = addOutputPort(portName.c_str(), "Vec3", "vector data");
    }
    if (_portOutScal != 0)
        delete[] _portOutScal;
    _noPortsScal = 3;
    _portOutScal = new coOutputPort *[_noPortsScal];
    for (unsigned char i = 0; i < _noPortsScal; i++)
    {
        const std::string portName = getListName("ScalarData", i, false);
        _portOutScal[i] = addOutputPort(portName.c_str(), "Float", "scalar data");
    }
}

void ReadCOMSOL::writeMesh(MeshData *mesh)
{
    sendInfo("Write finite element mesh to output port.");
    const TimeSteps *timeSteps = _physics->getTimeSteps(false);
    const unsigned int noTimeSteps = timeSteps->getNoTimeSteps();
    coDistributedObject **allMeshes = new coDistributedObject *[noTimeSteps + 1];
    coDistributedObject **allDomains = new coDistributedObject *[noTimeSteps + 1];
    allMeshes[noTimeSteps] = NULL;
    allDomains[noTimeSteps] = NULL;
    const std::string meshPortName = _portOutMesh->getObjName();
    const std::string domainPortName = _portOutMeshDomain->getObjName();
    coObjInfo infoMeshPort(meshPortName);
    coObjInfo infoDomainPort(domainPortName);
    for (unsigned int t = 0; t < noTimeSteps; t++)
    {
        const std::string meshNameTimeStep = getListName(meshPortName, t, false);
        const std::string domainNameTimeStep = getListName(domainPortName, t, false);
        const unsigned long noTetra = mesh->getNoTetra4(t);
        const unsigned long noNodes = mesh->getNoNodes(t);
        coDoUnstructuredGrid *unstructuredGrid = new coDoUnstructuredGrid(meshNameTimeStep.c_str(), noTetra, noTetra * 4, noNodes, true);
        const int size = noTetra;
        coDoIntArr *domainNo = new coDoIntArr(domainNameTimeStep.c_str(), 1, &size);
        allMeshes[t] = unstructuredGrid;
        allDomains[t] = domainNo;
        int *elements = NULL;
        int *connectivity = NULL;
        float *x = NULL;
        float *y = NULL;
        float *z = NULL;
        unstructuredGrid->getAddresses(&elements, &connectivity, &x, &y, &z);
        int *typeList = NULL;
        unstructuredGrid->getTypeList(&typeList);
        int *domains = domainNo->getAddress();
        for (unsigned long i = 0; i < noTetra; i++)
        {
            typeList[i] = TYPE_TETRAHEDER;
            elements[i] = i * 4;
        }
        if (noTetra > 0)
        {
            for (unsigned long i = 0; i < noTetra; i++)
            {
                Tetra4 element = mesh->getTetra4(t, i);
                connectivity[i * 4] = element.node1;
                connectivity[i * 4 + 1] = element.node2;
                connectivity[i * 4 + 2] = element.node3;
                connectivity[i * 4 + 3] = element.node4;
                domains[i] = element.domainNo;
            }
        }
        for (unsigned long i = 0; i < noNodes; i++)
        {
            CartesianCoordinates node = mesh->getNode(t, i);
            x[i] = node.x;
            y[i] = node.y;
            z[i] = node.z;
        }
    }
    coDoSet *meshes = new coDoSet(infoMeshPort, allMeshes);
    coDoSet *domains = new coDoSet(infoDomainPort, allDomains);
    if (noTimeSteps > 1)
    {
        const std::string attribute = "1 " + noTimeSteps;
        meshes->addAttribute("TIMESTEP", attribute.c_str());
        domains->addAttribute("TIMESTEP", attribute.c_str());
    }
    //_portOutMesh->setCurrentObject(meshes);
    //_portOutMeshDomain->setCurrentObject(domains);
    delete[] allMeshes;
    delete[] allDomains;
}

void ReadCOMSOL::writeData(PhysicalValues *physicalValues)
{
    sendInfo("Write data to output ports.");
    const TimeSteps *timeSteps = _physics->getTimeSteps(false);
    const unsigned int noTimeSteps = timeSteps->getNoTimeSteps();
    const unsigned char noPhysics = _physics->getNoPhysics();
    unsigned char setNoPhysicalValues = 0;
    unsigned char currentPortNoVec = 0;
    unsigned char currentPortNoScal = 0;
    for (unsigned char i = 0; i < noPhysics; i++)
    {
        const unsigned char noValue = _physics->getNoPhysicalValuesPhysic(i);
        for (unsigned char j = 0; j < noValue; j++)
        {
            if (_physics->computePhysicalValue(i, j))
            {
                const bool vector = physicalValues->isVectorDataSet(setNoPhysicalValues);
                coDistributedObject **allData = new coDistributedObject *[noTimeSteps + 1];
                allData[noTimeSteps] = NULL;
                std::string namePort = "";
                if (vector)
                {
                    namePort = _portOutVec[currentPortNoVec]->getObjName();
                    currentPortNoVec++;
                }
                else
                {
                    namePort = _portOutScal[currentPortNoScal]->getObjName();
                    currentPortNoScal++;
                }
                coObjInfo info(namePort.c_str());
                for (unsigned int t = 0; t < noTimeSteps; t++)
                {
                    const unsigned int noEvaluationPoints = physicalValues->getNoEvaluationPoints(t);
                    const std::string objName = getListName(namePort, t, false);
                    coDistributedObject *data = 0;
                    if (vector)
                    {
                        coDoVec3 *dataVec = new coDoVec3(objName.c_str(), noEvaluationPoints);
                        float *u = NULL;
                        float *v = NULL;
                        float *w = NULL;
                        dataVec->getAddresses(&u, &v, &w);
                        for (unsigned long ep = 0; ep < noEvaluationPoints; ep++)
                        {
                            const double *value = physicalValues->getValue(t, setNoPhysicalValues, ep);
                            u[ep] = value[0];
                            v[ep] = value[1];
                            w[ep] = value[2];
                        }
                        data = dataVec;
                    }
                    else
                    {
                        coDoFloat *dataScal = new coDoFloat(objName.c_str(), noEvaluationPoints);
                        float *u = dataScal->getAddress();
                        for (unsigned long ep = 0; ep < noEvaluationPoints; ep++)
                        {
                            const double *value = physicalValues->getValue(t, setNoPhysicalValues, ep);
                            u[ep] = value[0];
                        }
                        data = dataScal;
                    }
                    allData[t] = data;
                }
                coDoSet *dataSet = new coDoSet(info, allData);
                if (noTimeSteps > 1)
                {
                    const std::string attribute = "1 " + noTimeSteps;
                    dataSet->addAttribute("TIMESTEP", attribute.c_str());
                }
                delete[] allData;
                setNoPhysicalValues++;
            }
        }
    }
}
