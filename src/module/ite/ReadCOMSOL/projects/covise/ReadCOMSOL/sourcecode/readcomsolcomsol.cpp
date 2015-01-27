/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "readcomsol.h"

void ReadCOMSOL::readComsolModel(const std::string fileName)
{
    if (fileName != _fileName)
    {
        _fileName = fileName;
        _interfaceMatlab->WriteString(_fileName.c_str(), "comsolFileName");
        _interfaceMatlab->Execute("clear comsolModel");
        sendInfo("Trying to load COMSOL Multiphysics model, please wait ...");
        _interfaceMatlab->Execute("comsolModel = mphload(comsolFileName, 'comsolModel')");
        _interfaceMatlab->Execute("clear comsolFileName");
        if (_interfaceMatlab->IsVariable("comsolModel"))
        {
            sendInfo("COMSOL Multiphysics model file has been successfully loaded");
            setParameterLists();
        }
        else
            sendError("COMSOL Multiphyiscs model file could not be opened");
    }
}

MeshData *ReadCOMSOL::readMesh(const TimeSteps *timeSteps, const bool movingMesh) const
{
    sendInfo("Read finite element mesh from COMSOL Multiphysics ...");
    MeshData *retVal = _communicationComsol->getMesh(timeSteps, movingMesh);
    sendInfo("Finished reading of finite element mesh from COMSOL Multiphysics.");
    return retVal;
}

PhysicalValues *ReadCOMSOL::readData(const MeshData *mesh, TimeSteps *timeSteps,
                                     std::vector<std::vector<unsigned int> > &values2SetNo) const
{
    sendInfo("Read data from COMSOL Multiphysics ...");
    std::vector<std::vector<bool> > exportList = getExportList();
    std::vector<unsigned long> noEvaluationPoints;
    noEvaluationPoints.resize(timeSteps->getNoTimeSteps());
    std::vector<std::vector<CartesianCoordinates> > evaluationPoints;
    evaluationPoints.resize(timeSteps->getNoTimeSteps());
    std::vector<std::vector<unsigned int> > domainNoEvaluationPoints;
    domainNoEvaluationPoints.resize(timeSteps->getNoTimeSteps());
    const double geomScaling = mesh->getScalingFactor();
    for (unsigned int t = 0; t < timeSteps->getNoTimeSteps(); t++)
    {
        unsigned long noPoints = 0;
        int *domainNo = 0;
        CartesianCoordinates *temp = getEvaluationPoints(mesh, noPoints, &domainNo, t);
        noEvaluationPoints[t] = noPoints;
        evaluationPoints[t].resize(noEvaluationPoints[t]);
        domainNoEvaluationPoints[t].resize(noEvaluationPoints[t]);
        for (unsigned long k = 0; k < noEvaluationPoints[t]; k++)
        {
            evaluationPoints[t][k] = temp[k];
            evaluationPoints[t][k].x /= geomScaling;
            evaluationPoints[t][k].y /= geomScaling;
            evaluationPoints[t][k].z /= geomScaling;
            domainNoEvaluationPoints[t][k] = domainNo[k];
        }
        delete[] temp;
        delete[] domainNo;
    }
    PhysicalValues *retVal = _communicationComsol->getPhysicalValues(timeSteps, _physics, exportList, values2SetNo, evaluationPoints, domainNoEvaluationPoints, noEvaluationPoints);
    sendInfo("Finished reading data from COMSOL Multiphysics.");
    return retVal;
}

void ReadCOMSOL::evaluateComsolModel(MeshData **mesh, PhysicalValues **physicalValues)
{
    sendInfo("Evaluate COMSOL Multiphysics model ...");
    setComputationPhysicalValues();
    _communicationComsol->evaluateModel(mesh, physicalValues, _physics);
    sendInfo("Evaluation of COMSOL Multiphysics model has been finished.");
}
