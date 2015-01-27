/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "communicationcomsolstandard.h"
#include <sstream>
#include <iostream>

CommunicationComsolStandard::CommunicationComsolStandard(const InterfaceMatlab *matlab)
    : _matlab(matlab)
{
}

CommunicationComsolStandard::~CommunicationComsolStandard()
{
}

ComsolPhysics *const CommunicationComsolStandard::readPhysics(TimeSteps::QualityTimeHarmonic qualityTimeHarmonic) const
{
    const unsigned int noPhysics = getNoPhysics();
    ComsolPhysics *retVal = ComsolPhysics::getInstance();
    for (unsigned int i = 0; i < noPhysics; i++)
    {
        std::string type;
        std::string tag;
        getPhysicsTypeTag(i, type, tag);
        retVal->addDataPhysics(type, tag);
    }
    TimeSteps *timeSteps = getTimeSteps(qualityTimeHarmonic);
    retVal->setTimeSteps(timeSteps);
    return retVal;
}

bool CommunicationComsolStandard::evaluateModel(MeshData **mesh, PhysicalValues **physicalValues, const ComsolPhysics *physics) const
{
    const std::string matlabCommand = getMatlabCommand(physics);
    _matlab->Execute(matlabCommand.c_str());
    *mesh = getPostProcessingMesh(physics);
    *physicalValues = getPostProcessingData(physics, *mesh);
    _matlab->Execute("clear postProcessingData");
    return true;
}

unsigned int CommunicationComsolStandard::getNoPhysics(void) const
{
    _matlab->Execute("physicsTags = comsolModel.physics.tags");
    _matlab->Execute("noPhysics = length(physicsTags)");
    _matlab->Execute("clear physicsTags");
    unsigned int retVal = _matlab->GetInteger("noPhysics");
    _matlab->Execute("clear noPhysics");
    return retVal;
}

void CommunicationComsolStandard::getPhysicsTypeTag(const unsigned int no, std::string &type, std::string &tag) const
{
    _matlab->Execute("physicsTags = comsolModel.physics.tags");
    std::ostringstream tempTag;
    tempTag << "tag = char(physicsTags(" << no + 1 << "))";
    std::string inputMatlab = tempTag.str();
    _matlab->Execute(inputMatlab.c_str());
    _matlab->Execute("clear physicsTags");
    tag = _matlab->GetString("tag");
    _matlab->Execute("physic = comsolModel.physics(tag)");
    _matlab->Execute("type = char(physic.getType())");
    type = _matlab->GetString("type");
    _matlab->Execute("clear tag");
    _matlab->Execute("clear physic");
    _matlab->Execute("clear type");
}

TimeSteps *CommunicationComsolStandard::getTimeSteps(TimeSteps::QualityTimeHarmonic qualityTimeHarmonic) const
{
    TimeSteps *retVal = TimeSteps::getInstance();
    _matlab->Execute("[sz,ndofs,data,name,type] = mphgetp(comsolModel)");
    std::string type = _matlab->GetString("type");
    if (type == "Stationary")
    {
        retVal->setType(TimeSteps::Type_static);
        retVal->setFirstTimeStepOnly(true);
        retVal->putValue(0);
    }
    else
    {
        if (type == "Time")
        {
            retVal->setType(TimeSteps::Type_transient);
            retVal->setFirstTimeStepOnly(false);
            unsigned long noRows = 0;
            unsigned long noColumns = 0;
            double *data = _matlab->GetDoubleMatrix("data", noRows, noColumns);
            for (unsigned long i = 0; i < noRows; i++)
                retVal->putValue(data[i]);
        }
        else
        {
            if (type == "Parametric")
            {
                std::string name = _matlab->GetString("name");
                if (name == "freq")
                {
                    retVal->setType(TimeSteps::Type_harmonic);
                    retVal->setFirstTimeStepOnly(false);
                    unsigned int noPhase = 1;
                    //switch (qualityTimeHarmonic)
                    //{
                    //case TimeSteps::QualityTimeHarmonic_Low:
                    //    noPhase = 13;
                    //    break;
                    //case TimeSteps::QualityTimeHarmonic_Medium:
                    //    noPhase = 21;
                    //    break;
                    //case TimeSteps::QualityTimeHarmonic_High:
                    //    noPhase = 33;
                    //    break;
                    //};
                    //const double delta = 2 * 3.141592653589793 / (noPhase - 1);
                    //for (unsigned int i = 0; i < noPhase; i++)
                    //    retVal->putValue(i * delta);
                    retVal->putValue(0);
                }
                else
                {
                    retVal->setType(TimeSteps::Type_parametric);
                    retVal->setFirstTimeStepOnly(false);
                    unsigned long noRows = 0;
                    unsigned long noColumns = 0;
                    double *data = _matlab->GetDoubleMatrix("data", noRows, noColumns);
                    for (unsigned long i = 0; i < noRows; i++)
                        retVal->putValue(i + 1);
                }
            }
            else
            {
                retVal->setType(TimeSteps::Type_static);
                retVal->setFirstTimeStepOnly(true);
                retVal->putValue(0);
            }
        }
    }
    _matlab->Execute("clear sz");
    _matlab->Execute("clear ndofs");
    _matlab->Execute("clear data");
    _matlab->Execute("clear name");
    _matlab->Execute("clear type");
    return retVal;
}

MeshData *const CommunicationComsolStandard::getMesh(const TimeSteps *timeSteps, const bool movingMesh) const
{
    MeshData *retVal = MeshData::getInstance(timeSteps->getNoTimeSteps());
    storeNodes(retVal, timeSteps->getNoTimeSteps());
    storeElements(retVal, timeSteps->getNoTimeSteps());
    if (movingMesh)
        MoveNodes(retVal, timeSteps);
    return retVal;
}

PhysicalValues *const CommunicationComsolStandard::getPhysicalValues(const TimeSteps *timeSteps, const ComsolPhysics *physics,
                                                                     const std::vector<std::vector<bool> > &exportList,
                                                                     std::vector<std::vector<unsigned int> > &value2SetNo,
                                                                     const std::vector<std::vector<CartesianCoordinates> > &evaluationPoints,
                                                                     const std::vector<std::vector<unsigned int> > &domainNoEvaluationPoints,
                                                                     const std::vector<unsigned long> &noEvaluationPoints) const
{
    PhysicalValues *retVal = PhysicalValues::getInstance(timeSteps->getNoTimeSteps(), noEvaluationPoints);
    const unsigned int noTimeSteps = timeSteps->getNoTimeSteps();
    bool useTimeSteps = false;
    std::string nameTimeStep = "";
    switch (timeSteps->getType())
    {
    case TimeSteps::Type_harmonic:
        nameTimeStep = "Phase";
        useTimeSteps = true;
        break;
    case TimeSteps::Type_transient:
        nameTimeStep = "t";
        useTimeSteps = true;
        break;
    case TimeSteps::Type_parametric:
        nameTimeStep = "Solnum";
        useTimeSteps = true;
    case TimeSteps::Type_static:
    default:
        break;
    };
    for (unsigned int i = 0; i < physics->getNoPhysics(); i++)
    {
        for (unsigned int j = 0; j < physics->getNoPhysicalValuesPhysic(i); j++)
        {
            if (exportList[i][j])
            {
                const std::string nameValue = physics->getNameValue(i, j);
                const std::string tag = physics->getTag(i);
                std::ostringstream name;
                name << tag << "." << nameValue;
                const bool vector = physics->isVector(i, j);
                retVal->addDataSet(name.str(), vector);
                value2SetNo[i][j] = retVal->getNoDataSets() - 1;
                const bool dof = physics->isDof(i, j);
                const bool average = physics->getAverage(i, j);
                for (unsigned int t = 0; t < noTimeSteps; t++)
                {
                    double *values = retVal->getAllValues(t, value2SetNo[i][j]);
                    if (vector)
                        getResultsVector(nameValue, tag, average, noEvaluationPoints[t], &evaluationPoints[t][0], &domainNoEvaluationPoints[t][0], values, timeSteps->getValue(t), useTimeSteps, nameTimeStep, dof);
                    else
                        getResultsScalar(nameValue, tag, average, noEvaluationPoints[t], &evaluationPoints[t][0], &domainNoEvaluationPoints[t][0], values, timeSteps->getValue(t), useTimeSteps, nameTimeStep, dof);
                }
            }
        }
    }
    return retVal;
}

void CommunicationComsolStandard::storeNodes(MeshData *meshData, const unsigned int noTimeSteps) const
{
    _matlab->Execute("geomTags = comsolModel.geom");
    _matlab->Execute("geomTag = char(geomTags(1).tags)");
    _matlab->Execute("geom = comsolModel.geom(geomTag)");
    _matlab->Execute("lengthUnit = char(geom.lengthUnit())");
    std::string lengthUnit = _matlab->GetString("lengthUnit");
    const double geomScaling = getGeomScaling(lengthUnit);
    meshData->setScalingFactor(geomScaling);
    _matlab->Execute("meshTags = comsolModel.mesh");
    _matlab->Execute("meshTag = char(meshTags(1).tags)");
    _matlab->Execute("mesh = comsolModel.mesh(meshTag)");
    _matlab->Execute("nodes = mesh.getVertex");
    unsigned long nodesNoRows = 0;
    unsigned long nodesNoColumns = 0;
    double *nodes = _matlab->GetDoubleMatrix("nodes", nodesNoRows, nodesNoColumns);
    for (unsigned int t = 0; t < noTimeSteps; t++)
    {
        for (unsigned long i = 0; i < nodesNoColumns; i++)
        {
            CartesianCoordinates node;
            node.x = nodes[i * nodesNoRows] * geomScaling;
            node.y = nodes[i * nodesNoRows + 1] * geomScaling;
            node.z = nodes[i * nodesNoRows + 2] * geomScaling;
            meshData->addNode(t, node);
        }
    }
    _matlab->Execute("clear geomTags");
    _matlab->Execute("clear geomTag");
    _matlab->Execute("clear geom");
    _matlab->Execute("clear lengthUnit");
    _matlab->Execute("clear meshTags");
    _matlab->Execute("clear meshTag");
    _matlab->Execute("clear mesh");
    _matlab->Execute("clear nodes");
    delete[] nodes;
}

void CommunicationComsolStandard::storeElements(MeshData *meshData, const unsigned int noTimeSteps) const
{
    _matlab->Execute("meshTags = comsolModel.mesh");
    _matlab->Execute("meshTag = char(meshTags(1).tags)");
    _matlab->Execute("mesh = comsolModel.mesh(meshTag)");
    _matlab->Execute("numTet = mesh.getNumElem('tet')");
    const int numTet = _matlab->GetInteger("numTet");
    if (numTet > 0)
    {
        _matlab->Execute("tet = double(mesh.getElem('tet'))");
        _matlab->Execute("tetDomain = double(mesh.getElemEntity('tet'))");
        unsigned long domainsNoRows = 0;
        unsigned long domainsNoColumns = 0;
        double *domains = _matlab->GetDoubleMatrix("tetDomain", domainsNoRows, domainsNoColumns);
        unsigned long tetNoRows = 0;
        unsigned long tetNoColumns = 0;
        double *tet = _matlab->GetDoubleMatrix("tet", tetNoRows, tetNoColumns);
        for (unsigned int t = 0; t < noTimeSteps; t++)
        {
            for (unsigned long j = 0; j < tetNoColumns; j++)
            {
                Tetra4 element;
                element.domainNo = (unsigned int)domains[j] - 1;
                element.node1 = (unsigned long)tet[j * tetNoRows + 1 - 1];
                element.node2 = (unsigned long)tet[j * tetNoRows + 2 - 1];
                element.node3 = (unsigned long)tet[j * tetNoRows + 3 - 1];
                element.node4 = (unsigned long)tet[j * tetNoRows + 4 - 1];
                meshData->addTetra4(t, element);
            }
        }
        _matlab->Execute("clear tet");
        _matlab->Execute("clear tetDomain");
    }
    _matlab->Execute("clear meshTags");
    _matlab->Execute("clear meshTag");
    _matlab->Execute("clear mesh");
    _matlab->Execute("clear numTet");
}

void CommunicationComsolStandard::MoveNodes(MeshData *meshData, const TimeSteps *timeSteps) const
{
    const unsigned int noTimeSteps = timeSteps->getNoTimeSteps();
    bool useTimeSteps = false;
    std::string nameTimeStep = "";
    switch (timeSteps->getType())
    {
    case TimeSteps::Type_harmonic:
        nameTimeStep = "Phase";
        useTimeSteps = true;
        break;
    case TimeSteps::Type_transient:
        nameTimeStep = "t";
        useTimeSteps = true;
        break;
    case TimeSteps::Type_static:
    default:
        break;
    };
    for (unsigned int t = 0; t < noTimeSteps; t++)
    {
        const unsigned noNodes = meshData->getNoNodes(t);
        const CartesianCoordinates *coordinates = meshData->getNodes(t);
        const int *temp = meshData->getDomainNoNodes(t);
        std::vector<unsigned int> domainNoCoordinates;
        domainNoCoordinates.resize(noNodes);
        unsigned int i = 0;
        for (i = 0; i < noNodes; i++)
            domainNoCoordinates[i] = temp[i];
        double *displacement = new double[noNodes * 3];
        getResultsVector("d", "ale", false, noNodes, coordinates, &domainNoCoordinates[0], displacement, timeSteps->getValue(t), useTimeSteps, nameTimeStep, true);
        for (i = 0; i < noNodes; i++)
        {
            CartesianCoordinates node = meshData->getNode(t, i);
            node.x += displacement[i * 3];
            node.y += displacement[i * 3 + 1];
            node.z += displacement[i * 3 + 2];
            meshData->replaceNodeCoordinates(t, i, node);
        }
        delete[] displacement;
    }
}

void CommunicationComsolStandard::getResultsScalar(const std::string nameValue, const std::string tag, const bool average, const unsigned int noPoints, const CartesianCoordinates coordinates[], const unsigned int domainNoCoordinates[], double values[], const double valueTimeStep, const bool useTimeStep, const std::string nameTimeStep, const bool dof) const
{
    std::ostringstream nameTmp;
    if (dof)
        nameTmp << nameValue;
    else
    {
        if (average)
            nameTmp << tag << "." << nameValue << "av";
        else
            nameTmp << tag << "." << nameValue;
    }
    std::string name = nameTmp.str();
    getSinglePhysicalValue(name.c_str(), noPoints, coordinates, domainNoCoordinates, values, valueTimeStep, useTimeStep, nameTimeStep);
}

void CommunicationComsolStandard::getResultsVector(const std::string nameValue, const std::string tag, const bool average, const unsigned int noPoints, const CartesianCoordinates coordinates[], const unsigned int domainNoCoordinates[], double values[], const double valueTimeStep, const bool useTimeStep, const std::string nameTimeStep, const bool dof) const
{
    double *resultsSingle = new double[noPoints];
    std::ostringstream xNameTmp, yNameTmp, zNameTmp;
    if (dof)
    {
        xNameTmp << nameValue << "x";
        yNameTmp << nameValue << "y";
        zNameTmp << nameValue << "z";
    }
    else
    {
        if (average)
        {
            xNameTmp << tag << "." << nameValue << "xav";
            yNameTmp << tag << "." << nameValue << "yav";
            zNameTmp << tag << "." << nameValue << "zav";
        }
        else
        {
            xNameTmp << tag << "." << nameValue << "x";
            yNameTmp << tag << "." << nameValue << "y";
            zNameTmp << tag << "." << nameValue << "z";
        }
    }
    std::string xName = xNameTmp.str();
    std::string yName = yNameTmp.str();
    std::string zName = zNameTmp.str();
    unsigned int i = 0;
    getSinglePhysicalValue(xName.c_str(), noPoints, coordinates, domainNoCoordinates, resultsSingle, valueTimeStep, useTimeStep, nameTimeStep);
    for (i = 0; i < noPoints; i++)
        values[3 * i] = resultsSingle[i];
    getSinglePhysicalValue(yName.c_str(), noPoints, coordinates, domainNoCoordinates, resultsSingle, valueTimeStep, useTimeStep, nameTimeStep);
    for (i = 0; i < noPoints; i++)
        values[3 * i + 1] = resultsSingle[i];
    getSinglePhysicalValue(zName.c_str(), noPoints, coordinates, domainNoCoordinates, resultsSingle, valueTimeStep, useTimeStep, nameTimeStep);
    for (i = 0; i < noPoints; i++)
        values[3 * i + 2] = resultsSingle[i];
    delete[] resultsSingle;
}

void CommunicationComsolStandard::getSinglePhysicalValue(const char *name, const unsigned int noPoints, const CartesianCoordinates coordinates[], const unsigned int domainNoCoordinates[], double values[], const double valueTimeStep, const bool useTimeStep, const std::string nameTimeStep) const
{
    unsigned int noDomains = 0;
    unsigned int i;
    for (i = 0; i < noPoints; i++)
    {
        if (domainNoCoordinates[i] > noDomains)
            noDomains = domainNoCoordinates[i];
        values[i] = 0;
    }
    noDomains += 1;
    for (i = 0; i < noDomains; i++)
    {
        unsigned int noPointsDomain = 0;
        unsigned int j = 0;
        for (j = 0; j < noPoints; j++)
        {
            if (domainNoCoordinates[j] == i)
                noPointsDomain++;
        }
        if (noPointsDomain > 0)
        {
            double *matlabCoordinates = new double[noPointsDomain * 3];
            unsigned int *matlab2Global = new unsigned int[noPointsDomain];
            unsigned int indexMatlab = 0;
            for (j = 0; j < noPoints; j++)
            {
                if (domainNoCoordinates[j] == i)
                {
                    matlabCoordinates[indexMatlab * 3] = coordinates[j].x;
                    matlabCoordinates[indexMatlab * 3 + 1] = coordinates[j].y;
                    matlabCoordinates[indexMatlab * 3 + 2] = coordinates[j].z;
                    matlab2Global[indexMatlab] = j;
                    indexMatlab++;
                }
            }
            _matlab->SetDoubleMatrix("evaluationPoints", matlabCoordinates, noPointsDomain, 3);
            _matlab->SetInteger("currentDomain", i + 1);
            std::ostringstream temp;
            if (useTimeStep)
                temp << "results = mphinterp(comsolModel, '" << name << "', 'coord', evaluationPoints, '" << nameTimeStep << "', " << valueTimeStep << ", 'Selection', currentDomain)";
            else
                temp << "results = mphinterp(comsolModel, '" << name << "', 'coord', evaluationPoints, 'Selection', currentDomain)";
            std::string inputMatlab = temp.str();
            _matlab->Execute(inputMatlab.c_str());
            unsigned long noRows;
            unsigned long noColumns;
            const double *results = _matlab->GetDoubleMatrix("results", noRows, noColumns);
            for (j = 0; j < noPointsDomain; j++)
                values[matlab2Global[j]] = results[j];
            _matlab->Execute("clear results");
            _matlab->Execute("clear evaluationPoints");
            _matlab->Execute("clear currentDomain");
            delete[] results;
            delete[] matlab2Global;
            delete[] matlabCoordinates;
        }
    }
}

double CommunicationComsolStandard::getGeomScaling(const std::string lengthUnit) const
{
    double retVal = 1.0;
    if (lengthUnit == "cm")
        retVal = 1e-2;
    else
    {
        if (lengthUnit == "mm")
            retVal = 1e-3;
        else
        {
            if (lengthUnit == "nm")
                retVal = 1e-9;
            else if (lengthUnit == "µm")
                retVal = 1e-6;
        }
    }
    return retVal;
}

std::string CommunicationComsolStandard::getMatlabCommand(const ComsolPhysics *physics) const
{
    std::ostringstream retVal;
    retVal << "postProcessingData = mpheval(comsolModel, {'dom'";
    const unsigned int noPhysics = physics->getNoPhysics();
    for (unsigned int i = 0; i < noPhysics; i++)
    {
        const std::string tag = physics->getTag(i);
        const unsigned int noPhysicalValues = physics->getNoPhysicalValuesPhysic(i);
        for (unsigned int j = 0; j < noPhysicalValues; j++)
        {
            if (physics->computePhysicalValue(i, j))
            {
                const std::string nameValue = physics->getNameValue(i, j);
                std::string prefix = "";
                if (!physics->isDof(i, j))
                    prefix = tag + ".";
                std::string postfix = "";
                if (physics->getAverage(i, j))
                    postfix = "av";
                if (physics->isVector(i, j))
                {
                    retVal << ", '" << prefix << nameValue << "x" << postfix << "'";
                    retVal << ", '" << prefix << nameValue << "y" << postfix << "'";
                    retVal << ", '" << prefix << nameValue << "z" << postfix << "'";
                }
                else
                    retVal << ", '" << prefix << nameValue << postfix << "'";
            }
        }
    }
    retVal << "}";
    TimeSteps *timeSteps = physics->getTimeSteps(false);
    //if (timeSteps->getType() == TimeSteps::Type_harmonic)
    //{
    //    const unsigned int noTimeSteps = timeSteps->getNoTimeSteps();
    //    double* valuesPhase = new double [noTimeSteps];
    //    for (unsigned int i = 0; i < noTimeSteps; i++)
    //        valuesPhase[i] = timeSteps->getValue(i);
    //    _matlab->SetDoubleMatrix("valuesPhase", valuesPhase, noTimeSteps, 1);
    //    retVal << ", 'Phase', valuesPhase";
    //    delete [] valuesPhase;
    //}
    retVal << ")";
    return retVal.str();
}

MeshData *CommunicationComsolStandard::getPostProcessingMesh(const ComsolPhysics *physics) const
{
    unsigned long domainsNoRows = 0;
    unsigned long domainsNoColumns = 0;
    _matlab->Execute("d1 = postProcessingData.d1");
    double *domains = _matlab->GetDoubleMatrix("d1", domainsNoRows, domainsNoColumns);
    _matlab->Execute("clear d1");
    MeshData *retVal = MeshData::getInstance(domainsNoRows);
    unsigned long nodesNoRows = 0;
    unsigned long nodesNoColumns = 0;
    _matlab->Execute("p = postProcessingData.p");
    double *nodes = _matlab->GetDoubleMatrix("p", nodesNoRows, nodesNoColumns);
    _matlab->Execute("clear p");
    unsigned long tetraNoRows = 0;
    unsigned long tetraNoColumns = 0;
    _matlab->Execute("t = postProcessingData.t");
    int *tetras = _matlab->GetIntegerMatrix("t", tetraNoRows, tetraNoColumns);
    _matlab->Execute("clear t");
    _matlab->Execute("geomTags = comsolModel.geom");
    _matlab->Execute("geomTag = char(geomTags(1).tags)");
    _matlab->Execute("geom = comsolModel.geom(geomTag)");
    _matlab->Execute("lengthUnit = char(geom.lengthUnit())");
    std::string lengthUnit = _matlab->GetString("lengthUnit");
    _matlab->Execute("clear geomTags");
    _matlab->Execute("clear geomTag");
    _matlab->Execute("clear geom");
    _matlab->Execute("clear lengthUnit");
    const double geomScaling = getGeomScaling(lengthUnit);
    const TimeSteps *timeSteps = physics->getTimeSteps(false);
    const unsigned int noTimeSteps = timeSteps->getNoTimeSteps();
    for (unsigned int i = 0; i < nodesNoColumns; i++)
    {
        CartesianCoordinates node;
        node.x = nodes[i * nodesNoRows + 0] * geomScaling;
        node.y = nodes[i * nodesNoRows + 1] * geomScaling;
        node.z = nodes[i * nodesNoRows + 2] * geomScaling;
        for (unsigned int j = 0; j < noTimeSteps; j++)
            retVal->addNode(j, node);
    }
    for (unsigned int i = 0; i < tetraNoColumns; i++)
    {
        Tetra4 tetra;
        tetra.node1 = tetras[i * tetraNoRows + 0];
        tetra.node2 = tetras[i * tetraNoRows + 1];
        tetra.node3 = tetras[i * tetraNoRows + 2];
        tetra.node4 = tetras[i * tetraNoRows + 3];
        tetra.domainNo = domains[tetra.node1 * domainsNoRows + 0];
        for (unsigned int j = 0; j < noTimeSteps; j++)
            retVal->addTetra4(j, tetra);
    }
    return retVal;
}

PhysicalValues *CommunicationComsolStandard::getPostProcessingData(const ComsolPhysics *physics, const MeshData *mesh) const
{
    const TimeSteps *timeSteps = physics->getTimeSteps(false);
    const unsigned int noTimeSteps = timeSteps->getNoTimeSteps();
    std::vector<unsigned long> noEvaluationPoints;
    noEvaluationPoints.resize(noTimeSteps);
    for (unsigned int i = 0; i < noTimeSteps; i++)
        noEvaluationPoints[i] = mesh->getNoNodes(i);
    PhysicalValues *retVal = PhysicalValues::getInstance(noTimeSteps, noEvaluationPoints);
    const unsigned int noPhysics = physics->getNoPhysics();
    unsigned char arrayNo = 2; // 1 for domains, 0 not used
    unsigned int dataSet = 0;
    const std::string stringAssignment = " = postProcessingData.";
    for (unsigned int i = 0; i < noPhysics; i++)
    {
        const unsigned int noPhysicalValues = physics->getNoPhysicalValuesPhysic(i);
        const std::string namePhysic = physics->getTag(i);
        for (unsigned int j = 0; j < noPhysicalValues; j++)
        {
            if (physics->computePhysicalValue(i, j))
            {
                const std::string nameDataSet = namePhysic + "." + physics->getNameValue(i, j);
                const bool vector = physics->isVector(i, j);
                retVal->addDataSet(nameDataSet, vector);
                if (vector)
                {
                    std::ostringstream nameMatlabX;
                    nameMatlabX << "d" << (int)arrayNo;
                    arrayNo++;
                    std::ostringstream nameMatlabY;
                    nameMatlabY << "d" << (int)arrayNo;
                    arrayNo++;
                    std::ostringstream nameMatlabZ;
                    nameMatlabZ << "d" << (int)arrayNo;
                    arrayNo++;
                    const std::string extractX = nameMatlabX.str() + stringAssignment + nameMatlabX.str();
                    const std::string extractY = nameMatlabY.str() + stringAssignment + nameMatlabY.str();
                    const std::string extractZ = nameMatlabZ.str() + stringAssignment + nameMatlabZ.str();
                    _matlab->Execute(extractX.c_str());
                    _matlab->Execute(extractY.c_str());
                    _matlab->Execute(extractZ.c_str());
                    unsigned long noRowsX = 0;
                    unsigned long noColumnsX = 0;
                    double *allValuesX = _matlab->GetDoubleMatrix(nameMatlabX.str().c_str(), noRowsX, noColumnsX);
                    unsigned long noRowsY = 0;
                    unsigned long noColumnsY = 0;
                    double *allValuesY = _matlab->GetDoubleMatrix(nameMatlabY.str().c_str(), noRowsY, noColumnsY);
                    unsigned long noRowsZ = 0;
                    unsigned long noColumnsZ = 0;
                    double *allValuesZ = _matlab->GetDoubleMatrix(nameMatlabZ.str().c_str(), noRowsZ, noColumnsZ);
                    const std::string deleteX = "clear " + nameMatlabX.str();
                    const std::string deleteY = "clear " + nameMatlabY.str();
                    const std::string deleteZ = "clear " + nameMatlabZ.str();
                    _matlab->Execute(deleteX.c_str());
                    _matlab->Execute(deleteY.c_str());
                    _matlab->Execute(deleteZ.c_str());
                    for (unsigned int noValue = 0; noValue < noColumnsX; noValue++)
                    {
                        for (unsigned int timeStep = 0; timeStep < noRowsX; timeStep++)
                        {
                            double value[3];
                            value[0] = allValuesX[noValue * noRowsX + timeStep];
                            value[1] = allValuesY[noValue * noRowsY + timeStep];
                            value[2] = allValuesZ[noValue * noRowsZ + timeStep];
                            retVal->setValue(timeStep, dataSet, noValue, value);
                        }
                    }
                    delete[] allValuesX;
                    delete[] allValuesY;
                    delete[] allValuesZ;
                }
                else
                {
                    std::ostringstream nameMatlab;
                    nameMatlab << "d" << (int)arrayNo;
                    arrayNo++;
                    const std::string extract = nameMatlab.str() + stringAssignment + nameMatlab.str();
                    _matlab->Execute(extract.c_str());
                    unsigned long noRows = 0;
                    unsigned long noColumns = 0;
                    double *allValues = _matlab->GetDoubleMatrix(nameMatlab.str().c_str(), noRows, noColumns);
                    const std::string deleteD = "clear " + nameMatlab.str();
                    _matlab->Execute(deleteD.c_str());
                    for (unsigned int noValue = 0; noValue < noColumns; noValue++)
                    {
                        for (unsigned int timeStep = 0; timeStep < noRows; timeStep++)
                        {
                            double value[3];
                            value[0] = allValues[noValue * noRows + timeStep];
                            value[1] = 0;
                            value[2] = 0;
                            retVal->setValue(timeStep, dataSet, noValue, value);
                        }
                    }
                    delete[] allValues;
                }
                dataSet++;
            }
        }
    }
    return retVal;
}
