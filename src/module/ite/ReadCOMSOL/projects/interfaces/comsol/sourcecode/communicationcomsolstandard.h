/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// communication with COMSOL Multphysics implemented based on MATLAB Livelink
// author: Andre Buchau
// 21.01.2011: modified

#pragma once

#include "../include/communicationcomsol.hxx"

class CommunicationComsolStandard : public CommunicationComsol
{
public:
    CommunicationComsolStandard(const InterfaceMatlab *matlab);
    ~CommunicationComsolStandard();
    ComsolPhysics *const readPhysics(TimeSteps::QualityTimeHarmonic qualityTimeHarmonic) const;
    bool evaluateModel(MeshData **mesh, PhysicalValues **physicalValues, const ComsolPhysics *physics) const;
    MeshData *const getMesh(const TimeSteps *timeSteps, const bool movingMesh) const;
    PhysicalValues *const getPhysicalValues(const TimeSteps *timeSteps, const ComsolPhysics *physics,
                                            const std::vector<std::vector<bool> > &exportList,
                                            std::vector<std::vector<unsigned int> > &value2SetNo,
                                            const std::vector<std::vector<CartesianCoordinates> > &evaluationPoints,
                                            const std::vector<std::vector<unsigned int> > &domainNoEvaluationPoints,
                                            const std::vector<unsigned long> &noEvaluationPoints) const;

protected:
private:
    const InterfaceMatlab *_matlab;
    unsigned int getNoPhysics(void) const;
    void getPhysicsTypeTag(const unsigned int no, std::string &type, std::string &tag) const;
    TimeSteps *getTimeSteps(TimeSteps::QualityTimeHarmonic qualityTimeHarmonic) const;
    void storeNodes(MeshData *meshData, const unsigned int noTimeSteps) const;
    void storeElements(MeshData *meshData, const unsigned int noTimeSteps) const;
    void MoveNodes(MeshData *meshData, const TimeSteps *timeSteps) const;
    void getResultsScalar(const std::string nameValue, const std::string tag, const bool average, const unsigned int noPoints, const CartesianCoordinates coordinates[], const unsigned int domainNoCoordinates[], double values[], const double valueTimeStep, const bool useTimeStep, const std::string nameTimeStep, const bool dof) const;
    void getResultsVector(const std::string nameValue, const std::string tag, const bool average, const unsigned int noPoints, const CartesianCoordinates coordinates[], const unsigned int domainNoCoordinates[], double values[], const double valueTimeStep, const bool useTimeStep, const std::string nameTimeStep, const bool dof) const;
    void getSinglePhysicalValue(const char *name, const unsigned int noPoints, const CartesianCoordinates coordinates[], const unsigned int domainNoCoordinates[], double values[], const double valueTimeStep, const bool useTimeStep, const std::string nameTimeStep) const;
    double getGeomScaling(const std::string lengthUnit) const;
    std::string getMatlabCommand(const ComsolPhysics *physics) const;
    MeshData *getPostProcessingMesh(const ComsolPhysics *physics) const;
    PhysicalValues *getPostProcessingData(const ComsolPhysics *physics, const MeshData *mesh) const;
};
