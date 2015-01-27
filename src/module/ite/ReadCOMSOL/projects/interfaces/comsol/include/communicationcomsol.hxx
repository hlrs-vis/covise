// communication with COMSOL Multiphysics
// author: Andre Buchau
// 21.01.2011: modified

#pragma once

#include "../../matlab/include/interfacematlab.hxx"
#include "comsolphysics.hxx"
#include "../../data/include/data.hxx"
#include "../sourcecode/dllapi.h"

class API_INTERFACECOMSOL CommunicationComsol
{
public:
    CommunicationComsol();
    ~CommunicationComsol();
    static CommunicationComsol* getInstance(const InterfaceMatlab* matlab);
    virtual ComsolPhysics* const readPhysics(TimeSteps::QualityTimeHarmonic qualityTimeHarmonic) const = 0;
    virtual bool evaluateModel(MeshData** mesh, PhysicalValues** physicalValues, const ComsolPhysics* physics) const = 0;
    virtual MeshData* const getMesh(const TimeSteps* timeSteps, const bool movingMesh) const = 0;
    virtual PhysicalValues* const getPhysicalValues(const TimeSteps* timeSteps, const ComsolPhysics* physics,
                                                    const std::vector<std::vector<bool> >& exportList,
                                                    std::vector<std::vector<unsigned int> >& value2SetNo,
                                                    const std::vector <std::vector <CartesianCoordinates> > &evaluationPoints,
                                                    const std::vector <std::vector <unsigned int> > &domainNoEvaluationPoints,
                                                    const std::vector <unsigned long> &noEvaluationPoints) const = 0;
protected:
private:
};
