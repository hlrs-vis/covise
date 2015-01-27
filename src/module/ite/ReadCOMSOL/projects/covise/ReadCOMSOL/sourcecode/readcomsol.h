/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ReadCOMSOL: HLRS COVISE module, which reads data from COMSOL Multiphysics
// author: Andre Buchau
// 21.01.2011: modified

#pragma once

#include <api/coModule.h>
#include "../../../interfaces/data/include/data.hxx"
#include "../../../interfaces/matlab/include/interfacematlab.hxx"
#include "../../../interfaces/comsol/include/interfacecomsollib.hxx"

using namespace covise;

class ReadCOMSOL : public coFunctionModule
{
public:
    ReadCOMSOL(int argc, char **argv);
    virtual ~ReadCOMSOL();

protected:
private:
    // COVISE interface functions
    virtual void postInst();
    virtual void param(const char *paramName, bool inMapLoading);
    virtual int compute(const char *port);
    // member variables
    InterfaceMatlab *_interfaceMatlab;
    CommunicationComsol *_communicationComsol;
    ComsolPhysics *_physics;
    coOutputPort *_portOutMesh;
    coOutputPort *_portOutMeshDomain;
    unsigned char _noPortsVec;
    unsigned char _noPortsScal;
    coOutputPort **_portOutVec;
    coOutputPort **_portOutScal;
    coFileBrowserParam *_fileBrowser;
    std::string _fileName;
    coChoiceParam **_paramOutVec;
    coChoiceParam **_paramOutScal;
    struct ApplicationModeData
    {
        std::string name;
        unsigned char applicationMode;
        unsigned char physicalValue;
        bool compute;
    };
    std::vector<ApplicationModeData> _paramDataVec;
    std::vector<ApplicationModeData> _paramDataScal;
    std::vector<string> _namesApplicationModes;
    // initialization
    void initializeLocalVariables();
    void deleteLocalVariables();
    // output ports
    void createOutputPorts();
    void writeMesh(MeshData *mesh);
    void writeData(PhysicalValues *physicalValues);
    // tools
    std::string getListName(const std::string baseName, const unsigned int no, const bool space) const;
    CartesianCoordinates *getEvaluationPoints(const MeshData *mesh, unsigned long &noEvaluationPoints, int **domainNoEvaluationPoints, const unsigned int timeStep) const;
    // dialog elements
    void createDialogElements();
    void setParameterLists(void);
    std::vector<std::vector<bool> > getExportList() const;
    void setComputationPhysicalValues();
    // COMSOL Multiphysics interaction
    void readComsolModel(const std::string fileName);
    MeshData *readMesh(const TimeSteps *timeSteps, const bool movingMesh) const;
    PhysicalValues *readData(const MeshData *mesh, TimeSteps *timeSteps,
                             std::vector<std::vector<unsigned int> > &values2SetNo) const;
    void evaluateComsolModel(MeshData **mesh, PhysicalValues **physicalValues);
};
