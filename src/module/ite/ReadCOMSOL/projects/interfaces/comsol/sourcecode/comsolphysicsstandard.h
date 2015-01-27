/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// physics of COMSOL Multiphysics standard implementation
// author: André Buchau
// 18.10.2010: adapted to version 4.0 of COMSOL Multiphysics
// 18.11.2010: modified DLL

#pragma once

#include "../include/comsolphysics.hxx"
#include "../../data/include/time.hxx"

class ComsolPhysicsStandard : public ComsolPhysics
{
public:
    ComsolPhysicsStandard();
    ~ComsolPhysicsStandard();
    void addDataPhysics(const std::string type, const std::string tag);
    void setTimeSteps(TimeSteps *timeSteps);
    unsigned int getNoPhysics(void) const;
    unsigned int getNoPhysicalValuesPhysic(const unsigned int no) const;
    std::string getTextPhysicalValue(const unsigned int noPhysic, const unsigned int noPhysicalValue) const;
    bool isVector(const unsigned int noPhysic, const unsigned int noPhysicalValue) const;
    std::string getNameValue(const unsigned int noPhysic, const unsigned int noPhysicalValue) const;
    std::string getTag(const unsigned int noPhysic) const;
    bool getAverage(const unsigned int noApplicationMode, const unsigned int noPhysicalValue) const;
    TimeSteps *getTimeSteps(const bool firstTimeStepOnly) const;
    bool isDof(const unsigned int noPhysic, const unsigned int noPhysicalValue) const;
    bool isMovingMesh() const;
    bool computePhysicalValue(const unsigned int noPhysic, const unsigned int noPhysicalValue) const;
    void setComputePhysicalValue(const unsigned int noPhysic, const unsigned int noPhysicalValue, const bool value);
    struct DataPhysicalValue
    {
        std::string comsolName;
        std::string description;
        bool vector;
        bool average;
        bool dof;
        bool compute;
    };
    struct DataPhysics
    {
        std::string tag;
        std::string type;
        std::vector<DataPhysicalValue> physicalValues;
    };

protected:
private:
    std::vector<DataPhysics> _physics;
    TimeSteps *_timeSteps;
    void setPhysicalValuesElectrostatics(DataPhysics &physic);
    void setPhysicalValuesMagnetostatics(DataPhysics &physic, const bool currents);
    void setPhysicalValuesSteadyCurrents(DataPhysics &physic);
    void setPhysicalValuesElectromagneticWaves(DataPhysics &physic);
    void setPhysicalValuesNavierStokes(DataPhysics &physic);
    void setPhysicalValuesHeatTransfer(DataPhysics &physic);
    void setPhysicalValuesMovingMesh(DataPhysics &physic);
};
