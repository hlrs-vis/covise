// physics of COMSOL Multiphysics
// author: Andre Buchau
// 18.10.2010: adapted to version 4.0 of COMSOL Multiphysics
// 18.11.2010: modified DLL

#pragma once

#include <string>
#include <vector>
#include "../../data/include/time.hxx"
#include "../sourcecode/dllapi.h"

class API_INTERFACECOMSOL ComsolPhysics
{
public:
    ComsolPhysics();
    virtual ~ComsolPhysics();
    static ComsolPhysics* getInstance();
    virtual void addDataPhysics(const std::string type, const std::string tag) = 0;
    virtual void setTimeSteps(TimeSteps* timeSteps) = 0;
    virtual unsigned int getNoPhysics(void) const = 0;
    virtual unsigned int getNoPhysicalValuesPhysic(const unsigned int no) const = 0;
    virtual std::string getTextPhysicalValue(const unsigned int noPhysic, const unsigned int noPhysicalValue) const = 0;
    virtual bool isVector(const unsigned int noPhysic, const unsigned int noPhysicalValue) const = 0;
    virtual std::string getNameValue(const unsigned int noPhysic, const unsigned int noPhysicalValue) const = 0;
    virtual std::string getTag(const unsigned int noPhysic) const = 0;
    virtual bool getAverage(const unsigned int noApplicationMode, const unsigned int noPhysicalValue) const = 0;
    virtual TimeSteps* getTimeSteps(const bool firstTimeStepOnly) const = 0;
    virtual bool isDof(const unsigned int noPhysic, const unsigned int noPhysicalValue) const = 0;
    virtual bool isMovingMesh() const = 0;
    virtual bool computePhysicalValue(const unsigned int noPhysic, const unsigned int noPhysicalValue) const = 0;
    virtual void setComputePhysicalValue(const unsigned int noPhysic, const unsigned int noPhysicalValue, const bool value) = 0;
    struct DataPhysicalValue
    {
        std::string comsolName;
        std::string description;
        bool vector;
        bool average;
        bool dof;
    };
    struct DataPhysics
    {
        std::string tag;
        std::string type;
        std::vector <DataPhysicalValue> physicalValues;
    };
protected:
private:
};
