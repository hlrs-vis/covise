// container for mesh data
// author: A. Buchau
// modified: 18.11.2010

#pragma once

#include "cartesian.hxx"
#include "basicelements.hxx"
#include "../sourcecode/dllapi.h"

class API_DATA MeshData
{
public:
    MeshData();
    virtual ~MeshData();
    static MeshData* getInstance(const unsigned int noTimeSteps);
    virtual void addNode(const unsigned int timeStep, const CartesianCoordinates coordinates) = 0;
    virtual void addTetra4(const unsigned int timeStep, const Tetra4 element) = 0;
    virtual unsigned long getNoNodes(const unsigned int timeStep) const = 0;
    virtual CartesianCoordinates getNode(const unsigned int timeStep, const unsigned int no) const = 0;
    virtual const CartesianCoordinates* getNodes(const unsigned int timeStep) const = 0;
    virtual int const * getDomainNoNodes(const unsigned int timeStep) const = 0;
    virtual unsigned long getNoTetra4(const unsigned int timeStep) const = 0;
    virtual Tetra4 getTetra4(const unsigned int timeStep, const unsigned int no) const = 0;
    virtual unsigned int getMaxDomainNo(const unsigned int timeStep) const = 0;
    virtual unsigned long getSeparatedNode(const unsigned int timeStep, const unsigned long nodeNo, const unsigned int domainNo) = 0;
    virtual void setScalingFactor(const double value) = 0;
    virtual double getScalingFactor() const = 0;
    virtual void replaceNodeCoordinates(const unsigned int timeStep, const unsigned int no, const CartesianCoordinates coordinates) = 0;
protected:
private:
};
