/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// container for mesh data standard implementation
// author: A. Buchau
// modified: 18.11.2010

#pragma once

#include "../include/meshdata.hxx"
#include <vector>

class MeshDataStandard : public MeshData
{
public:
    MeshDataStandard(const unsigned int noTimeSteps);
    ~MeshDataStandard();
    void addNode(const unsigned int timeStep, const CartesianCoordinates coordinates);
    void addTetra4(const unsigned int timeStep, const Tetra4 element);
    unsigned long getNoNodes(const unsigned int timeStep) const;
    CartesianCoordinates getNode(const unsigned int timeStep, const unsigned int no) const;
    const CartesianCoordinates *getNodes(const unsigned int timeStep) const;
    int const *getDomainNoNodes(const unsigned int timeStep) const;
    unsigned long getNoTetra4(const unsigned int timeStep) const;
    Tetra4 getTetra4(const unsigned int timeStep, const unsigned int no) const;
    unsigned int getMaxDomainNo(const unsigned int timeStep) const;
    unsigned long getSeparatedNode(const unsigned int timeStep, const unsigned long nodeNo, const unsigned int domainNo);
    void setScalingFactor(const double value);
    double getScalingFactor() const;
    void replaceNodeCoordinates(const unsigned int timeStep, const unsigned int no, const CartesianCoordinates coordinates);

protected:
private:
    struct dataTimeStep
    {
        std::vector<CartesianCoordinates> _nodes;
        std::vector<int> _domainNoNodes;
        std::vector<std::vector<unsigned int> > _setNodeNo2NodeNo;
        std::vector<Tetra4> _tetra4;
    };
    std::vector<dataTimeStep> _data;
    double _scalingFactor;
};
