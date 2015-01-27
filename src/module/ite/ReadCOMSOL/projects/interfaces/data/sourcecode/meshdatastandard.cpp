/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "meshdatastandard.h"
#include <stdlib.h>
#include <iostream>

MeshDataStandard::MeshDataStandard(const unsigned int noTimeSteps)
{
    _scalingFactor = 1.0;
    _data.resize(noTimeSteps);
    for (unsigned int i = 0; i < noTimeSteps; i++)
    {
        _data[i]._domainNoNodes.resize(0);
        _data[i]._nodes.resize(0);
        _data[i]._setNodeNo2NodeNo.resize(0);
        _data[i]._tetra4.resize(0);
    }
}

MeshDataStandard::~MeshDataStandard()
{
}

void MeshDataStandard::addNode(const unsigned int timeStep, const CartesianCoordinates coordinates)
{
    if (timeStep < _data.size())
    {
        _data[timeStep]._nodes.push_back(coordinates);
        _data[timeStep]._domainNoNodes.push_back(-1);
        const unsigned int index = _data[timeStep]._setNodeNo2NodeNo.size();
        _data[timeStep]._setNodeNo2NodeNo.resize(index + 1);
        _data[timeStep]._setNodeNo2NodeNo[index].resize(1, index);
    }
}

void MeshDataStandard::addTetra4(const unsigned int timeStep, const Tetra4 element)
{
    if (timeStep < _data.size())
    {
        const unsigned int index = _data[timeStep]._tetra4.size();
        _data[timeStep]._tetra4.resize(index + 1);
        _data[timeStep]._tetra4[index].domainNo = element.domainNo;
        _data[timeStep]._tetra4[index].node1 = getSeparatedNode(timeStep, element.node1, element.domainNo);
        _data[timeStep]._tetra4[index].node2 = getSeparatedNode(timeStep, element.node2, element.domainNo);
        _data[timeStep]._tetra4[index].node3 = getSeparatedNode(timeStep, element.node3, element.domainNo);
        _data[timeStep]._tetra4[index].node4 = getSeparatedNode(timeStep, element.node4, element.domainNo);
    }
}

unsigned long MeshDataStandard::getNoNodes(const unsigned int timeStep) const
{
    unsigned long retVal = 0;
    if (timeStep < _data.size())
        retVal = _data[timeStep]._nodes.size();
    return retVal;
}

CartesianCoordinates MeshDataStandard::getNode(const unsigned int timeStep, const unsigned int no) const
{
    CartesianCoordinates retVal;
    retVal.x = 0;
    retVal.y = 0;
    retVal.z = 0;
    if (timeStep < _data.size())
    {
        if (no < _data[timeStep]._nodes.size())
            retVal = _data[timeStep]._nodes[no];
    }
    return retVal;
}

const CartesianCoordinates *MeshDataStandard::getNodes(const unsigned int timeStep) const
{
    return &_data[timeStep]._nodes[0];
}

int const *MeshDataStandard::getDomainNoNodes(const unsigned int timeStep) const
{
    return &_data[timeStep]._domainNoNodes[0];
}

unsigned long MeshDataStandard::getNoTetra4(const unsigned int timeStep) const
{
    unsigned long retVal = 0;
    if (timeStep < _data.size())
        retVal = _data[timeStep]._tetra4.size();
    return retVal;
}

Tetra4 MeshDataStandard::getTetra4(const unsigned int timeStep, const unsigned int no) const
{
    Tetra4 retVal;
    retVal.domainNo = 0;
    retVal.node1 = 0;
    retVal.node2 = 0;
    retVal.node3 = 0;
    retVal.node4 = 0;
    if (timeStep < _data.size())
    {
        if (no < _data[timeStep]._tetra4.size())
            retVal = _data[timeStep]._tetra4[no];
    }
    return retVal;
}

unsigned int MeshDataStandard::getMaxDomainNo(const unsigned int timeStep) const
{
    unsigned int retVal = 0;
    if (timeStep < _data.size())
    {
        for (unsigned long i = 0; i < _data[timeStep]._tetra4.size(); i++)
        {
            const unsigned int domainNo = _data[timeStep]._tetra4[i].domainNo;
            if (domainNo > retVal)
                retVal = domainNo;
        }
    }
    return retVal;
}

unsigned long MeshDataStandard::getSeparatedNode(const unsigned int timeStep, const unsigned long nodeNo, const unsigned int domainNo)
{
    unsigned long retVal = nodeNo;
    if (_data[timeStep]._domainNoNodes[nodeNo] == -1)
        _data[timeStep]._domainNoNodes[nodeNo] = domainNo;
    else
    {
        bool separatedNodeExists = false;
        for (unsigned int i = 0; i < _data[timeStep]._setNodeNo2NodeNo[nodeNo].size(); i++)
        {
            if (_data[timeStep]._domainNoNodes[_data[timeStep]._setNodeNo2NodeNo[nodeNo][i]] == domainNo)
            {
                separatedNodeExists = true;
                retVal = _data[timeStep]._setNodeNo2NodeNo[nodeNo][i];
                break;
            }
        }
        if (!separatedNodeExists)
        {
            retVal = _data[timeStep]._nodes.size();
            _data[timeStep]._nodes.push_back(_data[timeStep]._nodes[nodeNo]);
            _data[timeStep]._domainNoNodes.push_back(domainNo);
            _data[timeStep]._setNodeNo2NodeNo[nodeNo].push_back(retVal);
        }
    }
    return retVal;
}

void MeshDataStandard::setScalingFactor(const double value)
{
    _scalingFactor = value;
}

double MeshDataStandard::getScalingFactor() const
{
    return _scalingFactor;
}

void MeshDataStandard::replaceNodeCoordinates(const unsigned int timeStep, const unsigned int no, const CartesianCoordinates coordinates)
{
    if (timeStep < _data.size())
    {
        if (no < _data[timeStep]._nodes.size())
            _data[timeStep]._nodes[no] = coordinates;
    }
}
