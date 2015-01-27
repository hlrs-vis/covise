/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "readcomsol.h"

std::string ReadCOMSOL::getListName(const std::string baseName, const unsigned int no, const bool space) const
{
    std::stringstream temp;
    temp << baseName;
    if (space)
        temp << " ";
    else
        temp << "_";
    temp << no + 1;
    return temp.str();
}

CartesianCoordinates *ReadCOMSOL::getEvaluationPoints(const MeshData *mesh, unsigned long &noEvaluationPoints, int **domainNoEvaluationPoints, const unsigned int timeStep) const
{
    noEvaluationPoints = mesh->getNoNodes(timeStep);
    *domainNoEvaluationPoints = new int[noEvaluationPoints];
    CartesianCoordinates *retVal = new CartesianCoordinates[noEvaluationPoints];
    const int *domainNoNodes = mesh->getDomainNoNodes(timeStep);
    int *refDomainNo = *domainNoEvaluationPoints;
    for (unsigned long i = 0; i < noEvaluationPoints; i++)
    {
        retVal[i] = mesh->getNode(timeStep, i);
        refDomainNo[i] = domainNoNodes[i];
    }
    return retVal;
}
