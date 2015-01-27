/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SimReference.h"

using namespace opencover;

SimReference::SimReference(const char *SimuPath, const char *SimuName)
    : osg::Referenced()
{

    SimPath.push_back(SimuPath);
    SimName.push_back(SimuName);
}

std::vector<std::string> SimReference::getSimPath()
{
    return SimPath;
}
std::vector<std::string> SimReference::getSimName()
{
    return SimName;
}
void SimReference::addSim(const char *SimuPath, const char *SimuName)
{
    SimPath.push_back(SimuPath);
    SimName.push_back(SimuName);
}
