/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "readcomsol.h"

void ReadCOMSOL::initializeLocalVariables()
{
    _interfaceMatlab = 0;
    _communicationComsol = 0;
    _physics = 0;
    _portOutMesh = 0;
    _portOutMeshDomain = 0;
    _noPortsVec = 0;
    _noPortsScal = 0;
    _portOutVec = 0;
    _portOutScal = 0;
    _fileBrowser = 0;
    _fileName = "";
    _paramOutVec = 0;
    _paramOutScal = 0;
    _paramDataVec.resize(0);
    _paramDataScal.resize(0);
    _namesApplicationModes.resize(0);
}

void ReadCOMSOL::deleteLocalVariables()
{
    if (_interfaceMatlab != 0)
        delete _interfaceMatlab;
    if (_communicationComsol != 0)
        delete _communicationComsol;
    if (_physics != 0)
        delete _physics;
    if (_portOutVec != 0)
        delete[] _portOutVec;
    if (_portOutScal != 0)
        delete[] _portOutScal;
    if (_paramOutVec != 0)
        delete[] _paramOutVec;
    if (_paramOutScal != 0)
        delete[] _paramOutScal;
}
