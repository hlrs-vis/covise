/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "readcomsol.h"

ReadCOMSOL::ReadCOMSOL(int argc, char *argv[])
    : coFunctionModule(argc, argv, "ReadCOMSOL")
{
    initializeLocalVariables();
    createOutputPorts();
    createDialogElements();
}

ReadCOMSOL::~ReadCOMSOL()
{
    deleteLocalVariables();
}

void ReadCOMSOL::postInst(void)
{
    sendInfo("ITE ReadCOMSOL (C) Institut für Theorie der Elektrotechnik");
    _interfaceMatlab = InterfaceMatlab::getInstance();
    if (_interfaceMatlab->Connect())
    {
        sendInfo("A connection to MATLAB has been successfully established.");
        _interfaceMatlab->Execute("emptyModel = ModelUtil.create('emptyModel')");
        if (_interfaceMatlab->IsVariable("emptyModel"))
        {
            sendInfo("A connection to COMSOL Multiphysics has been successfully established.");
            _communicationComsol = CommunicationComsol::getInstance(_interfaceMatlab);
            _interfaceMatlab->Execute("clear emptyModel");
        }
        else
        {
            sendError("A connection to COMSOL Multiphysics could not be established.");
            sendError("Start COMSOL Multiphysics with MATLAB and run enableservice('automationserver', true)");
        }
    }
    else
        sendError("A connection to MATLAB could not be established.");
}

void ReadCOMSOL::param(const char *paramName, bool inMapLoading)
{
    if (!inMapLoading && (_fileBrowser != 0))
    {
        if (strcmp(paramName, _fileBrowser->getName()) == 0)
            readComsolModel(_fileBrowser->getValue());
    }
}

int ReadCOMSOL::compute(const char *port)
{
    MeshData *mesh = 0;
    PhysicalValues *physicalValues = 0;
    evaluateComsolModel(&mesh, &physicalValues);
    writeMesh(mesh);
    writeData(physicalValues);
    delete mesh;
    delete physicalValues;
    return SUCCESS;
}

MODULE_MAIN(IO, ReadCOMSOL)
