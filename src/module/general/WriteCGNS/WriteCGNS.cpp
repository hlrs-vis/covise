/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <sstream>
#include <do/coDoGeometry.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoTexture.h>
#include "WriteCGNS.h"

WriteCGNS::WriteCGNS(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Write Unstructured Grid to cgns file")
{

    // ports
    gridPort = addInputPort("GridIn", "UnstructuredGrid", "unstructured grid input");
    gridPort->setRequired(true);

    // params
    m_FileNameParam = addFileBrowserParam("FileName", "Specify file name of cgns file to write.");
    m_FileNameParam->setValue(".", "*.cgns/*");
}

WriteCGNS::~WriteCGNS()
{
}

int WriteCGNS::compute(const char *)
{
    if (const coDoUnstructuredGrid *grid = dynamic_cast<const coDoUnstructuredGrid *>(gridPort->getCurrentObject()))
    {
        // write cgns file
        int error;
        error = cg_open(m_FileNameParam->getValue(), CG_MODE_WRITE, &cgnsFile);
        if (error != 0)
        {
            sendError("unambe to open file %s", m_FileNameParam->getValue());
            return STOP_PIPELINE;
        }

        error = cg_close(cgnsFile);
    }
    else
    {
        sendError("incompatible data type on input GridIn, expected coDoUnstructuredGrid");
        return STOP_PIPELINE;
    }

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, WriteCGNS)
