/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2010 RRZK  **
 **                                                                          **
 ** Description: Test module for coVTK class                                 **
 **              converts from COVISE to VTK and back                        **
 **                                                                          **
 ** Name:        TestVtk                                                     **
 ** Category:    examples                                                    **
 **                                                                          **
 ** Author: Martin Aumueller <aumueller@uni-koeln.de>                        **
 **                                                                          **
\****************************************************************************/

#include <do/coDoData.h>
#include <vtk/coVtk.h>
#include "TestVtk.h"

TestVtk::TestVtk(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Convert from COVISE to VTK data and back")
{
    input = addInputPort("GridIn0", "UniformGrid|RectilinearGrid|StructuredGrid|UnstructuredGrid"
                                    "|Polygons|Lines|TriangleStrips|Points",
                         "input grid");

    output = addOutputPort("GridOut0", "UniformGrid|RectilinearGrid|StructuredGrid|UnstructuredGrid"
                                       "|Polygons|Lines|TriangleStrips|Points",
                           "output grid");
}

int TestVtk::compute(const char *port)
{
    (void)port;

    const coDistributedObject *in = input->getCurrentObject();
    coDistributedObject *out = NULL;
    vtkDataSet *vtk = coVtk::covise2Vtk(in);

    if (vtk)
        out = coVtk::vtkGrid2Covise(output->getNewObjectInfo(), vtk);

    output->setCurrentObject(out);

    return SUCCESS;
}

TestVtk::~TestVtk()
{
}

MODULE_MAIN(Examples, TestVtk)
