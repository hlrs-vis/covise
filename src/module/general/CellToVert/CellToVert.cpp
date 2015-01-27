/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                     (C)2005 Visenso ++
// ++ Description: Interpolation from Cell Data to Vertex Data            ++
// ++                 ( CellToVert module functionality  )                ++
// ++                                                                     ++
// ++ Author: Sven Kufer( sk@visenso.de)                                  ++
// ++                                                                     ++
// ++**********************************************************************/

#include "CellToVert.h"
#include <util/coviseCompat.h>
#include <alg/coCellToVert.h>
#include <do/coDoData.h>

using namespace covise;

CellToVert::CellToVert(int argc, char *argv[])
    : coSimpleModule(argc, argv, "interpolates per-cell data to per-vertex data")
{
    // initialize

    // input
    grid_in = addInputPort("GridIn0", "UnstructuredGrid|Polygons|Lines", "mesh");
    data_in = addInputPort("DataIn0", "Float|Vec3", "data");

    // parameters
    algorithm = addChoiceParam("algorithm", "how to do the interpolation");

    const char *algoChoices[] = { "SqrWeight", "Simple" };
    algorithm->setValue(2, algoChoices, 0);

    // output
    data_out = addOutputPort("DataOut0", "Float|Vec3", "data");

    setCopyAttributes(1);
}

////// hello
int CellToVert::compute(const char *)
{
    // module settings
    coCellToVert::Algorithm algo_option;

    // output
    coDistributedObject *returnObject;

    // get parameter
    if (algorithm->getValue() == 0)
        algo_option = coCellToVert::SQR_WEIGHT;
    else
        algo_option = coCellToVert::SIMPLE;

    coCellToVert fct;
    // here we go
    returnObject = fct.interpolate(grid_in->getCurrentObject(), data_in->getCurrentObject(), data_out->getObjName(), algo_option);

    if (!returnObject)
    {
        Covise::sendWarning("Failed to interpolate data.");
        if (dynamic_cast<const coDoFloat *>(data_in->getCurrentObject()))
            returnObject = new coDoFloat(data_out->getObjName(), 0);
        else
            returnObject = new coDoVec3(data_out->getObjName(), 0);
    }

    copyAttributes(returnObject, data_in->getCurrentObject());
    data_out->setCurrentObject(returnObject);
    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Interpolator, CellToVert)
