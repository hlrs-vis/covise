/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class SplitGeometry                      ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 04.10.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <do/coDoGeometry.h>
#include <do/coDoSet.h>
#include <do/coDoIntArr.h>
#include <do/coDoData.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoTriangleStrips.h>
#include "SplitGeometry.h"
#include <string>

//#define VERBOSE

//
// Constructor
//
SplitGeometry::SplitGeometry(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Split Geometry objects into grid, colors, normals, and texture")
{
    // Ports
    m_inPort = addInputPort("DataIn0", "Geometry", "data set");
    m_outPortGrid = addOutputPort("GridOut0", "UniformGrid|RectilinearGrid|StructuredGrid|UnstructuredGrid|Points|Spheres|Lines|Polygons|TriangleStrips", "grid");
    m_outPortData = addOutputPort("DataOut0", "RGBA|Int|Float|Vec2|Vec3|Tensor", "mapped data/colors");
    m_outPortNormals = addOutputPort("DataOut1", "Vec3", "normal data");
    m_outPortTexture = addOutputPort("DataOut2", "Texture", "texture data");
}

//
// compute method
//
int
SplitGeometry::compute(const char *)
{
    ////////////////////////////////////////////////////
    //       P O R T S
    ////////////////////////////////////////////////////

    const coDistributedObject *gridInObj = m_inPort->getCurrentObject();
    if (!gridInObj)
    {
        sendError("no input");
        return STOP_PIPELINE;
    }

    const coDoGeometry *geom = dynamic_cast<const coDoGeometry *>(gridInObj);
    if (!geom)
    {
        sendError("input must be Geometry");
        return STOP_PIPELINE;
    }

    for (int i = 0; i < 4; ++i)
    {
        // is the index reasonable?
        const coDistributedObject *shmSetElem = NULL;
        coOutputPort *port = NULL;
        switch (i)
        {
        case 0:
            shmSetElem = geom->getGeometry();
            port = m_outPortGrid;
            break;
        case 1:
            shmSetElem = geom->getColors();
            port = m_outPortData;
            break;
        case 2:
            shmSetElem = geom->getNormals();
            port = m_outPortNormals;
            break;
        case 3:
            shmSetElem = geom->getTexture();
            port = m_outPortTexture;
            break;
        default:
            shmSetElem = NULL;
            port = NULL;
        }

        if (shmSetElem)
        {
            port->setCurrentObject(shmSetElem->clone(port->getObjName()));
        }
    }
    return CONTINUE_PIPELINE;
}

//
// Destructor
//
SplitGeometry::~SplitGeometry()
{
}

MODULE_MAIN(Tools, SplitGeometry)
