/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2002 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class coRenderer                      ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 03.09.02                                                      ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <do/coDistributedObject.h>
#include "coRenderer.h"

using namespace covise;
//
// Constructor
//
coRenderer::coRenderer(int argc, char *argv[])
    : coModule(argc, argv)
    , inPort_(addInputPort("RenderData", "Geometry|Points|Lines|Polygons|TriangleStrips|Spheres", "input geometry"))
{

    inPort_->setRequired(0);
}

int
coRenderer::compute(const char *port)
{
    (void)port;
    sendError("All Renderers have to be put into Renderer group");
    return CONTINUE_PIPELINE; // dummy, should never be called
}

const coDistributedObject *
coRenderer::getInObj(const coObjInfo &info)
{
    const coDistributedObject *retObj = NULL;

    if (info.getName())
    {
        retObj = coDistributedObject::createFromShm(info);
    }

    return retObj;
}

//
// Destructor
//
coRenderer::~coRenderer()
{
}
