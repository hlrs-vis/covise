/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FixUsg.h"
#include <api/coFeedback.h>
#include <util/coviseCompat.h>
#include <alg/coFixUsg.h>

FixUSG::FixUSG(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Filter for unnecessary coordinates")
{
    int i;
    ptrDataInPort = new coInputPort *[num_ports + 1];
    ptrDataOutPort = new coOutputPort *[num_ports + 1];

    inMeshPort = addInputPort("GridIn0", "UnstructuredGrid|Polygons|Lines", "Unstructured Grid or Polygons");

    for (i = 0; i < num_ports; i++)
    {
        char portname[20];

        sprintf(portname, "DataIn%i", i);
        ptrDataInPort[i] = addInputPort(portname, "Float|Vec3", "data");
        ptrDataInPort[i]->setRequired(0);
    }

    paramMaxvertices = addInt32Param("maxvertices", "min. number of vertices in box for further recursion");
    paramMaxvertices->setValue(50);

    paramDelta = addFloatParam("delta", "max. distance between two vertices");
    paramDelta->setValue(0.0);

    paramAlgorithm = addChoiceParam("algorithm", "choose your favorite algorithm");
    const char *alg_choice[] = { "BoundingBox", "None" };
    paramAlgorithm->setValue(2, alg_choice, 0);

    paramOptimize = addChoiceParam("optimize", "should we care 'bout RAM or not");
    const char *opt_choice[] = { "speed", "memory" };
    paramOptimize->setValue(2, opt_choice, 0);
    outMeshPort = addOutputPort("GridOut0", "UnstructuredGrid|Polygons|Lines", "filtered USG or Polygons");

    for (i = 0; i < num_ports; i++)
    {
        char portname[20];
        sprintf(portname, "DataOut%i", i);
        ptrDataOutPort[i] = addOutputPort(portname, "Float|Vec3", "data");
        ptrDataOutPort[i]->setDependencyPort(ptrDataInPort[i]);
    }
    setCopyAttributes(1);

    return;
};

FixUSG::~FixUSG()
{
    delete[] ptrDataInPort;
    delete[] ptrDataOutPort;
}

int FixUSG::compute(const char *)
{

    coFixUsg fct(paramMaxvertices->getValue(), paramDelta->getValue(),
                 paramOptimize->getValue() != 0 /* false means optimize for speed */);

    const coDistributedObject **in_data = new const coDistributedObject *[num_ports];
    const char **outNames = new const char *[num_ports];
    const char *outGridName = outMeshPort->getObjName();
    int *tgt2src = new int[num_ports];
    int i, num_used = 0;
    for (i = 0; i < num_ports; i++)
    {
        if (ptrDataInPort[i]->getCurrentObject())
        {
            in_data[num_used] = ptrDataInPort[i]->getCurrentObject();
            outNames[num_used] = ptrDataOutPort[i]->getObjName();
            tgt2src[num_used] = i;
            num_used++;
        }
    }

    // malloc
    coDistributedObject **returnObject = new coDistributedObject *[num_used];
    coDistributedObject *geo_out;
    bool fail = false;
    int num_red;
    if ((num_red = fct.fixUsg(inMeshPort->getCurrentObject(), &geo_out, outGridName, num_used, in_data, returnObject, outNames)) != coFixUsg::FIX_ERROR)
    {
        for (i = 0; i < num_used; i++)
        {
            ptrDataOutPort[tgt2src[i]]->setCurrentObject(returnObject[i]);
        }
        outMeshPort->setCurrentObject(geo_out);
        Covise::sendInfo("removed %d points", num_red);
    }
    else
    {
        fail = true;
        Covise::sendError("Failure.");
    }
    delete[] in_data;
    delete[] outNames;
    delete[] tgt2src;
    delete[] returnObject;

    if (!fail)
        return CONTINUE_PIPELINE;
    else
        return STOP_PIPELINE;
}

MODULE_MAIN(Tools, FixUSG)
