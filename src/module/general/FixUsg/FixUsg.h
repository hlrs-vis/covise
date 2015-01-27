/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__FIXUSG_H)
#define __FIXUSG_H

#include <api/coSimpleModule.h>
using namespace covise;

class FixUSG : public coSimpleModule
{

private:
    coInputPort *inMeshPort;
    coOutputPort *outMeshPort;
    char **inPortNames;
    char **outPortNames;
    coInputPort **ptrDataInPort;
    coOutputPort **ptrDataOutPort;
    coIntScalarParam *paramMaxvertices;
    coFloatParam *paramDelta;

    coChoiceParam *paramAlgorithm, *paramOptimize;
    const char *strOutPortName;

protected:
    // number of additional data-ports

    enum
    {
        num_ports = 5
    };

public:
    FixUSG(int argc, char *argv[]);
    virtual ~FixUSG();

    virtual int compute(const char *port);
};
#endif // __FIXUSG_H
