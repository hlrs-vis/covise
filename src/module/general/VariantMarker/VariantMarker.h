/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VARIANTMARKER_H
#define _VARIANTMARKER_H
#include <api/coModule.h>
using namespace covise;
#include <do/coDoData.h>
class VariantMarker : public coModule
{

public:
    VariantMarker(int argc, char **argv);
    virtual ~VariantMarker();

private:
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);
    coInputPort *p_Inport;
    coOutputPort *p_Outport;
    coStringParam *p_varName;
};

#endif
