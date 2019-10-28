// Unification Library for Modular Visualization Systems
//
// System
//
// CGL ETH Zuerich
// Filip Sadlo 2006 - 2007

// Usage: define either AVS or COVISE or VTK
//        define also COVISE5 for Covise 5

#ifndef _UNISYS_H_
#define _UNISYS_H_

#ifdef AVS
#include <avs/avs.h>
#endif

#ifdef COVISE
#ifdef COVISE5
#include <coModule.h>
#else
#include <api/coModule.h>
#endif
#endif

#ifdef VISTLE
#include <module/module.h>
#endif

#ifdef VTK
#include "vtkAlgorithm.h"
#endif

#ifdef VISTLE
#include "../vistle_ext/export.h"

class V_UNIVIZEXPORT UniSys
#else
class UniSys
#endif
{

private:
#ifdef COVISE
    covise::coModule *covModule;
#endif

#ifdef VISTLE
    vistle::Module *vistleModule;
#endif

#ifdef VTK
    vtkAlgorithm *vtkAlg;
#endif

public:
#ifdef AVS
    UniSys()
    {
        ;
    }
#endif

#ifdef COVISE
    UniSys(covise::coModule *m)
    {
        covModule = m;
    }
#endif

#ifdef VISTLE
    UniSys(vistle::Module *mod)
    {
        vistleModule = mod;
    }
#endif

#ifdef VTK
    UniSys(vtkAlgorithm *a)
    {
        vtkAlg = a;
    }
#endif

    void info(const char *str, ...);
    void warning(const char *str, ...);
    void error(const char *str, ...);
    void moduleStatus(const char *str, int perc);
    bool inputChanged(const char *name, int connection);
    bool parameterChanged(const char *name);
};

#endif // _UNISYS_H_
