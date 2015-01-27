/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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

#ifdef VTK
#include "vtkAlgorithm.h"
#endif

using namespace std;

class UniSys
{

private:
#ifdef COVISE
    covise::coModule *covModule;
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
