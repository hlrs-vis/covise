/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LoadCadData_H
#define _LoadCadData_H
#include <api/coModule.h>
using namespace covise;
#include <do/coDoData.h>
class LoadCadData : public coModule
{

public:
    LoadCadData(int argc, char **argv);
    virtual ~LoadCadData();

private:
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);

    coOutputPort *p_pointName;
    coFileBrowserParam *p_modelPath;
    coFloatParam *p_scale;

    coFloatVectorParam *p_resize;
    coFloatVectorParam *p_tansvec;
    coFloatParam *p_rotangle;
    coFloatVectorParam *p_rotvec;
    coBooleanParam *p_backface;
    coBooleanParam *p_orientation_iv;
    coBooleanParam *p_convert_xforms_iv;

    char modelPath[1024];
    char *pointName;
    coDoPoints *point;
};

#endif
