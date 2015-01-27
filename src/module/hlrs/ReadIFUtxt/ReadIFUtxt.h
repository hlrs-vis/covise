/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_IFU_TXT_H
#define _READ_IFU_TXT_H
/**************************************************************************\
**                                                   	      (C)2007 HLRS  **
**                                                                        **
** Description: READ IFU Measurements and simulation results              **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: Uwe Woessner                                                   **                             **
**                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <api/coModule.h>
using namespace covise;

class ReadIFUtxt : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);

    //  member data
    coOutputPort *p_mesh;
    coOutputPort *p_mesh2;
    coOutputPort *p_data;
    coOutputPort *p_data2;
    coOutputPort *p_velo;
    coOutputPort *p_velo2;
    coOutputPort *p_dx_12;
    coFileBrowserParam *p_fileParam;
    coFileBrowserParam *p_GridPath;
    coBooleanParam *p_read_GridPath2;
    coFileBrowserParam *p_GridPath2;
    coFileBrowserParam *p_dirPath;
    coFileBrowserParam *p_vPath;
    coIntScalarParam *p_xGridSize;
    coIntScalarParam *p_yGridSize;
    coFloatParam *p_xSize;
    coFloatParam *p_ySize;
    coFloatParam *p_zScale;

    coFloatVectorParam *p_offset;

public:
    ReadIFUtxt(int argc, char *argv[]);
    virtual ~ReadIFUtxt();
};

#endif
