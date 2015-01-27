/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_DASIM_H
#define _READ_DASIM_H
/**************************************************************************\
**                                                   	      (C)2002 RUS **
**                                                                        **
** Description: READ SoundVol result files             	                  **
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

class ReadMap : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);

    //  member data
    coOutputPort *p_mesh;
    coOutputPort *p_data;
    coOutputPort *p_velo;
    coFileBrowserParam *p_fileParam;
    coFileBrowserParam *p_GridPath;
    coFileBrowserParam *p_dirPath;
    coFileBrowserParam *p_vPath;
    coIntScalarParam *p_xGridSize;
    coIntScalarParam *p_yGridSize;
    coFloatParam *p_xSize;
    coFloatParam *p_ySize;
    coFloatParam *p_zScale;

public:
    ReadMap(int argc, char *argv[]);
    virtual ~ReadMap();
};

#endif
