/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TRANS3D_H
#define _TRANS3D_H
/**************************************************************************\ 
 **                                                   	      (C)2000 RUS **
 **                                                                        **
 ** Description:  Trans3D  Simulation Module            	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Uwe Woessner                                                   **
 ** History:                                                               **
 ** Apr 00         v1                                                      **                               **
 **                                                                        **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;

#include "trans3DInterface.h"

class Trans3D : public coModule
{

private:
    //  member functions
    virtual int compute();
    virtual void quit();
    virtual float idle();

    void createObjects();
    void updateParameters();

    int xDim, yDim, zDim;
    int timestep;
    bool calculating;

    //  member data
    const char *filename; // Trans3D file name
    const char *inifilename; // Trans3D init file name
    FILE *fp;

    coOutputPort *gridPort;
    coOutputPort *TDataPort;
    coOutputPort *QDataPort;
    coFileBrowserParam *iniFileParam;
    coFileBrowserParam *fileParam;
    coIntScalarParam *numVisStepsParam;
    coFloatParam *radius;
    coFloatParam *intensitaet;
    coFloatParam *divergenz;
    coFloatParam *wellenlaenge;
    coFloatParam *strahlradius;
    coFloatParam *fokuslage;
    coFloatVectorParam *laserPos;

public:
    Trans3D();
    virtual ~Trans3D();
};
#endif
