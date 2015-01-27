/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2008 Visenso  **
 **                                                                        **
 ** Description: ReadBifBof Plugin                                         **
 **              Reads in bif and bof files                                **
 **              Output: points or unstructured grid                       **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _RWBIFBOF_H
#define _RWBIFBOF_H

#include <api/coModule.h>
using namespace covise;
#include "covise/covise.h"
//#include <covise/covise_config.h>

#include <string>
#include <vector>

#include "BifBofInterface.h"
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include "BifElement.h"
#include "BifNodalPoints.h"
#include "BifGeoElements.h"
//#include "BifPartDefinition.h"

#include "BofScalarData.h"
#include "BofVectorData.h"

class ReadBIFBOF : public coModule
{

public:
    ReadBIFBOF(int argc, char **argv);
    virtual ~ReadBIFBOF();

private:
    BifBof::Word headBuffer[30];
    // compute callback
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);
    // parameter

    coFileBrowserParam *p_bifFileParam;
    coFileBrowserParam *p_bofFileParam;

    // ports
    coOutputPort *p_3D_gridOutPort;
    coOutputPort *p_2D_gridOutPort;
    coOutputPort *p_VectorData;
    coOutputPort *p_ScalarData;

    // Methods
    int getHeaderIntWord(int num);
    float getHeaderFloatWord(int num);
    char *getHeaderCharWord(int num);
    int handleError();
    void makeDseleMap();
    // Variables
    BifBof *bifBof;
    int readingComplete;
    float *scalarValues;
    float *xValues;
    float *yValues;
    float *zValues;
    // float           *uValues;
    // float           *vValues;
    // float           *wValues;
    std::string speciesName;
    //int glob_minNodeID;
    //int glob_maxNodeID;
};

#endif
