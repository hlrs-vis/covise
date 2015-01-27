/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_RFL_H
#define _READ_RFL_H
/**************************************************************************\ 
 **                                                   	      (C)2001     **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Björn Sander/Uwe                                               **
 ** =============================================================================
 ** READRFL Modul zum Lesen von ANSYS RFL-Ergebnisfiles (FLOWTRAN)
 ** -----------------------------------------------------------------------------
 ** 17.9.2001  Björn Sander
 ** =============================================================================
 **                           **
 **                                                                        **
\**************************************************************************/

#include <api/coModule.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
using namespace covise;
#include "ReadRflFile.h"

#define NUMPORTS 5

class ReadRFL : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void quit();
    virtual void param(const char *paramName, bool inMapLoading);
    void updateChoices();

    //  member data
    const char *filename; // obj file name
    FILE *fp;

    READRFL *rflFile;
    coOutputPort *gridPort;
    coFileBrowserParam *rflFileParam;
    coDoUnstructuredGrid *gridObject;
    int numVertices;
    int *el, *vl, *tl;
    float *x_c, *y_c, *z_c;
    int *typeList;
    coChoiceParam *dofs[NUMPORTS];
    coOutputPort *data[NUMPORTS];
    coIntSliderParam *datasetNum;
    coIntSliderParam *numTimesteps;
    coIntScalarParam *numSkip;

public:
    ReadRFL(int argc, char *argv[]);
    virtual ~ReadRFL();
};
#endif
