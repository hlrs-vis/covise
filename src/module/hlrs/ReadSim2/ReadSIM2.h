/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READSIM2_H
#define _READSIM2_H
/**********************************************************************************\
 **                                                                    (C)2009   **
 **                                                                              **
 ** Description: VTK data reading Module                                         **
 **              reads data in vtk format                                        **
 **              either a single file or a set of files (different timesteps)    **
 **                                                                              **
 ** Name:        ReadSIM                                                    **
 ** Category:    IO                                                              **
 **                                                                              **
 ** Author:      Julia Portl                                                     **
 **              Visualization and Numerical Geometry Group, IWR                 **
 **              Heidelberg University                                           **
 **                                                                              **
 ** History:     April 2009                                                      **

 Modified by Daniel Jungblut: G-CSC Frankfurt University
 October 2009



 **                                                                              **
 **                                                                              **
 \**********************************************************************************/

#include <api/coSimpleModule.h>
#include <api/coFileBrowserParam.h>
#include <api/coFeedback.h>

// Used Datatypes
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>

using namespace covise;

class ReadSIM2 : public coSimpleModule
{

private:
    //  member functions
    virtual int compute(const char *);

    // Params
    coIntScalarParam *maxTimeSteps;
    coFileBrowserParam *fileBrowser;
    coBooleanParam *processAllTimeSteps;

    //  Ports
    coOutputPort *GridOutPort;
    coOutputPort *VectorOutPort;
    coOutputPort *ScalarOutPort;

public:
    ReadSIM2(int argc, char *argv[]);
};

#endif
