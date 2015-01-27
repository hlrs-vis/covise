/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READDX_H
#define _READDX_H
/****************************************************************************\ 
 **                                                            (C)1999 RUS   **
 **                                                                          **
 ** Description: Reader for the IBM Dataexploerer format                     **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 ** Name:        ReadDx                                                      **
 ** Category:                                                                **
 **                                                                          **
 ** Author: C. Schwenzer		                                                **
 **                                                                          **
 ** History:  								                                **
 ** September-99  					       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
//covise specific includes
#include <api/coModule.h>
using namespace covise;
#include <api/coStepFile.h>
//Module specific includes
#include "parser.h"
#include "action.h"
#include "MultiGridMember.h"

const int maxDataPorts = 5;
class ReadDx : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);
    bool computeTimeStep(const char *fileName,
                         int timeStepNo,
                         coDistributedObject **resultGrid,
                         const char *gridObjName,
                         coDistributedObject **timeData[maxDataPorts],
                         int &numberOfDataSets);

    //  Ports
    coOutputPort *p_UnstrGrid;
    coOutputPort *p_ScalarData[maxDataPorts];

    // Determine how timesteps are handled.
    // Possible values
    // "None", "StepFile", "Normal"
    //None: There aren't any timestep
    //StepFile: Each timestep is in a separate file
    //This means you get a DO_set with timestep attribute
    //consisting of Distributed objects which may be coDoSets if
    //they in turn consist of parts
    //You get a coDoSet without timestep attribute
    //consisting of coDoSets with timestep attribute
    coChoiceParam *p_timeStepMode;

    //only valid for "StepFile" mode
    //denotes the number of files to be read in

    coIntScalarParam *p_timeSteps;

    //only valid for "StepFile" mode
    //determines how many files to skip
    coIntScalarParam *p_skip;

    //Path of the file(s) to be read in
    coFileBrowserParam *p_stepPath;

    // Users can stretch or squeeze a
    // grid which is read int
    coFloatSliderParam *p_xScale;
    coFloatSliderParam *p_yScale;
    coFloatSliderParam *p_zScale;
    // Determines whether the default byte order of data
    // to be read in is little endian or not
    // this value is only used if the byte order
    // int the dx file to be read in is unspecified
    // THIS HAS NOTHING TO DO WITH THE BYTEORDER OF
    // THE MACHINE THIS MODULE IS RUNNING ON.
    coBooleanParam *p_defaultIsLittleEndian;

public:
    static const int NONE = 0;
    static const int STEPFILE = 1;
    static const int NORMAL = 2;

    ReadDx(int argc, char *argv[]);
    virtual ~ReadDx();
    /** make a set of unstructured grids by reading the data
          @param objname Name of the distributed object
          @param
          @param m current multigrid to be read in
          @param arrays map of arrays belonging to this member
          @param fields map of fields belonging to this member
          @param dxPath path to the dx file to be read
          @param number
          @param reverse array
          * reverse[i] means the number of the vertex belonging to the
          * connection n
      @param reverseSize size of the reverse list
      */
    bool makeGridSet(const char *objname,
                     coDistributedObject **d,
                     MultiGridMember *m,
                     DxObjectMap &arrays,
                     DxObjectMap &fields,
                     const char *dxPath,
                     int number,
                     int *&reverse, int &reverseSize);
    /** make a set of data objects by reading the data
          @param objname Name of the distributed object
          @param d
          @param m current multigrid to be read in
          @param arrays map of arrays belonging to this member
          @param fields map of fields belonging to this member
          @param dxPath path to the dx file to be read
          @param number  number of the current part
          @param timeStepNo number of the current timestep
          @param reverse array
          * reverse[i] means the number of the vertex belonging to the
      * datavalue with number n
      @param reverseSize size of the reverse list
      */
    bool makeDataSet(coDistributedObject ***d,
                     MultiGridMember *m,
                     DxObjectMap &arrays,
                     const char *dxPath,
                     int partNumber,
                     int timeStepNo,
                     int &numberOfDataSets,
                     int *reverse, int reverseSize);
};

//MultiGridMember &m,
#endif
