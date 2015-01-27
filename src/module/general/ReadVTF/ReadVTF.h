/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef READ_VTF_H
#define READ_VTF_H

/**************************************************************************\ 
 **                                                           (C)2001    **
 ** Description: Reading VTF binary data, in this file we analyze data   **
 **                                                                      **
 ** Author:                                                              **
 **                            Karin Mueller                             **
 **                                             Vircinity                **
 **                            Technologiezentrum                        **
 **                            70550 Stuttgart                           **
 ** Date:  01.10.01                                                      **
\**************************************************************************/

#include "ReadVTFData.h"
#include <util/coRestraint.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>

// structs for drawing polygons

typedef struct IntegerList *pt_integerList;

struct IntegerList
{
    int data;
    pt_integerList next;
};

typedef struct FloatList *pt_floatList;

struct FloatList
{
    float data;
    pt_floatList next;
};

//==============================================================================

class ReadVTF : public coModule
{
public:
    ReadVTF(int argc, char *argv[]);
    virtual ~ReadVTF();

private:
    //**** funktions
    virtual int compute(const char *port);
    // if a parameter is changed, this funktion will be called
    void param(const char *paramname, bool inMapLoading);
    int readFile();
    FILE *openFile(const char *filename);
    int getInfoAboutFile();
    ReadVTFData *tReadData;

    void drawResults();
    void drawPolygons();
    int findTransformation(int elementID, int timeStep);
    void transformCoords(float *x, float *y, float *z, int block, int step, bool IFS);
    int getUSGType(int iType); // get unstructured grid type

    // functions for working with lists
    pt_integerList newIntListElement(int data);
    bool putIntListElemAtEnd(pt_integerList last, int data);
    pt_floatList newFloatListElement(float data);
    bool putFloatListElemAtEnd(pt_floatList last, float data);
    void deleteIntListElem(pt_integerList ptr);
    void deleteFloatListElem(pt_floatList ptr);

    //**** ports
    coFileBrowserParam *ptBrowserParam;
    coOutputPort *pt_outPort;
    coOutputPort *pt_outPoly;
    coOutputPort *pt_outResults[NUMPORTS]; //scalar or vector data
    coChoiceParam *pt_choice[NUMPORTS];
    coStringParam *p_Selection;

    coRestraint sel;

    // variables for the port
    coDistributedObject **obj;
    coDoUnstructuredGrid *poly;
    coDoFloat *S3DRes[NUMPORTS];
    coDoVec3 *V3DRes[NUMPORTS];
    coDoVec3 *V3DResNew[NUMPORTS];

    //**** member variables
    const char *m_cPath;
    int *m_iNumCorners; // num_corners in coDoUnstructuredGrid
    int *m_iNumPoints;
    int *m_iNumPolygons;
    int m_iNumTimeSteps;
};
#endif
