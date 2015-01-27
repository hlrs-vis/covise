/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CHECK_USG_H
#define CHECK_USG_H
/**************************************************************************\ 
 **                                                           (C)2001	  	  **
 ** Description: FixUSG delets points which are more than one time 		  **
 **				 in an UnstructuredGrid 											     **
 **                                                                        **
 ** Author:                                                                **
 **                            Karin Müller                             	  **
 **                				Vircinity               						  **
 **                            Technologiezentrum                          **
 **                            70550 Stuttgart                             **
 ** Date:  01.10.01		                                                  **
\**************************************************************************/

#include <do/coDoData.h>

namespace covise
{

// variables
int numCoordToRemove;
int *replBy;

int *filtered2source;
int *source2filtered;
int numFiltered;

// functions
// this is the main function which must be used, if you want to fix USG
ALGEXPORT coDistributedObject *checkUSG(coDistributedObject *DistrObj, const coObjInfo &outInfo);

// these are helping funktions for checkUSG
coDistributedObject *filterCoordinates(coDistributedObject *obj_in, const coObjInfo &outInfo,
                                       int master_num_coord, float *xcoord, float *ycoord, float *zcoord);

void boundingBox(float **x, float **y, float **z, int *c, int n,
                 float *bbx1, float *bby1, float *bbz1,
                 float *bbx2, float *bby2, float *bbz2);

int isEqual(float x1, float y1, float z1,
            float x2, float y2, float z2, float dist);

int getOctant(float x, float y, float z, float ox, float oy, float oz);

void getOctantBounds(int o, float ox, float oy, float oz, float bbx1, float bby1, float bbz1, float bbx2, float bby2, float bbz2, float *bx1, float *by1, float *bz1, float *bx2, float *by2, float *bz2);

void computeCell(float *xcoord, float *ycoord, float *zcoord,
                 int *coordInBox, int numCoordInBox,
                 float bbx1, float bby1, float bbz1,
                 float bbx2, float bby2, float bbz2,
                 int optimize, float maxDistanceSqr, int maxCoord);

void computeCell(float *xcoord, float *ycoord, float *zcoord,
                 int *coordInBox, int numCoordInBox,
                 float bbx1, float bby1, float bbz1,
                 float bbx2, float bby2, float bbz2,
                 int maxCoord,
                 int *replBy, int &numCoordToRemove);
void computeWorkingLists(int num_coord);

void computeReplaceLists(int num_coord, int *replBy,
                         int *&src2filt, int *&filtered2source);
}
#endif
