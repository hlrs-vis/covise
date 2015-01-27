/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// helpers for AVS
// filip sadlo
// cgl eth 2006
// $Id: avs_ext.h,v 1.2 2006/07/05 17:38:11 sadlof Exp $
//
// $Log: avs_ext.h,v $
// Revision 1.2  2006/07/05 17:38:11  sadlof
// Minor changes.
//
//

#ifndef _AVS_EXT_H_
#define _AVS_EXT_H_

//#include <vector>
#include "linalg.h"
#include <avs/avs.h>
#include <avs/ucd_defs.h>

using namespace std;

// Basic
int findInIntArray(int key, int *arr, int arr_len);
bool findInString(char *key, char sep, char *string);
bool findInString(int key, char sep, char *string);

// UCD related
int ucd_nodeCompNb(UCD_structure *ucd, int veclen = -1);
char ucd_nodeLabelDelim(UCD_structure *ucd);
int ucd_nodeCompLabel(UCD_structure *ucd, int comp_idx, char *dest);
int ucd_findNodeCompByVeclen(UCD_structure *ucd, int dest_veclen, int dest_idx,
                             int compare = 0);

//UCD_structure* ucdClone(UCD_structure* ucd, int veclen, char* label);
// set veclen to zero for all components
UCD_structure *ucdClone(UCD_structure *ucd, int veclen, char *name);
UCD_structure *ucdClone(UCD_structure *ucd, int componentCnt, int *components, char *name, char *labels = NULL, char *labelSep = ".");
// to be removed, don't use
//UCD_structure* ucdClone(UCD_structure* ucd, int veclen, char* label, bool allComponents=false) {
//  if (allComponents) ucdClone(ucd, 0, label);
//  else ucdClone(ucd, veclen, label);
//}
UCD_structure *generateUniformUCD(char *name,
                                  float originX, float originY, float originZ,
                                  int cellsX, int cellsY, int cellsZ,
                                  float cellSize,
                                  int componentCnt, int *components,
                                  char *labels, char *labelSep);

// GUI helpers
// updates AVS choice and determines selected UCD component
int ucd_processCompChoice(UCD_structure *ucd, char *ucd_name,
                          char *choice, char *choice_name, int dest_veclen,
                          int defaultComp = -1);

#endif // _AVS_EXT_H_
