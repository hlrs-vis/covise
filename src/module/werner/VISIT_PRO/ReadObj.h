/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_OBJ_H
#define _READ_OBJ_H
/**************************************************************************\ 
 **                                                   	      (C)1999 RUS **
 **                                                                        **
 ** Description: Reader for Wavefront OBJ Format	                          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: D. Rainer		                                              **
 **                                                                        **
 ** History:  								                              **
 ** 01-September-99 v1					       		                      **
 **                                                                        **
 **                                                                        **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;
class ReadObj
{
public:
    static coDistributedObject *read(const char *objFile,
                                     const char *objName,
                                     const char *unit);

    typedef char mtlNameType[100];

private:
    static FILE *openFile(const char *filename);
    static coDistributedObject *readObjFile(const char *objName, float scale);
    static void readMtlFile();
    static int makePackedColor(float r, float g, float b, float a);
    static int getCurrentColor(char *mtlName);

    //  member data
    static char *objFile, *mtlFile; // obj file name
    static FILE *objFp, *mtlFp;
    static int numMtls; // no of materials in the mtl file
    static int *pcList; // list of packed colors from the mtl file
    static int currentColor; // current color in packed format
    static mtlNameType *mtlNameList; // list of material names in the mtl file
};
#endif
