/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_OBJ_SIMPLE_H
#define _READ_OBJ_SIMPLE_H
/**************************************************************************\ 
 **                                                   	      (C)1999 RUS **
 **                                                                        **
 ** Description: Simple Reader for Wavefront OBJ Format	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: D. Rainer                                                      **
 **                                                                        **
 ** History:                                                               **
 ** April 99         v1                                                    **
 ** September 99     new covise api                                        **                               **
 **                                                                        **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;

class ReadObjSimple : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void quit();

    bool openFile();
    void readFile();

    //  member data
    const char *filename; // obj file name
    FILE *fp;

    coOutputPort *polygonPort;
    coFileBrowserParam *objFileParam;

public:
    ReadObjSimple(int argc, char *argv[]);
    virtual ~ReadObjSimple();
};
#endif
