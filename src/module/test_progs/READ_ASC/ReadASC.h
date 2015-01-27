/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_ASC_H
#define _READ_ASC_H
/**************************************************************************\ 
 **                                                   	        (C)1999 RUS **
 **                                                                        **
 ** Description: Read data from ASCII format                               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: A. Werner                                                      **
 **                                                                        **
 ** History:                                                               **
 ** April 99         v1                                                    **
 ** September 99     new covise api                                        **                               **
 **                                                                        **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;

class ReadASC : public coModule
{

private:
    //  member functions
    virtual int compute();

    //  ports
    coOutputPort *p_data;
    coFileBrowserParam *p_filename;

    // (recursively) read an object from the file: return NULL on error
    coDistributedObject *readObj(const char *name, istream &str);

    // Read special types
    coDistributedObject *readGeom(const char *name, char *command, istream &str);
    coDistributedObject *readUSG(const char *name, char *command, istream &str);
    coDistributedObject *readV3D(const char *name, char *command, istream &str);
    coDistributedObject *readS3D(const char *name, char *command, istream &str);
    coDistributedObject *readSet(const char *name, char *command, istream &str);

public:
    ReadASC();
    virtual ~ReadASC();
};
#endif
