/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _WritePolygon_H
#define _RW_ASCII_H
/**************************************************************************\ 
 **                                                           (C)2008 HLRS **
 **                                                                        **
 ** Description: Write Polygondata to files (VRML, STL, ***)               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Bruno Burbaum                               **
 **                                 HLRS                                   **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  08.10.08  V0.1                                                  **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;

class WritePolygon : public coModule
{
private:
    // member functions
    virtual int compute(const char *port);
    enum fileformats {TYPE_WRL=0,TYPE_STL,TYPE_STL_TRI};

    // ports
    //      coOutputPort *p_dataOut;
    coInputPort *p_dataIn;

    coFileBrowserParam *p_filename;
    coBooleanParam *p_newFile;
    coChoiceParam *p_fileFormat;
    const coDistributedObject *currentObject;


    // write object
    void writeObj(const char *offset, const coDistributedObject *new_data);
    void writeSTLObj(const char *offset, const coDistributedObject *new_data);

    void writeFileBegin();
    void writeFileEnd();
    FILE *file;
    int outputtype;

public:
    // constructor
    WritePolygon(int argc, char *argv[]);

    // destructor
    virtual ~WritePolygon();
};
#endif
