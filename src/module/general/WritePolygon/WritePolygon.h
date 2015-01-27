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

    // ports
    //      coOutputPort *p_dataOut;
    coInputPort *p_dataIn;

    coFileBrowserParam *p_filename;
    coBooleanParam *p_newFile;

    // read object
    coDistributedObject *readObj(const char *name, istream &str);

    // write object
    void writeObj(const char *offset, const coDistributedObject *new_data, FILE *file);

    void writeFileBegin(int outputtype, FILE *file);
    void writeFileEnd(int outputtype, FILE *file);

    // read covise objects
    coDistributedObject *readPOLYGN(const char *name, char *command, istream &str);
    coDistributedObject *readLINES(const char *name, char *command, istream &str);
    coDistributedObject *readUNSGRD(const char *name, char *command, istream &str);
    coDistributedObject *readPOINTS(const char *name, char *command, istream &str);
    coDistributedObject *readUSTSDT(const char *name, char *command, istream &str);
    coDistributedObject *readUSTVDT(const char *name, char *command, istream &str);
    coDistributedObject *readSETELE(const char *name, char *command, istream &str);
    coDistributedObject *readTRIANG(const char *name, char *command, istream &str);
    coDistributedObject *readUNIGRD(const char *name, char *command, istream &str);
    coDistributedObject *readRCTGRD(const char *name, char *command, istream &str);
    coDistributedObject *readSTRGRD(const char *name, char *command, istream &str);
    coDistributedObject *readSTRSDT(const char *name, char *command, istream &str);
    coDistributedObject *readSTRVDT(const char *name, char *command, istream &str);
    coDistributedObject *readRGBADT(const char *name, char *command, istream &str);
    coDistributedObject *readINTARR(const char *name, char *command, istream &str);

public:
    // constructor
    WritePolygon(int argc, char *argv[]);

    // destructor
    virtual ~WritePolygon();
};
#endif
