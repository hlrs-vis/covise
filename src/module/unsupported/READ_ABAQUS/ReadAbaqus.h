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
#include <fstream.h>

class ReadAbaqus_Stream
{
private:
    ifstream *d_filePtr;
    char *d_filename;
    int d_isBinary;
    int d_bytesInBlock; // number of bytes still in block

    // read n bytes across FTN block
    // return != 0 if istream->bad()
    int readBytesBin(void *buffer, int len);

public:
    ReadAbaqus_Stream(const char *filename);
    int isGood()
    {
        return d_filePtr && *d_filePtr && d_isBinary >= 0;
    }
    int readRec(char *buffer); // return type, convert ascii to binary
    void rewind();
};

class ReadAbaqus : public coModule
{

private:
    //  member functions
    virtual int compute();
    void readFile(ReadAbaqus_Stream *abaqIn);

    //  member data
    coOutputPort *p_mesh;
    coOutputPort *p_data;
    coFileBrowserParam *p_fileParam;

public:
    ReadAbaqus();
    virtual ~ReadAbaqus();
};
#endif
