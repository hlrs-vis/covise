/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_ACOUSTICS_H
#define _READ_ACOUSTICS_H
/**************************************************************************\ 
 **                                   (C)2000 VirCinity IT-Consulting GmbH **
 **                                                                        **
 ** Description: Simple Reader for Head Audio Akustik	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: A. Wierse                                                      **
 **                                                                        **
 ** History:                                                               **
 ** May 00           v1                                                    **
 **                                                                        **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;

class ReadAcoustics : public coModule
{

private:
    //  member functions
    virtual int compute();
    virtual void quit();

    int openFile();
    void readFile();

    //  member data
    const char *filename; // obj file name
    FILE *fp;
    int xdim, ydim;

    coOutputPort *matrix1Port;
    coOutputPort *matrix2Port;
    coOutputPort *data1Port;
    coOutputPort *data2Port;
    coFileBrowserParam *objFileParam;

public:
    ReadAcoustics();
    virtual ~ReadAcoustics();
};
#endif
