/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_CSV_POLY_H
#define _READ_CSV_POLY_H
/**************************************************************************\ 
 **                                                   	      (C)2015 HLRS **
 **                                                                        **
 ** Description: Simple Reader CSV files storing polygon information       **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: D. Rainer                                                      **
 **                                                                        **
 ** History:                                                               **
 ** December 15         v1                                                 **
 **                                                                        **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;

class ReadCSVPoly : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void quit();

    bool openFile();
    void readFiles();

    //  member data
    const char *coordFilename; // CSV file name
    const char *facesFilename; // CSV file name
    FILE *fp1;
    FILE *fp2;

    coOutputPort *polygonPort;
    coFileBrowserParam *coordFileParam;
    coFileBrowserParam *facesFileParam;

public:
    ReadCSVPoly(int argc, char *argv[]);
    virtual ~ReadCSVPoly();
};
#endif
