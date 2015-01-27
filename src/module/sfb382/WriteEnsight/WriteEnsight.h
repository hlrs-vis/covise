/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _WRITE_ENSIGHT_H
#define _WRITE_ENSIGHT_H
/**************************************************************************\ 
 **                                                           (C)2001 RUS  **
 **                                                                        **
 ** Description: Write coDoLines in Ensight ASCII data format               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                     Juergen Schulze-Doebold                            **
 **     High Performance Computing Center University of Stuttgart          **
 **                         Allmandring 30                                 **
 **                         70550 Stuttgart                                **
 **                                                                        **
 ** Cration Date: 06.04.01                                                 **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <api/coSimpleModule.h>
using namespace covise;

class coWriteEnsight : public coModule
{
private:
    coFileBrowserParam *pa_filename; // file name of ASCII output file
    coInputPort *po_lines; // receive line data
    coInputPort *po_colors; // receive colors data
    int timesteps; // number of time steps

    int compute();
    void quit();
    void writeGeometryFile(FILE *);
    void writeVariablesFile(FILE *);
    void writeCaseFile(FILE *, char *, char *);
    void strcpyTail(char *, const char *, char);

public:
    coWriteEnsight();
    ~coWriteEnsight();
};
#endif
